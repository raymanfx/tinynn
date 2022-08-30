use std::{
    fs::File,
    io::{BufRead, BufReader},
    ops::{Mul, Sub},
};

use num_traits::{FromPrimitive, Zero};
use plotters::{prelude::*, style::full_palette::ORANGE};

use tinynn::{stat, Array};

fn load_boston(path: &str) -> (Vec<Vec<f32>>, Vec<f32>) {
    // Boston housing prices dataset has 13 distinct features (plus one for price).
    const FEATURES: usize = 13;

    // open the dataset on the filesystem
    let file = File::open(path).unwrap();
    // collect all lines which start with a float
    let data = BufReader::new(file).lines().into_iter().filter_map(|line| {
        let line = if let Ok(line) = line {
            line
        } else {
            return None;
        };

        match line.split_whitespace().peekable().peek() {
            Some(elem) => match elem.parse::<f32>() {
                Ok(_) => Some(line),
                Err(_) => None,
            },
            None => None,
        }
    });

    // Map each line into a vector of floats.
    // If you think of the resulting vector as a 2D array, the rows represent the observations and
    // the columns represent the features.
    let (data, target) = data
        .into_iter()
        .map(|line| {
            let mut elems = line.split_whitespace();

            let mut data = Vec::new();
            for _ in 0..FEATURES {
                let next = elems.next().unwrap();
                data.push(next.parse::<f32>().unwrap());
            }
            let target = elems.next().unwrap().parse::<f32>().unwrap();

            assert_eq!(data.len(), FEATURES);
            (data, target)
        })
        .fold(
            (Vec::new(), Vec::new()),
            |(mut data, mut target), (x, y)| {
                data.push(x);
                target.push(y);
                (data, target)
            },
        );

    (data, target)
}

fn main() {
    // parse the dataset (Boston house-price data of Harrison and Rubinfeld)
    let (data, target) = load_boston("tinynn/examples/boston.txt");

    // compute stats
    let data_iter = data.iter().map(|row| row.iter().copied());
    let means = stat::mean2(data_iter.clone(), 1);
    let vars = stat::var2(data_iter.clone(), 1, 1.0);
    let stds = stat::std2(data_iter.clone(), 1, 1.0);
    println!("n={}", data.len());
    println!("means={:?}", means);
    println!("var={:?}", vars);
    println!("stdev={:?}", stds);

    // standardize data: remove the mean and scale to unit variance
    // z = (x - u) / s
    // where u: mean, s: standard deviation
    let data: Vec<Vec<f32>> = data
        .into_iter()
        .map(|row| {
            row.into_iter()
                .zip(means.iter().zip(stds.iter()))
                .map(|(x, (mean, stddev))| (x - mean) / stddev)
                .collect()
        })
        .collect();

    // split into train and test data sets
    let (x_train, x_test) = data.iter().enumerate().fold(
        (Vec::new(), Vec::new()),
        |(mut train, mut test), (i, row)| {
            let row = row.clone();
            if i % 3 == 0 {
                test.push(row);
            } else {
                train.push(row);
            }

            (train, test)
        },
    );
    let (y_train, y_test) = target.iter().enumerate().fold(
        (Vec::new(), Vec::new()),
        |(mut train, mut test), (i, row)| {
            let row = row.clone();
            if i % 3 == 0 {
                test.push(row);
            } else {
                train.push(row);
            }

            (train, test)
        },
    );

    assert_eq!(x_train.len(), y_train.len());
    let x_train_len = x_train.len();
    let y_train_len = y_train.len();
    let y_test_len = y_test.len();

    // input
    let x = Array::from(x_train);
    // target
    let y = Array::from(y_train).reshape(vec![y_train_len, 1]);
    // weights
    let mut w = Array::new(vec![x.shape()[1], 1], 1.0);
    // intercept
    let mut b = 0.0;

    // forward pass
    fn forward<T, A>(x_batch: Array<T, A>, weights: Array<T, A>, bias: T) -> Array<T, Vec<T>>
    where
        T: Mul<Output = T> + Copy + Mul + Zero,
        A: AsRef<[T]>,
    {
        // matrix dot product of observations and weights
        let N = x_batch.dot(&weights);
        // add the bias to compute the predicitons
        let P = N + bias;
        P
    }

    // backward pass
    fn backward<T, A>(
        x_batch: Array<T, A>,
        y_batch: Array<T, A>,
        predictions: Array<T, A>,
    ) -> (Array<T, Vec<T>>, T)
    where
        T: Mul<Output = T> + Sub<Output = T> + Copy + Zero + FromPrimitive,
        A: AsRef<[T]>,
        Array<T, Vec<T>>:
            Mul<T, Output = Array<T, Vec<T>>> + Mul<Array<T, Vec<T>>, Output = Array<T, Vec<T>>>,
    {
        // 1. Loss function with respect to matrix product plus intercept (P)
        // L(P, Y)  = (Y - P)^2
        // L'(P, Y) = (Y - P) * 2 * -1
        let dL_dP = (y_batch.clone() - predictions) * T::from_i8(-2).unwrap();

        // 2. Sum with respect to matrix product (N)
        // P(N, B)  = N + B
        // P'(N, B) = 1
        let dP_dN = Array::new(vec![x_batch.shape()[0], 1], T::from_u8(1).unwrap());

        // 3. Sum with respect to bias (B)
        // P(N, B)  = N + B
        // P'(N, B) = 1
        let dP_dB = T::from_u8(1).unwrap();

        // 4. Matrix product with respect to weights (W)
        // N(x, w)  = x * w
        // N'(x, w) = x (transposed)
        let dN_dW = x_batch.transpose(vec![(1, 0)]);

        let dL_dN = dL_dP.clone() * dP_dN;
        let dL_dW = dN_dW.dot(&dL_dN);
        let dL_dB = (dL_dP * dP_dB)
            .as_ref()
            .iter()
            .fold(T::zero(), |accum, x| accum + *x);

        (dL_dW, dL_dB)
    }

    // train both, the weights and the bias
    let batch_size = 23;
    let learning_rate = 0.001;
    for i in 0..1000 {
        // choose batches
        let mut start = i % x_train_len;
        let diff = start + batch_size;
        if diff > x_train_len {
            start -= diff - x_train_len;
        }
        let end = start + batch_size;
        let x_batch = x.as_matrix().slice(start..end);
        let y_batch = y.as_matrix().slice(start..end);

        // make a prediction using the current parameters
        let predictions = forward(x_batch.view(), w.view(), b);

        // calculate the loss aka difference between predictions and targets
        let loss = stat::mse(predictions.iter(), y_batch.iter());

        // calculate the gradients
        let (grad_weights, grad_bias) =
            backward(x_batch.view(), y_batch.view(), predictions.view());

        // adjust parameters using gradients
        w = w - grad_weights * learning_rate;
        b = b - grad_bias * learning_rate;

        println!("i: {}, loss: {}", i, loss);
    }

    // calculate the final predictions
    let x = Array::from(x_test.clone());
    let y = Array::from(y_test.clone()).reshape(vec![y_test_len, 1]);
    let predictions = x.dot(&w) + b;
    println!("MAE  : {}", stat::mae(predictions.iter(), y.iter()));
    println!("RMSE : {}", stat::rmse(predictions.iter(), y.iter()));

    // get the most important feature
    // This is only possible since we are doing a linear regression *and* we scaled the features
    // before passing them through our model.
    let most_important_feature =
        w.iter()
            .enumerate()
            .fold((0_usize, 0.0_f32), |mut accum, (i, item)| {
                let abs_coeff = item.abs();
                if abs_coeff > accum.1 {
                    accum.0 = i;
                    accum.1 = abs_coeff;
                }
                accum
            });

    // gather the original data points for the most important feature
    let x_test_means = stat::mean2(x_test.clone(), 1);
    let x_test_important_feature = x_test.iter().cloned().map(|mut row| {
        (&mut row).into_iter().enumerate().for_each(|(i, feature)| {
            if i != most_important_feature.0 {
                *feature = x_test_means[i];
            }
        });
        row
    });

    let x = Array::from(x_test_important_feature.collect::<Vec<Vec<f32>>>());
    let predictions_most_important_feature = x.dot(&w) + b;

    let root_drawing_area =
        BitMapBackend::new("tinynn/examples/linear_regression.png", (1000, 1500))
            .into_drawing_area()
            .titled("Bostn Housing Prices", ("sans-serif", 60))
            .unwrap();

    // paint the canvas
    root_drawing_area.fill(&WHITE).unwrap();

    // divide into upper and lower half
    let (upper, lower) = root_drawing_area.split_vertically(750);

    let mut chart = ChartBuilder::on(&upper)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("Model Performance", ("sans-serif", 40))
        .build_cartesian_2d(0_f32..50_f32, 0_f32..50_f32)
        .unwrap();

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Prediction")
        .y_desc("Target")
        .axis_desc_style(("sans-serif", 15))
        .draw()
        .unwrap();

    chart
        .draw_series(
            predictions
                .iter()
                .zip(y.as_ref().iter().cloned())
                .map(|point| Circle::new(point, 5i32, *&BLUE.filled())),
        )
        .unwrap()
        .label("Prediction vs. Target")
        .legend(|(x, y)| Circle::new((x, y), 5i32, *&BLUE.filled()));

    chart
        .draw_series(LineSeries::new(
            (0..50).map(|x| (x as f32, x as f32)),
            &ORANGE,
        ))
        .unwrap()
        .label("Perfect fit")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &ORANGE));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()
        .unwrap();

    let mut chart = ChartBuilder::on(&lower)
        .margin(5)
        .set_all_label_area_size(50)
        .caption(
            "Most important feature vs target / predictions",
            ("sans-serif", 40),
        )
        .build_cartesian_2d(-2_f32..4_f32, 0_f32..50_f32)
        .unwrap();

    chart
        .draw_series(
            x_test
                .iter()
                .map(|row| row[most_important_feature.0])
                .zip(y_test.iter().cloned())
                .map(|point| Circle::new(point, 5i32, *&BLUE.filled())),
        )
        .unwrap()
        .label("Prediction vs. Target")
        .legend(|(x, y)| Circle::new((x, y), 5i32, *&BLUE.filled()));

    chart
        .draw_series(LineSeries::new(
            x_test
                .iter()
                .map(|row| row[most_important_feature.0])
                .zip(predictions_most_important_feature.iter()),
            &ORANGE,
        ))
        .unwrap()
        .label("Perfect fit")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &ORANGE));

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Most important feature (normalized)")
        .y_desc("Target / Predictions")
        .axis_desc_style(("sans-serif", 15))
        .draw()
        .unwrap();
}
