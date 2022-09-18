use num_traits::{FromPrimitive, Pow, Zero};
use std::ops::{Add, Div, Sub};

/// Helper: swap rows and cols of a 2-dimensional iterator.
/// This essentially performs a matrix transpose.
///
/// # Arguments
///
/// * `values` - Input values (2-dimensional iterator)
fn swap_rows_cols<T>(values: impl IntoIterator<Item = impl IntoIterator<Item = T>>) -> Vec<Vec<T>> {
    values.into_iter().fold(Vec::new(), |mut accum, row| {
        row.into_iter().enumerate().for_each(|(i, col)| {
            if accum.len() <= i {
                accum.push(Vec::new());
            }
            accum[i].push(col);
        });

        accum
    })
}

/// Returns the sum of all elements.
///
/// # Arguments
///
/// * `values` - Input values
pub fn sum<T>(values: impl IntoIterator<Item = T>) -> (usize, T)
where
    T: Add<Output = T> + Zero,
{
    let (count, sum) = values
        .into_iter()
        .enumerate()
        .fold((0, T::zero()), |(_, accum), (i, x)| (i, accum + x));

    (count + 1, sum)
}

/// Returns the mean of all elements.
///
/// # Arguments
///
/// * `values` - Input values
pub fn mean<T>(values: impl IntoIterator<Item = T>) -> T
where
    T: Add<Output = T> + Zero + Div<T, Output = T> + FromPrimitive,
{
    let (count, sum) = sum(values);
    sum / T::from_usize(count).unwrap()
}

/// Returns the variance of all elements.
///
/// # Arguments
///
/// * `values` - Input values
/// * `ddof` - Delta degrees of freedom (0 for population, 1 for sample)
pub fn var<T>(values: impl IntoIterator<Item = T>, ddof: T) -> T
where
    T: Copy
        + Add<Output = T>
        + Zero
        + Div<T, Output = T>
        + FromPrimitive
        + Sub<Output = T>
        + Pow<u8, Output = T>,
{
    let values: Vec<T> = values.into_iter().collect();
    let mean = mean(values.clone());

    let squared = values.into_iter().map(|x| (x - mean).pow(2));
    let (count, sum) = sum(squared);
    sum / (T::from_usize(count).unwrap() - ddof)
}

/// Returns the standard deviation of all elements.
///
/// # Arguments
///
/// * `values` - Input values
/// * `ddof` - Delta degrees of freedom (0 for population, 1 for sample)
pub fn std<T>(values: impl IntoIterator<Item = T>, ddof: T) -> T
where
    T: Copy
        + Add<Output = T>
        + Div<T, Output = T>
        + FromPrimitive
        + Sub<Output = T>
        + Pow<u8, Output = T>
        + Into<f32>
        + Zero,
{
    let var = var(values, ddof);
    T::from_f32(var.into().sqrt()).unwrap()
}

/// Returns the individual sums of all elements of an axis.
///
/// # Arguments
///
/// * `values` - Input values
/// * `axis` - Axis to process (0 for rows, 1 for cols)
pub fn sum2<T>(
    values: impl IntoIterator<Item = impl IntoIterator<Item = T>>,
    axis: usize,
) -> (Vec<usize>, Vec<T>)
where
    T: Add<Output = T> + Zero,
{
    let values = match axis {
        0 => values
            .into_iter()
            .map(|row| row.into_iter().collect())
            .collect(),
        1 => swap_rows_cols(values),
        _ => panic!("invalid axis"),
    };

    values
        .into_iter()
        .fold((Vec::new(), Vec::new()), |mut accum, col| {
            let (count, sum) = sum(col);
            accum.0.push(count);
            accum.1.push(sum);
            accum
        })
}

/// Returns the individual means of all elements of an axis.
///
/// # Arguments
///
/// * `values` - Input values
/// * `axis` - Axis to process (0 for rows, 1 for cols)
pub fn mean2<T>(
    values: impl IntoIterator<Item = impl IntoIterator<Item = T>>,
    axis: usize,
) -> Vec<T>
where
    T: Copy + Add<Output = T> + Zero + Div<T, Output = T> + FromPrimitive,
{
    let values = match axis {
        0 => values
            .into_iter()
            .map(|row| row.into_iter().collect())
            .collect(),
        1 => swap_rows_cols(values),
        _ => panic!("invalid axis"),
    };

    values.into_iter().fold(Vec::new(), |mut accum, col| {
        accum.push(mean(col));
        accum
    })
}

/// Returns the individual variances of all elements of an axis.
///
/// # Arguments
///
/// * `values` - Input values
/// * `axis` - Axis to process (0 for rows, 1 for cols)
/// * `ddof` - Delta degrees of freedom (0 for population, 1 for sample)
pub fn var2<T>(
    values: impl IntoIterator<Item = impl IntoIterator<Item = T>>,
    axis: usize,
    ddof: T,
) -> Vec<T>
where
    T: Copy
        + Add<Output = T>
        + Zero
        + Div<T, Output = T>
        + FromPrimitive
        + Sub<Output = T>
        + Pow<u8, Output = T>,
{
    let values = match axis {
        0 => values
            .into_iter()
            .map(|row| row.into_iter().collect())
            .collect(),
        1 => swap_rows_cols(values),
        _ => panic!("invalid axis"),
    };

    values.into_iter().fold(Vec::new(), |mut accum, col| {
        accum.push(var(col, ddof));
        accum
    })
}

/// Returns the individual standard deviations of all elements of an axis.
///
/// # Arguments
///
/// * `values` - Input values
/// * `axis` - Axis to process (0 for rows, 1 for cols)
/// * `ddof` - Delta degrees of freedom (0 for population, 1 for sample)
pub fn std2<T>(
    values: impl IntoIterator<Item = impl IntoIterator<Item = T>>,
    axis: usize,
    ddof: T,
) -> Vec<T>
where
    T: Copy
        + Add<Output = T>
        + Div<T, Output = T>
        + FromPrimitive
        + Sub<Output = T>
        + Pow<u8, Output = T>
        + Into<f32>
        + Zero,
{
    let values = match axis {
        0 => values
            .into_iter()
            .map(|row| row.into_iter().collect())
            .collect(),
        1 => swap_rows_cols(values),
        _ => panic!("invalid axis"),
    };

    values.into_iter().fold(Vec::new(), |mut accum, col| {
        accum.push(std(col, ddof));
        accum
    })
}

/// Standardizes a given matrix.
///
/// Remove the mean and scale to unit variance
/// z = (x - u) / s
/// where u: mean, s: standard deviation
///
/// # Arguments
///
/// * `values` - Input values
pub fn standardize<T>(values: impl IntoIterator<Item = T>) -> Vec<T>
where
    T: Copy
        + Add<Output = T>
        + Zero
        + Div<T, Output = T>
        + FromPrimitive
        + Sub<Output = T>
        + Pow<u8, Output = T>
        + Into<f32>,
{
    let values: Vec<T> = values.into_iter().collect();

    // compute stats
    let mean = mean(values.clone());
    let std = std(values.clone(), T::zero());

    values.into_iter().map(|x| (x - mean) / std).collect()
}

/// Standardizes a given matrix.
///
/// Remove the mean and scale to unit variance
/// z = (x - u) / s
/// where u: mean, s: standard deviation
///
/// The outer iterator is supposed to go thorugh rows, the inner one through columns.
/// By default, the mean and standard deviation are computed per column.
///
/// # Arguments
///
/// * `values` - Input values
/// * `axis` - Axis to process (0 for rows, 1 for cols)
pub fn standardize2<T>(
    values: impl IntoIterator<Item = impl IntoIterator<Item = T>>,
    axis: usize,
) -> Vec<Vec<T>>
where
    T: Copy
        + Add<Output = T>
        + Zero
        + Div<T, Output = T>
        + FromPrimitive
        + Sub<Output = T>
        + Pow<u8, Output = T>
        + Into<f32>,
{
    let values = match axis {
        0 => values
            .into_iter()
            .map(|row| row.into_iter().collect())
            .collect(),
        1 => swap_rows_cols(values),
        _ => panic!("invalid axis"),
    };

    // standardize each row
    let values = values.into_iter().map(|row| standardize(row));

    // restore original axis layout if necessary
    match axis {
        0 => values.collect(),
        1 => swap_rows_cols(values),
        _ => panic!("invalid axis"),
    }
}

/// Returns the mean average error.
///
/// # Arguments
///
/// * `preds` - Predictions
/// * `targets` - Target values
pub fn mae<T>(preds: impl IntoIterator<Item = T>, targets: impl IntoIterator<Item = T>) -> f32
where
    T: Copy + Into<f32>,
{
    let (count, error) =
        preds
            .into_iter()
            .zip(targets.into_iter())
            .fold((0, 0.0), |(count, accum), (p, y)| {
                let count = count + 1;
                let accum = accum + f32::abs(y.into() - p.into());
                (count, accum)
            });
    error / count as f32
}

/// Returns the mean squared error.
///
/// # Arguments
///
/// * `preds` - Predictions
/// * `targets` - Target values
pub fn mse<T>(preds: impl IntoIterator<Item = T>, targets: impl IntoIterator<Item = T>) -> f32
where
    T: Into<f32>,
{
    let (count, error) =
        preds
            .into_iter()
            .zip(targets.into_iter())
            .fold((0, 0.0), |(count, accum), (p, y)| {
                let count = count + 1;
                let accum = accum + f32::pow(y.into() - p.into(), 2);
                (count, accum)
            });
    error / count as f32
}

/// Returns the root mean squared error.
///
/// # Arguments
///
/// * `preds` - Predictions
/// * `targets` - Target values
pub fn rmse<T>(preds: impl IntoIterator<Item = T>, targets: impl IntoIterator<Item = T>) -> f32
where
    T: Into<f32>,
{
    mse(preds, targets).sqrt()
}

#[cfg(test)]
mod tests {
    #[test]
    fn sum() {
        let data = vec![1, 2, 3, 4, 5];
        let (_, sum) = super::sum(data);
        assert_eq!(sum, 15);
    }

    #[test]
    fn mean() {
        let data = vec![1, 2, 3, 4, 5];
        let mean = super::mean(data);
        assert_eq!(mean, 3);
    }

    #[test]
    fn var() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let var = super::var(data.clone(), 0.0);
        assert_eq!(var, 10.0 / 5.0);
        let var = super::var(data, 1.0);
        assert_eq!(var, 10.0 / 4.0);
    }

    #[test]
    fn std() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let std = super::std(data.clone(), 0.0);
        assert_eq!(std, (10.0_f32 / 5.0).sqrt());
        let std = super::std(data, 1.0);
        assert_eq!(std, (10.0_f32 / 4.0).sqrt());
    }

    #[test]
    fn standardize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mean = 4.0;
        let _var = 4.0;
        let std = 2.0;
        let standardized = super::standardize(data.clone());
        standardized
            .into_iter()
            .enumerate()
            .for_each(|(i, x)| assert_eq!(x, (data[i] - mean) / std));
    }

    #[test]
    fn standardize2() {
        let data = vec![vec![1.0f32, 7.0f32], vec![2.0f32, 5.0f32]];
        let standardized = super::standardize2(data.clone(), 0);
        let means = super::mean2(data.clone(), 0);
        let stds = super::std2(data.clone(), 0, 0.0);
        standardized.into_iter().enumerate().for_each(|(i, row)| {
            row.into_iter()
                .enumerate()
                .for_each(|(j, x)| assert_eq!(x, (data[i][j] - means[i]) / stds[i]))
        });

        let data = vec![vec![1.0f32, 2.0f32], vec![7.0f32, 5.0f32]];
        let standardized = super::standardize2(data.clone(), 1);
        let means = super::mean2(data.clone(), 1);
        let stds = super::std2(data.clone(), 1, 0.0);
        standardized.into_iter().enumerate().for_each(|(i, row)| {
            row.into_iter()
                .enumerate()
                .for_each(|(j, x)| assert_eq!(x, (data[i][j] - means[j]) / stds[j]));
        });
    }

    #[test]
    fn mae() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![2.0, 4.0, 1.0, 3.0];
        let mae = super::mae(predictions, targets);
        assert_eq!(mae, 6.0 / 4.0);
    }

    #[test]
    fn mse() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![2.0, 4.0, 1.0, 3.0];
        let mse = super::mse(predictions, targets);
        assert_eq!(mse, 10.0 / 4.0);
    }

    #[test]
    fn rmse() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![2.0, 4.0, 1.0, 3.0];
        let rmse = super::rmse(predictions, targets);
        assert_eq!(rmse, (10.0_f32 / 4.0).sqrt());
    }
}
