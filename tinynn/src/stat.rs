use num_traits::{Float, FromPrimitive, Pow, Zero};
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
        + Float,
{
    var(values, ddof).sqrt()
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
        + Float,
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
}
