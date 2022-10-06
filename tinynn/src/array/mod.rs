use std::marker::PhantomData;

mod matrix;
use matrix::Matrix;

mod iter;
mod ops;

/// A generic n-dimensional array.
///
/// This kind of array owns its data.
/// Elements are stored in a linear array.
#[derive(Debug)]
pub struct Array<T, A> {
    _marker: PhantomData<T>,
    array: A,
    shape: Vec<usize>,
}

impl<T, A> Array<T, A> {
    /// Returns the shape of the array.
    ///
    /// Each axis is represented by an integer.
    /// E.g. `vec![4]` indicates an one-dimensional array with four elements, while for example
    ///      `vec![3, 3]` represents a matrix with three rows and three columns.
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }
}

impl<T> Array<T, Vec<T>> {
    /// Creates an array of the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - A Vec describing the length of each axis
    /// * `value` - Initial value for a new element
    ///
    /// # Examples
    ///
    /// ```
    /// use tinynn::Array;
    /// let array = Array::new(vec![2, 2], 0.0);
    /// ```
    pub fn new(shape: Vec<usize>, value: T) -> Self
    where
        T: Clone,
    {
        let len = shape
            .clone()
            .into_iter()
            .reduce(|accum, item| accum * item)
            .expect("invalid shape");

        let mut array = Vec::new();
        array.resize(len, value);
        Array {
            _marker: PhantomData,
            array,
            shape,
        }
    }

    /// Returns an empty array.
    ///
    /// # Examples
    ///
    /// ```
    /// use tinynn::Array;
    /// let array = Array::<f32, Vec<f32>>::empty();
    /// ```
    pub fn empty() -> Self {
        Array {
            _marker: PhantomData,
            array: Vec::new(),
            shape: Vec::new(),
        }
    }
}

impl<T, A> AsRef<[T]> for Array<T, A>
where
    A: AsRef<[T]>,
{
    fn as_ref(&self) -> &[T] {
        self.array.as_ref()
    }
}

impl<T, A> AsMut<[T]> for Array<T, A>
where
    A: AsMut<[T]>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.array.as_mut()
    }
}

impl<T, A> Array<T, A>
where
    A: AsRef<[T]>,
{
    /// Returns a read-only representation of the array.
    pub fn view(&self) -> Array<T, &[T]> {
        let array = self.array.as_ref();
        let shape = self.shape.clone();
        Array {
            _marker: PhantomData,
            array,
            shape,
        }
    }

    /// Returns a 2D representation on the array.
    ///
    /// Panics if the shape is not 2-dimensional.
    pub fn as_matrix(&self) -> Matrix<T, &[T]> {
        assert_eq!(self.shape.len(), 2);
        let rows = self.shape[0];
        let cols = self.shape[1];
        Matrix {
            _marker: PhantomData,
            array: self.array.as_ref(),
            rows,
            cols,
        }
    }
}

impl<T, A> Array<T, A>
where
    A: AsMut<[T]>,
{
    /// Returns a 2D representation on the array, allowing for mutation.
    ///
    /// Panics if the shape is not 2-dimensional.
    pub fn as_mut_matrix(&mut self) -> Matrix<T, &mut [T]> {
        assert_eq!(self.shape.len(), 2);
        let rows = self.shape[0];
        let cols = self.shape[1];
        Matrix {
            _marker: PhantomData,
            array: self.array.as_mut(),
            rows,
            cols,
        }
    }
}

impl<T, A> Array<T, A>
where
    T: Copy,
    A: AsRef<[T]>,
{
    /// Returns a deep copy of self.
    ///
    /// The resulting array instance owns the element buffer.
    pub fn clone(&self) -> Array<T, Vec<T>> {
        let array = self.array.as_ref().to_vec();
        let shape = self.shape.clone();
        Array {
            _marker: PhantomData,
            array,
            shape,
        }
    }

    /// Change the layout of the array, potentially altering its dimensions.
    ///
    /// Panics if the new shape would yield a different number of elements than the array
    /// currently holds.
    ///
    /// # Arguments
    ///
    /// * `shape` - A Vec describing the length of each axis
    pub fn reshape(mut self, shape: Vec<usize>) -> Self {
        let len = shape
            .clone()
            .into_iter()
            .reduce(|accum, item| accum * item)
            .expect("invalid shape");
        assert_eq!(self.array.as_ref().len(), len);
        self.shape = shape;
        self
    }

    /// Returns the transpose of the array.
    ///
    /// This operation might affect the internal shape. In some cases, elements will be
    /// reordered (copied) as well.
    ///
    /// # Arguments
    ///
    /// * `axes` - A Vec describing which axes to swap
    pub fn transpose(&self, axes: Vec<(usize, usize)>) -> Array<T, Vec<T>> {
        // for vectors (1D), this is a noop
        if self.shape.len() == 1 {
            return Array::from(self.array.as_ref());
        }

        // for matrices (2D), we just do standard matrix transpose
        if self.shape.len() == 2 {
            // if this is a square matrix, we can just swap its elements
            if self.shape[0] == self.shape[1] {
                // make sure we handle all the elements we need to handle
                let rows = self.shape[0];
                let mut output = self.clone();
                let mut matrix = output.as_mut_matrix();
                for i in 1..rows {
                    for j in 0..i {
                        let tmp = matrix[i][j];
                        matrix[i][j] = matrix[j][i];
                        matrix[j][i] = tmp;
                    }
                }
                return output;
            }

            // this is not a square matrix ..
            // allocate an output matrix and perform the naive transpose
            let mut output = Array::new(vec![self.shape[1], self.shape[0]], self.array.as_ref()[0]);

            // transpose the matrix
            let in_mat = self.as_matrix();
            let mut out_mat = output.as_mut_matrix();
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    out_mat[j][i] = in_mat[i][j];
                }
            }
            return output;
        }

        // silence compiler warning about unused variable
        let _ = axes;

        // for n-dimensional arrays, we permute the axes as desired
        panic!(
            "transpose() is not implemented for n-dimensional arrays (shape: {:?})",
            self.shape
        );
    }
}

impl<T, const D1: usize> From<[T; D1]> for Array<T, Vec<T>> {
    fn from(data: [T; D1]) -> Self {
        let shape = vec![data.len()];
        Array {
            _marker: PhantomData,
            array: Vec::from(data),
            shape,
        }
    }
}

impl<T, const D1: usize, const D2: usize> From<[[T; D1]; D2]> for Array<T, Vec<T>>
where
    T: Clone,
{
    fn from(data: [[T; D1]; D2]) -> Self {
        let mut array = Vec::with_capacity(D1 * D2);
        for i in 0..D2 {
            array.extend_from_slice(&data[i]);
        }
        let shape = vec![D2, D1];
        Array {
            _marker: PhantomData,
            array,
            shape,
        }
    }
}

impl<T> From<Vec<T>> for Array<T, Vec<T>>
where
    T: Clone,
{
    fn from(array: Vec<T>) -> Self {
        let shape = vec![array.len()];
        Array {
            _marker: PhantomData,
            array,
            shape,
        }
    }
}

impl<T> From<Vec<Vec<T>>> for Array<T, Vec<T>>
where
    T: Clone,
{
    fn from(array: Vec<Vec<T>>) -> Self {
        let rows = array.len();
        let cols = array[0].len();
        let array = array.into_iter().fold(Vec::new(), |mut accum, row| {
            accum.extend(row);
            accum
        });
        let shape = vec![rows, cols];
        Array {
            _marker: PhantomData,
            array,
            shape,
        }
    }
}

impl<T> From<&[T]> for Array<T, Vec<T>>
where
    T: Clone,
{
    fn from(array: &[T]) -> Self {
        let array: Vec<T> = array.to_vec();
        let shape = vec![array.len()];
        Array {
            _marker: PhantomData,
            array,
            shape,
        }
    }
}

impl<'a, T> From<&'a [T]> for Array<T, &'a [T]> {
    fn from(array: &'a [T]) -> Self {
        let shape = vec![array.len()];
        Array {
            _marker: PhantomData,
            array: array,
            shape,
        }
    }
}

impl<'a, T> From<&'a mut [T]> for Array<T, &'a mut [T]> {
    fn from(array: &'a mut [T]) -> Self {
        let shape = vec![array.len()];
        Array {
            _marker: PhantomData,
            array: array,
            shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Array;

    #[test]
    fn new() {
        let array = Array::new(vec![2, 3], 0);
        assert!(!array.shape().is_empty());
        assert_eq!(array.shape(), &vec![2, 3]);
    }

    #[test]
    fn empty() {
        let array: Array<u8, _> = Array::empty();
        assert!(array.shape().is_empty());
    }

    #[test]
    fn reshape() {
        let array = Array::new(vec![2, 3], 0);
        assert_eq!(array.shape(), &vec![2, 3]);
        let array = array.reshape(vec![3, 2]);
        assert_eq!(array.shape(), &vec![3, 2]);
    }

    #[test]
    fn from_array_1d() {
        let data = [1, 2, 3];
        let array = Array::from(data);
        assert_eq!(array.shape().len(), 1);
        assert_eq!(array.shape()[0], 3);
    }

    #[test]
    fn from_array_2d() {
        let data = [[1, 2, 3], [4, 5, 6]];
        let array: Array<u8, Vec<u8>> = Array::from(data);
        assert_eq!(array.shape().len(), 2);
        assert_eq!(array.shape()[0], 2);
        assert_eq!(array.shape()[1], 3);
    }

    #[test]
    fn transpose_1d() {
        let array = Array::from([1, 2, -3]);
        let array = array.transpose(vec![]);
        assert_eq!(array.shape(), &vec![3]);
        assert_eq!(array.as_ref(), &vec![1, 2, -3]);
    }

    #[test]
    fn transpose_matrix() {
        let array: Array<i8, Vec<i8>> = Array::from([[1, 2, -3], [3, 4, -5]]);
        let array = array.transpose(vec![]);
        assert_eq!(array.shape(), &vec![3, 2]);
        let matrix = array.as_matrix();
        assert_eq!(matrix[0], [1, 3]);
        assert_eq!(matrix[1], [2, 4]);
        assert_eq!(matrix[2], [-3, -5]);
    }

    #[test]
    fn transpose_matrix_square() {
        let array: Array<i8, Vec<i8>> = Array::from([[1, 2], [3, 4]]);
        let array = array.transpose(vec![]);
        assert_eq!(array.shape(), &vec![2, 2]);
        let matrix = array.as_matrix();
        assert_eq!(matrix[0], [1, 3]);
        assert_eq!(matrix[1], [2, 4]);

        let array: Array<i8, Vec<i8>> = Array::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let array = array.transpose(vec![]);
        assert_eq!(array.shape(), &vec![3, 3]);
        let matrix = array.as_matrix();
        assert_eq!(matrix[0], [1, 4, 7]);
        assert_eq!(matrix[1], [2, 5, 8]);
        assert_eq!(matrix[2], [3, 6, 9]);
    }
}
