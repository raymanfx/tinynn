use std::mem;

mod matrix;
use matrix::Matrix;

mod ops;

/// A generic n-dimensional array.
///
/// This kind of array owns its data.
/// Elements are stored in a linear buffer which can be grown and shrinked as needed.
#[derive(Debug, Clone)]
pub struct Array<T> {
    buf: Vec<T>,
    shape: Vec<usize>,
}

impl<T> Array<T> {
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
    /// let array = Array::<f32>::new(vec![2, 2], 0.0);
    /// ```
    pub fn new(shape: Vec<usize>, value: T) -> Array<T>
    where
        T: Clone,
    {
        let len = shape
            .clone()
            .into_iter()
            .reduce(|accum, item| accum * item)
            .expect("invalid shape");

        let mut buf = Vec::new();
        buf.resize(len, value);
        Array { buf, shape }
    }

    /// Returns an empty array.
    ///
    /// # Examples
    ///
    /// ```
    /// use tinynn::Array;
    /// let array = Array::<f32>::empty();
    /// ```
    pub fn empty() -> Array<T> {
        Array {
            buf: Vec::new(),
            shape: Vec::new(),
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
    pub fn reshape(mut self, shape: Vec<usize>) -> Array<T> {
        let len = shape
            .clone()
            .into_iter()
            .reduce(|accum, item| accum * item)
            .expect("invalid shape");
        assert_eq!(self.buf.len(), len);
        self.shape = shape;
        self
    }

    /// Returns the backing buffer.
    pub fn as_slice(&self) -> &[T] {
        &self.buf
    }

    /// Returns the backing buffer, allowing for mutation.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buf
    }

    /// Returns a 2D representation on the array.
    ///
    /// Panics if the shape is not 2-dimensional.
    pub fn as_matrix(&self) -> Matrix<&Self> {
        assert_eq!(self.shape.len(), 2);
        Matrix { array: self }
    }

    /// Returns a 2D representation on the array, allowing for mutation.
    ///
    /// Panics if the shape is not 2-dimensional.
    pub fn as_mut_matrix(&mut self) -> Matrix<&mut Self> {
        assert_eq!(self.shape.len(), 2);
        Matrix { array: self }
    }

    /// Returns the shape of the array.
    ///
    /// Each axis is represented by an integer.
    /// E.g. `vec![4]` indicates an one-dimensional array with four elements, while for example
    ///      `vec![3, 3]` represents a matrix with three rows and three columns.
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    /// Returns an iterator over the array elements.
    pub fn iter(&self) -> Iter<T> {
        Iter {
            array: self,
            offset: 0,
        }
    }

    /// Returns an iterator that allows modifying each value.
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            array: self,
            offset: 0,
        }
    }
}

impl<T> Array<T>
where
    T: Copy,
{
    /// Returns the transpose of the array.
    ///
    /// This operation might affect the internal shape. In some cases, elements will be
    /// reordered (copied) as well.
    ///
    /// # Arguments
    ///
    /// * `axes` - A Vec describing which axes to swap
    pub fn transpose(mut self, axes: Vec<(usize, usize)>) -> Array<T> {
        // for vectors (1D), this is a noop
        if self.shape.len() == 1 {
            return self;
        }

        // for matrices (2D), we just do standard matrix transpose
        if self.shape.len() == 2 {
            // if this is a square matrix, we can just swap its elements
            if self.shape[0] == self.shape[1] {
                // make sure we handle all the elements we need to handle
                let rows = self.shape[0];
                let mut matrix = self.as_mut_matrix();
                for i in 1..rows {
                    for j in 0..i {
                        let tmp = matrix[i][j];
                        matrix[i][j] = matrix[j][i];
                        matrix[j][i] = tmp;
                    }
                }
                return self;
            }

            // this is not a square matrix ..
            // allocate an output matrix and perform the naive transpose
            let mut output = Array::new(vec![self.shape[1], self.shape[0]], self.buf[0]);

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

impl<T, const D1: usize> From<[T; D1]> for Array<T> {
    fn from(data: [T; D1]) -> Array<T> {
        let shape = vec![data.len()];
        Array {
            buf: Vec::from(data),
            shape,
        }
    }
}

impl<T, const D1: usize, const D2: usize> From<[[T; D1]; D2]> for Array<T>
where
    T: Clone,
{
    fn from(data: [[T; D1]; D2]) -> Array<T> {
        let mut buf = Vec::with_capacity(D1 * D2);
        for i in 0..D2 {
            buf.extend_from_slice(&data[i]);
        }
        let shape = vec![D2, D1];
        Array { buf, shape }
    }
}

impl<T> From<Vec<T>> for Array<T> {
    fn from(buf: Vec<T>) -> Array<T> {
        let shape = vec![buf.len()];
        Array { buf, shape }
    }
}

impl<T> From<Vec<Vec<T>>> for Array<T> {
    fn from(buf: Vec<Vec<T>>) -> Array<T> {
        if buf.is_empty() || buf[0].is_empty() {
            return Array::empty();
        }

        let rows = buf.len();
        let cols = buf[0].len();
        let buf = buf
            .into_iter()
            .reduce(|mut accum, item| {
                assert_eq!(item.len(), cols);
                accum.extend(item);
                accum
            })
            .unwrap();
        let shape = vec![rows, cols];
        Array { buf, shape }
    }
}

pub struct Iter<'a, T> {
    array: &'a Array<T>,
    offset: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.offset;
        self.offset += 1;

        if offset == self.array.buf.len() {
            return None;
        }

        Some(&self.array.buf[offset])
    }
}

pub struct IterMut<'a, T> {
    array: &'a mut Array<T>,
    offset: usize,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.offset;
        self.offset += 1;

        if offset == self.array.buf.len() {
            return None;
        }

        // This is safe because...
        // (from http://stackoverflow.com/questions/25730586):
        // The Rust compiler does not know that when you ask a mutable iterator for the next
        // element, that you get a different reference every time and never the same reference
        // twice. Of course, we know that such an iterator won't give you the same reference twice.
        unsafe { Some(mem::transmute(&mut self.array.buf[offset])) }
    }
}

#[cfg(test)]
mod tests {
    use super::Array;

    #[test]
    fn new() {
        let array: Array<u8> = Array::new(vec![2, 3], 0);
        assert!(!array.shape().is_empty());
        assert_eq!(array.shape(), &vec![2, 3]);
    }

    #[test]
    fn empty() {
        let array: Array<u8> = Array::empty();
        assert!(array.shape().is_empty());
    }

    #[test]
    fn reshape() {
        let array: Array<u8> = Array::new(vec![2, 3], 0);
        assert_eq!(array.shape(), &vec![2, 3]);
        let array = array.reshape(vec![3, 2]);
        assert_eq!(array.shape(), &vec![3, 2]);
    }

    #[test]
    fn from_array_1d() {
        let data = [1, 2, 3];
        let array: Array<u8> = Array::from(data);
        assert_eq!(array.shape().len(), 1);
        assert_eq!(array.shape()[0], 3);
    }

    #[test]
    fn from_array_2d() {
        let data = [[1, 2, 3], [4, 5, 6]];
        let array: Array<u8> = Array::from(data);
        assert_eq!(array.shape().len(), 2);
        assert_eq!(array.shape()[0], 2);
        assert_eq!(array.shape()[1], 3);
    }

    #[test]
    fn transpose_1d() {
        let array: Array<i8> = Array::from([1, 2, -3]);
        let array = array.transpose(vec![]);
        assert_eq!(array.shape(), &vec![3]);
        assert_eq!(array.as_slice(), &vec![1, 2, -3]);
    }

    #[test]
    fn transpose_matrix() {
        let array: Array<i8> = Array::from([[1, 2, -3], [3, 4, -5]]);
        let array = array.transpose(vec![]);
        assert_eq!(array.shape(), &vec![3, 2]);
        let matrix = array.as_matrix();
        assert_eq!(matrix[0], vec![1, 3]);
        assert_eq!(matrix[1], vec![2, 4]);
        assert_eq!(matrix[2], vec![-3, -5]);
    }

    #[test]
    fn transpose_2d_square() {
        let array: Array<i8> = Array::from([[1, 2], [3, 4]]);
        let array = array.transpose(vec![]);
        assert_eq!(array.shape(), &vec![2, 2]);
        let matrix = array.as_matrix();
        assert_eq!(matrix[0], vec![1, 3]);
        assert_eq!(matrix[1], vec![2, 4]);

        let array: Array<i8> = Array::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let array = array.transpose(vec![]);
        assert_eq!(array.shape(), &vec![3, 3]);
        let matrix = array.as_matrix();
        assert_eq!(matrix[0], vec![1, 4, 7]);
        assert_eq!(matrix[1], vec![2, 5, 8]);
        assert_eq!(matrix[2], vec![3, 6, 9]);
    }
}
