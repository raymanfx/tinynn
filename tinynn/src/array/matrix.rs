use std::{
    marker::PhantomData,
    mem,
    ops::{Index, IndexMut, Range},
};

use crate::array::Array;

/// Matrix representation of an array.
///
/// This struct cannot be instantiated.
/// Instead, use the `as_matrix()` or `as_mut_matrix()` methods of `Array<T>`.
pub struct Matrix<T, A> {
    pub(crate) _marker: PhantomData<T>,
    pub(crate) array: A,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl<T, A> Matrix<T, A>
where
    T: Copy,
    A: AsRef<[T]>,
{
    /// Returns a partial view of the original matrix.
    ///
    /// # Arguments
    ///
    /// * `range` - Range specifying which rows to expose
    pub fn slice(&self, range: Range<usize>) -> Array<T, Vec<T>> {
        assert!(range.end <= self.rows);
        let array: Vec<T> = self
            .rows()
            .enumerate()
            .filter(|&(i, _)| i >= range.start && i < range.end)
            .map(&|(_, x)| x)
            .fold(Vec::new(), |mut accum, row| {
                accum.extend(row);
                accum
            });
        let shape = vec![range.len(), self.cols];
        Array {
            _marker: PhantomData,
            array,
            shape,
        }
    }

    // Returns the rows of the matrix.
    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        let mut slices = Vec::new();
        for i in 0..self.rows {
            slices.push(&self[i]);
        }
        slices.into_iter()
    }
}

impl<T, A> Matrix<T, A>
where
    T: Copy,
    A: AsRef<[T]> + AsMut<[T]>,
{
    /// Returns a partial view of the original matrix, allowing for mutation.
    ///
    /// # Arguments
    ///
    /// * `range` - Range specifying which rows to expose
    pub fn slice_mut(&self, range: Range<usize>) -> Array<T, Vec<T>> {
        assert!(range.end <= self.rows);
        let array: Vec<T> = self
            .rows()
            .enumerate()
            .filter(|&(i, _)| i >= range.start && i < range.end)
            .map(&|(_, x)| x)
            .fold(Vec::new(), |mut accum, row| {
                accum.extend(row);
                accum
            });
        let shape = vec![range.len(), self.cols];
        Array {
            _marker: PhantomData,
            array,
            shape,
        }
    }

    // Returns the rows of the matrix, allowing for mutation.
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        let mut slices = Vec::new();
        for i in 0..self.rows {
            // This is safe because...
            // (from http://stackoverflow.com/questions/25730586):
            // The Rust compiler does not know that when you ask a mutable iterator for the next
            // element, that you get a different reference every time and never the same reference
            // twice. Of course, we know that such an iterator won't give you the same reference twice.
            let slice = unsafe { mem::transmute(&mut self[i]) };
            slices.push(slice);
        }
        slices.into_iter()
    }
}

impl<T, A> From<Array<T, A>> for Matrix<T, A> {
    fn from(array: Array<T, A>) -> Self {
        assert_eq!(array.shape.len(), 2);
        let rows = array.shape[0];
        let cols = array.shape[1];
        Matrix {
            _marker: PhantomData,
            array: array.array,
            rows,
            cols,
        }
    }
}

impl<T, A> Index<usize> for Matrix<T, A>
where
    A: AsRef<[T]>,
{
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let array = self.array.as_ref();
        let offset = index * self.cols;
        &array[offset..offset + self.cols]
    }
}

impl<T, A> IndexMut<usize> for Matrix<T, A>
where
    A: AsRef<[T]> + AsMut<[T]>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let array = self.array.as_mut();
        let offset = index * self.cols;
        &mut array[offset..offset + self.cols]
    }
}
