use std::{
    mem,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use crate::array::Array;

/// Matrix representation of an array.
///
/// This struct cannot be instantiated.
/// Instead, use the `as_matrix()` or `as_mut_matrix` methods of `Array<T>`.
pub struct Matrix<A> {
    pub(crate) array: A,
}

impl<T> Matrix<&Array<T>> {
    // Returns the rows of the matrix.
    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        let rows = self.array.shape[0];
        let mut slices = Vec::new();
        for i in 0..rows {
            slices.push(&self[i]);
        }
        slices.into_iter()
    }
}

impl<T> Matrix<&mut Array<T>> {
    // Returns the rows of the matrix, allowing for mutation.
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        let rows = self.array.shape[0];
        let mut slices = Vec::new();
        for i in 0..rows {
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

impl<T> Index<usize> for Matrix<&Array<T>> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let cols = self.array.shape[1];
        let offset = index * cols;
        &self.array.buf[offset..offset + cols]
    }
}

impl<T> Index<usize> for Matrix<&mut Array<T>> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        let cols = self.array.shape[1];
        let offset = index * cols;
        &self.array.buf[offset..offset + cols]
    }
}

impl<T> IndexMut<usize> for Matrix<&mut Array<T>> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let cols = self.array.shape[1];
        let offset = index * cols;
        &mut self.array.buf[offset..offset + cols]
    }
}

impl<T> Deref for Matrix<&Array<T>> {
    type Target = Array<T>;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl<T> Deref for Matrix<&mut Array<T>> {
    type Target = Array<T>;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl<T> DerefMut for Matrix<&mut Array<T>> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.array
    }
}
