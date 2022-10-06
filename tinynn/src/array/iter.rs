use std::mem;

use crate::array::Array;

impl<T, A> Array<T, A> {
    /// Returns an iterator over the array elements.
    pub fn iter(&self) -> Iter<T, A> {
        let mut axis_offsets = self.shape.clone();
        axis_offsets.fill(0);

        Iter {
            array: self,
            axis_offsets,
        }
    }

    /// Returns an iterator that allows modifying each value.
    pub fn iter_mut(&mut self) -> IterMut<T, A> {
        let mut axis_offsets = self.shape.clone();
        axis_offsets.fill(0);

        IterMut {
            array: self,
            axis_offsets,
        }
    }
}

pub struct Iter<'a, T, A> {
    array: &'a Array<T, A>,
    axis_offsets: Vec<usize>,
}

impl<'a, T, A> Iter<'a, T, A> {
    /// Returns the offset within a linear array given a shape.
    ///
    /// Since we need to handle arbitrary dimensions, this gets a little tricky.
    /// For a three dimensional array of shape `[3, 3]`, the offset would be calculated as
    /// follows:
    ///
    ///     offset = row_index * cols_per_row + col_index
    ///
    /// For example, for the element at `[1, 2]`, that is second row and third column, we get:
    ///
    ///     offset = 1 * 3 + 2 = 5
    ///
    /// Generalizing this to n dimensions yields the following expression:
    ///
    ///     offset = axis_1_index * axis_2_len
    ///            + axis_2_index * axis_3_len
    ///         .. + axis_n-1_index * axis_n_len
    ///            + axis_n_index
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the input array
    /// * `axis_offsets` - Offset per axis
    fn linear_offset(shape: &[usize], axis_offsets: &[usize]) -> Option<usize> {
        let offset = axis_offsets
            .iter()
            .enumerate()
            .fold(0, |accum, (i, index)| {
                let premul = shape.iter().skip(i + 1).fold(1, |accum, len| accum * len);
                accum + index * premul
            });

        if offset <= shape.iter().fold(1, |accum, len| accum * len) {
            Some(offset)
        } else {
            None
        }
    }

    /// Modifies the axis offsets such that the next element of the n-dimensional array can be
    /// computed.
    ///
    /// No bounds checking is performed, so the final axis offsets should be checked
    /// before computing a linear offset from them.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the input array
    /// * `axis_offsets` - Offset per axis
    fn advance_offsets(shape: &[usize], axis_offsets: &mut [usize]) {
        for i in (0..axis_offsets.len()).rev() {
            if axis_offsets[i] < shape[i] - 1 {
                axis_offsets[i] += 1;
                break;
            } else {
                axis_offsets[i] = 0;
            }
        }
    }
}

impl<'a, T, A> Iterator for Iter<'a, T, A>
where
    T: Copy,
    A: AsRef<[T]>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // get the linear offset
        let offset = if let Some(offset) =
            Iter::<T, A>::linear_offset(&self.array.shape, &self.axis_offsets)
        {
            offset
        } else {
            return None;
        };

        // advance the axis indices so that the next element will be accessed the next time this
        // function is called
        Iter::<T, A>::advance_offsets(&self.array.shape, &mut self.axis_offsets);

        Some(self.array.as_ref()[offset])
    }
}

pub struct IterMut<'a, T, A> {
    array: &'a mut Array<T, A>,
    axis_offsets: Vec<usize>,
}

impl<'a, T, A> Iterator for IterMut<'a, T, A>
where
    A: AsMut<[T]>,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        // get the linear offset
        let offset = if let Some(offset) =
            Iter::<T, A>::linear_offset(&self.array.shape, &self.axis_offsets)
        {
            offset
        } else {
            return None;
        };

        // advance the axis indices so that the next element will be accessed the next time this
        // function is called
        Iter::<T, A>::advance_offsets(&self.array.shape, &mut self.axis_offsets);

        // This is safe because...
        // (from http://stackoverflow.com/questions/25730586):
        // The Rust compiler does not know that when you ask a mutable iterator for the next
        // element, that you get a different reference every time and never the same reference
        // twice. Of course, we know that such an iterator won't give you the same reference twice.
        unsafe { Some(mem::transmute(&mut self.array.as_mut()[offset])) }
    }
}

#[cfg(test)]
mod tests {
    use super::Array;

    #[test]
    fn iter_2d() {
        let data = [[1, 2, 3], [4, 5, 6]];
        let array: Array<u8, Vec<u8>> = Array::from(data);
        let mut iter = array.iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), Some(6));
    }

    #[test]
    fn iter_3d() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let array = Array::from(data).reshape(vec![2, 2, 2]);
        let mut iter = array.iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), Some(6));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(8));
    }
}
