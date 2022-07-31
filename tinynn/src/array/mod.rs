use std::mem;

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

    /// Returns the backing buffer.
    pub fn as_slice(&self) -> &[T] {
        &self.buf
    }

    /// Returns the backing buffer, allowing for mutation.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buf
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
    /// Creates an array of the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - A Vec describing the length of each axis
    /// * `value` - Initial value for a new element
    pub fn with_shape(shape: Vec<usize>, value: T) -> Array<T> {
        let len = shape
            .clone()
            .into_iter()
            .reduce(|accum, item| accum * item)
            .expect("invalid shape");

        let mut buf = Vec::new();
        buf.resize(len, value);
        Array { buf, shape }
    }

    /// Change the layout of the array, potentially altering its dimensions.
    ///
    /// Panics if the new shape would yield a different number of elements than the array
    /// currently holds.
    ///
    /// # Arguments
    ///
    /// * `shape` - A Vec describing the length of each axis
    pub fn reshape(&mut self, shape: Vec<usize>) {
        let len = shape
            .clone()
            .into_iter()
            .reduce(|accum, item| accum * item)
            .expect("invalid shape");
        assert_eq!(self.buf.len(), len);

        self.shape = shape;
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
    fn empty() {
        let array: Array<u8> = Array::empty();
        assert!(array.shape().is_empty());
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
    fn reshape() {
        let mut array: Array<u8> = Array::with_shape(vec![2, 3], 0);
        array.reshape(vec![3, 2]);
        assert_eq!(array.shape(), &vec![3, 2]);
    }
}
