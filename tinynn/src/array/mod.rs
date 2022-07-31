/// A generic n-dimensional array.
///
/// This kind of array owns its data.
/// Elements are stored in a linear buffer which can be grown and shrinked as needed.
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

    /// Returns the shape of the array.
    ///
    /// Each axis is represented by an integer.
    /// E.g. `vec![4]` indicates an one-dimensional array with four elements, while for example
    ///      `vec![3, 3]` represents a matrix with three rows and three columns.
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    /// Returns a flat representation of the array. All dimensions are squashed into a single,
    /// contiguous vector.
    pub fn flat(&self) -> &Vec<T> {
        &self.buf
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
        buf.extend_from_slice(&data[0]);
        buf.extend_from_slice(&data[1]);
        let shape = vec![data[0].len(), data[1].len()];
        Array { buf, shape }
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
        assert_eq!(array.shape()[0], 3);
        assert_eq!(array.shape()[1], 3);
    }

    #[test]
    fn flat() {
        let data = [1, 2, 3];
        let array: Array<u8> = Array::from(data);
        assert_eq!(array.flat(), &data);
    }
}
