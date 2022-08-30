use num_traits::Zero;
use std::ops::{Add, Mul, Sub};

use crate::array::Array;

impl<T, A> Array<T, A>
where
    T: Copy + Mul<Output = T> + Zero,
    A: AsRef<[T]>,
{
    /// Returns the dot product of the two arrays.
    ///
    /// Both self and rhs must be of the correct shape.
    /// E.g. if both are 1D arrays, they must have the same length.
    /// Similarly, if self is a 2D array with shape M x K, rhs must be of shape K x N and the
    /// resulting output array will be of shape M x N.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right hand side array
    pub fn dot<A2>(&self, rhs: &Array<T, A2>) -> Array<T, Vec<T>>
    where
        A2: AsRef<[T]>,
    {
        // helper: 1D x 1D dot product (inner product of vectors)
        fn dot_1d<T: Copy + Zero + Mul<Output = T>>(lhs: &[T], rhs: &[T]) -> T {
            lhs.iter()
                .zip(rhs.iter())
                .fold(T::zero(), |acc, (lhs, rhs)| acc + (*lhs * *rhs))
        }

        // helper: 2D x 2D dot product (inner product of matrices)
        fn dot_2d<T: Copy + Zero + Mul<Output = T>, A: AsRef<[T]>, A2: AsRef<[T]>>(
            lhs: &Array<T, A>,
            rhs: &Array<T, A2>,
        ) -> Array<T, Vec<T>> {
            // ensure we are dealing with matrices
            assert_eq!(lhs.shape.len(), 2);
            assert_eq!(rhs.shape.len(), 2);

            // ensure compatible shapes
            assert_eq!(lhs.shape[1], rhs.shape[0]);

            // allocate output array
            let mut output = Array::new(vec![lhs.shape[0], rhs.shape[1]], T::zero());

            // perform the actual matrix multiplication
            let mut out_mat = output.as_mut_matrix();
            let lhs_mat = lhs.as_matrix();
            let rhs_mat = rhs.as_matrix();
            for i in 0..lhs.shape[0] {
                for j in 0..rhs.shape[1] {
                    let mut sum = T::zero();
                    for k in 0..rhs.shape[0] {
                        sum = sum + lhs_mat[i][k] * rhs_mat[k][j];
                    }
                    out_mat[i][j] = sum;
                }
            }

            output
        }

        // in case of 1D x 1D, we want the inner product of vectors (aka dot product)
        if self.shape.len() == 1 && rhs.shape.len() == 1 {
            // the axis length has to match
            let lhs_len = self.shape.first().unwrap();
            let rhs_len = rhs.shape.first().unwrap();
            assert_eq!(
                lhs_len,
                rhs_len,
                "1D vector product requires matching axis length (lhs: {:?}, rhs: {:?})",
                self.shape(),
                rhs.shape()
            );

            let dot = dot_1d(self.as_ref(), rhs.as_ref());
            return Array::from([dot]);
        }

        // in case of 2D x 2D, we want the innter product of matrices
        if self.shape.len() == 2 && rhs.shape.len() == 2 {
            // the axis length has to match
            assert_eq!(
                self.shape[1],
                rhs.shape[0],
                "2D matrix product requires matching inner axis length (lhs: {:?}, rhs: {:?})",
                self.shape(),
                rhs.shape()
            );

            return dot_2d(&self, &rhs);
        }

        // mD x nD: tensor contraction requires the last axis of lhs to match the first axis of rhs
        panic!(
            "incompatible shapes, tensor contraction not implemented (lhs: {:?}, rhs: {:?})",
            self.shape(),
            rhs.shape()
        );
    }
}

impl<T, A, A2> Add<Array<T, A2>> for Array<T, A>
where
    T: Copy + Add<Output = T>,
    A: AsMut<[T]>,
    A2: AsRef<[T]>,
{
    type Output = Self;

    fn add(mut self, rhs: Array<T, A2>) -> Self::Output {
        // if the shapes do not match, this operation is illegal
        if self.shape() != rhs.shape() {
            panic!(
                "shapes do not match (lhs: {:?}, rhs: {:?})",
                self.shape(),
                rhs.shape()
            );
        }

        // fill the output array
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(lhs, rhs)| *lhs = *lhs + rhs);
        self
    }
}

impl<T, A> Add<T> for Array<T, A>
where
    T: Copy + Add<Output = T>,
    A: AsMut<[T]>,
{
    type Output = Self;

    fn add(mut self, rhs: T) -> Self::Output {
        // add rhs to each element in self
        self.iter_mut().for_each(|lhs| *lhs = *lhs + rhs);
        self
    }
}

impl<T, A, A2> Mul<Array<T, A2>> for Array<T, A>
where
    T: Copy + Mul<Output = T>,
    A: AsMut<[T]>,
    A2: AsRef<[T]>,
{
    type Output = Self;

    fn mul(mut self, rhs: Array<T, A2>) -> Self::Output {
        // if the shapes do not match, this operation is illegal
        if self.shape() != rhs.shape() {
            panic!(
                "shapes do not match (lhs: {:?}, rhs: {:?})",
                self.shape(),
                rhs.shape()
            );
        }

        // fill the output array
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(lhs, rhs)| *lhs = *lhs * rhs);
        self
    }
}

impl<T, A> Mul<T> for Array<T, A>
where
    T: Copy + Mul<Output = T>,
    A: AsMut<[T]>,
{
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self::Output {
        // multiply rhs with each element in self
        self.iter_mut().for_each(|lhs| *lhs = *lhs * rhs);
        self
    }
}

impl<T, A, A2> Sub<Array<T, A2>> for Array<T, A>
where
    T: Copy + Sub<Output = T>,
    A: AsMut<[T]>,
    A2: AsRef<[T]>,
{
    type Output = Self;

    fn sub(mut self, rhs: Array<T, A2>) -> Self::Output {
        // if the shapes do not match, this operation is illegal
        if self.shape() != rhs.shape() {
            panic!(
                "shapes do not match (lhs: {:?}, rhs: {:?})",
                self.shape(),
                rhs.shape()
            );
        }

        // fill the output array
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(lhs, rhs)| *lhs = *lhs - rhs);
        self
    }
}

impl<T, A> Sub<T> for Array<T, A>
where
    T: Copy + Sub<Output = T>,
    A: AsMut<[T]>,
{
    type Output = Self;

    fn sub(mut self, rhs: T) -> Self::Output {
        // substract rhs from each element in self
        self.iter_mut().for_each(|lhs| *lhs = *lhs - rhs);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::Array;

    #[test]
    fn add() {
        let array = Array::from([1, 2, 3]);
        let array = array + Array::from([2, 1, 0]);
        assert_eq!(array.as_ref(), &vec![3, 3, 3]);
    }

    #[test]
    fn add_scalar() {
        let array = Array::from([1, 2, 3]);
        let array = array + 3;
        assert_eq!(array.as_ref(), &vec![4, 5, 6]);
    }

    #[test]
    fn mul_1d() {
        let array = Array::from([1, 2, 3]);
        let array = array * Array::from([2, 3, 4]);
        assert_eq!(array.as_ref(), &vec![2, 6, 12]);
    }

    #[test]
    fn mul_2d() {
        let array: Array<i32, Vec<i32>> = Array::from([[1, 2], [2, 3]]);
        let array = array * Array::from([[3, 4], [4, 5]]);
        assert_eq!(array.as_ref(), Array::from([[3, 8], [8, 15]]).as_ref());
    }

    #[test]
    fn mul_scalar() {
        let array = Array::from([1, 2, -3]);
        let array = array * -1;
        assert_eq!(array.as_ref(), &vec![-1, -2, 3]);
    }

    #[test]
    fn dot_1d() {
        let array = Array::from([1, 2, 3]);
        let array = array.dot(&Array::from([2, 3, 4]));
        assert_eq!(array.as_ref(), &vec![20]);
    }

    #[test]
    fn dot_2d() {
        let array: Array<i32, Vec<i32>> = Array::from([[1, 2], [2, 3]]);
        let array = array.dot(&Array::from([[3, 4], [4, 5]]));
        assert_eq!(array.as_ref(), Array::from([[11, 14], [18, 23]]).as_ref());

        let array: Array<i32, Vec<i32>> = Array::from([[1, 2], [2, 3], [3, 4]]);
        let array = array.dot(&Array::from([[3, 4, 5], [5, 6, 7]]));
        assert_eq!(
            array.as_ref(),
            Array::from([[13, 16, 19], [21, 26, 31], [29, 36, 43]]).as_ref()
        );
    }

    #[test]
    fn sub() {
        let array = Array::from([1, 2, 3]);
        let array = array - Array::from([2, 1, 0]);
        assert_eq!(array.as_ref(), &vec![-1, 1, 3]);
    }

    #[test]
    fn sub_scalar() {
        let array = Array::from([1, 2, 3]);
        let array = array - 3;
        assert_eq!(array.as_ref(), &vec![-2, -1, 0]);
    }
}
