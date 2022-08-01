use num_traits::Zero;
use std::ops::{Add, Mul};

use crate::array::Array;

impl<T> Add for Array<T>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        // if there are no elements to add, we can bail out early
        if self.shape().len() == 0 || rhs.shape().len() == 0 {
            return self;
        }

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
            .for_each(|(lhs, rhs)| *lhs = *lhs + *rhs);
        self
    }
}

impl<T> Add<T> for Array<T>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(mut self, rhs: T) -> Self::Output {
        // add rhs to each element in self
        self.iter_mut().for_each(|lhs| *lhs = *lhs + rhs);
        self
    }
}

impl<T> Mul for Array<T>
where
    T: Copy + Mul<Output = T> + Zero,
{
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        // if there are no elements to add, we can bail out early
        if self.shape().len() == 0 || rhs.shape().len() == 0 {
            return self;
        }

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
            .for_each(|(lhs, rhs)| *lhs = *lhs * *rhs);
        self
    }
}

impl<T> Mul<T> for Array<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self::Output {
        // multiply rhs with each element in self
        self.iter_mut().for_each(|lhs| *lhs = *lhs * rhs);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::Array;

    #[test]
    fn add() {
        let array: Array<u8> = Array::from([1, 2, 3]);
        let array = array + Array::from([2, 1, 0]);
        assert_eq!(array.as_slice(), &vec![3, 3, 3]);
    }

    #[test]
    fn add_scalar() {
        let array: Array<u8> = Array::from([1, 2, 3]);
        let array = array + 3;
        assert_eq!(array.as_slice(), &vec![4, 5, 6]);
    }

    #[test]
    fn mul_1d() {
        let array: Array<u8> = Array::from([1, 2, 3]);
        let array = array * Array::from([2, 3, 4]);
        assert_eq!(array.as_slice(), &vec![2, 6, 12]);
    }

    #[test]
    fn mul_2d() {
        let array: Array<u8> = Array::from([[1, 2], [2, 3]]);
        let array = array * Array::from([[3, 4], [4, 5]]);
        assert_eq!(array.as_slice(), Array::from([[3, 8], [8, 15]]).as_slice());
    }

    #[test]
    fn mul_scalar() {
        let array: Array<i8> = Array::from([1, 2, -3]);
        let array = array * -1;
        assert_eq!(array.as_slice(), &vec![-1, -2, 3]);
    }
}
