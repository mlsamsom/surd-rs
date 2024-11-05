pub use crate::error::Result;

pub fn log_ns<T>(v: T) -> T
where
    T: num_traits::Float,
{
    if v != T::zero() && v.is_finite() {
        v.log2()
    } else {
        T::zero()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, Array};

    use super::*;

    #[test]
    fn test_log_ns() {
        let x_f64 = Array::from_vec(vec![1.0_f64, 2.0, 0.0, f64::NAN, f64::INFINITY]);
        let x_f32 = Array::from_vec(vec![1.0_f32, 2.0, 0.0, f32::NAN, f32::INFINITY]);
        assert!(x_f64.mapv(log_ns) == arr1(&[0.0_f64, 1.0, 0.0, 0.0, 0.0]));
        assert!(x_f32.mapv(log_ns) == arr1(&[0.0_f32, 1.0, 0.0, 0.0, 0.0]));
    }
}
