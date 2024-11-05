use ndarray::{Array1, ArrayD};

pub use crate::error::Result;

/// Compute the logarithm in base 2 avoiding singularities
///
/// # Usage
/// let x_f64 = Array::from_vec(vec![1.0_f64, 2.0, 0.0, f64::NAN, f64::INFINITY]);
/// let log_x = x_f64.mapv(log_ns);
/// // log_x = [0.0, 1.0, 0.0, 0.0, 0.0];
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

/// Compute the Shannon entropy of a discrete probability distribution function
/// of arbitrary dimension.
pub fn entropy_any_dim<T>(p: &ArrayD<T>) -> T
where
    T: num_traits::Float,
{
    -(p * p.mapv(log_ns)).sum()
}

/// Compute the Shannon entropy of a 1D discrete probability distribution function
pub fn entropy_1d<T>(p: &Array1<T>) -> T
where
    T: num_traits::Float,
{
    -(p * p.mapv(log_ns)).sum()
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use ndarray::{arr1, arr2, Array};

    use super::*;

    #[test]
    fn test_log_ns() {
        let x_f64 = Array::from_vec(vec![1.0_f64, 2.0, 0.0, f64::NAN, f64::INFINITY]);
        let x_f32 = Array::from_vec(vec![1.0_f32, 2.0, 0.0, f32::NAN, f32::INFINITY]);
        assert!(x_f64.mapv(log_ns) == arr1(&[0.0_f64, 1.0, 0.0, 0.0, 0.0]));
        assert!(x_f32.mapv(log_ns) == arr1(&[0.0_f32, 1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_entropy_any_dim() {
        let a = arr2(&[[1.0, 3.4, 2.4], [1.0, 2.4, 1.4]]);
        let ent = entropy_any_dim(&a.into_dyn());
        assert_approx_eq!(ent, -12.74498f64, 1e-3f64);
    }

    #[test]
    fn test_entropy_1d() {
        let a = arr1(&[1.0, 3.4, 2.4]);
        let ent = entropy_1d(&a);
        assert_approx_eq!(ent, -9.0341007f64, 1e-3f64);
    }
}
