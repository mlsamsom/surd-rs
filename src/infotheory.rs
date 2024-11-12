use std::{
    fmt::Debug,
    ops::{AddAssign, DivAssign},
};

use ndarray::{Array1, ArrayD};
use ndarray_stats::QuantileExt;

pub use crate::error::{Error, Result};

/// Compute the normalized empirical probability distribution
pub fn normalized_histogram<T>(p: &Array1<T>, bins: usize) -> Result<(Array1<T>, Array1<T>)>
where
    T: num_traits::Float,
    T: AddAssign,
    T: DivAssign,
    T: Debug,
    T: ndarray::ScalarOperand,
{
    let min = p.min().expect("Cannot find min of input array");
    let max = p.max().unwrap();
    let bins_float = T::from(bins).expect("Conversion from usize to T failed");

    let bin_width = (*max - *min) / bins_float;

    let mut histogram: Array1<T> = Array1::zeros(bins);
    let mut edges: Array1<T> = Array1::zeros(bins);
    edges[0] = *min;
    for i in 1..bins {
        edges[i] = edges[i - 1] + bin_width;
    }

    for &v in p.iter() {
        let bin_index = (((v - *min) / bin_width).floor())
            .to_usize()
            .expect("Conversion from T to usize failed");

        if bin_index < bins {
            histogram[bin_index] += T::from(1).expect("Conversion to T failed");
        } else {
            histogram[bins - 1] += T::from(1).expect("Conversion to T failed");
        }
    }
    histogram += T::from(1e-14).expect("Failed to convert float to T");
    histogram /= histogram.sum();

    Ok((histogram, edges))
}

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
    use anyhow::Result;
    use assert_approx_eq::assert_approx_eq;
    use ndarray::{arr1, arr2, Array};

    use super::*;

    #[test]
    fn test_histogram() -> Result<()> {
        let a = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let r = normalized_histogram(&a, 3)?;
        //assert!(r.0 == arr1(&[0.375, 0.25, 0.375]));
        assert_approx_eq!(r.0[0] as f64, 0.375f64, 1e-3f64);
        assert_approx_eq!(r.0[1] as f64, 0.25f64, 1e-3f64);
        assert_approx_eq!(r.0[2] as f64, 0.375f64, 1e-3f64);
        Ok(())
    }

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
