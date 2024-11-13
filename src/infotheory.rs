use std::{
    collections::HashSet,
    fmt::Debug,
    ops::{AddAssign, DivAssign},
};

use ndarray::{Array, Array1, ArrayD, Axis};
use ndarray_stats::QuantileExt;

pub use crate::error::{Error, Result};

/// Compute the normalized empirical probability distribution for a 1D array of data
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
    T: Debug,
{
    -(p * p.mapv(log_ns)).sum()
}

/// Compute the Shannon entropy of a 1D discrete probability distribution function
///
/// # Usage
///
pub fn entropy_1d<T>(p: &Array1<T>) -> T
where
    T: num_traits::Float,
{
    -(p * p.mapv(log_ns)).sum()
}

/// Compute the joint entropy for specific dimensions of a probability distribution.
///
/// # Usage
///
///  Example: compute the joint entropy H(X0,X3,X7)
///  >>> joint_entropy_any_dim(p, vec![0,3,7])
pub fn joint_entropy_any_dim<T>(p: &ArrayD<T>, indices: Vec<usize>) -> T
where
    T: num_traits::Float,
    T: Debug,
{
    let s1: HashSet<usize> = (0..p.ndim()).collect();
    let s2: HashSet<usize> = indices.into_iter().collect();

    let mut excl_indices = s1.difference(&s2).collect::<Vec<&usize>>();
    excl_indices.sort_unstable();

    let mut marg_dist = p.clone();
    for (i, axis) in excl_indices.into_iter().enumerate() {
        marg_dist = marg_dist.sum_axis(Axis(*axis - i));
    }

    entropy_any_dim(&marg_dist)
}

/// Compute the conditional entropy between two sets of variables.
///
/// # Usage
///
/// Example: compute the conditional entropy H(X0,X2|X7)
/// >>> cond_entropy(p, (0, 2), (7,))
pub fn conditional_entropy<T>(
    p: &ArrayD<T>,
    target_indices: Vec<usize>,
    condition_indices: Vec<usize>,
) -> T
where
    T: num_traits::Float,
    T: Debug,
{
    // get indices
    let target_ind_set: HashSet<usize> = target_indices.into_iter().collect();
    let condition_ind_set: HashSet<usize> = condition_indices.clone().into_iter().collect();

    let all_inds: Vec<usize> = target_ind_set
        .union(&condition_ind_set)
        .collect::<Vec<&usize>>()
        .into_iter()
        .copied()
        .collect();

    let joint_entropy = joint_entropy_any_dim(p, all_inds);
    let conditioned_entropy = joint_entropy_any_dim(p, condition_indices);

    joint_entropy - conditioned_entropy
}

/// Compute the mutual information between two sets of variables.
///
/// # Usage
///
/// Example: compute the mutual information I(X0,X5;X4,X2)
/// >>> mutual_info(p, vec![0, 5], vec![4, 2])
pub fn mutual_information<T>(p: &ArrayD<T>, set1_indices: Vec<usize>, set2_indices: Vec<usize>) -> T
where
    T: num_traits::Float,
    T: Debug,
{
    let entropy_set1 = joint_entropy_any_dim(p, set1_indices.clone());
    let conditional_entropy = conditional_entropy(p, set1_indices, set2_indices);

    entropy_set1 - conditional_entropy
}

/// Compute the conditional mutual information between two sets of variables
/// conditioned to a third set.
///
/// # Usage
///
/// Example: compute the conditional mutual information I(X0,X5;X4,X2|X1)
/// cond_mutual_info(p, vec![0, 5], vec![4, 2], vec[1,]))
pub fn conditional_mutual_information<T>(
    p: &ArrayD<T>,
    ind1: Vec<usize>,
    ind2: Vec<usize>,
    ind3: Vec<usize>,
) -> T
where
    T: num_traits::Float,
    T: Debug,
{
    // merge indices 2 and 3
    let ind2_set: HashSet<usize> = ind2.into_iter().collect();
    let ind3_set: HashSet<usize> = ind3.clone().into_iter().collect();

    let combined_inds: Vec<usize> = ind2_set
        .union(&ind3_set)
        .collect::<Vec<&usize>>()
        .into_iter()
        .copied()
        .collect();

    conditional_entropy(p, ind1.clone(), ind3) - conditional_entropy(p, ind1, combined_inds)
}

/// Calculate the transfer entropy from each input variable to the target variable.
pub fn transfer_entropy<T>(p: &ArrayD<T>, target_var: usize) -> Array1<T>
where
    T: num_traits::Float,
    T: Debug,
{
    let num_vars = p.ndim() - 1;
    let mut trans_ent = Array1::zeros(num_vars);

    for i in 1..num_vars + 1 {
        let present_indices: Vec<usize> = (1..num_vars + 1).collect();

        // indices should not contain present variable
        let mut conditioned_indices = vec![target_var];
        for j in 1..num_vars + 1 {
            if j != target_var && j != i {
                conditioned_indices.push(j);
            }
        }

        // conditional entropy of the future state of the target variable given it's own past
        let cond_ent_target_given_past =
            conditional_entropy(p, vec![target_var], conditioned_indices);

        //  Conditional entropy of the future state of the target variable given its own past and the ith input variable
        let cond_ent_target_given_past_and_input =
            conditional_entropy(p, vec![target_var], present_indices);

        trans_ent[i - 1] = cond_ent_target_given_past - cond_ent_target_given_past_and_input;
    }

    trans_ent
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use assert_approx_eq::assert_approx_eq;
    use ndarray::{arr1, arr2, array, Array};

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

    #[test]
    fn test_joint_entropy_any_dim() {
        let p = array![
            [2., 2., 5., 67.],
            [4., 2., 6., 7.],
            [1., 23., 6., 98.],
            [2., 2., 5., 67.]
        ];
        let ent1 = joint_entropy_any_dim(&p.clone().into_dyn(), vec![0]);
        assert_approx_eq!(ent1, -1926.3956f64, 1e-3f64);
    }

    #[test]
    fn test_conditional_entropy() {
        let p: Array<f64, _> = Array::ones((5, 5, 5, 5, 5, 5));
        let ent1 = conditional_entropy(&p.clone().into_dyn(), vec![0, 2], vec![5]);
        assert_approx_eq!(ent1, 72560.252965f64, 1e-3f64);
    }

    #[test]
    fn test_mutual_entropy() {
        let p: Array<f64, _> = Array::ones((5, 5, 5, 5, 5, 5));
        let mi = mutual_information(&p.clone().into_dyn(), vec![0, 5], vec![4, 2]);
        assert_approx_eq!(mi, -217680.75889f64, 1e-3f64);
    }

    #[test]
    fn test_conditional_mutual_entropy() {
        let p: Array<f64, _> = Array::ones((5, 5, 5, 5, 5, 5));
        let mi =
            conditional_mutual_information(&p.clone().into_dyn(), vec![0, 5], vec![4, 2], vec![1]);
        assert_approx_eq!(mi, 0.0, 1e-3f64);
    }

    #[test]
    fn test_transfer_entropy() {
        let p = array![
            [2., 2., 5., 67.],
            [4., 2., 6., 7.],
            [1., 23., 6., 98.],
            [2., 2., 5., 67.]
        ];
        let te = transfer_entropy(&p.clone().into_dyn(), 0);
        assert_approx_eq!(te[0], -498.79567f64, 1e-3f64);
    }
}
