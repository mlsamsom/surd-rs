pub mod _dev_utils;
pub mod error;
pub mod infotheory;
pub mod numpy;

use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    ops::{AddAssign, DivAssign},
};

use infotheory::{conditional_entropy, joint_entropy_any_dim, log_ns, sum_axes};
use itertools::Itertools;
use ndarray::{arr1, concatenate, s, stack, Array, Array1, ArrayD, Axis};
use ndarray_slice::Slice1Ext;
use numpy::{apply_mask, argsort1d, diff1d, indices_where, squeeze, sum_axes_keepdims};

pub use crate::error::{Error, Result};

/// Makes sure there are no zeros in a (all positive or zero) array
/// also normalizes the array
pub fn clean_probability_dist<T>(p: &mut ArrayD<T>) -> Result<()>
where
    T: num_traits::Float,
    T: AddAssign,
    T: DivAssign,
    T: Debug,
    T: ndarray::ScalarOperand,
{
    *p += T::from(1e-14).expect("Failed to convert epsilon to T");
    *p /= p.sum();

    Ok(())
}

pub struct InfoResult<T> {
    pub variables: Vec<usize>,
    pub value: T,
}

pub struct SurdResult<T>
where
    T: num_traits::Float,
{
    pub information_redundancy: Vec<InfoResult<T>>,
    pub information_synergy: Vec<InfoResult<T>>,
    pub mutual_information: Vec<InfoResult<T>>,
    pub information_leak: T,
}

/// Decompose the mutual information between a target variable and a set
/// of agent variables into three terms: Redundancy (I_R), Synergy (I_S),
/// and Unique (I_U) information.
///    
/// The surd function is designed to compute a decomposition of
/// the mutual information between a target variable T
/// (signal in the future) and agent variables A (signals in the present).
/// This decomposition results in terms related to redundancy
/// (overlapping information), synergy (information that arises only when
/// considering multiple variables together), and unique information.
pub fn surd<T>(p: &ArrayD<T>) -> Result<SurdResult<T>>
where
    T: num_traits::Float,
    T: AddAssign,
    T: DivAssign,
    T: PartialOrd,
    T: Debug,
    T: ndarray::ScalarOperand,
{
    // Extract dimension numbers
    let total_dimensions = p.ndim();
    let n_agent_vars = total_dimensions - 1;
    let n_targets = p.shape()[0];
    let inds: Vec<usize> = (1..total_dimensions).collect();

    // Calculate information leak
    let h = joint_entropy_any_dim(p, vec![0]);
    let h_c = conditional_entropy(p, vec![0], inds.clone());
    let info_leak = h / h_c;

    // Compute the marginal distribution of the target
    let p_s = sum_axes_keepdims(p, inds.clone());
    let mut combs: Vec<Vec<usize>> = Vec::new();

    // specific mutual information
    let mut i_specific: HashMap<Vec<usize>, Array1<T>> = HashMap::new();

    for i in &inds {
        for j in inds.clone().into_iter().combinations(*i) {
            combs.push(j.clone());
            let ind_set: HashSet<usize> = inds.clone().into_iter().collect();
            let j_set: HashSet<usize> = j.clone().into_iter().collect();
            let noj = ind_set.difference(&j_set).collect::<Vec<&usize>>();
            let noj: Vec<usize> = noj.iter().map(|&&x| x).collect();

            // Compute joint and conditional distributions for current combination
            let mut noj_w_0: Vec<usize> = vec![0];
            noj_w_0.extend(&noj);
            let p_a = sum_axes_keepdims(p, noj_w_0);
            let p_as = sum_axes_keepdims(p, noj);

            let p_a_s = &p_as / &p_s;
            let p_s_a = &p_as / &p_a;
            let log_p_s_a = p_s_a.mapv(log_ns);
            let log_p_s = p_s.mapv(log_ns);
            let info = sum_axes(&(p_a_s * (log_p_s_a - log_p_s)), j.to_vec());
            i_specific.insert(
                j,
                info.to_shape(info.len())
                    .expect("Failed to change array shape")
                    .to_owned(),
            );
        }
    }

    // Compute mutual info for each combination
    let mut mi: HashMap<Vec<usize>, T> = HashMap::new();
    let p_s_squeezed = squeeze(&p_s);
    let p_s_squeezed_sum = p_s_squeezed.sum();
    for k in i_specific.keys() {
        mi.insert(k.to_vec(), p_s_squeezed_sum);
    }

    // Redundancy term
    let mut i_redundancy: HashMap<Vec<usize>, T> = HashMap::new();
    for cc in &combs {
        i_redundancy.insert(
            cc.to_vec(),
            T::from(0.0).expect("Failed to convert 0.0 to T"),
        );
    }

    // Synergy term
    let mut i_synergy: HashMap<Vec<usize>, T> = HashMap::new();
    for cc in &combs[n_agent_vars..] {
        i_synergy.insert(
            cc.to_vec(),
            T::from(0.0).expect("Failed to convert 0.0 to T"),
        );
    }

    // Process each value of the target variable
    for t in 1..n_targets {
        // Extract specific mutual information for current target
        let mut i1 = Vec::new();
        for ii in i_specific.values() {
            i1.push(ii[t]);
        }
        let mut i1 = Array::from_vec(i1);

        // Sort specific mutual information
        let i1_argsort = argsort1d(&i1);
        let mut lab = Vec::new();
        for i_ in &i1_argsort {
            lab.push(combs[*i_].clone());
        }

        let mut lens = Vec::new();
        for l in &lab {
            lens.push(l.len());
        }

        // Update specific mutual information based on existing maximum values
        i1.sort_by(|a, b| a.partial_cmp(b).expect("Unable to compare during sort"));
        for l in 1..*lens.iter().max().unwrap() {
            let inds_l2 = Array::from_iter(indices_where(&lens, |&v| v == l + 1));

            let mask: Array1<bool> = Array::from_iter(lens.iter().map(|x| *x == l));
            let il1max = apply_mask(&i1, &mask)
                .into_iter()
                .max_by(|a, b| a.partial_cmp(b).expect("Failed to compare in max"))
                .expect("Unable to find max of il");

            let i1_inds_l2: Vec<T> = inds_l2.iter().map(|i| i1[*i]).collect();
            let mask: Array1<bool> = Array::from_iter(i1_inds_l2.iter().map(|x| *x < il1max));
            let inds_ = apply_mask(&inds_l2, &mask);
            for i in inds_ {
                i1[i] = T::from(0.0).expect("Failed to create generic 0.0");
            }
        }

        // Recompute sorting of updated specific mutual information values
        let i1_argsort = argsort1d(&i1);
        let mut lab2 = Vec::new();
        for i in i1_argsort {
            lab2.push(&lab[i]);
        }

        // Compute differences in sorted specific mutual information values
        i1.sort_by(|a, b| a.partial_cmp(b).expect("Unable to compare during sort"));
        let d_i = concatenate(
            ndarray::Axis(0),
            &[
                arr1(&[T::from(0.0).expect("Failed to create generic 0.0")]).view(),
                diff1d(&i1).view(),
            ],
        )
        .expect("Failed to prepend [0]");
        let red_vars: Vec<usize> = inds.clone();

        // Distribute mutual information to redundancy and synergy terms

        for (i, ll) in lab.into_iter().enumerate() {
            let info = d_i[i] * p_s_squeezed[t];
            if ll.len() == 1 {
                i_redundancy.insert(
                    red_vars.clone(),
                    *i_redundancy
                        .get(&red_vars)
                        .expect("Value in i_redundancy does not exist")
                        + info,
                );
            } else {
                i_synergy.insert(
                    ll.clone(),
                    *i_synergy
                        .get(&ll)
                        .expect("Value in i_synergy does not exist")
                        + info,
                );
            }
        }
    }

    // package for output
    let information_redundancy: Vec<InfoResult<T>> = i_redundancy
        .into_iter()
        .map(|(variables, value)| InfoResult {
            variables: variables.clone(),
            value,
        })
        .collect();
    let information_synergy: Vec<InfoResult<T>> = i_synergy
        .into_iter()
        .map(|(variables, value)| InfoResult {
            variables: variables.clone(),
            value,
        })
        .collect();
    let mutual_information: Vec<InfoResult<T>> = mi
        .into_iter()
        .map(|(variables, value)| InfoResult {
            variables: variables.clone(),
            value,
        })
        .collect();

    Ok(SurdResult {
        information_redundancy,
        information_synergy,
        mutual_information,
        information_leak: info_leak,
    })
}

fn run_surd<T>(p: &ArrayD<T>, nlag: usize, nbins: usize)
where
    T: num_traits::Float,
{
    let nvars: usize = p.shape()[0];

    for i in 0..nvars {
        // Organize data (0 target variable, 1: agent variables)
        let r_slice = p.slice(s![i, 1..]).to_owned();
        let c_slice = p.slice(s![.., ..-1]).to_owned();
        let y = stack![Axis(0), r_slice.insert_axis(Axis(0)), c_slice];
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array};

    use super::*;

    #[test]
    fn test_clean_probability_dist() {
        let mut p = array![
            [1., 1., 0., 2.],
            [4., 2., 6., 7.],
            [1., 23., 6., 98.],
            [2., 2., 5., 67.]
        ]
        .into_dyn();
        let _ = clean_probability_dist(&mut p);
        assert!(p.sum() == 1.);
    }

    #[test]
    fn test_surd() {
        let mut p: Array<f64, _> = Array::ones((5, 5, 5, 5, 5, 5)).into_dyn();
        let _ = clean_probability_dist(&mut p);
        let _ = surd(&p);
    }
}
