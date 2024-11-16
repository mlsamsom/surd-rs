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
use ndarray::{Array1, ArrayD};
use numpy::{argsort, indices_where, squeeze, sum_axes_keepdims};

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
pub fn surd<T>(p: &ArrayD<T>) -> Result<()>
where
    T: num_traits::Float,
    T: AddAssign,
    T: DivAssign,
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
    println!("{:?}", inds);
    let p_s = sum_axes_keepdims(p, inds.clone());
    let mut combs: Vec<Vec<&usize>> = Vec::new();

    // specific mutual information
    let mut i_specific: HashMap<Vec<&usize>, Array1<T>> = HashMap::new();

    for &i in inds.iter() {
        for j in inds.iter().combinations(i) {
            combs.push(j.clone());
            let ind_set: HashSet<usize> = inds.clone().into_iter().collect();
            let j_set: HashSet<usize> = j.clone().into_iter().copied().collect();
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
            let info = sum_axes(
                &(p_a_s * (log_p_s_a - log_p_s)),
                j.iter().map(|&&x| x).collect(),
            );
            i_specific.insert(
                j,
                info.to_shape(info.len())
                    .expect("Failed to change array shape")
                    .to_owned(),
            );
        }
    }

    // Compute mutual info for each combination
    let mut mi: HashMap<Vec<&usize>, T> = HashMap::new();
    for k in i_specific.keys() {
        let p_s_squeezed = squeeze(&p_s).sum();
        mi.insert(k.to_vec(), p_s_squeezed);
    }

    // Redundancy term
    let mut i_redundancy: HashMap<Vec<&usize>, T> = HashMap::new();
    for cc in &combs {
        i_redundancy.insert(
            cc.to_vec(),
            T::from(0.0).expect("Failed to convert 0.0 to T"),
        );
    }

    // Synergy term
    let mut i_synergy: HashMap<Vec<&usize>, T> = HashMap::new();
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

        // Sort specific mutual information
        let i1_argsort = argsort(&i1);
        let mut lab = Vec::new();
        for i_ in &i1_argsort {
            lab.push(combs[*i_].clone());
        }

        let mut lens = Vec::new();
        for l in lab {
            lens.push(l.len());
        }

        // Update specific mutual information based on existing maximum values
        let i1_sorted: Vec<T> = i1_argsort.into_iter().map(|i| i1[i]).collect();

        for l in 1..*lens.iter().max().unwrap() {
            let mut inds_l2: Vec<usize> = Vec::new();
            for x in indices_where(&lens, |&v| v == l + 1) {
                inds_l2.push(x);
            }
        }
    }

    Ok(())
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
