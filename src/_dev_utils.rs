use ndarray::{Array, Array2};
use ndarray_rand::{rand_distr::Normal, RandomExt};

pub fn mediator(n: usize) -> Array2<f64> {
    let mut q: Array2<f64> = Array::zeros((3, n));
    let w: Array2<f64> = Array::random(
        (3, n),
        Normal::new(0.0, 1.0).expect("Failed to create normal dist"),
    );

    for n in 0..n - 1 {
        q[[0, n + 1]] = f64::sin(q[[1, n]]) + 0.001 * w[[0, n]];
        q[[1, n + 1]] = f64::cos(q[[2, n]]) + 0.01 * w[[1, n]];
        q[[2, n + 1]] = 0.5 * q[[2, n]] + 0.1 * w[[2, n]];
    }

    q
}
