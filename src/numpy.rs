use ndarray::{ArrayD, Axis};

pub fn sum_axes<T>(p: &ArrayD<T>, mut inds: Vec<usize>) -> ArrayD<T>
where
    T: num_traits::Float,
{
    let mut sum_arr = p.clone();
    inds.sort();
    for (i, axis) in inds.into_iter().enumerate() {
        sum_arr = sum_arr.sum_axis(Axis(axis - i));
    }
    sum_arr
}

pub fn sum_axes_keepdims<T>(p: &ArrayD<T>, mut inds: Vec<usize>) -> ArrayD<T>
where
    T: num_traits::Float,
{
    let mut sum_arr = p.clone();
    let mut current_shape: Vec<usize> = p.shape().to_vec();
    inds.sort();
    for axis in inds.into_iter() {
        current_shape[axis] = 1;
        sum_arr = sum_arr
            .sum_axis(Axis(axis))
            .to_shape(current_shape.clone())
            .unwrap()
            .into_owned();
    }
    sum_arr
}

pub fn squeeze<T>(p: &ArrayD<T>) -> ArrayD<T>
where
    T: num_traits::Float,
{
    let new_shape: Vec<usize> = p.shape().iter().filter(|&&x| x != 1).copied().collect();
    let squeezed = p
        .to_shape(new_shape)
        .expect("Failed to squeeze array")
        .into_owned();
    squeezed
}

pub fn argsort<T>(v: &[T]) -> Vec<usize>
where
    T: PartialOrd,
{
    let mut indices: Vec<usize> = (0..v.len()).collect();

    indices.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).expect("Failed to compare"));

    indices
}

pub fn indices_where<T>(
    input: impl IntoIterator<Item = T>,
    mut where_: impl FnMut(T) -> bool,
) -> impl Iterator<Item = usize> {
    input
        .into_iter()
        .enumerate()
        .filter_map(move |(index, elt)| where_(elt).then_some(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array};

    #[test]
    fn test_sum_axes() {
        let p = array![
            [2., 2., 5., 67.],
            [4., 2., 6., 7.],
            [1., 23., 6., 98.],
            [2., 2., 5., 67.]
        ];

        let s = sum_axes(&p.into_dyn(), vec![1]);
        assert!(s.sum() == 299.0);
    }

    #[test]
    fn test_sum_axes_keepdims() {
        let p = array![
            [2., 2., 5., 67.],
            [4., 2., 6., 7.],
            [1., 23., 6., 98.],
            [2., 2., 5., 67.]
        ];

        let s = sum_axes_keepdims(&p.into_dyn(), vec![1]);
        assert!(s.sum() == 299.0);
        assert!(s.ndim() == 2);
    }

    #[test]
    fn test_squeeze() {
        let p: Array<f64, _> = Array::ones((5, 1, 5, 1, 5, 1)).into_dyn();
        let squeezed = squeeze(&p);
        assert!(squeezed.shape().len() == 3);
    }

    #[test]
    fn test_argsort() {
        let p = array![2., 2., 5., 67., 4.];
        let p_sort = argsort(&p.to_vec());
        assert!(p_sort == vec![0, 1, 4, 2, 3]);
    }

    #[test]
    fn test_indices_where() {
        let v = vec![3, 42, 3, 5, 7, 30];
        let mut r = Vec::new();
        for x in indices_where(&v, |&val| val > 20) {
            r.push(x);
        }

        assert!(r == vec![1, 5]);
    }
}