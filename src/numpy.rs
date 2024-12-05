use ndarray::{s, Array, Array1, Array2, ArrayD, Axis, IxDyn};
use ndarray_stats::QuantileExt;

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

pub fn argsort1d<T>(v: &Array1<T>) -> Vec<usize>
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

pub fn apply_mask<T>(arr: &Array1<T>, mask: &Array1<bool>) -> Array1<T>
where
    T: Copy,
{
    Array::from_iter(arr.iter().zip(mask).filter(|&(_, m)| *m).map(|(d, _)| *d))
}

pub fn diff1d<T>(arr: &Array1<T>) -> Array1<T>
where
    T: num_traits::Float + num_traits::Num,
{
    let s1 = arr.slice(s![1..]);
    let s2 = arr.slice(s![..-1]);
    &s1 - &s2
}

pub fn histogramdd<T>(data: &Array2<T>, bins: &[usize]) -> (ArrayD<usize>, Vec<Vec<T>>)
where
    T: num_traits::Float,
{
    let ndim = data.shape()[1];
    assert_eq!(
        ndim,
        bins.len(),
        "The number of dimensions must match the number of bins"
    );

    // Compute range for each dimension
    let mut edges: Vec<Vec<T>> = Vec::new();
    for (dim, &bin) in bins.iter().enumerate() {
        let col = data.slice(s![.., dim]);
        let min = col.min().expect("Unable to find min in col");
        let max = col.max().expect("Unable to find max in col");
        let step = (*max - *min) / T::from(bin).unwrap();
        edges.push(
            (0..(bins[dim] + 1))
                .map(|i| *min + T::from(i).unwrap() * step)
                .collect(),
        );
    }

    // Bin the data points
    let mut histogram = ArrayD::<usize>::zeros(IxDyn(bins));
    for row in data.outer_iter() {
        let mut index = Vec::with_capacity(ndim);
        for (dim, &val) in row.iter().enumerate() {
            let bin_edges = &edges[dim];
            let bin = match bin_edges.binary_search_by(|edge| {
                edge.partial_cmp(&val)
                    .expect("Unable to perform compare to find edge for value")
            }) {
                Ok(i) => i.min(bins[dim] - 1),
                Err(i) => i.saturating_sub(1).min(bins[dim] - 1),
            };
            index.push(bin);
        }
        *histogram.get_mut(IxDyn(&index)).unwrap() += 1;
    }

    (histogram, edges)
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
        let p_sort = argsort1d(&p);
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

    #[test]
    fn test_apply_mask() {
        let v = array![3, 42, 3, 5, 7, 30];
        let mask = array![true, false, false, true, false, true];
        let a = apply_mask(&v, &mask);
        assert!(a == array![3, 5, 30]);
    }

    #[test]
    fn test_diff() {
        let v = array![1.0, 2.0, 4.0, 5.0];
        let d = diff1d(&v);
        assert!(d == array![1.0, 2.0, 1.0]);
    }

    #[test]
    fn test_histogramdd() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 1.5, 2.5, 3.0, 3.5, 2.0, 1.0, 3.0, 2.0],
        )
        .unwrap();

        let bins = vec![4, 4];

        let (hist, edges) = histogramdd(&data, &bins);

        let answer: ArrayD<usize> =
            array![[0, 1, 0, 0,], [0, 0, 1, 0,], [1, 0, 0, 0,], [0, 1, 0, 1]].into_dyn();
        assert!(hist == answer);
        assert!(edges[0] == vec![1.0, 1.5, 2.0, 2.5, 3.0]);
        assert!(edges[1] == vec![1.0, 1.625, 2.25, 2.875, 3.5]);
    }
}
