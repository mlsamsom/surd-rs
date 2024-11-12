use ndarray_stats::histogram::errors::BinNotFound;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    // -- Information Theory Sub-routines
    UnequalBinDims { nbins: usize, arr_dim: usize },
    NoMinOrMaxValue,
    FailedToGenerateHist,

    // -- External Modules
    BinNotFound(String),
}

impl From<BinNotFound> for Error {
    fn from(val: BinNotFound) -> Self {
        Self::BinNotFound(val.to_string())
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error> {
        write!(fmt, "{self:?}")
    }
}

impl std::error::Error for Error {}
