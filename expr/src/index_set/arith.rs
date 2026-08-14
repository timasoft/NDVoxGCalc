use std::iter::FusedIterator;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use super::IndexSet;
use crate::ArithIndexSetTryFromError;

/// A type around [`IndexSet`] providing BigUint-like arithmetic.
///
/// Treats the bitset as an unsigned integer:
/// `Add` / `AddAssign` perform binary addition with carry propagation;
/// `Sub` / `SubAssign` perform binary subtraction with borrow (panics on underflow).
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
#[expect(clippy::module_name_repetitions)]
pub struct ArithIndexSet(pub IndexSet);

impl From<IndexSet> for ArithIndexSet {
    #[inline]
    fn from(set: IndexSet) -> Self {
        Self(set)
    }
}

impl From<ArithIndexSet> for IndexSet {
    #[inline]
    fn from(set: ArithIndexSet) -> Self {
        set.0
    }
}

impl From<u8> for ArithIndexSet {
    #[inline]
    fn from(value: u8) -> Self {
        Self(IndexSet::Small(u32::from(value)))
    }
}

impl From<u16> for ArithIndexSet {
    #[inline]
    fn from(value: u16) -> Self {
        Self(IndexSet::Small(u32::from(value)))
    }
}

impl From<u32> for ArithIndexSet {
    #[inline]
    fn from(value: u32) -> Self {
        Self(IndexSet::Small(value))
    }
}

impl From<u64> for ArithIndexSet {
    #[inline]
    fn from(value: u64) -> Self {
        Self(IndexSet::Medium(value))
    }
}

impl From<u128> for ArithIndexSet {
    #[inline]
    fn from(value: u128) -> Self {
        Self(IndexSet::Large(value))
    }
}

impl From<usize> for ArithIndexSet {
    #[inline]
    #[cfg(target_pointer_width = "32")]
    fn from(value: usize) -> Self {
        Self(IndexSet::Small(value as u32))
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    fn from(value: usize) -> Self {
        Self(IndexSet::Medium(value as u64))
    }
}

#[inline]
const fn add_small(a: u32, b: u32) -> (IndexSet, bool) {
    let (wrapped, overflow) = a.overflowing_add(b);
    if overflow {
        (IndexSet::Medium((1_u64 << 32) | (wrapped as u64)), true)
    } else {
        (IndexSet::Small(wrapped), false)
    }
}

#[inline]
const fn add_medium(a: u64, b: u64) -> (IndexSet, bool) {
    let (wrapped, overflow) = a.overflowing_add(b);
    if overflow {
        (IndexSet::Large((1_u128 << 64) | (wrapped as u128)), true)
    } else {
        (IndexSet::Medium(wrapped), false)
    }
}

#[inline]
fn add_large(a: u128, b: u128) -> (IndexSet, bool) {
    let (wrapped, overflow) = a.overflowing_add(b);
    if overflow {
        (
            IndexSet::Heap(vec![wrapped as u64, (wrapped >> 64) as u64, 1]),
            true,
        )
    } else {
        (IndexSet::Large(wrapped), false)
    }
}

#[inline]
fn add_heap(mut a: Vec<u64>, b: u64) -> (IndexSet, bool) {
    let mut carry = b;
    for item in &mut a {
        let (sum, ov) = item.overflowing_add(carry);
        *item = sum;
        carry = u64::from(ov);
        if carry == 0 {
            break;
        }
    }
    let overflow = carry > 0;
    if overflow {
        a.push(carry);
    }
    (IndexSet::Heap(a), overflow)
}

#[inline]
fn sub_heap(a: Vec<u64>, b: u64) -> IndexSet {
    let mut result = a;
    let mut borrow = b;
    for item in &mut result {
        let (diff, ov) = item.overflowing_sub(borrow);
        *item = diff;
        borrow = u64::from(ov);
        if borrow == 0 {
            break;
        }
    }
    debug_assert_eq!(
        borrow, 0,
        "subtraction underflow should have been caught earlier"
    );
    IndexSet::Heap(result)
}

impl ArithIndexSet {
    /// Convert to `u128` if the value fits in 128 bits.
    ///
    /// Returns `None` for `Heap` variants with more than 2 chunks.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::ArithIndexSet;
    /// let a = ArithIndexSet::from(42u64);
    /// assert_eq!(a.to_u128(), Some(42u128));
    /// ```
    #[inline]
    #[must_use]
    pub fn to_u128(&self) -> Option<u128> {
        match &self.0 {
            IndexSet::Small(v) => Some(u128::from(*v)),
            IndexSet::Medium(v) => Some(u128::from(*v)),
            IndexSet::Large(v) => Some(*v),
            IndexSet::Heap(v) => {
                let required = v.iter().rposition(|&chunk| chunk != 0).map_or(0, |i| i + 1);
                match required {
                    0 => Some(0),
                    #[expect(clippy::indexing_slicing)]
                    1 => Some(u128::from(v[0])),
                    #[expect(clippy::indexing_slicing)]
                    2 => Some(u128::from(v[0]) | (u128::from(v[1]) << 64)),
                    _ => None,
                }
            }
        }
    }

    /// Integer addition with overflow. Returns `(sum, overflowed)`
    ///
    /// `overflowed` is `true` if the result promoted to a larger variant.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::ArithIndexSet;
    /// let a = ArithIndexSet::from(u32::MAX);
    /// let b = ArithIndexSet::from(1u32);
    /// let (sum, overflowed) = a.overflowing_add(b);
    /// assert_eq!(sum, ArithIndexSet::from(u32::MAX as u64 + 1));
    /// assert!(overflowed);
    /// ```
    #[must_use]
    pub fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let (inner, overflow) =
            match (self.0, rhs.0) {
                (IndexSet::Small(a), IndexSet::Small(b)) => add_small(a, b),
                (IndexSet::Small(b), IndexSet::Medium(a))
                | (IndexSet::Medium(a), IndexSet::Small(b)) => add_medium(a, u64::from(b)),
                (IndexSet::Medium(a), IndexSet::Medium(b)) => add_medium(a, b),
                (IndexSet::Small(b), IndexSet::Large(a))
                | (IndexSet::Large(a), IndexSet::Small(b)) => add_large(a, u128::from(b)),
                (IndexSet::Medium(b), IndexSet::Large(a))
                | (IndexSet::Large(a), IndexSet::Medium(b)) => add_large(a, u128::from(b)),
                (IndexSet::Large(a), IndexSet::Large(b)) => add_large(a, b),
                (IndexSet::Small(b), IndexSet::Heap(a))
                | (IndexSet::Heap(a), IndexSet::Small(b)) => add_heap(a, u64::from(b)),
                (IndexSet::Medium(b), IndexSet::Heap(a))
                | (IndexSet::Heap(a), IndexSet::Medium(b)) => add_heap(a, b),
                (a, b) => {
                    let a_vec: Vec<u64> = a.into();
                    let b_vec: Vec<u64> = b.into();
                    let max_len = a_vec.len().max(b_vec.len());
                    let mut result = Vec::with_capacity(max_len + 1);
                    let mut carry = false;
                    for i in 0..max_len {
                        let av = a_vec.get(i).copied().unwrap_or(0);
                        let bv = b_vec.get(i).copied().unwrap_or(0);
                        let (sum, ov1) = av.overflowing_add(bv);
                        let (sum2, ov2) = sum.overflowing_add(u64::from(carry));
                        result.push(sum2);
                        carry = ov1 || ov2;
                    }
                    if carry {
                        result.push(1);
                    }
                    (IndexSet::Heap(result), carry)
                }
            };
        (Self(inner), overflow)
    }

    /// Checked addition. Returns `None` if the result would overflow the
    /// representation (would require promotion to a larger variant).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::ArithIndexSet;
    /// let a = ArithIndexSet::from(5u32);
    /// let b = ArithIndexSet::from(3u32);
    /// assert_eq!(a.checked_add(b), Some(ArithIndexSet::from(8u32)));
    /// let big = ArithIndexSet::from(u32::MAX);
    /// assert_eq!(big.checked_add(ArithIndexSet::from(1u32)), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        let (result, overflow) = self.overflowing_add(rhs);
        if overflow {
            None
        } else {
            Some(Self(result.0.minimized()))
        }
    }

    /// Integer subtraction with underflow. Returns `(diff, underflow)`
    ///
    /// `underflow` is `true` if `self < rhs` (result clamped to 0).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::ArithIndexSet;
    /// let a = ArithIndexSet::from(3u32);
    /// let b = ArithIndexSet::from(5u32);
    /// let (diff, underflow) = a.overflowing_sub(b);
    /// assert_eq!(diff, ArithIndexSet::from(0u32));
    /// assert!(underflow);
    /// ```
    #[must_use]
    pub fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        if self.0 < rhs.0 {
            return (Self(IndexSet::Small(0)), true);
        }
        let inner = match (self.0, rhs.0) {
            (IndexSet::Small(a), IndexSet::Small(b)) => IndexSet::Small(a - b),
            (IndexSet::Small(a), IndexSet::Medium(b)) => IndexSet::Small(a - b as u32),
            (IndexSet::Medium(a), IndexSet::Small(b)) => IndexSet::Medium(a - u64::from(b)),
            (IndexSet::Medium(a), IndexSet::Medium(b)) => IndexSet::Medium(a - b),
            (IndexSet::Small(a), IndexSet::Large(b)) => IndexSet::Small(a - b as u32),
            (IndexSet::Large(a), IndexSet::Small(b)) => IndexSet::Large(a - u128::from(b)),
            (IndexSet::Medium(a), IndexSet::Large(b)) => IndexSet::Medium(a - b as u64),
            (IndexSet::Large(a), IndexSet::Medium(b)) => IndexSet::Large(a - u128::from(b)),
            (IndexSet::Large(a), IndexSet::Large(b)) => IndexSet::Large(a - b),
            (IndexSet::Small(a), IndexSet::Heap(b)) => {
                IndexSet::Small(a - b.first().copied().unwrap_or(0) as u32)
            }
            (IndexSet::Heap(a), IndexSet::Small(b)) => sub_heap(a, u64::from(b)),
            (IndexSet::Medium(a), IndexSet::Heap(b)) => {
                IndexSet::Medium(a - b.first().copied().unwrap_or(0))
            }
            (IndexSet::Heap(a), IndexSet::Medium(b)) => sub_heap(a, b),
            (a, b) => {
                let a_vec: Vec<u64> = a.into();
                let b_vec: Vec<u64> = b.into();
                let mut result = Vec::with_capacity(a_vec.len());
                let mut borrow = false;
                for i in 0..a_vec.len() {
                    let av = a_vec.get(i).copied().unwrap_or(0);
                    let bv = b_vec.get(i).copied().unwrap_or(0);
                    let (diff, ov1) = av.overflowing_sub(bv);
                    let (diff2, ov2) = diff.overflowing_sub(u64::from(borrow));
                    result.push(diff2);
                    borrow = ov1 || ov2;
                }
                debug_assert!(
                    !borrow,
                    "subtraction underflow should have been caught earlier"
                );
                IndexSet::Heap(result)
            }
        }
        .minimized();
        (Self(inner), false)
    }

    /// Checked subtraction. Returns `None` if `self < rhs` (underflow).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::ArithIndexSet;
    /// let a = ArithIndexSet::from(8u32);
    /// let b = ArithIndexSet::from(3u32);
    /// assert_eq!(a.clone().checked_sub(b.clone()), Some(ArithIndexSet::from(5u32)));
    /// assert!(b.checked_sub(a).is_none());
    /// ```
    #[inline]
    #[must_use]
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        let (result, underflow) = self.overflowing_sub(rhs);
        if underflow { None } else { Some(result) }
    }

    /// Creates an iterator over `self..end` (exclusive).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::ArithIndexSet;
    /// let start = ArithIndexSet::from(2u32);
    /// let end = ArithIndexSet::from(5u32);
    /// let values: Vec<ArithIndexSet> = start.range(end).collect();
    /// assert_eq!(values.len(), 3);
    /// assert_eq!(values[0], ArithIndexSet::from(2u32));
    /// assert_eq!(values[2], ArithIndexSet::from(4u32));
    /// ```
    #[inline]
    #[must_use]
    pub const fn range(self, end: Self) -> RangeIter {
        RangeIter { start: self, end }
    }

    /// Creates an iterator over `self..` (infinite, starting from `self`).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::ArithIndexSet;
    /// let values: Vec<ArithIndexSet> = ArithIndexSet::from(3u32).range_from().take(3).collect();
    /// assert_eq!(values, vec![
    ///     ArithIndexSet::from(3u32),
    ///     ArithIndexSet::from(4u32),
    ///     ArithIndexSet::from(5u32),
    /// ]);
    /// ```
    #[inline]
    #[must_use]
    pub const fn range_from(self) -> RangeFrom {
        RangeFrom { current: self }
    }

    /// Creates an iterator over `0..self` (exclusive, from zero).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::ArithIndexSet;
    /// let results: Vec<ArithIndexSet> = ArithIndexSet::from(4u32).range_to().collect();
    /// assert_eq!(results.len(), 4);
    /// assert_eq!(results[0], ArithIndexSet::from(0u32));
    /// assert_eq!(results[3], ArithIndexSet::from(3u32));
    /// ```
    #[inline]
    #[must_use]
    pub fn range_to(self) -> RangeIter {
        RangeIter {
            start: Self::default(),
            end: self,
        }
    }
}

macro_rules! impl_try_from_arith {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl TryFrom<ArithIndexSet> for $ty {
                type Error = ArithIndexSetTryFromError;

                #[inline]
                fn try_from(value: ArithIndexSet) -> Result<Self, Self::Error> {
                    let v = value.to_u128().ok_or(ArithIndexSetTryFromError::Overflow)?;
                    <$ty>::try_from(v).map_err(|_| ArithIndexSetTryFromError::Overflow)
                }
            }
        )+
    };
}

macro_rules! impl_try_from_int_to_arith {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl TryFrom<$ty> for ArithIndexSet {
                type Error = ArithIndexSetTryFromError;

                #[inline]
                fn try_from(value: $ty) -> Result<Self, Self::Error> {
                    if value < 0 {
                        Err(ArithIndexSetTryFromError::Negative)
                    } else {
                        Ok(ArithIndexSet::from(value.unsigned_abs()))
                    }
                }
            }
        )+
    };
}

impl_try_from_arith!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
impl_try_from_int_to_arith!(i8, i16, i32, i64, i128, isize);

impl std::ops::Deref for ArithIndexSet {
    type Target = IndexSet;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for ArithIndexSet {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Add for ArithIndexSet {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.overflowing_add(rhs).0
    }
}

impl AddAssign for ArithIndexSet {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self) + rhs;
    }
}

impl Sub for ArithIndexSet {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (result, underflow) = self.overflowing_sub(rhs);
        assert!(!underflow, "attempt to subtract with overflow");
        result
    }
}

impl SubAssign for ArithIndexSet {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self) - rhs;
    }
}

/// A range of [`ArithIndexSet`] values that can be iterated over.
///
/// Created with [`ArithIndexSet::range`].
#[derive(Debug, Clone)]
pub struct RangeIter {
    start: ArithIndexSet,
    end: ArithIndexSet,
}

/// An infinite iterator over [`ArithIndexSet`] values starting from a given value.
///
/// Created with [`ArithIndexSet::range_from`].
#[derive(Debug, Clone)]
pub struct RangeFrom {
    current: ArithIndexSet,
}

impl Iterator for RangeIter {
    type Item = ArithIndexSet;

    #[inline]
    fn next(&mut self) -> Option<ArithIndexSet> {
        if self.start < self.end {
            let current = self.start.clone();
            self.start += ArithIndexSet::from(1_u32);
            Some(current)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let diff =
            if let (Some(end_val), Some(start_val)) = (self.end.to_u128(), self.start.to_u128()) {
                end_val.saturating_sub(start_val).into()
            } else {
                self.end
                    .clone()
                    .checked_sub(self.start.clone())
                    .unwrap_or_default()
            };
        usize::try_from(diff).map_or((usize::MAX, None), |len| (len, Some(len)))
    }
}

impl DoubleEndedIterator for RangeIter {
    #[inline]
    fn next_back(&mut self) -> Option<ArithIndexSet> {
        if self.start < self.end {
            self.end -= ArithIndexSet::from(1_u32);
            Some(self.end.clone())
        } else {
            None
        }
    }
}

impl FusedIterator for RangeIter {}

impl Iterator for RangeFrom {
    type Item = ArithIndexSet;

    #[inline]
    fn next(&mut self) -> Option<ArithIndexSet> {
        let current = self.current.clone();
        self.current += ArithIndexSet::from(1_u32);
        Some(current)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}

impl FusedIterator for RangeFrom {}
