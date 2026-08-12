//! Bitset and arithmetic types for CSE slot management.
//!
//! [`IndexSet`] tracks which variable slots a sub-expression depends on.
//! [`ArithIndexSet`] wraps it with BigUint-like arithmetic for enumeration.

use std::{
    hash::{Hash, Hasher},
    iter::{ExactSizeIterator, FusedIterator},
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Shl, ShlAssign, Shr, ShrAssign},
};

pub mod arith;
pub use arith::{ArithIndexSet, RangeFrom as ArithRangeFrom, RangeIter as ArithRangeIter};

/// A compact bitset for tracking slot indices in CSE.
///
/// Uses stack-optimized representations for small sizes (up to 32, 64, and 128
/// bits) and falls back to a heap-allocated `Vec<u64>` for larger sets.
#[derive(Debug, Clone, Eq)]
pub enum IndexSet {
    /// Up to 32 bits stored inline.
    Small(u32),
    /// 33–64 bits stored inline.
    Medium(u64),
    /// 65–128 bits stored inline.
    Large(u128),
    /// More than 128 bits, stored as a heap-allocated vector of 64-bit chunks.
    Heap(Vec<u64>),
}

impl Default for IndexSet {
    #[inline]
    fn default() -> Self {
        Self::Small(0)
    }
}

impl IndexSet {
    /// Create a set containing exactly one slot index.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::IndexSet;
    /// let s = IndexSet::singleton(5);
    /// assert!(s.contains(5));
    /// assert!(!s.contains(4));
    /// ```
    #[inline]
    #[must_use]
    pub fn singleton(slot: usize) -> Self {
        #[expect(clippy::indexing_slicing)]
        if slot < 32 {
            Self::Small(1_u32 << slot)
        } else if slot < 64 {
            Self::Medium(1_u64 << slot)
        } else if slot < 128 {
            Self::Large(1_u128 << slot)
        } else {
            let idx = slot / 64;
            let chunks = idx + 1;
            let mut vec = vec![0_u64; chunks];
            vec[idx] |= 1_u64 << (slot % 64);
            Self::Heap(vec)
        }
    }

    /// Insert or remove a slot index.
    ///
    /// Automatically promotes the representation when the slot exceeds the
    /// current variant's capacity.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::IndexSet;
    /// let mut s = IndexSet::default();
    /// s.insert(1, true);
    /// assert!(s.contains(1));
    /// s.insert(1, false);
    /// assert!(!s.contains(1));
    /// ```
    pub fn insert(&mut self, slot: usize, value: bool) {
        match self {
            Self::Small(bits) => {
                if slot < 32 {
                    if value {
                        *bits |= 1_u32 << slot;
                    } else {
                        *bits &= !(1_u32 << slot);
                    }
                } else if value {
                    #[expect(clippy::indexing_slicing)]
                    if slot < 64 {
                        *self = Self::Medium((u64::from(*bits)) | (1_u64 << slot));
                    } else if slot < 128 {
                        *self = Self::Large((u128::from(*bits)) | (1_u128 << slot));
                    } else {
                        let idx = slot / 64;
                        let chunks = idx + 1;
                        let mut vec = vec![0_u64; chunks];
                        vec[0] = u64::from(*bits);
                        vec[idx] |= 1_u64 << (slot % 64);
                        *self = Self::Heap(vec);
                    }
                }
            }
            Self::Medium(bits) => {
                if slot < 64 {
                    if value {
                        *bits |= 1_u64 << slot;
                    } else {
                        *bits &= !(1_u64 << slot);
                    }
                } else if value {
                    #[expect(clippy::indexing_slicing)]
                    if slot < 128 {
                        *self = Self::Large((u128::from(*bits)) | (1_u128 << slot));
                    } else {
                        let idx = slot / 64;
                        let chunks = idx + 1;
                        let mut vec = vec![0_u64; chunks];
                        vec[0] = *bits;
                        vec[idx] |= 1_u64 << (slot % 64);
                        *self = Self::Heap(vec);
                    }
                }
            }
            #[expect(clippy::indexing_slicing)]
            Self::Large(bits) => {
                if slot < 128 {
                    if value {
                        *bits |= 1_u128 << slot;
                    } else {
                        *bits &= !(1_u128 << slot);
                    }
                } else if value {
                    let idx = slot / 64;
                    let chunks = idx + 1;
                    let mut vec = vec![0_u64; chunks];
                    vec[0] = *bits as u64;
                    vec[1] = (*bits >> 64_usize) as u64;
                    vec[idx] |= 1_u64 << (slot % 64);
                    *self = Self::Heap(vec);
                }
            }
            Self::Heap(vec) => {
                let idx = slot / 64;
                let bit = slot % 64;
                #[expect(clippy::indexing_slicing)]
                if value {
                    if idx >= vec.len() {
                        vec.resize(idx + 1, 0);
                    }
                    vec[idx] |= 1_u64 << bit;
                } else if idx < vec.len() {
                    vec[idx] &= !(1_u64 << bit);
                }
            }
        }
    }

    /// Returns `true` if the two sets have no elements in common.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::IndexSet;
    /// let a = IndexSet::singleton(0);
    /// let b = IndexSet::singleton(1);
    /// assert!(a.is_disjoint(&b));
    /// let c = IndexSet::singleton(0);
    /// assert!(!a.is_disjoint(&c));
    /// ```
    #[must_use]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => (a & b) == 0,
            (Self::Small(b), Self::Medium(a)) | (Self::Medium(a), Self::Small(b)) => {
                (a & u64::from(*b)) == 0
            }
            (Self::Small(b), Self::Large(a)) | (Self::Large(a), Self::Small(b)) => {
                (a & u128::from(*b)) == 0
            }
            (Self::Medium(a), Self::Medium(b)) => (a & b) == 0,
            (Self::Medium(b), Self::Large(a)) | (Self::Large(a), Self::Medium(b)) => {
                (a & u128::from(*b)) == 0
            }
            (Self::Large(a), Self::Large(b)) => (a & b) == 0,
            (Self::Heap(a), Self::Heap(b)) => {
                let min_len = a.len().min(b.len());
                for i in 0..min_len {
                    #[expect(clippy::indexing_slicing)]
                    if (a[i] & b[i]) != 0 {
                        return false;
                    }
                }
                true
            }
            (Self::Small(b), Self::Heap(a)) | (Self::Heap(a), Self::Small(b)) => {
                a.first().is_none_or(|&x| (x & u64::from(*b)) == 0)
            }
            (Self::Medium(b), Self::Heap(a)) | (Self::Heap(a), Self::Medium(b)) => {
                a.first().is_none_or(|&x| (x & b) == 0)
            }
            (Self::Large(b), Self::Heap(a)) | (Self::Heap(a), Self::Large(b)) => {
                if a.is_empty() {
                    return true;
                }
                #[expect(clippy::indexing_slicing)]
                if (a[0] & (*b as u64)) != 0 {
                    return false;
                }
                #[expect(clippy::indexing_slicing)]
                if a.len() > 1 && (a[1] & ((*b >> 64) as u64)) != 0 {
                    return false;
                }
                true
            }
        }
    }

    /// Returns `true` if the given slot index is present in the set.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::IndexSet;
    /// let s = IndexSet::singleton(3);
    /// assert!(s.contains(3));
    /// assert!(!s.contains(0));
    /// ```
    #[inline]
    #[must_use]
    pub fn contains(&self, slot: usize) -> bool {
        match self {
            Self::Small(bits) => slot < 32 && (*bits & (1_u32 << slot)) != 0,
            Self::Medium(bits) => slot < 64 && (*bits & (1_u64 << slot)) != 0,
            Self::Large(bits) => slot < 128 && (*bits & (1_u128 << slot)) != 0,
            #[expect(clippy::indexing_slicing)]
            Self::Heap(vec) => {
                let idx = slot / 64;
                let bit = slot % 64;
                idx < vec.len() && (vec[idx] & (1_u64 << bit)) != 0
            }
        }
    }

    /// Iterate over all slot indices present in the set, in ascending order.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::IndexSet;
    /// let mut s = IndexSet::singleton(2);
    /// s.insert(5, true);
    /// let slots: Vec<usize> = s.iter().collect();
    /// assert_eq!(slots, vec![2, 5]);
    /// ```
    #[inline]
    #[must_use]
    pub fn iter(&self) -> IndexSetIter<'_> {
        self.into_iter()
    }

    #[inline]
    fn get_first_chunk(&self) -> u64 {
        match self {
            Self::Small(b) => u64::from(*b),
            Self::Medium(b) => *b,
            Self::Large(b) => *b as u64,
            Self::Heap(v) => v.first().copied().unwrap_or(0),
        }
    }

    #[inline]
    const fn max_chunks(&self) -> usize {
        match self {
            Self::Small(_) | Self::Medium(_) => 1,
            Self::Large(_) => 2,
            Self::Heap(vec) => vec.len(),
        }
    }

    /// Returns the number of slot indices in the set (population count).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::IndexSet;
    /// let mut s = IndexSet::singleton(0);
    /// s.insert(2, true);
    /// s.insert(5, true);
    /// assert_eq!(s.count_ones(), 3);
    /// ```
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> usize {
        match self {
            Self::Small(bits) => bits.count_ones() as usize,
            Self::Medium(bits) => bits.count_ones() as usize,
            Self::Large(bits) => bits.count_ones() as usize,
            Self::Heap(vec) => vec.iter().map(|&x| x.count_ones() as usize).sum(),
        }
    }

    /// Returns `true` if the set contains no elements.
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::IndexSet;
    /// let mut s: IndexSet = Default::default();
    /// assert!(s.is_empty());
    /// s.insert(0, true);
    /// assert!(!s.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Small(bits) => *bits == 0,
            Self::Medium(bits) => *bits == 0,
            Self::Large(bits) => *bits == 0,
            Self::Heap(vec) => vec.iter().all(|&x| x == 0),
        }
    }

    /// Shrink to the smallest variant that can hold the current bits.
    ///
    /// Returns `self` for chaining; see also [`minimize`](Self::minimize).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::IndexSet;
    /// let s = IndexSet::Large(0).minimized();
    /// assert!(matches!(s, IndexSet::Small(_)));
    /// ```
    #[inline]
    #[must_use]
    pub fn minimized(mut self) -> Self {
        self.minimize();
        self
    }

    /// Shrink to the smallest variant that can hold the current bits.
    ///
    /// Mutates `self`; see also [`minimized`](Self::minimized).
    ///
    /// # Examples
    /// ```
    /// # use hypervox_expr::IndexSet;
    /// let mut s = IndexSet::Large(0);
    /// s.minimize();
    /// assert!(matches!(s, IndexSet::Small(_)));
    /// ```
    pub fn minimize(&mut self) {
        *self = match std::mem::take(self) {
            Self::Heap(mut vec) => {
                let last_option = vec.iter().rposition(|&x| x != 0);
                match last_option {
                    None => Self::Small(0),
                    Some(0) => {
                        #[expect(clippy::indexing_slicing)]
                        let v = vec[0];
                        let v_u32 = v as u32;
                        if u64::from(v_u32) == v {
                            Self::Small(v_u32)
                        } else {
                            Self::Medium(v)
                        }
                    }
                    #[expect(clippy::indexing_slicing)]
                    Some(1) => Self::Large(u128::from(vec[0]) | (u128::from(vec[1]) << 64)),
                    Some(last) => {
                        vec.truncate(last.saturating_add(1));
                        Self::Heap(vec)
                    }
                }
            }
            Self::Large(0) | Self::Medium(0) => Self::Small(0),
            Self::Large(v) if u128::from(v as u64) == v => {
                let low = v as u64;
                if u64::from(low as u32) == low {
                    Self::Small(low as u32)
                } else {
                    Self::Medium(low)
                }
            }
            Self::Medium(v) if u64::from(v as u32) == v => Self::Small(v as u32),
            other => other,
        };
    }
}

impl From<IndexSet> for Vec<u64> {
    fn from(set: IndexSet) -> Self {
        match set {
            IndexSet::Small(bits) => vec![u64::from(bits)],
            IndexSet::Medium(bits) => vec![bits],
            IndexSet::Large(bits) => vec![bits as u64, (bits >> 64) as u64],
            IndexSet::Heap(vec) => vec,
        }
    }
}

impl PartialEq for IndexSet {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => a == b,
            (Self::Medium(a), Self::Medium(b)) => a == b,
            (Self::Large(a), Self::Large(b)) => a == b,
            _ => self.iter().eq(other.iter()),
        }
    }
}

impl PartialOrd for IndexSet {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexSet {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Small(a), Self::Small(b)) => a.cmp(b),
            (Self::Medium(a), Self::Medium(b)) => a.cmp(b),
            (Self::Large(a), Self::Large(b)) => a.cmp(b),
            // Compare from the highest bit down — same as integer cmp.
            _ => self.iter().rev().cmp(other.iter().rev()),
        }
    }
}

impl Hash for IndexSet {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.count_ones().hash(state);

        for idx in self {
            idx.hash(state);
        }
    }
}

impl BitOr for IndexSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Small(a), Self::Small(b)) => Self::Small(a | b),
            (Self::Small(b), Self::Medium(a)) | (Self::Medium(a), Self::Small(b)) => {
                Self::Medium(a | u64::from(b))
            }
            (Self::Small(b), Self::Large(a)) | (Self::Large(a), Self::Small(b)) => {
                Self::Large(a | u128::from(b))
            }
            (Self::Small(b), Self::Heap(mut a)) | (Self::Heap(mut a), Self::Small(b)) => {
                #[expect(clippy::indexing_slicing)]
                if a.is_empty() {
                    a.push(u64::from(b));
                } else {
                    a[0] |= u64::from(b);
                }
                Self::Heap(a)
            }
            (Self::Medium(a), Self::Medium(b)) => Self::Medium(a | b),
            (Self::Medium(b), Self::Large(a)) | (Self::Large(a), Self::Medium(b)) => {
                Self::Large(a | u128::from(b))
            }
            (Self::Medium(b), Self::Heap(mut a)) | (Self::Heap(mut a), Self::Medium(b)) => {
                #[expect(clippy::indexing_slicing)]
                if a.is_empty() {
                    a.push(b);
                } else {
                    a[0] |= b;
                }
                Self::Heap(a)
            }
            (Self::Large(a), Self::Large(b)) => Self::Large(a | b),
            #[expect(clippy::indexing_slicing)]
            (Self::Large(b), Self::Heap(mut a)) | (Self::Heap(mut a), Self::Large(b)) => {
                if a.len() < 2 {
                    a.resize(2, 0);
                }
                a[0] |= b as u64;
                a[1] |= (b >> 64) as u64;
                Self::Heap(a)
            }
            (Self::Heap(mut a), Self::Heap(b)) => {
                if a.len() < b.len() {
                    a.resize(b.len(), 0);
                }
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x |= *y;
                }
                Self::Heap(a)
            }
        }
    }
}

impl BitOrAssign for IndexSet {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self) | rhs;
    }
}

impl BitAnd for IndexSet {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Small(a), Self::Small(b)) => Self::Small(a & b),
            (Self::Small(b), Self::Medium(a)) | (Self::Medium(a), Self::Small(b)) => {
                Self::Small((a & u64::from(b)) as u32)
            }
            (Self::Small(b), Self::Large(a)) | (Self::Large(a), Self::Small(b)) => {
                Self::Small((a & u128::from(b)) as u32)
            }
            #[expect(clippy::indexing_slicing)]
            (Self::Small(b), Self::Heap(a)) | (Self::Heap(a), Self::Small(b)) => {
                if a.is_empty() {
                    Self::default()
                } else {
                    Self::Small((a[0] & u64::from(b)) as u32)
                }
            }
            (Self::Medium(a), Self::Medium(b)) => Self::Medium(a & b),
            (Self::Medium(b), Self::Large(a)) | (Self::Large(a), Self::Medium(b)) => {
                Self::Medium((a & u128::from(b)) as u64)
            }
            #[expect(clippy::indexing_slicing)]
            (Self::Medium(b), Self::Heap(a)) | (Self::Heap(a), Self::Medium(b)) => {
                if a.is_empty() {
                    Self::Medium(0)
                } else {
                    Self::Medium(a[0] & b)
                }
            }
            (Self::Large(a), Self::Large(b)) => Self::Large(a & b),
            (Self::Large(b), Self::Heap(a)) | (Self::Heap(a), Self::Large(b)) => {
                let lo = a.first().map_or(0, |&x| x & (b as u64));
                let hi = a.get(1).map_or(0, |&x| x & ((b >> 64_usize) as u64));
                if hi == 0 {
                    Self::Medium(lo)
                } else {
                    Self::Large(u128::from(lo) | (u128::from(hi) << 64))
                }
            }
            (Self::Heap(mut a), Self::Heap(b)) => {
                let min_len = a.len().min(b.len());
                a.truncate(min_len);
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x &= *y;
                }
                Self::Heap(a)
            }
        }
        .minimized()
    }
}

impl BitAndAssign for IndexSet {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self) & rhs;
    }
}

#[inline]
fn heap_shl(vec: Vec<u64>, rhs: usize) -> IndexSet {
    let chunk_shift = rhs / 64;
    let bit_shift = rhs % 64;
    let mut new_vec = Vec::with_capacity(chunk_shift + vec.len() + 1);
    new_vec.extend(std::iter::repeat_n(0_u64, chunk_shift));
    if bit_shift == 0 {
        new_vec.extend(vec);
    } else {
        let mut carry = 0_u64;
        for v in vec {
            let val = (v << bit_shift) | carry;
            carry = v >> (64 - bit_shift);
            new_vec.push(val);
        }
        if carry != 0 {
            new_vec.push(carry);
        }
    }
    IndexSet::Heap(new_vec)
}

impl Shl<usize> for IndexSet {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self {
        if rhs == 0 || self.is_empty() {
            return self;
        }
        match self {
            Self::Small(bits) => {
                let val = u128::from(bits);
                if (val.leading_zeros() as usize) < rhs {
                    heap_shl(vec![u64::from(bits)], rhs)
                } else {
                    let shifted = val << rhs;
                    if u128::from(shifted as u32) == shifted {
                        Self::Small(shifted as u32)
                    } else if u128::from(shifted as u64) == shifted {
                        Self::Medium(shifted as u64)
                    } else {
                        Self::Large(shifted)
                    }
                }
            }
            Self::Medium(bits) => {
                let val = u128::from(bits);
                if (val.leading_zeros() as usize) < rhs {
                    heap_shl(vec![bits], rhs)
                } else {
                    let shifted = val << rhs;
                    if u128::from(shifted as u64) == shifted {
                        Self::Medium(shifted as u64)
                    } else {
                        Self::Large(shifted)
                    }
                }
            }
            Self::Large(bits) => {
                if (bits.leading_zeros() as usize) < rhs {
                    let lo = bits as u64;
                    let hi = (bits >> 64) as u64;
                    let vec = if hi != 0 { vec![lo, hi] } else { vec![lo] };
                    heap_shl(vec, rhs)
                } else {
                    Self::Large(bits << rhs)
                }
            }
            Self::Heap(vec) => heap_shl(vec, rhs),
        }
    }
}

impl ShlAssign<usize> for IndexSet {
    #[inline]
    fn shl_assign(&mut self, rhs: usize) {
        *self = std::mem::take(self) << rhs;
    }
}

impl Shr<usize> for IndexSet {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self {
        if rhs == 0 || self.is_empty() {
            return self;
        }
        match self {
            Self::Small(bits) => {
                if rhs >= 32 {
                    Self::Small(0)
                } else {
                    Self::Small(bits >> rhs)
                }
            }
            Self::Medium(bits) => {
                if rhs >= 64 {
                    Self::Small(0)
                } else {
                    Self::Medium(bits >> rhs)
                }
            }
            Self::Large(bits) => {
                if rhs >= 128 {
                    Self::Small(0)
                } else {
                    Self::Large(bits >> rhs)
                }
            }
            Self::Heap(vec) => {
                let chunk_shift = rhs / 64;
                let bit_shift = rhs % 64;
                if chunk_shift >= vec.len() {
                    return Self::Small(0);
                }
                #[expect(clippy::indexing_slicing)]
                let remaining = &vec[chunk_shift..];
                if bit_shift == 0 {
                    Self::Heap(remaining.to_vec())
                } else {
                    let mut new_vec = Vec::with_capacity(remaining.len());
                    for i in 0..remaining.len() {
                        #[expect(clippy::indexing_slicing)]
                        let mut val = remaining[i] >> bit_shift;
                        #[expect(clippy::indexing_slicing)]
                        if i.saturating_add(1) < remaining.len() {
                            val |= remaining[i.saturating_add(1)] << (64 - bit_shift);
                        }
                        new_vec.push(val);
                    }
                    Self::Heap(new_vec)
                }
            }
        }
        .minimized()
    }
}

impl ShrAssign<usize> for IndexSet {
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        *self = std::mem::take(self) >> rhs;
    }
}

/// An iterator over the slot indices contained in an [`IndexSet`].
///
/// Produces indices in ascending order.
#[derive(Debug, Clone)]
pub struct IndexSetIter<'a> {
    inner: &'a IndexSet,
    front_chunk: usize,
    front_bits: u64,
    back_chunk: usize,
    back_bits: u64,
    remaining: usize,
}

impl IndexSetIter<'_> {
    #[inline]
    fn chunk_bits(&self, chunk: usize) -> u64 {
        match self.inner {
            IndexSet::Small(bits) if chunk == 0 => u64::from(*bits),
            IndexSet::Medium(bits) if chunk == 0 => *bits,
            IndexSet::Large(bits) if chunk == 0 => *bits as u64,
            IndexSet::Large(bits) if chunk == 1 => (*bits >> 64) as u64,
            IndexSet::Heap(vec) => vec.get(chunk).copied().unwrap_or(0),
            _ => 0,
        }
    }
}

impl<'a> IntoIterator for &'a IndexSet {
    type Item = usize;

    type IntoIter = IndexSetIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let max = self.max_chunks().saturating_sub(1);
        IndexSetIter {
            inner: self,
            front_chunk: 0,
            front_bits: self.get_first_chunk(),
            back_chunk: max,
            back_bits: if max == 0 {
                self.get_first_chunk()
            } else {
                match self {
                    IndexSet::Large(b) => (*b >> 64) as u64,
                    IndexSet::Heap(v) => v.last().copied().unwrap_or(0),
                    _ => unreachable!("Small/Medium have only one chunk"),
                }
            },
            remaining: self.count_ones(),
        }
    }
}

impl Iterator for IndexSetIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.front_bits != 0 {
                let tz = self.front_bits.trailing_zeros() as usize;
                let mask = 1_u64 << tz;
                self.front_bits &= !mask;
                if self.front_chunk == self.back_chunk {
                    self.back_bits &= !mask;
                }
                self.remaining = self.remaining.wrapping_sub(1);
                return Some(self.front_chunk.wrapping_mul(64).wrapping_add(tz));
            }

            if self.remaining == 0 {
                return None;
            }

            self.front_chunk = self.front_chunk.wrapping_add(1);
            self.front_bits = self.chunk_bits(self.front_chunk);
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl DoubleEndedIterator for IndexSetIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if self.back_bits != 0 {
                let lz = self.back_bits.ilog2() as usize;
                let mask = 1_u64 << lz;
                self.back_bits &= !mask;
                if self.front_chunk == self.back_chunk {
                    self.front_bits &= !mask;
                }
                self.remaining = self.remaining.wrapping_sub(1);
                return Some(self.back_chunk.wrapping_mul(64).wrapping_add(lz));
            }

            if self.remaining == 0 {
                return None;
            }

            if self.back_chunk == 0 {
                self.remaining = 0;
                return None;
            }
            self.back_chunk = self.back_chunk.wrapping_sub(1);
            self.back_bits = self.chunk_bits(self.back_chunk);
        }
    }
}

impl ExactSizeIterator for IndexSetIter<'_> {}

impl FusedIterator for IndexSetIter<'_> {}
