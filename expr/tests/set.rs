#![expect(clippy::indexing_slicing, clippy::panic, clippy::shadow_reuse)]

use std::hash::{Hash as _, Hasher as _};

use hypervox_expr::*;

#[test]
fn singleton_and_contains() {
    let s = IndexSet::singleton(0);
    assert!(s.contains(0));
    assert!(!s.contains(1));

    let s = IndexSet::singleton(31);
    assert!(s.contains(31));

    let s = IndexSet::singleton(63);
    assert!(s.contains(63));

    let s = IndexSet::singleton(200);
    assert!(s.contains(200));
}

#[test]
fn insert_and_growth() {
    let mut s = IndexSet::Small(0);
    s.insert(0, true);
    assert!(s.contains(0));

    s.insert(33, true);
    assert!(matches!(s, IndexSet::Medium(_)));
    assert!(s.contains(33));

    s.insert(70, true);
    assert!(matches!(s, IndexSet::Large(_)));
    assert!(s.contains(70));

    s.insert(150, true);
    assert!(matches!(s, IndexSet::Heap(_)));
    assert!(s.contains(150));

    s.insert(150, false);
    assert!(!s.contains(150));
}

#[test]
fn iter() {
    let mut s = IndexSet::Small(0);
    s.insert(2, true);
    s.insert(5, true);
    s.insert(0, true);
    let v: Vec<_> = s.iter().collect();
    assert_eq!(v, vec![0, 2, 5]);
}

#[test]
fn iter_large() {
    let mut s = IndexSet::singleton(200);
    s.insert(100, true);
    s.insert(0, true);
    let v: Vec<_> = s.iter().collect();
    assert_eq!(v, vec![0, 100, 200]);
}

#[test]
fn iter_rev_and_exact() {
    let mut s = IndexSet::singleton(200);
    s.insert(100, true);
    s.insert(0, true);
    assert_eq!(s.iter().rev().collect::<Vec<_>>(), vec![200, 100, 0]);
    let mut it = s.iter();
    it.next();
    assert_eq!(it.len(), 2);
}

#[test]
fn iter_mixed() {
    let mut s = IndexSet::singleton(200);
    s.insert(5, true);
    s.insert(0, true);
    let mut it = s.iter();
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next_back(), Some(200));
    assert_eq!(it.next(), Some(5));
    assert_eq!(it.next(), None);
    assert_eq!(it.next_back(), None);
}

#[test]
fn count_ones_and_is_empty() {
    assert!(IndexSet::Small(0).is_empty());
    assert_eq!(IndexSet::Small(0).count_ones(), 0);

    let mut s = IndexSet::Small(0);
    s.insert(0, true);
    s.insert(1, true);
    s.insert(2, true);
    assert_eq!(s.count_ones(), 3);
    assert!(!s.is_empty());

    let mut s = IndexSet::singleton(200);
    s.insert(201, true);
    assert_eq!(s.count_ones(), 2);
}

#[test]
fn is_disjoint() {
    let a = IndexSet::singleton(0);
    let b = IndexSet::singleton(1);
    assert!(a.is_disjoint(&b));

    let b2 = IndexSet::singleton(0);
    assert!(!a.is_disjoint(&b2));
}

#[test]
fn is_disjoint_cross_variant() {
    let s = IndexSet::singleton(0);
    let m = IndexSet::singleton(33);
    let l = IndexSet::singleton(70);
    let h = IndexSet::singleton(200);

    assert!(s.is_disjoint(&m));
    assert!(m.is_disjoint(&l));
    assert!(l.is_disjoint(&h));

    let overlap = IndexSet::singleton(0);
    assert!(!s.is_disjoint(&overlap));
}

#[test]
fn minimize() {
    // Large with value that fits in Small
    let mut s = IndexSet::Large(0);
    s.insert(5, true);
    let s = s.minimized();
    assert!(matches!(s, IndexSet::Small(_)));

    // Heap that can downgrade to Large
    let mut s = IndexSet::Heap(vec![0, 0]);
    s.insert(70, true);
    let s = s.minimized();
    assert!(matches!(s, IndexSet::Large(_)));

    // Already minimal
    let s = IndexSet::singleton(0).minimized();
    assert!(matches!(s, IndexSet::Small(_)));
}

#[test]
fn bitor() {
    let a = IndexSet::singleton(0);
    let b = IndexSet::singleton(1);
    let c = a | b;
    assert!(c.contains(0));
    assert!(c.contains(1));

    let v: Vec<_> = c.iter().collect();
    assert_eq!(v, vec![0, 1]);
}

#[test]
fn bitand() {
    let a = IndexSet::singleton(0) | IndexSet::singleton(1);
    let b = IndexSet::singleton(1) | IndexSet::singleton(2);
    let c = a & b;
    assert!(!c.contains(0));
    assert!(c.contains(1));
    assert!(!c.contains(2));
}

#[test]
fn partial_eq_across_variants() {
    let s = IndexSet::singleton(0);
    let m = IndexSet::Medium(1);
    assert_eq!(s, m);

    let m2 = IndexSet::Medium(1);
    assert_eq!(s, m2);
}

#[test]
fn hash_consistency() {
    use std::collections::hash_map::DefaultHasher;
    let a = IndexSet::singleton(0) | IndexSet::singleton(1);
    let b = IndexSet::Medium(0b11);
    let ha = {
        let mut h = DefaultHasher::new();
        a.hash(&mut h);
        h.finish()
    };
    let hb = {
        let mut h = DefaultHasher::new();
        b.hash(&mut h);
        h.finish()
    };
    assert_eq!(ha, hb);
}

#[test]
fn default_is_empty() {
    let s = IndexSet::default();
    assert!(s.is_empty());
    assert!(matches!(s, IndexSet::Small(0)));
}

#[test]
fn shl_small() {
    let s = IndexSet::singleton(0) | IndexSet::singleton(2);
    let shifted = s << 3;
    assert!(shifted.contains(3));
    assert!(shifted.contains(5));
    assert!(!shifted.contains(0));
}

#[test]
fn shl_overflow_small() {
    let s = IndexSet::singleton(0) | IndexSet::singleton(31);
    let shifted = s << 1;
    assert!(shifted.contains(1));
    assert!(shifted.contains(32));
}

#[test]
fn shl_large_to_heap() {
    let s = IndexSet::singleton(120);
    let shifted = s << 10;
    assert!(shifted.contains(130));
}

#[test]
fn shr_small() {
    let s = IndexSet::singleton(5) | IndexSet::singleton(8);
    let shifted = s >> 3;
    assert!(shifted.contains(2));
    assert!(shifted.contains(5));
    assert!(!shifted.contains(0));
}

#[test]
fn shl_shr_empty() {
    let s = IndexSet::Small(0);
    assert!((s.clone() << 5).is_empty());
    assert!((s >> 5).is_empty());
}

#[test]
fn shl_zero_shift() {
    let s = IndexSet::singleton(0) | IndexSet::singleton(5) | IndexSet::singleton(100);
    assert_eq!(s.clone() << 0, s);
}

#[test]
fn shr_overflow() {
    let s = IndexSet::singleton(0)
        | IndexSet::singleton(33)
        | IndexSet::singleton(70)
        | IndexSet::singleton(200);
    assert!((s >> 300).is_empty());
}

#[test]
fn shl_shr_roundtrip() {
    let s = IndexSet::singleton(0) | IndexSet::singleton(5) | IndexSet::singleton(100);
    let shifted = s.clone() << 7 >> 7;
    assert_eq!(shifted, s);
}

#[test]
fn ord_same_variant() {
    assert!(IndexSet::Small(0b001) < IndexSet::Small(0b010));
    assert!(IndexSet::Small(0b010) > IndexSet::Small(0b001));
    assert_eq!(IndexSet::Small(0b101), IndexSet::Small(0b101));
    assert!(IndexSet::Medium(0b001) < IndexSet::Medium(0b010));
    assert!(IndexSet::Large(0b001) < IndexSet::Large(0b010));
}

#[test]
fn ord_cross_variant() {
    assert_eq!(IndexSet::Small(0b11), IndexSet::Medium(0b11));
    assert_eq!(IndexSet::Medium(0b11), IndexSet::Large(0b11));
    assert_eq!(IndexSet::Large(0b11), IndexSet::Heap(vec![0b11]));
    assert!(IndexSet::Small(0b100) > IndexSet::Large(0b010));
}

#[test]
fn arith_add_small() {
    let a = ArithIndexSet(IndexSet::Small(5));
    let b = ArithIndexSet(IndexSet::Small(3));
    let c = a + b;
    assert_eq!(c.0, IndexSet::Small(8));
}

#[test]
fn arith_add_with_carry() {
    let a = ArithIndexSet(IndexSet::Small(u32::MAX));
    let b = ArithIndexSet(IndexSet::Small(1));
    let c = a + b;
    assert!(!matches!(c.0, IndexSet::Small(_)));
    assert_eq!(c.0.count_ones(), 1);
    assert!(c.0.contains(32));
}

#[test]
fn arith_add_large_overflow() {
    let a = ArithIndexSet(IndexSet::Large(u128::MAX));
    let b = ArithIndexSet(IndexSet::Large(1));
    let c = a + b;
    if let IndexSet::Heap(v) = &c.0 {
        assert_eq!(v.len(), 3);
        assert_eq!(v[2], 1);
    } else {
        panic!("expected Heap variant");
    }
}

#[test]
fn arith_sub_small() {
    let a = ArithIndexSet(IndexSet::Small(8));
    let b = ArithIndexSet(IndexSet::Small(3));
    let c = a - b;
    assert_eq!(c.0, IndexSet::Small(5));
}

#[test]
fn arith_sub_underflow_panics() {
    let a = ArithIndexSet(IndexSet::Small(3));
    let b = ArithIndexSet(IndexSet::Small(8));
    assert!(std::panic::catch_unwind(move || a - b).is_err());
}

#[test]
fn arith_from_conversion() {
    let s = IndexSet::Small(42);
    let a: ArithIndexSet = s.into();
    let back: IndexSet = a.into();
    assert_eq!(back, IndexSet::Small(42));
}

#[test]
fn arith_deref() {
    let a = ArithIndexSet(IndexSet::Small(0b101));
    assert!(a.contains(0));
    assert!(a.contains(2));
}

#[test]
fn arith_add_heap_fallback() {
    let a = ArithIndexSet(IndexSet::Heap(vec![u64::MAX, u64::MAX, 1]));
    let b = ArithIndexSet(IndexSet::Heap(vec![1, 0, 1]));
    let c = a + b;
    if let IndexSet::Heap(v) = c.0 {
        assert_eq!(v, vec![0, 0, 3]);
    } else {
        panic!("expected Heap variant");
    }
}

#[test]
fn bitor_assign() {
    let mut a = IndexSet::singleton(0);
    let b = IndexSet::singleton(1);
    a |= b;
    assert!(a.contains(0));
    assert!(a.contains(1));
}

#[test]
fn arith_assign() {
    let mut a = ArithIndexSet(IndexSet::Small(5));
    a += ArithIndexSet(IndexSet::Small(3));
    assert_eq!(a.0, IndexSet::Small(8));
    a -= ArithIndexSet(IndexSet::Small(3));
    assert_eq!(a.0, IndexSet::Small(5));
}

#[test]
fn arith_sub_heap_fallback() {
    let a = ArithIndexSet(IndexSet::Heap(vec![2, 1, 1]));
    let b = ArithIndexSet(IndexSet::Heap(vec![1, 1, 0]));
    let c = a - b;
    if let IndexSet::Heap(v) = c.0 {
        assert_eq!(v, vec![1, 0, 1]);
    } else {
        panic!("expected Heap variant");
    }
}

#[test]
fn try_from_arith_to_int() {
    assert_eq!(u32::try_from(ArithIndexSet(IndexSet::Small(42))), Ok(42));
    assert_eq!(
        u8::try_from(ArithIndexSet(IndexSet::Small(256))),
        Err(ArithIndexSetTryFromError::Overflow)
    );
    assert_eq!(i16::try_from(ArithIndexSet(IndexSet::Small(100))), Ok(100));
    assert_eq!(
        i8::try_from(ArithIndexSet(IndexSet::Small(200))),
        Err(ArithIndexSetTryFromError::Overflow)
    );
}

#[test]
fn try_from_int_to_arith() {
    assert_eq!(
        ArithIndexSet::try_from(-1_i32),
        Err(ArithIndexSetTryFromError::Negative)
    );
    assert_eq!(
        ArithIndexSet::try_from(42_i32),
        Ok(ArithIndexSet::from(42_u32))
    );
}

#[test]
fn arith_range_iter() {
    let start = ArithIndexSet::from(2_u32);
    let end = ArithIndexSet::from(7_u32);
    let results: Vec<ArithIndexSet> = start.range(end).collect();
    assert_eq!(results.len(), 5);
    assert_eq!(results[0], ArithIndexSet::from(2_u32));
    assert_eq!(results[1], ArithIndexSet::from(3_u32));
    assert_eq!(results[2], ArithIndexSet::from(4_u32));
    assert_eq!(results[3], ArithIndexSet::from(5_u32));
    assert_eq!(results[4], ArithIndexSet::from(6_u32));
}

#[test]
fn arith_range_empty() {
    let start = ArithIndexSet::from(5_u32);
    let end = ArithIndexSet::from(3_u32);
    let mut range = start.range(end);
    assert!(range.next().is_none() && range.next_back().is_none());
}

#[test]
fn arith_range_rev() {
    let start = ArithIndexSet::from(0_u32);
    let end = ArithIndexSet::from(4_u32);
    let results: Vec<ArithIndexSet> = start.range(end).rev().collect();
    assert_eq!(results.len(), 4);
    assert_eq!(results[0], ArithIndexSet::from(3_u32));
    assert_eq!(results[3], ArithIndexSet::from(0_u32));
}

#[test]
fn arith_range_to() {
    let results: Vec<ArithIndexSet> = ArithIndexSet::from(4_u32).range_to().collect();
    assert_eq!(results.len(), 4);
    assert_eq!(results[0], ArithIndexSet::from(0_u32));
    assert_eq!(results[1], ArithIndexSet::from(1_u32));
    assert_eq!(results[2], ArithIndexSet::from(2_u32));
    assert_eq!(results[3], ArithIndexSet::from(3_u32));
}

#[test]
fn arith_range_to_zero() {
    let mut range_to = ArithIndexSet::from(0_u32).range_to();
    assert!(range_to.next().is_none());
}

#[test]
fn arith_range_from_take() {
    let results: Vec<ArithIndexSet> = ArithIndexSet::from(3_u32).range_from().take(4).collect();
    assert_eq!(results.len(), 4);
    assert_eq!(results[0], ArithIndexSet::from(3_u32));
    assert_eq!(results[1], ArithIndexSet::from(4_u32));
    assert_eq!(results[2], ArithIndexSet::from(5_u32));
    assert_eq!(results[3], ArithIndexSet::from(6_u32));
}
