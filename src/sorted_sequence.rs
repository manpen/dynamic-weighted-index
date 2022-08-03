#![allow(dead_code)]
use smallvec::SmallVec;

#[derive(Debug, Default, Clone)]
pub struct SortedSequence<T>
where
    T: Ord + Default,
{
    elements: SmallVec<[T; 32]>,
}

impl<T> SortedSequence<T>
where
    T: Ord + Default,
{
    pub fn insert(&mut self, elem: T) -> bool {
        match self.elements.binary_search(&elem) {
            Ok(_) => false,
            Err(pos) => {
                self.elements.insert(pos, elem);
                true
            }
        }
    }

    pub fn remove(&mut self, elem: &T) -> bool {
        match self.elements.binary_search(elem) {
            Ok(pos) => {
                self.elements.remove(pos);
                true
            }
            Err(_) => false,
        }
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn contains(&self, elem: &T) -> bool {
        self.elements.binary_search(elem).is_ok()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }

    pub fn iter_rev(&self) -> impl Iterator<Item = &T> {
        self.elements.iter().rev()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use itertools::Itertools;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    #[test]
    fn insert() {
        let mut seq = SortedSequence::default();

        assert!(seq.insert(10));
        assert!(seq.insert(1));
        assert!(!seq.insert(1));
        assert!(seq.insert(5));

        assert_eq!(seq.iter().copied().collect_vec(), [1, 5, 10]);
        assert_eq!(seq.iter_rev().copied().collect_vec(), [10, 5, 1]);
    }

    #[test]
    fn contains() {
        let mut seq = SortedSequence::default();

        assert!(!seq.contains(&10));
        assert!(seq.insert(10));
        assert!(seq.contains(&10));
        assert!(!seq.contains(&5));
    }

    #[test]
    fn remove() {
        let mut seq = SortedSequence::default();

        assert!(seq.insert(10));
        assert!(seq.insert(1));
        assert!(!seq.insert(1));
        assert!(seq.remove(&1));
        assert!(!seq.remove(&1));
        assert!(seq.insert(5));

        assert_eq!(seq.iter().copied().collect_vec(), [5, 10]);
        assert_eq!(seq.iter_rev().copied().collect_vec(), [10, 5]);
    }

    #[test]
    fn len_is_empty() {
        let mut seq = SortedSequence::default();
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);

        seq.insert(123);

        assert!(!seq.is_empty());
        assert_eq!(seq.len(), 1);
    }

    fn is_sorted<T>(data: &[T]) -> bool
    where
        T: Ord,
    {
        data.windows(2).all(|w| w[0] <= w[1])
    }

    fn is_sorted_rev<T>(data: &[T]) -> bool
    where
        T: Ord,
    {
        data.windows(2).all(|w| w[0] >= w[1])
    }

    #[test]
    fn randomized() {
        let mut rng = pcg_rand::Pcg64::seed_from_u64(0x1234);

        for n in 1..50 {
            let mut seq = SortedSequence::default();
            let mut set = HashSet::new();

            for _ in 0..2 * n * n {
                let elem = rng.gen_range(0..=n);

                assert_eq!(seq.contains(&elem), set.contains(&elem));
                if rng.gen_bool(0.7) {
                    // insert
                    assert_eq!(seq.insert(elem), set.insert(elem));
                } else {
                    // delete
                    assert_eq!(seq.remove(&elem), set.remove(&elem));
                }

                assert_eq!(seq.contains(&elem), set.contains(&elem));
                assert_eq!(seq.len(), set.len());
                assert_eq!(seq.is_empty(), set.is_empty());

                assert!(is_sorted(&seq.iter().copied().collect_vec()));
                assert!(is_sorted_rev(&seq.iter_rev().copied().collect_vec()));
            }
        }
    }
}
