use super::sorted_sequence::SortedSequence;
use super::Weight;
use rand::prelude::{Distribution, SliceRandom};
use rand::Rng;

pub struct DynamicWeightedIndex<W: Weight> {
    indices: Vec<Option<RangeIndex>>,
    levels: Vec<Level<W>>,
    total_weight: W,
}

impl<W: Weight> DynamicWeightedIndex<W> {
    /// Constructs a [`DynamicWeightedIndex`] on `n` elements; initially all elements have weight 0.
    ///
    /// # Example
    /// ```should_panic
    /// use dynamic_weighted_index::DynamicWeightedIndex;
    /// let mut dyn_idx = DynamicWeightedIndex::new(2);
    ///
    /// dyn_idx.set_weight(1, 123); // okay
    /// dyn_idx.set_weight(0, 314); // okay
    /// dyn_idx.set_weight(2, 123); // panic: out-of-bounds
    /// ```
    pub fn new(n: usize) -> Self {
        Self {
            indices: vec![None; n],
            levels: vec![Default::default(); 5],
            total_weight: W::ZERO,
        }
    }

    /// Updates the weight of the `idx`-th element
    ///
    /// # Example
    /// ```
    /// use dynamic_weighted_index::DynamicWeightedIndex;
    /// let mut dyn_idx = DynamicWeightedIndex::new(2);
    ///
    /// assert_eq!(dyn_idx.weight(0), 0);
    /// assert_eq!(dyn_idx.weight(1), 0);
    ///
    /// dyn_idx.set_weight(1, 31);
    /// assert_eq!(dyn_idx.weight(0), 0);
    /// assert_eq!(dyn_idx.weight(1), 31);
    /// ```
    pub fn set_weight(&mut self, index: usize, weight: W) {
        if weight == W::ZERO {
            return self.remove_weight(index);
        }

        assert!(weight > W::ZERO);

        let new_range_id = weight.compute_range_index();

        // if element is already assigned
        if let Some(prev_range_index) = self.indices[index] {
            // short-cut: element stays in the same range; just update its weight
            if prev_range_index.range_index == new_range_id {
                return self.update_leaf_weight(prev_range_index, weight);
            } else {
                self.remove_weight(index);
            }
        }

        // element is unassigned at this point
        self.initialize_weight(new_range_id, IndexAndWeight { weight, index });
    }

    /// Functionally equivalent to `self.set_weight(idx, W::ZERO)`
    ///
    /// # Example
    /// ```
    /// use dynamic_weighted_index::DynamicWeightedIndex;
    /// let mut dyn_idx = DynamicWeightedIndex::new(2);
    ///
    /// dyn_idx.set_weight(1, 31);
    /// assert_eq!(dyn_idx.weight(1), 31);
    ///
    /// dyn_idx.remove_weight(1);
    /// assert_eq!(dyn_idx.weight(1), 0);
    /// ```
    pub fn remove_weight(&mut self, index: usize) {
        if let Some(range_index) = self.indices[index] {
            self.total_weight -= self.weight(index);

            self.indices[index] = None;
            self.remove_element_from_range(0, range_index);
        }
    }

    /// Returns the weight of the `index`-th element (Zero if the element is uninitialized)
    ///
    /// # Example
    /// ```
    /// use dynamic_weighted_index::DynamicWeightedIndex;
    /// let mut dyn_idx = DynamicWeightedIndex::new(2);
    ///
    /// assert_eq!(dyn_idx.weight(0), 0);
    /// assert_eq!(dyn_idx.weight(1), 0);
    ///
    /// dyn_idx.set_weight(1, 31);
    /// assert_eq!(dyn_idx.weight(0), 0);
    /// assert_eq!(dyn_idx.weight(1), 31);
    /// ```
    pub fn weight(&self, idx: usize) -> W {
        self.indices[idx].map_or(W::ZERO, |idx| {
            self.get_range(0, idx.range_index).elements[idx.index_in_range].weight
        })
    }

    /// Returns the non-negative sum `S` of all weights assigned to this data structure. This value
    /// can be interpreted as the normalization constant for sampling; i.e. an entry with weight `w`
    /// is sampled with probability `w/S`. If no weights are assigned, the total weight is `W::ZERO`.
    ///
    /// # Example
    /// ```
    /// use dynamic_weighted_index::DynamicWeightedIndex;
    /// let mut dyn_idx = DynamicWeightedIndex::new(2);
    ///
    /// // index is empty, so return 0
    /// assert_eq!(dyn_idx.total_weight(), 0);
    ///
    /// // set exactly one element to a positive weight
    /// dyn_idx.set_weight(1, 31);
    /// assert_eq!(dyn_idx.total_weight(), 31);
    ///
    /// // set another element
    /// dyn_idx.set_weight(0, 3);
    /// assert_eq!(dyn_idx.total_weight(), 34);
    /// ```
    pub fn total_weight(&self) -> W {
        self.total_weight
    }

    /// Samples an element and returns it including its weight. Returns `None` iff the data
    /// structure is empty.
    ///
    /// # Example
    /// ```
    /// use dynamic_weighted_index::DynamicWeightedIndex;
    /// use rand;
    ///
    /// let mut rng = rand::thread_rng();
    /// let mut dyn_idx = DynamicWeightedIndex::new(2);
    ///
    /// // index is empty, so return None
    /// assert!(dyn_idx.sample_index_and_weight(&mut rng).is_none());
    ///
    /// // set exactly one element to a positive weight
    /// dyn_idx.set_weight(1, 3.14159);
    /// assert_eq!(dyn_idx.sample_index_and_weight(&mut rng).unwrap().index, 1);
    /// assert_eq!(dyn_idx.sample_index_and_weight(&mut rng).unwrap().weight, 3.14159);
    ///
    /// // set another element with a MUCH larger weight; this test will fail in 1 out of N attempts
    /// // where N is larger than the number of atoms in the universe.
    /// dyn_idx.set_weight(0, 3.14159e100);
    /// assert_eq!(dyn_idx.sample_index_and_weight(&mut rng).unwrap().index, 0);
    /// ```
    pub fn sample_index_and_weight<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Option<IndexAndWeight<W>> {
        if self.total_weight <= W::ZERO {
            return None;
        }

        let mut sample = || {
            let mut level_idx = self.sample_level_index(rng)?;
            let mut root = self.levels[level_idx].sample_root(rng)?;

            while level_idx > 0 {
                let child = root.sample_child(rng).unwrap();
                level_idx -= 1;
                root = &self.levels[level_idx].ranges[child.index];
            }

            root.sample_child(rng)
        };

        loop {
            if let Some(x) = sample() {
                return Some(x);
            }
        }
    }
}

// PRIVATE IMPLEMENTATION DETAILS //////////////////////////////////////////////////////////////////

#[derive(Default, Debug, Eq, PartialEq, Clone, Copy)]
struct RangeIndex {
    range_index: u32,
    index_in_range: usize,
}

#[derive(Debug, Clone)]
struct Level<W: Weight> {
    ranges: Vec<Range<W>>,
    roots: SortedSequence<u32>,
    roots_weight: W,
}

impl<W: Weight> Level<W> {
    fn sample_root<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<&Range<W>> {
        linear_sampling_from_iterator(
            rng,
            self.roots_weight,
            self.roots.iter_rev().map(|&i| {
                let range = &self.ranges[i as usize];
                (range, range.weight)
            }),
        )
    }
}

impl<W: Weight> Default for Level<W> {
    fn default() -> Self {
        Level {
            ranges: (0..W::NUM_RANGES)
                .into_iter()
                .map(|range_index| Range {
                    max_element_weight: W::max_weight_of_range_index(range_index as u32),
                    ..Default::default()
                })
                .collect(),
            roots: Default::default(),
            roots_weight: W::ZERO,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexAndWeight<W: Weight> {
    pub weight: W,
    pub index: usize,
}

#[derive(Default, Debug, Clone)]
struct Range<W: Weight> {
    parent: Option<RangeIndex>,
    weight: W,
    max_element_weight: W,
    elements: Vec<IndexAndWeight<W>>,
}

impl<W: Weight> Range<W> {
    fn sample_child<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<IndexAndWeight<W>> {
        if self.elements.len() == 1 {
            Some(self.elements[0])
        } else {
            loop {
                let &element = self.elements.as_slice().choose(rng)?;

                if W::sample_from_ratio(rng, element.weight, self.max_element_weight) {
                    break Some(element);
                }
            }
        }
    }

    fn increase_weight_by(&mut self, delta: W) -> (W, W) {
        let old = self.weight;
        let new = self.weight + delta;
        self.weight = new;
        (old, new)
    }

    fn decrease_weight_by(&mut self, delta: W) -> (W, W) {
        let old = self.weight;
        let new = self.weight - delta;
        self.weight = new;
        (old, new)
    }

    fn update_partial_weight(&mut self, partial_from: W, partial_to: W) -> (W, W) {
        let old = self.weight;
        let new = self.weight - partial_from + partial_to;
        self.weight = new;
        (old, new)
    }
}

impl<W: Weight> DynamicWeightedIndex<W> {
    fn initialize_weight(&mut self, range_id: u32, new_element: IndexAndWeight<W>) {
        debug_assert!(self.indices[new_element.index].is_none());
        self.total_weight += new_element.weight;
        let index_in_range = self.push_element_to_range(0, range_id, new_element);

        self.indices[new_element.index] = Some(RangeIndex {
            range_index: range_id,
            index_in_range,
        });
    }

    fn push_element_to_range(
        &mut self,
        level_index: u32,
        range_index: u32,
        new_element: IndexAndWeight<W>,
    ) -> usize {
        let range = self.get_range_mut(level_index, range_index);
        let old_num_elements = range.elements.len();

        // handle weights
        let (old_weight, new_weight) = range.increase_weight_by(new_element.weight);

        // push element
        let elements_new_index = range.elements.len();
        range.elements.push(new_element);

        // recursive update; since we added an element, we know that ..
        match old_num_elements {
            0 => {
                // .. this range was previously empty; make it a root
                self.make_root_range(level_index, range_index);
            }
            1 => {
                // .. this range previously had a one element and therefore was a root
                let new_parent_range_index = new_weight.compute_range_index();
                self.remove_root_range(level_index, range_index, old_weight);
                self.push_range_to_parent(
                    level_index,
                    range_index,
                    new_parent_range_index,
                    new_weight,
                );
            }
            _ => {
                // .. this range had and has multiple elements and therefore a parent. Let's see whether it changed.
                self.update_range_weight(level_index, range_index, old_weight);
            }
        }

        elements_new_index
    }

    fn remove_element_from_range(&mut self, level_index: u32, range_index: RangeIndex) {
        let range = self.get_range_mut(level_index, range_index.range_index);
        let parent = range.parent;
        let old_num_elements = range.elements.len();

        // update weight
        let weight_removed = range.elements[range_index.index_in_range].weight;
        let old_weight = range.decrease_weight_by(weight_removed).0;

        // remove entry from elements
        let last_element_is_removed = range_index.index_in_range + 1 == range.elements.len();
        if last_element_is_removed {
            range.elements.pop();
        } else {
            let element_moved_to_front = range.elements.last().unwrap().index;
            range.elements.swap_remove(range_index.index_in_range);

            if level_index == 0 {
                self.indices[element_moved_to_front] = Some(range_index);
            } else {
                self.set_range_parent(level_index - 1, element_moved_to_front as u32, range_index);
            }
        }

        // recursive update; since we deleted an element, we know that ..
        match old_num_elements {
            1 => {
                // .. this was a root range that is now empty; remove it.
                self.remove_root_range(level_index, range_index.range_index, old_weight);
                self.get_range_mut(level_index, range_index.range_index)
                    .weight = W::ZERO;
            }
            2 => {
                // .. this range previously had a two element and therefore a parent.
                // Remove parent link and make range a root
                self.remove_element_from_range(level_index + 1, parent.unwrap());
                self.make_root_range(level_index, range_index.range_index);
            }
            _ => {
                // .. this range still has at least two elements and has a parent.
                // Recursively update the parent's weight infos
                self.update_range_weight(level_index, range_index.range_index, old_weight);
            }
        };
    }

    fn push_range_to_parent(
        &mut self,
        level_index: u32,
        range_index: u32,
        parent_range_index: u32,
        weight: W,
    ) {
        let range_as_element = IndexAndWeight {
            weight,
            index: range_index as usize,
        };

        let index_in_range =
            self.push_element_to_range(level_index + 1, parent_range_index, range_as_element);

        self.set_range_parent(
            level_index,
            range_index,
            RangeIndex {
                range_index: parent_range_index,
                index_in_range,
            },
        );
    }

    fn update_range_weight(&mut self, level: u32, range_index: u32, old_weight: W) {
        let range = self.get_range(level, range_index);
        let new_weight = range.weight;

        if range.elements.len() == 1 {
            self.levels[level as usize].roots_weight -= old_weight;
            self.levels[level as usize].roots_weight += new_weight;
        } else {
            let parent = range.parent.unwrap();

            let new_parent_range_index = new_weight.compute_range_index();

            if parent.range_index == new_parent_range_index {
                let parent_range = self.get_range_mut(level + 1, parent.range_index);

                let (old_parent_weight, _) =
                    parent_range.update_partial_weight(old_weight, new_weight);
                parent_range.elements[parent.index_in_range].weight = new_weight;

                self.update_range_weight(level + 1, parent.range_index, old_parent_weight);
            } else {
                self.remove_element_from_range(level + 1, parent);
                self.push_range_to_parent(level, range_index, new_parent_range_index, new_weight);
            }
        }
    }

    fn update_leaf_weight(&mut self, parent_range_index: RangeIndex, new_weight: W) {
        let parent = self.get_range_mut(0, parent_range_index.range_index);

        // update element's weight
        let old_weight = parent.elements[parent_range_index.index_in_range as usize].weight;
        parent.elements[parent_range_index.index_in_range].weight = new_weight;

        // update parent's total weight
        let (old_parent_weight, _) = parent.update_partial_weight(old_weight, new_weight);
        self.total_weight -= old_weight;
        self.total_weight += new_weight;

        self.update_range_weight(0, parent_range_index.range_index, old_parent_weight);
    }

    fn get_range(&self, level: u32, range: u32) -> &Range<W> {
        &self.levels[level as usize].ranges[range as usize]
    }

    fn get_range_mut(&mut self, level: u32, range: u32) -> &mut Range<W> {
        &mut self.levels[level as usize].ranges[range as usize]
    }

    fn set_range_parent(&mut self, level: u32, range_index: u32, parent: RangeIndex) {
        self.get_range_mut(level, range_index).parent = Some(parent);
    }

    fn make_root_range(&mut self, level: u32, range_index: u32) {
        let range = self.get_range_mut(level, range_index);
        range.parent = None;
        let weight = range.weight;
        self.levels[level as usize].roots_weight += weight;

        self.levels[level as usize].roots.insert(range_index);
    }

    fn remove_root_range(&mut self, level: u32, range_index: u32, weight: W) {
        self.levels[level as usize].roots_weight -= weight;

        self.levels[level as usize].roots.remove(&range_index);
    }

    fn sample_level_index<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<usize> {
        linear_sampling_from_iterator(
            rng,
            self.total_weight,
            self.levels.iter().map(|l| l.roots_weight).enumerate(),
        )
    }
}

impl<W: Weight> Distribution<Option<usize>> for DynamicWeightedIndex<W> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<usize> {
        self.sample_index_and_weight(rng).map(|x| x.index)
    }
}

fn linear_sampling_from_iterator<T, R: Rng + ?Sized, W: Weight>(
    rng: &mut R,
    total_weight: W,
    iter: impl Iterator<Item = (T, W)>,
) -> Option<T> {
    let mut search = rng.gen_range(W::ZERO..total_weight);

    for (res, w) in iter {
        if search < w {
            return Some(res);
        }

        search -= w;
    }

    None
}

#[cfg(test)]
mod test {
    use super::*;
    use pcg_rand::Pcg64;
    use rand::{Rng, SeedableRng};
    use statrs::distribution::{Binomial, DiscreteCDF};

    mod test_floating_point {
        use super::*;
        use assert_float_eq::*;

        const DUMMY_WEIGHT_VALUE: f64 = 3.14159;
        const DUMMY_WEIGHT_VALUE1: f64 = 123.0;

        #[test]
        fn set_weight() {
            let mut dyn_index = DynamicWeightedIndex::new(2);
            assert_eq!(dyn_index.weight(0), 0.0);
            assert_eq!(dyn_index.weight(1), 0.0);

            dyn_index.set_weight(0, DUMMY_WEIGHT_VALUE);
            assert_f64_near!(dyn_index.weight(0), DUMMY_WEIGHT_VALUE);
            assert_eq!(dyn_index.weight(1), 0.0);

            dyn_index.set_weight(1, DUMMY_WEIGHT_VALUE1);
            assert_f64_near!(dyn_index.weight(0), DUMMY_WEIGHT_VALUE);
            assert_f64_near!(dyn_index.weight(1), DUMMY_WEIGHT_VALUE1);
        }

        #[test]
        fn update_weight() {
            let mut dyn_index = DynamicWeightedIndex::new(2);
            assert_eq!(dyn_index.weight(0), 0.0);

            dyn_index.set_weight(0, DUMMY_WEIGHT_VALUE);
            assert_f64_near!(dyn_index.weight(0), DUMMY_WEIGHT_VALUE);
            assert_f64_near!(dyn_index.total_weight(), DUMMY_WEIGHT_VALUE);

            dyn_index.set_weight(0, DUMMY_WEIGHT_VALUE1);
            assert_f64_near!(dyn_index.weight(0), DUMMY_WEIGHT_VALUE1);
            assert_f64_near!(dyn_index.total_weight(), DUMMY_WEIGHT_VALUE1);
        }

        #[test]
        fn total_weight() {
            let mut dyn_index = DynamicWeightedIndex::new(2);
            assert_f64_near!(dyn_index.total_weight(), 0.0);

            dyn_index.set_weight(0, DUMMY_WEIGHT_VALUE);
            assert_f64_near!(dyn_index.total_weight(), DUMMY_WEIGHT_VALUE);

            dyn_index.set_weight(1, DUMMY_WEIGHT_VALUE1);
            assert_f64_near!(
                dyn_index.total_weight(),
                DUMMY_WEIGHT_VALUE + DUMMY_WEIGHT_VALUE1
            );
        }

        #[test]
        fn set_weight_randomized() {
            let mut rng = pcg_rand::Pcg64::seed_from_u64(0x12345678);

            for n in [2, 10, 100] {
                let mut weights = vec![None; n];
                let mut dyn_index = DynamicWeightedIndex::new(n);

                for _round in 0..10 * n {
                    let index = rng.gen_range(0..n);
                    let new_weight = rng.gen();

                    dyn_index.set_weight(index, new_weight);

                    // update own copy of weights
                    weights[index] = Some(new_weight);

                    for (i, w) in weights.iter().enumerate() {
                        if let Some(pw) = w.clone() {
                            let dyn_weight = dyn_index.weight(i);
                            assert_f64_near!(pw, dyn_weight);
                        } else {
                            assert_eq!(dyn_index.weight(i), 0.0);
                        }
                    }

                    let total_weight: f64 = weights.iter().map(|w| w.unwrap_or(0.0)).sum();
                    assert_f64_near!(dyn_index.total_weight(), total_weight, 1000);
                }
            }
        }

        fn get_used_index(rng: &mut impl Rng, n: usize) -> DynamicWeightedIndex<f64> {
            let mut dyn_index = DynamicWeightedIndex::new(n);
            for _ in 0..10 * n {
                let weight = 10.0_f64.powf(rng.gen_range(-10.0..10.0));
                let index = rng.gen_range(0..n);
                dyn_index.set_weight(index, weight);
            }
            dyn_index
        }

        #[test]
        fn consistent_root_weights() {
            let mut rng = pcg_rand::Pcg64::seed_from_u64(0x12345);

            for n in 2..100 {
                let dyn_index = get_used_index(&mut rng, n);

                let root_weights: f64 = dyn_index.levels.iter().map(|l| l.roots_weight).sum();
                assert_float_relative_eq!(dyn_index.total_weight, root_weights, 1e-6);
            }
        }

        #[test]
        fn consistent_range_weights() {
            let mut rng = pcg_rand::Pcg64::seed_from_u64(0x12345);

            for n in 2..100 {
                let dyn_index = get_used_index(&mut rng, n);

                for level in &dyn_index.levels {
                    for range in level.ranges.iter() {
                        let elem_sum: f64 = range.elements.iter().map(|e| e.weight).sum();
                        assert_float_relative_eq!(range.weight, elem_sum, 1e-6);
                    }
                }
            }
        }

        #[test]
        fn consistent_element_weights() {
            let mut rng = pcg_rand::Pcg64::seed_from_u64(0x12345);

            for n in 2..100 {
                let dyn_index = get_used_index(&mut rng, n);

                for level in dyn_index.levels.iter() {
                    for (range_id, range) in level.ranges.iter().enumerate() {
                        for e in range.elements.iter() {
                            assert!(e.weight <= range.max_element_weight);
                            assert!(e.weight >= range.max_element_weight / 2.0);
                        }
                    }
                }
            }
        }

        #[test]
        fn sample_single() {
            sample_single_base::<f64>(|rng: &mut Pcg64| rng.gen_range(1e-100..1e100));
        }

        #[test]
        fn sample_multiple() {
            sample_multiple_base::<f64>(|rng: &mut Pcg64| rng.gen_range(1e-100..1e100));
        }
    }

    mod test_integer {
        use super::*;

        const DUMMY_WEIGHT_VALUE: u64 = 314159;
        const DUMMY_WEIGHT_VALUE1: u64 = 123;

        #[test]
        fn set_weight() {
            let mut dyn_index = DynamicWeightedIndex::new(2);
            assert_eq!(dyn_index.weight(0), 0);
            assert_eq!(dyn_index.weight(1), 0);

            dyn_index.set_weight(0, DUMMY_WEIGHT_VALUE);
            assert_eq!(dyn_index.weight(0), DUMMY_WEIGHT_VALUE);
            assert_eq!(dyn_index.weight(1), 0);

            dyn_index.set_weight(1, DUMMY_WEIGHT_VALUE1);
            assert_eq!(dyn_index.weight(0), DUMMY_WEIGHT_VALUE);
            assert_eq!(dyn_index.weight(1), DUMMY_WEIGHT_VALUE1);
        }

        #[test]
        fn update_weight() {
            let mut dyn_index = DynamicWeightedIndex::new(2);
            assert_eq!(dyn_index.weight(0), 0);

            dyn_index.set_weight(0, DUMMY_WEIGHT_VALUE);
            assert_eq!(dyn_index.weight(0), DUMMY_WEIGHT_VALUE);
            assert_eq!(dyn_index.total_weight(), DUMMY_WEIGHT_VALUE);

            dyn_index.set_weight(0, DUMMY_WEIGHT_VALUE1);
            assert_eq!(dyn_index.weight(0), DUMMY_WEIGHT_VALUE1);
            assert_eq!(dyn_index.total_weight(), DUMMY_WEIGHT_VALUE1);
        }

        #[test]
        fn total_weight() {
            let mut dyn_index = DynamicWeightedIndex::new(2);
            assert_eq!(dyn_index.total_weight(), 0);

            dyn_index.set_weight(0, DUMMY_WEIGHT_VALUE);
            assert_eq!(dyn_index.total_weight(), DUMMY_WEIGHT_VALUE);

            dyn_index.set_weight(1, DUMMY_WEIGHT_VALUE1);
            assert_eq!(
                dyn_index.total_weight(),
                DUMMY_WEIGHT_VALUE + DUMMY_WEIGHT_VALUE1
            );
        }

        #[test]
        fn set_weight_randomized() {
            let mut rng = pcg_rand::Pcg64::seed_from_u64(0x12345678);

            for n in [2, 10, 100] {
                let mut weights = vec![None; n];
                let mut dyn_index = DynamicWeightedIndex::new(n);

                for _round in 0..10 * n {
                    let index = rng.gen_range(0..n);
                    let new_weight = rng.gen::<u32>() as u64;

                    dyn_index.set_weight(index, new_weight);

                    // update own copy of weights
                    weights[index] = Some(new_weight);

                    for (i, w) in weights.iter().enumerate() {
                        if let Some(pw) = w.clone() {
                            let dyn_weight = dyn_index.weight(i);
                            assert_eq!(pw, dyn_weight);
                        } else {
                            assert_eq!(dyn_index.weight(i), 0);
                        }
                    }

                    let total_weight = weights.iter().map(|w| w.unwrap_or(0)).sum();
                    assert_eq!(dyn_index.total_weight(), total_weight);
                }
            }
        }

        fn get_used_index(rng: &mut impl Rng, n: usize) -> DynamicWeightedIndex<u64> {
            let mut dyn_index = DynamicWeightedIndex::new(n);
            for _ in 0..10 * n {
                let weight = rng.gen::<u32>() as u64;
                let index = rng.gen_range(0..n);
                dyn_index.set_weight(index, weight);
            }
            dyn_index
        }

        #[test]
        fn consistent_root_weights() {
            let mut rng = pcg_rand::Pcg64::seed_from_u64(0x12345);

            for n in 2..100 {
                let dyn_index = get_used_index(&mut rng, n);

                let root_weights = dyn_index.levels.iter().map(|l| l.roots_weight).sum();
                assert_eq!(dyn_index.total_weight, root_weights);
            }
        }

        #[test]
        fn consistent_range_weights() {
            let mut rng = pcg_rand::Pcg64::seed_from_u64(0x12345);

            for n in 2..100 {
                let dyn_index = get_used_index(&mut rng, n);

                for level in &dyn_index.levels {
                    for range in level.ranges.iter() {
                        let elem_sum = range.elements.iter().map(|e| e.weight).sum();
                        assert_eq!(range.weight, elem_sum);
                    }
                }            }
        }

        #[test]
        fn consistent_element_weights() {
            let mut rng = pcg_rand::Pcg64::seed_from_u64(0x12345);

            for n in 2..100 {
                let dyn_index = get_used_index(&mut rng, n);
                for level in dyn_index.levels.iter() {
                    for (range_id, range) in level.ranges.iter().enumerate() {
                        assert_eq!(
                            range.max_element_weight.compute_range_index(),
                            range_id as u32
                        );

                        for e in range.elements.iter() {
                            assert_eq!(e.weight.compute_range_index(), range_id as u32);
                            assert!(e.weight <= range.max_element_weight);
                        }
                    }
                }
            }
        }

        #[test]
        fn sample_single() {
            sample_single_base::<u32>(|rng: &mut Pcg64| rng.gen::<u32>() >> 4);
        }

        #[test]
        fn sample_multiple() {
            sample_multiple_base::<u32>(|rng: &mut Pcg64| rng.gen::<u32>() >> 4);
        }
    }

    fn sample_single_base<W: Weight + std::iter::Sum + Into<f64>>(
        sampler: impl Fn(&mut Pcg64) -> W,
    ) {
        let mut rng = Pcg64::seed_from_u64(0x1234567);

        for n in [2, 10, 30] {
            let mut dyn_index = DynamicWeightedIndex::new(n);
            for _ in 0..10 * n {
                let index = rng.gen_range(0..n);
                let new_weight = sampler(&mut rng);

                // set a single weight; we always have to sample this one element
                dyn_index.set_weight(index, new_weight);
                for _j in 0..5 {
                    assert_eq!(dyn_index.sample(&mut rng), Some(index));
                }

                // unset weight --- no sample possible
                dyn_index.set_weight(index, W::ZERO);
                for _j in 0..5 {
                    assert_eq!(dyn_index.sample(&mut rng), None);
                }
            }
        }
    }

    fn sample_multiple_base<W: Weight + std::iter::Sum + Into<f64>>(
        sampler: impl Fn(&mut Pcg64) -> W,
    ) {
        const SAMPLES_PER_ELEMENT: usize = 25000;
        let mut rng = Pcg64::seed_from_u64(0x123456);

        for n in 2..15 {
            let mut dyn_index = DynamicWeightedIndex::new(n);
            let mut weights = vec![0.0; n];

            for _j in 0..n {
                let index = rng.gen_range(0..n);
                let new_weight = if rng.gen() {
                    sampler(&mut rng)
                } else {
                    W::ZERO
                }; // 50% to be empty
                dyn_index.set_weight(index, new_weight);
                weights[index] = new_weight.into();
            }

            if weights.iter().copied().sum::<f64>() == 0.0 {
                continue;
            }

            let mut counts = vec![0; n];
            for _j in 0..SAMPLES_PER_ELEMENT * n {
                counts[dyn_index.sample(&mut rng).unwrap()] += 1;
            }

            verify_multinomial(&weights, &counts, 0.01);
        }
    }

    fn verify_multinomial(weights: &[f64], counts: &[u64], significance: f64) {
        assert_eq!(weights.len(), counts.len());
        let num_total_counts = counts.iter().sum::<u64>();
        let total_weight: f64 = weights.iter().sum();

        // Bonferroni correction as we carry out {weights.len()}many independent trails
        let corrected_significance = significance / (weights.len() as f64);

        assert_eq!(num_total_counts == 0, total_weight == 0.0);

        for (&count, &weight) in counts.iter().zip(weights) {
            if weight == 0.0 {
                assert_eq!(count, 0);
            } else {
                let probabilty = weight / total_weight;
                let mean = probabilty * num_total_counts as f64;

                let distr = Binomial::new(probabilty, num_total_counts).unwrap();

                // compute two-sided p-value, i.e. the probability that more extreme values are
                // produced by the binomial distribution
                let pvalue = if mean >= count as f64 {
                    2.0 * distr.cdf(count)
                } else {
                    2.0 * (1.0 - distr.cdf(count - 1))
                };

                assert!(
                    pvalue >= corrected_significance,
                    "prob: {} mean: {} count: {} p-value: {} corrected-significance: {} n: {}",
                    probabilty,
                    mean,
                    count,
                    pvalue,
                    corrected_significance,
                    weights.len()
                );
            }
        }
    }
}
