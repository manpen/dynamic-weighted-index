use ez_bitset::bitset::BitSet;
use rand::prelude::{Distribution, SliceRandom};
use rand::Rng;
use std::collections::HashMap;

pub struct DynamicWeightedIndex {
    indices: Vec<Option<RangeIndex>>,
    levels: Vec<Level>,
    total_weight: f64,
    min_item_weight: f64,
    #[allow(dead_code)]
    max_total_weight: f64,
}

impl DynamicWeightedIndex {
    /// Constructs a [`DynamicWeightedIndex`] on `n` elements; initially all elements have weight 0.
    pub fn new(n: usize) -> Self {
        Self::with_limits(n, f64::MIN_POSITIVE, f64::MAX)
    }

    /// Constructs a [`DynamicWeightedIndex`] on `n` elements; initially all elements have weight `0.0`.
    /// Additionally `min_item_prob` gives a hint on the smallest positive weight that will be assigned
    /// to any element (that is, any weight `w` assigned satisfies either `w == 0` or `w >= min_item_prob`).
    /// The parameter `max_total_weight` is an upper bound on the sum of all weights assigned at any
    /// given time. Both parameters are hint to the data structure to optimize performance.
    pub fn with_limits(n: usize, min_item_weight: f64, max_total_weight: f64) -> Self {
        Self {
            indices: vec![None; n],
            levels: vec![Default::default(); 5],
            total_weight: 0.0,
            min_item_weight,
            max_total_weight,
        }
    }

    /// Updates the weight of the `idx`-th element
    pub fn set_weight(&mut self, index: usize, weight: f64) {
        assert!(weight == 0.0 || weight >= self.min_item_weight);
        self.total_weight += weight;

        let new_range_id = self.compute_range_index(weight);

        if let Some(prev_range_index) = self.indices[index] {
            let weight_before = self.weight(index).unwrap();
            self.total_weight -= weight_before;

            // short-cut: element stays in the same range; just update its weight
            if prev_range_index.range_index == new_range_id {
                let weight_increase = weight - weight_before;
                self.update_range_weight(0, new_range_id, weight_increase);
                return;
            }

            self.remove_element_from_range(0, prev_range_index);
        }

        // weight is unassigned at this point
        self.initialize_new_weight(new_range_id, WeightAndIndex::new(weight, index));
    }

    /// Returns the weight of the `index`-th element (`None` if the element is uninitialized)
    pub fn weight(&self, idx: usize) -> Option<f64> {
        let idx = self.indices[idx]?;
        Some(self.get_range_or_fail(0, idx.range_index).elements[idx.index_in_range].weight)
    }

    /// Returns the non-negative sum `S` of all weights assigned to this data structure. This value
    /// can be interpreted as the normalization constant for sampling; i.e. an entry with weight `w`
    /// is sampled with probability `w/S`. If no weights are assigned, the total weight is `0.0`.
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }
}

// PRIVATE IMPLEMENTATION DETAILS //////////////////////////////////////////////////////////////////

const FLOAT_EXP_BITS: usize = 11;

#[derive(Default, Debug, Eq, PartialEq, Clone, Copy)]
struct RangeIndex {
    range_index: u32,
    index_in_range: usize,
}

#[derive(Debug, Clone)]
struct Level {
    ranges: HashMap<usize, Range>,
    roots: BitSet,
    roots_weight: f64,
}

impl Level {
    fn sample_root<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<&Range> {
        linear_sampling_from_iterator(
            rng,
            self.roots_weight,
            self.roots.iter().map(|i| {
                let range = self.ranges.get(&i).unwrap();
                (range, range.weight)
            }),
        )
    }
}

impl Default for Level {
    fn default() -> Self {
        Level {
            ranges: Default::default(),
            roots: BitSet::new(1usize << FLOAT_EXP_BITS),
            roots_weight: 0.0,
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
struct WeightAndIndex {
    weight: f64,
    index: usize,
}

impl WeightAndIndex {
    fn new(weight: f64, index: usize) -> Self {
        Self { weight, index }
    }
}

#[derive(Default, Debug, Clone)]
struct Range {
    parent: Option<RangeIndex>,
    weight: f64,
    elements: Vec<WeightAndIndex>,
}

impl Range {
    fn sample_child<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<usize> {
        let max_weight = 2.0_f64.powf(self.elements.first()?.weight.log2() + 1.0);
        loop {
            let &element = self.elements.as_slice().choose(rng)?;
            if element.weight == max_weight || rng.gen_range(0.0..max_weight) < element.weight {
                return Some(element.index);
            }
        }
    }
}

impl DynamicWeightedIndex {
    fn initialize_new_weight(&mut self, range_id: u32, new_element: WeightAndIndex) {
        let index_in_range = self.push_to_parent(0, range_id, new_element);

        self.indices[new_element.index] = Some(RangeIndex {
            range_index: range_id,
            index_in_range,
        });
    }

    fn remove_element_from_range(&mut self, level_index: u32, range_index: RangeIndex) {
        let range = self.get_range_mut_or_insert(level_index, range_index.range_index);

        let remove_last_element = range_index.index_in_range + 1 == range.elements.len();

        let weight_removed = range.elements[range_index.index_in_range].weight;

        if remove_last_element {
            range.elements.pop();
        } else {
            let element_moved_to_front = range.elements.last().unwrap().index;
            range.elements.swap_remove(range_index.index_in_range);
            self.indices[element_moved_to_front] = Some(range_index);
        }

        self.update_range_weight(level_index, range_index.range_index, -weight_removed);
    }

    fn push_to_parent(
        &mut self,
        parent_level: u32,
        parent_range_index: u32,
        new_element: WeightAndIndex,
    ) -> usize {
        let range = self.get_range_mut_or_insert(parent_level, parent_range_index);
        let index_in_range = range.elements.len();
        range.elements.push(new_element);
        self.update_range_weight(parent_level, parent_range_index, new_element.weight);
        index_in_range
    }

    fn get_range_or_fail(&self, level: u32, range: u32) -> &Range {
        self.levels[level as usize]
            .ranges
            .get(&(range as usize))
            .unwrap()
    }

    fn get_range_mut_or_insert(&mut self, level: u32, range: u32) -> &mut Range {
        self.levels[level as usize]
            .ranges
            .entry(range as usize)
            .or_insert(Default::default())
    }

    fn update_range_weight(&mut self, level: u32, range_index: u32, weight_increase: f64) {
        let range_mut = self.get_range_mut_or_insert(level, range_index);
        let old_weight = range_mut.weight;
        debug_assert!(old_weight + weight_increase >= 0.0);
        range_mut.weight += weight_increase;

        match range_mut.elements.len() {
            0 => {
                if let Some(idx) = range_mut.parent {
                    self.remove_range_from_parent(level + 1, idx, old_weight);
                }
            }

            1 => self.update_root_range_weight(level, range_index, old_weight),

            _ => self.update_child_range_weight(level, range_index, old_weight),
        }
    }

    fn update_root_range_weight(&mut self, level: u32, range_index: u32, old_weight: f64) {
        let range = self.get_range_or_fail(level, range_index);
        let new_weight = range.weight;

        let level_weight_increase = if let Some(parent) = range.parent {
            self.remove_range_from_parent(level + 1, parent, old_weight);
            new_weight
        } else {
            new_weight - old_weight
        };

        self.make_root_range(level, range_index);
        self.levels[level as usize].roots_weight += level_weight_increase;
    }

    fn update_child_range_weight(&mut self, level: u32, range_index: u32, old_weight: f64) {
        let range = self.get_range_or_fail(level, range_index);
        let new_weight = range.weight;

        let new_parent_range_index = self.compute_range_index(range.weight);

        if let Some(parent) = range.parent {
            if parent.range_index == new_parent_range_index {
                let weight_increase = new_weight - old_weight;
                return self.update_range_weight(level + 1, parent.range_index, weight_increase);
            }

            self.remove_range_from_parent(level + 1, parent, old_weight);
        }

        // at this point the range has no parent setup
        let index_in_range = self.push_to_parent(
            level + 1,
            new_parent_range_index,
            WeightAndIndex::new(new_weight, range_index as usize),
        );

        self.set_range_parent(
            level,
            range_index,
            Some(RangeIndex {
                range_index: new_parent_range_index,
                index_in_range,
            }),
        );
    }

    fn remove_range_from_parent(&mut self, level: u32, parent: RangeIndex, child_weight: f64) {
        let range = self.get_range_mut_or_insert(level, parent.range_index);
        let swapped_with = range.elements.last().unwrap().index;
        range.elements.swap_remove(parent.index_in_range);
        let did_swap = range.elements.len() != parent.index_in_range;
        range.weight -= child_weight;

        if range.elements.len() == 1 {
            // TODO: actually, we want to reduce this path
            // make myself a root
            let weight = range.weight;
            let level = &mut self.levels[level as usize];

            level.roots.set_bit(parent.range_index as usize);
            level.roots_weight += weight;
        } else if did_swap {
            // keep index of moved range consistent
            let swapped_range = self.levels[level as usize - 1]
                .ranges
                .get_mut(&swapped_with)
                .unwrap();
            swapped_range.parent.as_mut().unwrap().index_in_range = parent.index_in_range;
        }
    }

    fn set_range_parent(&mut self, level: u32, range_index: u32, parent: Option<RangeIndex>) {
        self.get_range_mut_or_insert(level, range_index).parent = parent;
    }

    fn make_root_range(&mut self, level: u32, range_index: u32) -> bool {
        self.set_range_parent(level, range_index, None);

        self.levels[level as usize]
            .roots
            .set_bit(range_index as usize)
    }

    fn sample_level_index<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<usize> {
        linear_sampling_from_iterator(
            rng,
            self.total_weight,
            self.levels.iter().map(|l| l.roots_weight).enumerate(),
        )
    }

    fn compute_range_index(&self, weight: f64) -> u32 {
        const OFFSET: i32 = 1 << (FLOAT_EXP_BITS - 1);
        let log = weight.log2().floor() as i32;
        let result = log + OFFSET;
        assert!(result >= 0);
        result as u32
    }
}

impl Distribution<Option<usize>> for DynamicWeightedIndex {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<usize> {
        if self.total_weight <= 0.0 {
            return None;
        }

        let mut level_idx = self.sample_level_index(rng)?;
        let mut root = self.levels[level_idx].sample_root(rng)?;

        while level_idx > 1 {
            let child = root.sample_child(rng).unwrap();
            level_idx -= 1;
            root = self.levels[level_idx].ranges.get(&child).unwrap();
        }

        root.sample_child(rng)
    }
}

fn linear_sampling_from_iterator<T, R: Rng + ?Sized>(
    rng: &mut R,
    total_weight: f64,
    iter: impl Iterator<Item = (T, f64)>,
) -> Option<T> {
    let mut search = rng.gen_range(0.0..total_weight);

    for (res, w) in iter {
        if search < w {
            return Some(res);
        }

        search -= w;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_float_eq::*;
    use rand::{Rng, SeedableRng};

    const DUMMY_WEIGHT_VALUE: f64 = 3.14159;
    const DUMMY_WEIGHT_VALUE1: f64 = 123.0;

    #[test]
    fn set_weight() {
        let mut dyn_index = DynamicWeightedIndex::new(2);
        assert!(dyn_index.weight(0).is_none());
        assert!(dyn_index.weight(1).is_none());

        dyn_index.set_weight(0, DUMMY_WEIGHT_VALUE);
        assert_f64_near!(dyn_index.weight(0).unwrap(), DUMMY_WEIGHT_VALUE);
        assert!(dyn_index.weight(1).is_none());

        dyn_index.set_weight(1, DUMMY_WEIGHT_VALUE1);
        assert_f64_near!(dyn_index.weight(0).unwrap(), DUMMY_WEIGHT_VALUE);
        assert_f64_near!(dyn_index.weight(1).unwrap(), DUMMY_WEIGHT_VALUE1);
    }

    #[test]
    fn update_weight() {
        let mut dyn_index = DynamicWeightedIndex::new(2);
        assert!(dyn_index.weight(0).is_none());

        dyn_index.set_weight(0, DUMMY_WEIGHT_VALUE);
        assert_f64_near!(dyn_index.weight(0).unwrap(), DUMMY_WEIGHT_VALUE);
        assert_f64_near!(dyn_index.total_weight(), DUMMY_WEIGHT_VALUE);

        dyn_index.set_weight(0, DUMMY_WEIGHT_VALUE1);
        assert_f64_near!(dyn_index.weight(0).unwrap(), DUMMY_WEIGHT_VALUE1);
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
    #[ignore]
    fn set_weight_randomized() {
        let mut rng = pcg_rand::Pcg64::seed_from_u64(0x12345678);

        for n in [2, 10, 100] {
            let mut weights = vec![None; n];
            let mut dyn_index = DynamicWeightedIndex::new(n);

            for _round in 0..n * n {
                let index = rng.gen_range(0..n);
                let new_weight = rng.gen();

                dyn_index.set_weight(index, new_weight);

                // update own copy of weights
                weights[index] = Some(new_weight);

                for (i, w) in weights.iter().enumerate() {
                    if let Some(pw) = w.clone() {
                        let dyn_weight = dyn_index.weight(i).unwrap();
                        assert_f64_near!(pw, dyn_weight);
                    } else {
                        assert!(dyn_index.weight(i).is_none());
                    }
                }

                let total_weight: f64 = weights.iter().map(|w| w.unwrap_or(0.0)).sum();
                assert_f64_near!(dyn_index.total_weight(), total_weight);
            }
        }
    }
}
