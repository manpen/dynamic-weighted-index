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
    /// given time. Both parameters are hints to the data structure to optimize performance.
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
    pub fn set_weight(&mut self, index: usize, new_weight: f64) {
        if new_weight == 0.0 {
            return self.remove_weight(index);
        }

        assert!(new_weight >= self.min_item_weight);
        self.total_weight += new_weight;

        let new_range_id = self.compute_range_index(new_weight);

        // if element is already assigned, remove it from that range
        if let Some(prev_range_index) = self.indices[index] {
            let old_weight = self.weight(index).unwrap();
            self.total_weight -= old_weight;

            // short-cut: element stays in the same range; just update its weight
            if prev_range_index.range_index == new_range_id {
                let weight_increase = new_weight - old_weight;

                let parent = self.get_range_mut_or_fail(0, new_range_id);
                let (old_parent_weight, _) = parent.increase_weight(weight_increase);
                parent.elements[prev_range_index.index_in_range].weight = new_weight;

                self.update_range_weight(0, new_range_id, old_parent_weight);
                return;
            }

            self.remove_element_from_range(0, prev_range_index);
            self.indices[index] = None; // TODO: this may be optimized away by dropping the assertion in [`DynamicWeightedIndex::initialize_weight`]
        }

        // weight is unassigned at this point
        self.initialize_weight(
            new_range_id,
            IndexAndWeight {
                weight: new_weight,
                index,
            },
        );
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

    pub fn sample_index_and_weight<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<IndexAndWeight> {
        if self.total_weight <= 0.0 {
            return None;
        }

        let mut level_idx = self.sample_level_index(rng)?;
        let mut root = self.levels[level_idx].sample_root(rng)?;

        while level_idx > 0 {
            let child = root.sample_child(rng).unwrap();
            level_idx -= 1;
            root = self.levels[level_idx].ranges.get(&child.index).unwrap();
        }

        root.sample_child(rng)
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
pub struct IndexAndWeight {
    pub weight: f64,
    pub index: usize,
}

#[derive(Default, Debug, Clone)]
struct Range {
    parent: Option<RangeIndex>,
    weight: f64,
    max_element_weight: f64,
    elements: Vec<IndexAndWeight>,
}

impl Range {
    fn sample_child<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<IndexAndWeight> {
        if self.elements.len() == 1 {
            Some(self.elements[0])
        } else {
            debug_assert!(self
                .elements
                .iter()
                .all(|x| x.weight <= self.max_element_weight));

            loop {
                let &element = self.elements.as_slice().choose(rng)?;
                if element.weight == self.max_element_weight
                    || rng.gen_range(0.0..self.max_element_weight) < element.weight
                {
                    break Some(element);
                }
            }
        }
    }

    fn increase_weight(&mut self, delta: f64) -> (f64, f64) {
        let old = self.weight;
        let new = self.weight + delta;
        self.weight = new;
        (old, new)
    }
}

impl DynamicWeightedIndex {
    fn initialize_weight(&mut self, range_id: u32, new_element: IndexAndWeight) {
        debug_assert!(self.indices[new_element.index].is_none());
        let index_in_range = self.push_element_to_range(0, range_id, new_element);

        self.indices[new_element.index] = Some(RangeIndex {
            range_index: range_id,
            index_in_range,
        });
    }

    fn remove_weight(&mut self, index: usize) {
        if let Some(weight) = self.weight(index) {
            self.total_weight -= weight;
            let range_index = self.indices[index].unwrap();

            self.indices[index] = None;
            self.remove_element_from_range(0, range_index);
        }
    }

    fn push_element_to_range(
        &mut self,
        level_index: u32,
        range_index: u32,
        new_element: IndexAndWeight,
    ) -> usize {
        let range = self.get_range_mut_or_insert(level_index, range_index);
        let old_num_elements = range.elements.len();

        // handle weights
        let (old_weight, new_weight) = range.increase_weight(new_element.weight);

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
                let new_parent_range_index = self.compute_range_index(new_weight);
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
        let range = self.get_range_mut_or_fail(level_index, range_index.range_index);
        let parent = range.parent;
        let old_num_elements = range.elements.len();

        // update weight
        let weight_removed = range.elements[range_index.index_in_range].weight;
        let (old_weight, _new_weight) = range.increase_weight(-weight_removed);

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
        weight: f64,
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

    /// Updates the weight of a range in its parent's structure (the range's weight itself is not
    /// affected). Preconditions:
    ///  - The range must have had multiple children previously and has to keep them
    ///    (i.e. it is not a root)
    ///  - The weight record with the parent is unchanged and will be updated here
    fn update_range_weight(&mut self, level: u32, range_index: u32, old_weight: f64) {
        let range = self.get_range_or_fail(level, range_index);
        let new_weight = range.weight;

        if range.elements.len() == 1 {
            self.levels[level as usize].roots_weight += new_weight - old_weight;
        } else {
            let parent = range.parent.unwrap();

            let new_parent_range_index = self.compute_range_index(new_weight);

            if false {
                // TODO: implement short-cut "parent.range_index == new_parent_range_index"
            } else {
                self.remove_element_from_range(level + 1, parent);
                self.push_range_to_parent(level, range_index, new_parent_range_index, new_weight);
            }
        }
    }

    fn get_range_or_fail(&self, level: u32, range: u32) -> &Range {
        self.levels[level as usize]
            .ranges
            .get(&(range as usize))
            .unwrap()
    }

    fn get_range_mut_or_fail(&mut self, level: u32, range: u32) -> &mut Range {
        self.levels[level as usize]
            .ranges
            .get_mut(&(range as usize))
            .unwrap()
    }

    fn get_range_mut_or_insert(&mut self, level: u32, range_index: u32) -> &mut Range {
        self.levels[level as usize]
            .ranges
            .entry(range_index as usize)
            .or_insert_with(|| Range {
                max_element_weight: Self::max_weight_of_range_index(range_index),
                ..Default::default()
            })
    }

    fn set_range_parent(&mut self, level: u32, range_index: u32, parent: RangeIndex) {
        self.get_range_mut_or_insert(level, range_index).parent = Some(parent);
    }

    fn make_root_range(&mut self, level: u32, range_index: u32) {
        let range = self.get_range_mut_or_insert(level, range_index);
        range.parent = None;
        self.levels[level as usize].roots_weight += range.weight;

        self.levels[level as usize]
            .roots
            .set_bit(range_index as usize);
    }

    fn remove_root_range(&mut self, level: u32, range_index: u32, weight: f64) {
        self.levels[level as usize].roots_weight -= weight;

        self.levels[level as usize]
            .roots
            .unset_bit(range_index as usize);
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

    fn max_weight_of_range_index(index: u32) -> f64 {
        const OFFSET: i32 = 1 << (FLOAT_EXP_BITS - 1);
        let index = (index as i32) - OFFSET;
        2.0_f64.powi(index + 1)
    }
}

impl Distribution<Option<usize>> for DynamicWeightedIndex {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<usize> {
        self.sample_index_and_weight(rng).map(|x| x.index)
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
mod test {
    use super::*;
    use assert_float_eq::*;
    use rand::{Rng, SeedableRng};
    use statrs::distribution::{Binomial, DiscreteCDF};

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
                        let dyn_weight = dyn_index.weight(i).unwrap();
                        assert_f64_near!(pw, dyn_weight);
                    } else {
                        assert!(dyn_index.weight(i).is_none());
                    }
                }

                let total_weight: f64 = weights.iter().map(|w| w.unwrap_or(0.0)).sum();
                assert_f64_near!(dyn_index.total_weight(), total_weight, 1000);
            }
        }
    }

    #[test]
    fn sample_single() {
        let mut rng = pcg_rand::Pcg64::seed_from_u64(0x1234567);

        for n in [2, 10, 30] {
            let mut dyn_index = DynamicWeightedIndex::new(n);
            for _ in 0..10 * n {
                let index = rng.gen_range(0..n);
                let new_weight = rng.gen_range(1e-100..1e100);

                // set a single weight; we always have to sample this one element
                dyn_index.set_weight(index, new_weight);
                for _j in 0..5 {
                    assert_eq!(dyn_index.sample(&mut rng), Some(index));
                }

                // unset weight --- no sample possible
                dyn_index.set_weight(index, 0.0);
                for _j in 0..5 {
                    assert_eq!(dyn_index.sample(&mut rng), None);
                }
            }
        }
    }

    #[test]
    fn sample_multiple() {
        const SAMPLES_PER_ELEMENT: usize = 10000;
        let mut rng = pcg_rand::Pcg64::seed_from_u64(0x123456);

        for n in [2, 5, 20] {
            let mut dyn_index = DynamicWeightedIndex::new(n);
            let mut weights = vec![0.0; n];

            for _round in 0..10 {
                for _j in 0..n {
                    let index = rng.gen_range(0..n);
                    let new_weight = if rng.gen() { rng.gen() } else { 0.0 }; // 50% to be empty
                    dyn_index.set_weight(index, new_weight);
                    weights[index] = new_weight;
                }

                if weights.iter().copied().sum::<f64>() == 0.0 {
                    continue;
                }

                let mut counts = vec![0; n];
                for _j in 0..SAMPLES_PER_ELEMENT * n {
                    counts[dyn_index.sample(&mut rng).unwrap()] += 1;
                }

                verify_multinomial(&weights, &counts, 0.05);
            }
        }
    }

    fn verify_multinomial(weights: &[f64], counts: &[u64], significance: f64) {
        assert_eq!(weights.len(), counts.len());
        let num_total_counts = counts.iter().sum::<u64>();
        let total_weight = weights.iter().sum::<f64>();

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

                // compute two-sided p-value, i.e. the probabilty that more extreme values are
                // produced by the binomial distribution
                let pvalue = if mean >= count as f64 {
                    2.0 * distr.cdf(count)
                } else {
                    2.0 * (1.0 - distr.cdf(count - 1))
                };

                assert!(
                    pvalue >= corrected_significance,
                    "prob: {} mean: {} count: {} p-value: {}",
                    probabilty,
                    mean,
                    count,
                    pvalue
                );
            }
        }
    }
}
