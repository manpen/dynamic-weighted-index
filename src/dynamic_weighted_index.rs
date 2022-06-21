use ez_bitset::bitset::BitSet;
use rand::prelude::{Distribution, SliceRandom};
use rand::Rng;
use std::collections::HashMap;

pub struct DynamicWeightedIndex {
    indices: Vec<Option<RangeIndex>>,
    levels: Vec<Level>,
    total_weight: f64,
    min_item_weight: f64,
    max_total_weight: f64,
}

#[derive(Default, Debug, Eq, PartialEq, Clone, Copy)]
struct RangeIndex {
    range_index: u32,
    index_in_range: usize,
}

#[derive(Default, Debug, Clone)]
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

#[derive(Default, Debug, Clone)]
struct Range {
    parent: Option<RangeIndex>,
    weight: f64,
    elements: Vec<(f64, usize)>,
}

impl Range {
    fn sample_child<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<usize> {
        let max_weight = 2.0_f64.powf(self.elements.first()?.0.log2() + 1.0);
        loop {
            let &(weight, idx) = self.elements.as_slice().choose(rng)?;
            if weight == max_weight || rng.gen_range(0.0..max_weight) < weight {
                return Some(idx);
            }
        }
    }
}

impl DynamicWeightedIndex {
    /// Constructs a [`DynamicWeightedIndex`] on `n` elements; initially all elements have weight 0.
    pub fn new(n: usize) -> Self {
        Self::with_limits(n, f64::MIN_POSITIVE, f64::MAX)
    }

    /// Constructs a [`DynamicWeightedIndex`] on `n` elements; initially all elements have weight 0.
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

    /// Updates the weight of the `idx`-th element and returns the weight before the update
    /// (`None` if the weight was previously uninitialized)
    pub fn set_weight(&mut self, element_index: usize, new_weight: f64) -> Option<f64> {
        assert!(new_weight == 0.0 || new_weight >= self.min_item_weight);

        let range_index = self.indices[element_index];
        let weight_before = range_index
            .map(|idx| self.get_range(0, idx.range_index).elements[idx.index_in_range].0);

        let new_range_index = compute_range_index(new_weight);

        // already assigned (may change!)
        if let Some(range_index) = range_index {
            // element stays in the same range; just update it's weight
            if range_index.range_index == new_range_index {
                let delta = new_weight - weight_before.unwrap();
                self.update_range(0, new_range_index, delta);
                return weight_before;
            }

            // remove element from range
            let range = self.get_range_mut(0, range_index.range_index);
            let replaced = range.elements.last().unwrap().1;
            range.elements.swap_remove(range_index.index_in_range);

            if replaced != element_index {
                self.indices[replaced].as_mut().unwrap().index_in_range =
                    range_index.index_in_range;
            }

            self.total_weight -= weight_before.unwrap();
            self.update_range(0, range_index.range_index, -weight_before.unwrap());

            self.indices[element_index] = None; // TODO: debug only
        }

        // element is unassigned at this point
        self.total_weight += new_weight;
        let index_in_range = self.push_to_parent(0, new_range_index, element_index, new_weight);
        self.indices[element_index] = Some(RangeIndex {
            range_index: new_range_index,
            index_in_range,
        });

        weight_before
    }

    fn push_to_parent(
        &mut self,
        parent_level: u32,
        parent_range_index: u32,
        element_index: usize,
        weight: f64,
    ) -> usize {
        let range = self.get_range_mut(parent_level, parent_range_index);
        let index_in_range = range.elements.len();
        range.elements.push((weight, element_index));
        self.update_range(parent_level, parent_range_index, weight);
        index_in_range
    }

    fn get_range(&self, level: u32, range: u32) -> &Range {
        self.levels[level as usize]
            .ranges
            .get(&(range as usize))
            .unwrap()
    }

    fn get_range_mut(&mut self, level: u32, range: u32) -> &mut Range {
        self.levels[level as usize]
            .ranges
            .entry(range as usize)
            .or_insert(Default::default())
    }

    fn update_range(&mut self, level: u32, range_index: u32, delta: f64) {
        let was_root = self.levels[level as usize].roots[range_index as usize];

        let range_mut = self.get_range_mut(level, range_index);
        let old_weight = range_mut.weight;
        range_mut.weight += delta;
        let new_weight = range_mut.weight;
        let num_elements = range_mut.elements.len();
        let parent = range_mut.parent;

        if num_elements == 1 {
            // is root
            let was_root = self.levels[level as usize]
                .roots
                .set_bit(range_index as usize);

            if let Some(idx) = parent {
                self.remove_range_from_parent(
                    level + 1,
                    idx.range_index,
                    idx.index_in_range,
                    old_weight,
                );
            } else if was_root {
                self.levels[level as usize].roots_weight -= old_weight;
            }

            self.levels[level as usize].roots_weight += new_weight;
        } else if num_elements > 1 {
            // is child
            let new_parent_range_index = compute_range_index(range_mut.weight);

            if let Some(parent) = parent {
                if parent.range_index == new_parent_range_index {
                    return self.update_range(level + 1, parent.range_index, delta);
                }

                self.remove_range_from_parent(
                    level + 1,
                    parent.range_index,
                    parent.index_in_range,
                    old_weight,
                );
            }

            // at this point the range has no parent setup
            let index_in_range = self.push_to_parent(
                level + 1,
                new_parent_range_index,
                range_index as usize,
                new_weight,
            );

            self.get_range_mut(level, range_index).parent = Some(RangeIndex {
                range_index: new_parent_range_index,
                index_in_range,
            });
        } else {
            // is deleted
            assert_eq!(range_mut.weight, delta);
            if let Some(idx) = range_mut.parent {
                self.remove_range_from_parent(
                    level + 1,
                    idx.range_index,
                    idx.index_in_range,
                    old_weight,
                );
            }
        }
    }

    fn remove_range_from_parent(
        &mut self,
        level: u32,
        range: u32,
        index_in_range: usize,
        child_weight: f64,
    ) {
        let range = self.get_range_mut(level, range);
        let did_swap = range.elements.len() != index_in_range + 1;
        let swapped_with = range.elements.last().unwrap().1;
        range.elements.swap_remove(index_in_range);
        range.weight -= child_weight;

        if did_swap {}

        if range.elements.len() == 1 {
            // make myself a root
            let weight = range.weight;
            let level = &mut self.levels[level as usize];

            level.roots.set_bit(range as usize);
            level.roots_weight += weight;
        }
    }

    /// Returns the weight of the `idx`-th element (`None` if the element is uninitialized)
    pub fn get_weight(&self, idx: usize) -> Option<f64> {
        let idx = self.indices[idx]?;
        Some(self.levels[0].ranges[&(idx.range_index as usize)].elements[idx.index_in_range].0)
    }

    fn sample_level_index<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<usize> {
        linear_sampling_from_iterator(
            rng,
            self.total_weight,
            self.levels.iter().map(|l| l.roots_weight).enumerate(),
        )
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

fn compute_range_index(weight: f64) -> u32 {
    weight.log2().floor() as u32
}
