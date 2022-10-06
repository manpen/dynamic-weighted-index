use crate::numeric::FloatingPointParts;
use rand::Rng;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Sub, SubAssign};

pub trait Weight:
    Copy
    + Debug
    + Sized
    + SubAssign
    + PartialEq
    + PartialOrd
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + rand::distributions::uniform::SampleUniform
{
    const NUM_RANGES: u32;
    const ZERO: Self;

    fn compute_range_index(&self) -> u32;
    fn max_weight_of_range_index(index: u32) -> Self;
    fn sample_from_ratio<R: Rng + ?Sized>(rng: &mut R, numerator: Self, denominator: Self) -> bool;
}

macro_rules! weight_impl_int {
    ($t : ty, $test_name : ident) => {
        impl Weight for $t {
            const NUM_RANGES: u32 = <$t>::BITS;
            const ZERO: Self = 0;

            fn compute_range_index(&self) -> u32 {
                debug_assert!(*self > 0);
                Self::NUM_RANGES - 1 - self.leading_zeros()
            }

            fn max_weight_of_range_index(index: u32) -> Self {
                debug_assert!(index < Self::NUM_RANGES);
                ((2 as $t) << index).overflowing_sub(1).0
            }

            fn sample_from_ratio<R: Rng + ?Sized>(
                rng: &mut R,
                numerator: Self,
                denominator: Self,
            ) -> bool {
                debug_assert!(numerator <= denominator);
                rng.gen_range(0..denominator) < numerator
            }
        }

        #[cfg(test)]
        mod $test_name {
            use super::Weight;
            use pcg_rand::Pcg64;
            use rand::{Rng, SeedableRng};

            #[test]
            fn compute_range_index_rand() {
                let mut rng = Pcg64::seed_from_u64(0x1234);
                for _ in 0..1000 {
                    let num: $t = rng.gen();

                    if num == 0 {
                        continue;
                    }

                    let range = num.compute_range_index();

                    if num > 1 {
                        assert_eq!(range.saturating_sub(1), (num / 2).compute_range_index());
                    }

                    if num <= <$t>::MAX / 2 {
                        assert_eq!(range + 1, (num * 2).compute_range_index());
                    }
                }
            }

            #[test]
            fn compute_range_index_max() {
                for i in 0..<$t>::NUM_RANGES {
                    let max_weight = <$t>::max_weight_of_range_index(i);
                    assert_eq!(i, max_weight.compute_range_index());
                }
            }

            #[test]
            fn compute_range_index_max2() {
                for i in 0..(<$t>::NUM_RANGES - 1) {
                    let max_weight = <$t>::max_weight_of_range_index(i);
                    assert_eq!(i + 1, max_weight.compute_range_index() + 1);
                }
            }

            #[test]
            fn num_ranges() {
                assert_eq!(<$t>::MAX.compute_range_index() + 1, <$t>::NUM_RANGES);
            }
        }
    };
}

weight_impl_int!(u8, test_u8);
weight_impl_int!(u16, test_u16);
weight_impl_int!(u32, test_u32);
weight_impl_int!(u64, test_u64);
weight_impl_int!(u128, test_u128);
weight_impl_int!(usize, test_usize);

macro_rules! weight_impl_float {
    ($t : ty, $m : ty, $test_name : ident) => {
        impl Weight for $t {
            const NUM_RANGES: u32 = (<$t>::MAX_EXP - <$t>::MIN_EXP + 2) as u32;
            const ZERO: Self = 0.0;

            fn compute_range_index(&self) -> u32 {
                /*
                the portable way:
                    let log = weight.log2().floor() as i32;
                    let result = log - f64::MIN_EXP;
                    assert!(result >= 0);
                    result as u32
                */

                self.get_exponent() as u32
            }

            fn max_weight_of_range_index(index: u32) -> Self {
                Self::compose(0, 1 + index as $m, false)
            }

            fn sample_from_ratio<R: Rng + ?Sized>(
                rng: &mut R,
                numerator: Self,
                denominator: Self,
            ) -> bool {
                rng.gen_bool((numerator / denominator) as f64)
            }
        }

        #[cfg(test)]
        mod $test_name {
            use super::Weight;
            use pcg_rand::Pcg64;
            use rand::{Rng, SeedableRng};

            #[test]
            fn num_ranges() {
                assert_eq!(<$t>::MAX.compute_range_index() + 1, <$t>::NUM_RANGES);
            }

            #[test]
            fn compute_range_index_rand() {
                let mut rng = Pcg64::seed_from_u64(0x1234);
                for _ in 0..1000 {
                    let num = <$t>::from_bits(rng.gen::<$m>() >> 1);
                    if !num.is_finite() || !num.is_normal() {
                        continue;
                    }

                    let range = num.compute_range_index();

                    if range != 0 {
                        assert_eq!(range.saturating_sub(1), (num / 2.0).compute_range_index());
                    }

                    if range + 1 != <$t>::NUM_RANGES {
                        assert_eq!(
                            range + 1,
                            (num * 2.0).compute_range_index(),
                            "num: {}, 2*num: {}",
                            num,
                            num * 2.0
                        );
                    }
                }
            }
        }
    };
}

weight_impl_float!(f32, u32, test_f32);
weight_impl_float!(f64, u64, test_f64);
