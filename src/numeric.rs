pub trait FloatingPointParts {
    const BITS_MANTISSA: usize;
    const BITS_EXPONENT: usize;
    type BitsType;

    fn get_mantissa(&self) -> Self::BitsType;
    fn get_exponent(&self) -> Self::BitsType;
    fn get_sign(&self) -> bool;

    fn compose(mantissa: Self::BitsType, exponent: Self::BitsType, sign: bool) -> Self;
}

macro_rules! fpp_impl {
    () => {
        fn get_mantissa(&self) -> Self::BitsType {
            self.to_bits() & ((1 << Self::BITS_MANTISSA) - 1)
        }

        fn get_exponent(&self) -> Self::BitsType {
            (self.to_bits() >> Self::BITS_MANTISSA) & ((1 << Self::BITS_EXPONENT) - 1)
        }

        fn get_sign(&self) -> bool {
            (self.to_bits() >> (Self::BITS_EXPONENT + Self::BITS_MANTISSA)) == 1
        }

        fn compose(mantissa: Self::BitsType, exponent: Self::BitsType, sign: bool) -> Self {
            Self::from_bits(
                mantissa
                    | (exponent << Self::BITS_MANTISSA)
                    | ((sign as Self::BitsType) << (Self::BITS_EXPONENT + Self::BITS_MANTISSA)),
            )
        }
    };
}

impl FloatingPointParts for f32 {
    const BITS_MANTISSA: usize = 23;
    const BITS_EXPONENT: usize = 8;
    type BitsType = u32;

    fpp_impl!();
}

impl FloatingPointParts for f64 {
    const BITS_MANTISSA: usize = 52;
    const BITS_EXPONENT: usize = 11;
    type BitsType = u64;

    fpp_impl!();
}
