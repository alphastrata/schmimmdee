use std::ops::{Add, BitAnd, BitOr, BitXor, Not, Sub};

// Re-export for convenience
pub use crate::base64::*;
pub use crate::bit_gather::*;
pub use crate::branchless_select::*;
pub use crate::chess_moves::*;
pub use crate::crc32::*;
pub use crate::detect_duplicates::*;
pub use crate::extract_bitfields::*;
pub use crate::find_msb::*;
pub use crate::greyscale::*;
pub use crate::hamming_distance::*;
pub use crate::histogram::*;
pub use crate::minmax::*;
pub use crate::pack_saturating::*;
pub use crate::quantized_dot::*;
pub use crate::shuffle::*;
pub use crate::stringpattern::*;
pub use crate::transpose::*;
pub use crate::vfixupimm::*;
pub use crate::vpermt2::*;
pub use crate::vpshufbitqmb::*;
pub use crate::vpternlog::*;
pub use crate::vrange::*;

// Helper for formatting large numbers
pub fn format_number(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut formatted = Vec::with_capacity(s.len() + (s.len() - 1) / 3);
    let first_group_len = bytes.len() % 3;

    if first_group_len != 0 {
        for &b in &bytes[..first_group_len] {
            formatted.push(b);
        }
        if bytes.len() > first_group_len {
            formatted.push(b'\'');
        }
    }

    for (i, &b) in bytes[first_group_len..].iter().enumerate() {
        formatted.push(b);
        if (i + 1) % 3 == 0 && i + 1 != bytes.len() - first_group_len {
            formatted.push(b'\'');
        }
    }
    String::from_utf8(formatted).unwrap()
}

// Helper for formatting nanoseconds to more readable units
pub fn format_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{:.2} ns", ns)
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else {
        format!("{:.2} s", ns / 1_000_000_000.0)
    }
}

// Define a generic trait for SIMD operations
pub trait SimdVector<T>:
    Sized
    + Copy
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + Not<Output = Self>
{
    const LANES: usize;
    fn new(val: T) -> Self;
    fn splat(val: T) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn bitand(self, rhs: Self) -> Self;
    fn bitor(self, rhs: Self) -> Self;
    fn bitxor(self, rhs: Self) -> Self;
    fn not(self) -> Self;
    fn to_array(self) -> [T; 0]; // Placeholder, actual implementation will vary
}

// Placeholder for a generic SIMD type
pub struct Simd<T, const LANES: usize>([T; LANES]);

// Implementations for specific SIMD types would go here, e.g.,
// impl SimdVector<u32> for Simd<u32, 8> { ... }

// Placeholder for LOGICAL_LANES_ constant
const LOGICAL_LANES_: usize = 8; // Example value, adjust as needed

// Modules for each benchmark
mod base64;
mod bit_gather;
mod branchless_select;
mod chess_moves;
mod crc32;
mod detect_duplicates;
mod extract_bitfields;
mod find_msb;
mod greyscale;
mod hamming_distance;
mod histogram;
mod minmax;
mod pack_saturating;
mod quantized_dot;
mod shuffle;
mod stringpattern;
mod transpose;
mod vfixupimm;
mod vpermt2;
mod vpshufbitqmb;
mod vpternlog;
mod vrange;












