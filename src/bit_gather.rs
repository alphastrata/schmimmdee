// ---------- Bit Gather (PDEP/PEXT) Helpers ----------

/// Scalar implementation of bit gather (PDEP).
pub fn gather_scalar(data: u64, control: u64) -> u64 {
    let mut result = 0;
    let mut data_idx = 0;
    for i in 0..64 {
        if (control >> i) & 1 == 1 {
            if (data >> data_idx) & 1 == 1 {
                result |= 1 << i;
            }
            data_idx += 1;
        }
    }
    result
}

/// Scalar implementation of bit scatter (PEXT).
pub fn scatter_scalar(data: u64, control: u64) -> u64 {
    let mut result = 0;
    let mut result_idx = 0;
    for i in 0..64 {
        if (control >> i) & 1 == 1 {
            if (data >> i) & 1 == 1 {
                result |= 1 << result_idx;
            }
            result_idx += 1;
        }
    }
    result
}

/// SIMD implementation of bit gather (PDEP).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
pub unsafe fn gather_simd(data: u64, control: u64) -> u64 {
    use std::arch::x86_64::*;
    _pdep_u64(data, control)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn gather_simd(data: u64, control: u64) -> u64 {
    gather_scalar(data, control)
}

/// SIMD implementation of bit scatter (PEXT).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
pub unsafe fn scatter_simd(data: u64, control: u64) -> u64 {
    use std::arch::x86_64::*;
    _pext_u64(data, control)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn scatter_simd(data: u64, control: u64) -> u64 {
    scatter_scalar(data, control)
}
