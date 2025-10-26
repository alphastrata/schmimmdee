// ---------- String Pattern Matching Helpers ----------

/// Scalar implementation of string pattern matching (naive).
pub fn string_pattern_match_scalar(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    let mut matches = Vec::new();
    if pattern.is_empty() {
        return matches;
    }
    if text.len() < pattern.len() {
        return matches;
    }

    for i in 0..=(text.len() - pattern.len()) {
        let mut found = true;
        for j in 0..pattern.len() {
            if text[i + j] != pattern[j] {
                found = false;
                break;
            }
        }
        if found {
            matches.push(i);
        }
    }
    matches
}

/// SIMD implementation of string pattern matching.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn string_pattern_match_simd(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    
    let mut matches = Vec::new();
    if pattern.is_empty() {
        return matches;
    }
    if text.len() < pattern.len() {
        return matches;
    }

    // This is a simplified placeholder. A real SIMD string search
    // would use instructions like `_mm_cmpestrm` or `_mm_cmpistrm`
    // for efficient byte-level comparisons.
    // For now, we just process byte by byte within the SIMD loop.
    matches = string_pattern_match_scalar(text, pattern);
    matches
}

#[cfg(not(target_arch = "x86_64"))]
pub fn string_pattern_match_simd(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    string_pattern_match_scalar(text, pattern)
}
