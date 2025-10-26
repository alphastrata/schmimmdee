/// This benchmark demonstrates a "killer application" of the PEXT instruction
/// using chess move generation, which is a specific instance of a common problem:
/// finding adjacencies in a graph represented by a bitmask.
///
/// ## Graph Adjacency & Bitboards
/// A chessboard can be viewed as a graph where squares are nodes and piece moves
/// are edges. An adjacency matrix for this graph can be represented by "bitboards"
/// (64-bit integers), where each bit corresponds to a square.
///
/// To find a rook's moves from a square, we start with its "attack mask" - a
/// bitboard of all squares it could move to on an empty board. This is equivalent
/// to a row in an adjacency matrix.
///
/// ## Move Generation
/// To find a rook's moves, we need to find all squares it can travel to along
/// ranks and files until it hits the edge of the board or another piece.
///
/// ## The PEXT Approach
/// 1.  For a given rook square, we first get a "mask" of all squares it could
///     potentially move to on an empty board.
/// 2.  We then AND this mask with the bitboard of all occupied squares to find
///     the pieces that are actually blocking the rook.
/// 3.  This is where PEXT shines. `_pext_u64(blockers, mask)` creates a unique
///     index (a "perfect hash") for every possible configuration of blockers.
///     It does this by taking only the bits for the blocking pieces and packing
///     them together.
/// 4.  This index is then used to look up the correct move set from a pre-computed
///     attack table.
///
/// This method is extremely fast because it replaces complex, branchy loops
/// with a single instruction and a memory lookup. The PDEP instruction is the
/// logical inverse and is useful for creating the attack tables themselves.

use schmimmdee::{
    format_ns, format_number, rook_moves_pext, rook_moves_scalar,
    init_rook_attack_tables,
};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    if !std::is_x86_feature_detected!("bmi2") {
        eprintln!("Warning: BMI2 not detected. SIMD implementation will fall back to scalar.");
    }

    // Initialize the attack tables required for the PEXT-based move generation.
    println!("Pre-computing rook attack tables...");
    init_rook_attack_tables();
    println!("Done.\n");

    let trials = 100; // More trials, as the operation is very fast.
    let squares_to_test = 64;
    let occupancy_patterns = [
        0x0,                            // Empty board
        0x8040201008040201,             // Some diagonal pieces
        0xFFFFFFFFFFFFFFFF,             // Full board
        0x102000000800,                 // A typical mid-game occupancy
        0b1000000010001000000000100001, // Random-ish
    ];

    println!("{:-^80}", " Chess Rook Move Generation (PEXT) ");
    println!(
        "| {:>12} | {:>15} | {:>15} | {:>10} | {:>10} |",
        "Method", "Scalar", "PEXT", "Speedup", "Valid"
    );
    println!(
        "|{:-^14}|{:-^17}|{:-^17}|{:-^12}|{:-^12}|",
        "", "", "", "", ""
    );

    // Warmup
    black_box(rook_moves_scalar(0, occupancy_patterns[3]));
    unsafe {
        black_box(rook_moves_pext(0, occupancy_patterns[3]));
    }

    // Benchmark
    let scalar_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            for &occupancy in &occupancy_patterns {
                for square in 0..squares_to_test {
                    black_box(rook_moves_scalar(square as u8, occupancy));
                }
            }
            start.elapsed().as_nanos()
        })
        .sum();

    let pext_time: u128 = (0..trials)
        .map(|_| {
            let start = Instant::now();
            for &occupancy in &occupancy_patterns {
                for square in 0..squares_to_test {
                    unsafe {
                        black_box(rook_moves_pext(square as u8, occupancy));
                    }
                }
            }
            start.elapsed().as_nanos()
        })
        .sum();

    let total_ops = (trials * squares_to_test * occupancy_patterns.len()) as f64;
    let avg_scalar_per_op = scalar_time as f64 / total_ops;
    let avg_pext_per_op = pext_time as f64 / total_ops;
    let speedup = avg_scalar_per_op / avg_pext_per_op;

    // Verification
    let mut valid = true;
    'outer: for &occupancy in &occupancy_patterns {
        for square in 0..squares_to_test {
            let scalar_moves = rook_moves_scalar(square as u8, occupancy);
            let pext_moves = unsafe { rook_moves_pext(square as u8, occupancy) };
            if scalar_moves != pext_moves {
                valid = false;
                eprintln!("Validation failed for square {} with occupancy {:x}", square, occupancy);
                eprintln!("Scalar: {:064b}", scalar_moves);
                eprintln!("PEXT:   {:064b}", pext_moves);
                break 'outer;
            }
        }
    }
    assert!(valid, "Results do not match!");

    println!(
        "| {:>12} | {:>15} | {:>15} | {:>9.2}x | {:>9} |",
        "rook_moves",
        format_ns(avg_scalar_per_op),
        format_ns(avg_pext_per_op),
        speedup,
        if valid { "✓" } else { "✗" }
    );
    println!("{:-^80}", "");
    println!("(Times are per-operation averages over {} operations)", format_number(total_ops as usize));
    println!("\nBenchmark complete!");
}
