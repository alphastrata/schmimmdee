# SIMD experiments

This is just a resource for a blog post on SIMD.

The `./bins` directory contains all the experiments.

# minmax

Usage: `cargo run -r --bin minmax`

Compare SIMD and Scalar min/max finding functions on increasingly larger arrays of `f32`s.

<details><summary>Results:</summary>

> i7-4960HQ

|     Elements |          Scalar |            SIMD |    Speedup |
|--------------|-----------------|-----------------|------------|
|          1e3 |          2.57μs |        519.18ns |      4.96x |
|          1e4 |         21.50μs |          4.37μs |      4.92x |
|          1e5 |        224.34μs |         47.69μs |      4.70x |
|          1e6 |          2.13ms |        447.59μs |      4.76x |
|          1e7 |         21.38ms |          4.73ms |      4.51x |
|          1e8 |        213.87ms |         51.53ms |      4.15x |


<br>

> CPU: AMD Ryzen 9 5950X (32) @ 5.084GHz

|     Elements |          Scalar |            SIMD |    Speedup |
|--------------|-----------------|-----------------|------------|
|          1e3 |        937.00ns |        182.00ns |      5.15x |
|          1e4 |          9.34μs |          1.60μs |      5.84x |
|          1e5 |         88.22μs |         15.29μs |      5.77x |
|          1e6 |        905.53μs |        156.50μs |      5.79x |
|          1e7 |          9.20ms |          2.12ms |      4.35x |
|          1e8 |         90.77ms |         22.10ms |      4.11x |


</details>
<br>

---


# searching-pattern

Usage: `cargo run -r --bin string-pattern`

Uses this dataset: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz
(A list of the `titles` of all wikipedia articles).

<details><summary>Results:</summary>

> CPU: AMD Ryzen 9 5950X (32) @ 5.084GHz
> 'Path of Exile 2'  

|       Method |         Std Lib |            SIMD |    Speedup |      Valid |
|--------------|-----------------|-----------------|------------|------------|
|         find |        128.90ms |         48.90ms |      2.64x |         ✓ |

> 'AVX-512'  

|       Method |         Std Lib |            SIMD |    Speedup |      Valid |
|--------------|-----------------|-----------------|------------|------------|
|         find |          7.75ms |          8.56ms |      0.90x |         ✓ |

> 'Bannana'

|       Method |         Std Lib |            SIMD |    Speedup |      Valid |
|--------------|-----------------|-----------------|------------|------------|
|         find |         10.16ms |          9.21ms |      1.10x |         ✓ |

</details>

# histogram

Usage: `cargo run -r --bin histogram`

> this one isn't great.

Uses this dataset: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz
(A list of the `titles` of all wikipedia articles).

> CPU: AMD Ryzen 7 7840U
<details><summary>Results:</summary>

|     Elements |        Standard |            SIMD |    Speedup |      Valid |
|--------------|-----------------|-----------------|------------|------------|
|         1.0K |        398.90ns |        369.60ns |      1.08x |         ✓ |
|        10.0K |          3.22μs |          3.08μs |      1.05x |         ✓ |
|       100.0K |         32.53μs |         30.72μs |      1.06x |         ✓ |
|         1.0M |        345.88μs |        321.89μs |      1.07x |         ✓ |
|        10.0M |          3.19ms |          3.01ms |      1.06x |         ✓ |
|       404.2M |        128.03ms |        121.82ms |      1.05x |         ✓ |
</details>

# greyscale
> this one is my fav of all of the impls here

Usage: `cargo run -r --bin greyscale`

> CPU: AMD Ryzen 7 7840U
<details><summary>Results:</summary>

```txt
SIMD: 561.793µs, Baseline: 838.792µs
SIMD: 904.686µs, Baseline: 2.088766ms
```

Not bad!
</details>



# NOTES

**NOTE** that it has specific `rustflags` set in the `./cargo/config.toml` for studying the Assembly.
i.e 
```sh
cargo build -r --bin minmax
cat target/release/deps/schmimmdee-cd6afcb096a75bb5.s
```
- the hash will be different for every build.

- the linetables are included so you can match lines of code i.e:
```rust
    min_vec = min_vec.simd_min(values);
```

to lines of asm i.e:
```asm
LCPI7_1:
	.long	0xff7fffff
	.long	0xff7fffff
	.long	0xff7fffff
	.long	0xff7fffff
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_find_min_max_simd
	.p2align	4
_find_min_max_simd:
Lfunc_begin7:
	.file	18 "/Users/smak/Documents/schmimmdee" "src/main.rs"
	.loc	18 12 0 is_stmt 1
	.cfi_startproc
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset rbp, -16
	mov	rbp, rsp
	.cfi_def_cfa_register rbp
Ltmp45:
	.file	19 "/Users/smak/.rustup/toolchains/nightly-x86_64-apple-darwin/lib/rustlib/src/rust/library/core/src/slice" "iter.rs"
	.loc	19 1912 12 prologue_end
	mov	rax, rsi
	and	rax, -16
Ltmp46:
	je	LBB7_1
Ltmp47:
	.loc	19 0 12 is_stmt 0
	movaps	xmm5, xmmword ptr [rip + LCPI7_1]
	movaps	xmm10, xmmword ptr [rip + LCPI7_0]
	xor	ecx, ecx
	movaps	xmm0, xmm10
	movaps	xmm11, xmm10
	movaps	xmm9, xmm10
	movaps	xmm8, xmm5
	movaps	xmm7, xmm5
	movaps	xmm6, xmm5
```

For the uninitiated `xmm0` and the `xmm` like registers are from `sse`, `ymm` from `avx` and `avx2` and `zmm` would be from `avx512`.
