This is just a resource for a blog post on SIMD.

In a nutshell, it compraes a SIMD and Scalar min/max finding functions on increasingly larger arrays of `f32`s.

Results:
|     Elements |          Scalar |            SIMD |    Speedup |
|--------------|-----------------|-----------------|------------|
|          1e3 |          2.57μs |        519.18ns |      4.96x |
|          1e4 |         21.50μs |          4.37μs |      4.92x |
|          1e5 |        224.34μs |         47.69μs |      4.70x |
|          1e6 |          2.13ms |        447.59μs |      4.76x |
|          1e7 |         21.38ms |          4.73ms |      4.51x |
|          1e8 |        213.87ms |         51.53ms |      4.15x |

> 2013 Macbook Pro

NOTE that it has specific `rustflags` set in the `./cargo/config.toml` for studying the Assembly.
i.e 
```sh
cargo build -r
cat target/release/deps/schmimmdee-cd6afcb096a75bb5.s
```
- the hash will be different for every build.

- the linetables are inlcuded so you can match lines of code i.e:
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

Because I'm interested in it.
