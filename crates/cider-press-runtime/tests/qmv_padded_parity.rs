//! K-padding parity gates for `QuantizedWeight`, split into two honest tests
//! along the line where the math is actually exact.
//!
//! # Background: two kinds of "padding" claim
//!
//! Decode pads K from the model hidden dim (896) up to a kernel-friendly
//! width so the quantized matvec can run. There are TWO separable facts in
//! play, and the original single test conflated them:
//!
//!   1. **Padding algebra is bit-exact.** Appending zero-scale / zero-bias
//!      pad groups (with zero packed nibbles, and a zeroed activation tail)
//!      adds exactly `+0.0` to the output. Nothing about the real-element
//!      computation changes.
//!
//!   2. **Selecting a different kernel variant is NOT bit-exact.** The
//!      vendored dispatcher (`kernels/qmv.rs`) picks `affine_qmv_fast` when
//!      `N % BN == 0 && K % 512 == 0`, else the generic `affine_qmv`. The
//!      two tile K differently (generic: 8 values/thread, block 256; fast:
//!      16 values/thread, block 512), so the *grouping* of the float
//!      accumulation differs. Same logical inputs, different summation tree
//!      ⇒ 1–2 bf16-ULP drift on a minority of inputs. An empirical 300-seed
//!      sweep showed cross-variant byte-equality failing on ~6% of seeds;
//!      the seeds that pass byte-equal do so by luck, not by structure.
//!
//! The production decode path does BOTH at once (pad 896→1024, which also
//! crosses 1024 % 512 == 0 into the fast variant), so a naive "padded ==
//! unpadded, byte for byte" gate is a flaky test: it asserts (2) as if it
//! were (1).
//!
//! ## Test A — padding algebra is bit-exact (structural)
//!
//! [`padding_algebra_bit_exact_n896`] / [`..._n128`] pad the logical
//! `[N, 896]` weight to `[N, 960]`. 960 is a multiple of the group size (64)
//! but NOT of 512, so BOTH the unpadded (K=896) and padded (K=960) weights
//! dispatch the **generic** kernel. This isolates fact (1) with NO variant
//! swap, and under that regime byte-equality is *structural*, not luck:
//!
//!   - The generic kernel assigns the real elements at indices `0..896` to
//!     the same thread / block offsets whether the row length is 896 or 960
//!     (the per-thread element grouping is a function of the index, and the
//!     real indices are unchanged). So every per-thread partial over real
//!     elements is bit-identical between the two runs.
//!   - The pad elements at `896..960` contribute `scale·Σ(q·x) + bias·Σx`
//!     with `q = 0`, `scale = 0`, `bias = 0`, AND a zeroed activation tail.
//!     Each pad term is therefore an exact `+0.0`. Adding `+0.0` to a finite
//!     partial is bit-exact (`x + 0.0 == x` for all finite `x`), and the
//!     simd reduction over partials is the same tree on both sides with the
//!     extra terms exactly zero.
//!
//! Therefore all partials and the final reduction are bit-identical, and
//! byte-equality is a legitimate assertion HERE — unlike across variants.
//!
//! The negative controls run UNDER THIS K=960 generic-vs-generic regime:
//!
//!   - **NC1** nonzero pad SCALE, zero bias ⇒ byte-equal: `accum = Σ(q·x) =
//!     0` since every pad nibble is `q = 0`, so `scale·accum = 0` regardless
//!     of the scale value. (Relies on the tail being finite — `0 * NaN` would
//!     poison accum — which the alloc-site slack-zeroing guarantees.)
//!   - **NC2** zero scale, nonzero pad BIAS ⇒ byte-equal ONLY via the
//!     tail-zero invariant: the term is `bias·Σx_tail`, exactly zero IFF the
//!     activation tail reads 0.0. That is guaranteed at the allocation site
//!     (`cider_press_kernels::Device::alloc_buffer` zeroes the rounded
//!     slack), NOT inherited from any "Metal zero-fills allocations"
//!     assumption (Metal makes no such guarantee; the pool recycles). The
//!     LOADER contract stays zero pad-bias as defense-in-depth against a
//!     future allocation path that skipped slack-zeroing.
//!   - **NC3** corrupt a LOGICAL weight byte ⇒ byte-INEQUALITY: proves the
//!     comparison discriminates real weight differences.
//!
//! If Test A ever mismatches at K=960, the structural argument above is
//! false and something deeper is wrong — do not paper over it with a
//! tolerance.
//!
//! ## Test B — the production variant swap is ULP-bounded (tolerance)
//!
//! [`variant_swap_ulp_bounded`] runs the ACTUAL decode path: unpadded
//! `[N, 896]` through the generic kernel vs padded `[N, 1024]` through the
//! **fast** kernel (1024 % 512 == 0). Because the fast variant tiles K
//! differently, low-bit drift is expected and bounded, so this compares as
//! bf16 VALUES with the np.allclose combined bound `|a-b| ≤ atol + rtol·|b|`
//! (atol 0.005, rtol 0.01, per CLAUDE.md parity policy for composed chains)
//! rather than asserting bytes. The combined bound — not a fixed absolute
//! floor — is required because a qmv output reaches magnitude ~16, where a
//! single bf16 ULP is 0.125. It sweeps
//! several seeds so the tolerance claim is exercised beyond a single draw.
//! The product-level gate for this drift is the models-crate greedy-token
//! parity test, not byte-equality here.
//!
//! The slack-zeroing invariant itself is unit-tested directly at its only
//! enforcement point — see `cider-press-kernels/tests/alloc_slack_zeroed.rs`.

#![cfg(target_os = "macos")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names
)]

use cider_press_runtime::{Device, Quantization, QuantizedWeight, Tensor};
use half::bf16;

// Qwen2.5-0.5B decode shapes: q/o_proj have N=896, k/v_proj have N=128.
// Both have K=896 (model hidden dim = 896).
const K_LOGICAL: usize = 896;
const GROUP_SIZE: usize = 64;
const BITS: usize = 4;

// Test A pad target: a group-size multiple that is NOT a 512-multiple, so
// both sides stay on the GENERIC kernel (no variant swap → byte-exact).
const K_PAD_GENERIC: usize = 960;
// Test B pad target: the production width, a 512-multiple → FAST kernel.
const K_PAD_FAST: usize = 1024;

// Derived sizes for the unpadded [N, K_LOGICAL] layout.
// packed words per row: 896 * 4 / 32 = 112 u32; groups per row: 896 / 64 = 14.
const WORDS_PER_ROW_UNPADDED: usize = K_LOGICAL * BITS / 32;
const GROUPS_PER_ROW_UNPADDED: usize = K_LOGICAL / GROUP_SIZE;

// Test A (K=960): packed words 896→960 ⇒ 112→120 (+8 zero u32);
// groups 14→15 (+1 zero bf16 scale and +1 zero bf16 bias).
const WORDS_PER_ROW_GENERIC: usize = K_PAD_GENERIC * BITS / 32;
const GROUPS_PER_ROW_GENERIC: usize = K_PAD_GENERIC / GROUP_SIZE;

// Test B (K=1024): packed words 112→128 (+16); groups 14→16 (+2).
const WORDS_PER_ROW_FAST: usize = K_PAD_FAST * BITS / 32;
const GROUPS_PER_ROW_FAST: usize = K_PAD_FAST / GROUP_SIZE;

// Kernel-selection arithmetic, mirroring `kernels/qmv.rs`:
//   fast = N % BN == 0 && K % 512 == 0   (BN = 8 for these gs/bits)
// Test A MUST stay on the generic kernel — guard that K=960 is NOT a
// 512-multiple, so a future change to K_PAD_GENERIC can't silently slide it
// onto the fast path and re-introduce the cross-variant flake.
const _ASSERT_GENERIC_NOT_FAST: () = assert!(K_PAD_GENERIC % 512 != 0);
const _ASSERT_GENERIC_GROUP_ALIGNED: () = assert!(K_PAD_GENERIC % GROUP_SIZE == 0);
// Test B selects the fast kernel: 1024 % 512 == 0.
const _ASSERT_FAST_SELECTED: () = assert!(K_PAD_FAST % 512 == 0);

// Decode N values; both are multiples of 8 (the fast-path N condition).
const N_QO: usize = 896; // q_proj / o_proj
const N_KV: usize = 128; // k_proj / v_proj

/// Minimal deterministic LCG — no external crate needed.
fn lcg(state: &mut u64) -> u64 {
    // Knuth's multiplier, addend 1 (full-period LCG).
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    *state
}

/// Build deterministic packed-weight bytes for an [N, K] unpadded shape.
/// Returns (weight bytes, scale bytes, bias bytes).
///
/// Packed words are arbitrary nonzero bytes (int4 lanes — exact values
/// don't matter, only that they're consistent between runs). Scales are
/// small positive bf16 (~0.02–0.52); biases small signed bf16. The seeded
/// LCG makes every run produce the same bytes, so comparisons are
/// deterministic per seed.
fn make_unpadded_fixtures(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let words_per_row = k * BITS / 32;
    let groups_per_row = k / GROUP_SIZE;
    let mut state = seed;

    let mut w_bytes: Vec<u8> = Vec::with_capacity(n * words_per_row * 4);
    for _ in 0..n * words_per_row {
        let word = lcg(&mut state) as u32;
        w_bytes.extend_from_slice(&word.to_le_bytes());
    }

    let mut scales_bytes: Vec<u8> = Vec::with_capacity(n * groups_per_row * 2);
    for _ in 0..n * groups_per_row {
        let raw = lcg(&mut state);
        let frac = (raw & 0xFFFF) as f32 / 65536.0;
        let scale = bf16::from_f32(0.02 + 0.5 * frac);
        scales_bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
    }

    let mut biases_bytes: Vec<u8> = Vec::with_capacity(n * groups_per_row * 2);
    for _ in 0..n * groups_per_row {
        let raw = lcg(&mut state);
        let frac = (raw & 0xFFFF) as f32 / 65536.0;
        let bias = bf16::from_f32(-0.05 + 0.1 * frac);
        biases_bytes.extend_from_slice(&bias.to_bits().to_le_bytes());
    }

    (w_bytes, scales_bytes, biases_bytes)
}

/// Deterministic bf16 activation of shape `[1, 1, K_LOGICAL]`.
fn make_activation(seed: u64) -> Vec<bf16> {
    let mut state = seed;
    (0..K_LOGICAL)
        .map(|_| {
            let raw = lcg(&mut state);
            let frac = (raw & 0xFFFF) as f32 / 65536.0;
            bf16::from_f32(-1.0 + 2.0 * frac)
        })
        .collect()
}

/// Pad an unpadded (N × `K_LOGICAL`) weight to (N × `k_padded`), with pad
/// groups carrying `scale_val` / `bias_val`. Pad packed words are always
/// zero (`q = 0`). The two pad targets (960, 1024) differ only in how many
/// zero words / groups are appended.
fn pad_to(
    n: usize,
    k_padded: usize,
    w_bytes: &[u8],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    scale_val: f32,
    bias_val: f32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let words_per_row_padded = k_padded * BITS / 32;
    let groups_per_row_padded = k_padded / GROUP_SIZE;
    let pad_words = words_per_row_padded - WORDS_PER_ROW_UNPADDED;
    let pad_groups = groups_per_row_padded - GROUPS_PER_ROW_UNPADDED;

    let pad_scale_bytes = bf16::from_f32(scale_val).to_bits().to_le_bytes();
    let pad_bias_bytes = bf16::from_f32(bias_val).to_bits().to_le_bytes();

    let mut w_padded: Vec<u8> = Vec::with_capacity(n * words_per_row_padded * 4);
    let mut s_padded: Vec<u8> = Vec::with_capacity(n * groups_per_row_padded * 2);
    let mut b_padded: Vec<u8> = Vec::with_capacity(n * groups_per_row_padded * 2);

    for row in 0..n {
        let w_start = row * WORDS_PER_ROW_UNPADDED * 4;
        let w_end = w_start + WORDS_PER_ROW_UNPADDED * 4;
        w_padded.extend_from_slice(&w_bytes[w_start..w_end]);
        w_padded.extend(std::iter::repeat_n(0u8, pad_words * 4));

        let s_start = row * GROUPS_PER_ROW_UNPADDED * 2;
        let s_end = s_start + GROUPS_PER_ROW_UNPADDED * 2;
        s_padded.extend_from_slice(&scales_bytes[s_start..s_end]);
        for _ in 0..pad_groups {
            s_padded.extend_from_slice(&pad_scale_bytes);
        }

        let b_start = row * GROUPS_PER_ROW_UNPADDED * 2;
        let b_end = b_start + GROUPS_PER_ROW_UNPADDED * 2;
        b_padded.extend_from_slice(&biases_bytes[b_start..b_end]);
        for _ in 0..pad_groups {
            b_padded.extend_from_slice(&pad_bias_bytes);
        }
    }

    (w_padded, s_padded, b_padded)
}

/// Run `quantized_matmul` for both an unpadded `[N, 896]` weight and a
/// padded `[N, k_padded]` weight against the same activation; return the two
/// output tensors' raw byte vecs.
fn run_both(
    device: &Device,
    n: usize,
    k_padded: usize,
    w_bytes: &[u8],
    scales_bytes: &[u8],
    biases_bytes: &[u8],
    w_pad: &[u8],
    s_pad: &[u8],
    b_pad: &[u8],
    activation: &[bf16],
) -> (Vec<u8>, Vec<u8>) {
    let q = Quantization::Q4_GS64;

    let wq_unpadded = QuantizedWeight::from_bytes(
        device,
        [n, K_LOGICAL],
        q,
        w_bytes,
        scales_bytes,
        biases_bytes,
    )
    .expect("build unpadded QuantizedWeight");

    let wq_padded = QuantizedWeight::from_bytes_k_padded(
        device,
        [n, k_padded],
        K_LOGICAL,
        q,
        w_pad,
        s_pad,
        b_pad,
    )
    .expect("build padded QuantizedWeight");

    // Activation [1, 1, K_LOGICAL] — the decode shape.
    let x = Tensor::from_slice(device, activation, [1, 1, K_LOGICAL]).expect("upload activation");

    let y_unpadded = x
        .quantized_matmul(&wq_unpadded, true)
        .expect("schedule unpadded qmv");
    y_unpadded.eval().expect("eval unpadded");

    let y_padded = x
        .quantized_matmul(&wq_padded, true)
        .expect("schedule padded qmv");
    y_padded.eval().expect("eval padded");

    let out_unpadded = y_unpadded.cpu_bytes().expect("cpu_bytes unpadded").to_vec();
    let out_padded = y_padded.cpu_bytes().expect("cpu_bytes padded").to_vec();

    (out_unpadded, out_padded)
}

/// Reinterpret a little-endian bf16 byte vec as bf16 values.
fn as_bf16(bytes: &[u8]) -> Vec<bf16> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// np.allclose combined-bound tolerance comparison: each element must
/// satisfy `|a-b| ≤ abs_tol + rel_tol·|b|`. Sampled mismatch diagnostics
/// on failure.
///
/// NOTE: this deliberately diverges from the rope/softmax parity helpers,
/// which use the *separated* form `max_abs > abs_tol || max_rel > rel_tol`.
/// That form is fine for those kernels because their outputs are
/// bounded-magnitude (rope: rotations ~1; softmax: probabilities in [0,1]),
/// so a fixed `abs_tol ≈ 0.005` is ≈ 1 bf16 ULP across the whole range. A
/// qmv output is an 896-term accumulation that reaches magnitude ~16, where
/// one bf16 ULP is `16·2⁻⁷ = 0.125` — far above any fixed absolute floor.
/// The separated form would flag that genuine 1-ULP regroup drift as a
/// failure (and, worse, aggregates `max_abs`/`max_rel` across *different*
/// elements). CLAUDE.md prescribes the combined bound for exactly this case
/// ("long composed chains ... the np.allclose combined bound").
fn assert_within_tolerance(
    label: &str,
    got: &[bf16],
    expected: &[bf16],
    abs_tol: f32,
    rel_tol: f32,
) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mismatches = 0usize;
    let mut violations = 0usize;
    let mut samples: Vec<(usize, f32, f32, f32)> = Vec::new();
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        if a == b {
            continue;
        }
        mismatches += 1;
        let af = a.to_f32();
        let bf = b.to_f32();
        let abs = (af - bf).abs();
        let rel = if bf.abs() > 1e-6 { abs / bf.abs() } else { 0.0 };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        // np.allclose per-element combined bound.
        if abs > abs_tol + rel_tol * bf.abs() {
            violations += 1;
            if samples.len() < 4 {
                samples.push((i, af, bf, abs));
            }
        }
    }
    if violations > 0 {
        eprintln!(
            "{label}: {violations}/{mismatches} elements exceed combined bound \
             (of {} total); max_abs={max_abs:.6}, max_rel={max_rel:.6}",
            got.len(),
        );
        for (i, g, e, abs) in samples {
            eprintln!("  [{i}] got={g} expected={e} |diff|={abs}");
        }
        panic!(
            "{label}: tolerance exceeded — {violations} element(s) violate \
             |a-b| ≤ {abs_tol} + {rel_tol}·|b| (max_abs={max_abs}, max_rel={max_rel})"
        );
    }
}

// ── Test A: padding algebra is bit-exact (generic-vs-generic, K=960) ─────────

fn run_padding_algebra(n: usize) {
    // Both sides MUST land on the generic kernel: unpadded K=896 (896 % 512
    // != 0) and padded K=960 (960 % 512 != 0). No variant swap ⇒ byte-exact.
    assert!(n % 8 == 0, "N={n} multiple of 8 (decode shape)");
    // K=960 stays on the generic kernel; pad widths are as documented. These
    // are compile-time facts (see the module-level _ASSERT_* consts) — the
    // const blocks keep them visible at the use site without a runtime check.
    const {
        assert!(K_PAD_GENERIC % 512 != 0);
        assert!(WORDS_PER_ROW_GENERIC - WORDS_PER_ROW_UNPADDED == 8);
        assert!(GROUPS_PER_ROW_GENERIC - GROUPS_PER_ROW_UNPADDED == 1);
    }

    let device = Device::shared().expect("Metal device");

    let (w_bytes, scales_bytes, biases_bytes) =
        make_unpadded_fixtures(n, K_LOGICAL, 0xdead_beef_cafe_0042);
    let activation = make_activation(0xfeed_face_dead_1234);

    // ── Primary: zero-scale / zero-bias pad groups ──────────────────────────
    let (w_pad, s_pad, b_pad) = pad_to(
        n,
        K_PAD_GENERIC,
        &w_bytes,
        &scales_bytes,
        &biases_bytes,
        0.0,
        0.0,
    );
    let (out_unpadded, out_padded) = run_both(
        &device,
        n,
        K_PAD_GENERIC,
        &w_bytes,
        &scales_bytes,
        &biases_bytes,
        &w_pad,
        &s_pad,
        &b_pad,
        &activation,
    );
    assert_eq!(
        out_unpadded, out_padded,
        "N={n}: padding to K=960 (generic→generic, no variant swap) must be \
         byte-identical: real-element grouping is unchanged and pad terms add \
         exact +0.0"
    );
    println!(
        "N={n}: padding-algebra primary PASS ({} output bytes; \
         K=960 % 512 = {} ⇒ generic on both sides)",
        out_unpadded.len(),
        K_PAD_GENERIC % 512
    );

    // ── NC1: nonzero pad SCALE, zero bias ⇒ byte-equal ──────────────────────
    // accum = Σ(q·x) = 0 since every pad nibble q = 0, so scale·accum = 0
    // regardless of the scale value (finite tail guaranteed by alloc-site
    // slack-zeroing — else 0 * NaN would poison accum).
    let (wp_s, sp_s, bp_s) = pad_to(
        n,
        K_PAD_GENERIC,
        &w_bytes,
        &scales_bytes,
        &biases_bytes,
        0.5,
        0.0,
    );
    let (out_u1, out_p1) = run_both(
        &device,
        n,
        K_PAD_GENERIC,
        &w_bytes,
        &scales_bytes,
        &biases_bytes,
        &wp_s,
        &sp_s,
        &bp_s,
        &activation,
    );
    assert_eq!(
        out_u1, out_p1,
        "N={n}: NC1 nonzero pad scale=0.5, bias=0 ⇒ byte-equal \
         (scale·Σ(q·x) = scale·0 = 0)"
    );
    println!("N={n}: NC1 (pad scale=0.5, bias=0) PASS — byte-equal");

    // ── NC2: zero scale, nonzero pad BIAS ⇒ byte-equal via tail-zero ────────
    // contribution = scale·accum + bias·Σx_tail = 0 + bias·Σx_tail; this is
    // exactly 0 IFF the activation tail reads 0.0, which the alloc-site
    // slack-zeroing invariant guarantees (NOT inherited from "Metal zero-fills
    // allocations"; the pool recycles). The loader keeps zero pad-bias as
    // defense-in-depth against a future alloc path without slack-zeroing.
    let (wp_b, sp_b, bp_b) = pad_to(
        n,
        K_PAD_GENERIC,
        &w_bytes,
        &scales_bytes,
        &biases_bytes,
        0.0,
        1.0,
    );
    let (out_u2, out_p2) = run_both(
        &device,
        n,
        K_PAD_GENERIC,
        &w_bytes,
        &scales_bytes,
        &biases_bytes,
        &wp_b,
        &sp_b,
        &bp_b,
        &activation,
    );
    assert_eq!(
        out_u2, out_p2,
        "N={n}: NC2 zero scale, pad bias=1.0 ⇒ byte-equal ONLY via the \
         tail-zero invariant (bias·Σx_tail = bias·0 = 0)"
    );
    println!("N={n}: NC2 (pad scale=0, bias=1.0) PASS — byte-equal via tail-zero");

    // ── NC3: corrupted LOGICAL weight byte ⇒ byte-INEQUALITY ────────────────
    // Flip a byte in row 0, group 0 of the padded weight (within logical K).
    let mut w_corrupt = w_pad.clone();
    w_corrupt[0] ^= 0xFF;
    let (out_u3, out_p3) = run_both(
        &device,
        n,
        K_PAD_GENERIC,
        &w_bytes,
        &scales_bytes,
        &biases_bytes,
        &w_corrupt,
        &s_pad,
        &b_pad,
        &activation,
    );
    assert_ne!(
        out_u3, out_p3,
        "N={n}: NC3 corrupted logical weight byte must break byte-equality \
         (discrimination check)"
    );
    println!("N={n}: NC3 (corrupted logical weight) PASS — byte-equality correctly breaks");
}

#[test]
fn padding_algebra_bit_exact_n896() {
    run_padding_algebra(N_QO);
}

#[test]
fn padding_algebra_bit_exact_n128() {
    run_padding_algebra(N_KV);
}

// ── Test B: production variant swap is ULP-bounded (generic vs fast, K=1024) ──

fn run_variant_swap(n: usize) {
    // Unpadded K=896 → generic; padded K=1024 → fast (1024 % 512 == 0,
    // N % 8 == 0). The two kernels tile K differently, so byte-equality is
    // NOT guaranteed; assert bf16-ULP tolerance instead.
    assert!(n % 8 == 0, "N={n} multiple of 8 (fast-path N condition)");
    // K=1024 selects the fast kernel; pad widths are as documented.
    const {
        assert!(K_PAD_FAST % 512 == 0);
        assert!(WORDS_PER_ROW_FAST - WORDS_PER_ROW_UNPADDED == 16);
        assert!(GROUPS_PER_ROW_FAST - GROUPS_PER_ROW_UNPADDED == 2);
    }

    let device = Device::shared().expect("Metal device");

    // CLAUDE.md bf16-ULP parity policy for transcendental/regroup drift.
    let (abs_tol, rel_tol) = (0.005f32, 0.01f32);

    // Sweep several fixed seeds so the tolerance claim is exercised beyond a
    // single draw (each case is one tiny matvec — cheap). The ~6% of seeds
    // that drift cross-variant are exactly what this bound must absorb.
    let seeds: [u64; 8] = [
        0x0000_0000_0000_0001,
        0xdead_beef_cafe_0042,
        0x1234_5678_9abc_def0,
        0xfeed_face_dead_1234,
        0xa5a5_a5a5_a5a5_a5a5,
        0x0f0f_0f0f_0f0f_0f0f,
        0xcafe_d00d_0bad_f00d,
        0xffff_ffff_ffff_fffe,
    ];

    for (i, &seed) in seeds.iter().enumerate() {
        let (w_bytes, scales_bytes, biases_bytes) = make_unpadded_fixtures(n, K_LOGICAL, seed);
        let activation = make_activation(seed ^ 0x9e37_79b9_7f4a_7c15);

        let (w_pad, s_pad, b_pad) = pad_to(
            n,
            K_PAD_FAST,
            &w_bytes,
            &scales_bytes,
            &biases_bytes,
            0.0,
            0.0,
        );
        let (out_generic, out_fast) = run_both(
            &device,
            n,
            K_PAD_FAST,
            &w_bytes,
            &scales_bytes,
            &biases_bytes,
            &w_pad,
            &s_pad,
            &b_pad,
            &activation,
        );
        let generic = as_bf16(&out_generic);
        let fast = as_bf16(&out_fast);
        assert_within_tolerance(
            &format!("N={n} seed[{i}]=0x{seed:016x} generic-vs-fast"),
            &fast,
            &generic,
            abs_tol,
            rel_tol,
        );
        let exact = out_generic == out_fast;
        println!(
            "N={n} seed[{i}]=0x{seed:016x}: variant-swap within bf16-ULP tolerance \
             (byte-exact={exact})"
        );
    }
}

#[test]
fn variant_swap_ulp_bounded_n896() {
    run_variant_swap(N_QO);
}

#[test]
fn variant_swap_ulp_bounded_n128() {
    run_variant_swap(N_KV);
}
