//! Roundtrip tests for [`KvCache`].
//!
//! `KvCache` is a pure data-mover — no algorithm to match against MLX.
//! These tests cover the only correctness questions worth asking:
//!
//! - Writing N rows of distinguishable K/V values across multiple
//!   `update()` calls leaves the populated prefix readable through
//!   `keys_view` / `values_view`, with bytes equal to what we wrote.
//! - `position()` tracks correctly and rejects overflowing writes.
//! - `reset()` rewinds without leaking bytes from the prior run.
//! - Validation errors fire on dtype / shape mismatches. (The device
//!   check in `validate_update_input` is correct but not exercised:
//!   Apple Metal returns the same singleton from
//!   `Device::system_default()`, so two distinct device handles aren't
//!   constructible on a single-GPU Mac.)

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use cider_press_runtime::{DType, Device, KvCache, Tensor};
use half::bf16;

const MAX_TOKENS: usize = 8;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 4;

#[test]
fn roundtrip_appends_and_reads_back_populated_prefix() {
    let device = Device::system_default().expect("device");
    let mut cache =
        KvCache::new(&device, MAX_TOKENS, N_KV_HEADS, HEAD_DIM, DType::BF16).expect("alloc");
    assert_eq!(cache.position(), 0);

    // Append in three chunks (2 + 3 + 1 = 6 rows) so we exercise
    // multi-row `update` and a non-trivial running offset.
    //
    // KvCache's append-only contract requires that prior rows are committed
    // (via `eval`) before the next `update` — `update` bases its
    // `slice_update` op directly on the slab leaf, so unevaluated writes
    // from a prior step are not in the graph for the next step's base. Each
    // iteration evaluates the current prefix before appending the next chunk.
    let chunks: [usize; 3] = [2, 3, 1];
    let mut keys_expected: Vec<bf16> = Vec::new();
    let mut values_expected: Vec<bf16> = Vec::new();

    let mut next_value: f32 = 1.0;
    let mut chunks_pos = 0usize;
    for &step_t in &chunks {
        let k = make_chunk_tensor(&device, step_t, |row, h, d| {
            value(next_value, chunks_pos + row, h, d, 1.0)
        });
        let v = make_chunk_tensor(&device, step_t, |row, h, d| {
            value(next_value, chunks_pos + row, h, d, 100.0)
        });
        cache.update(&k, &v).expect("update");
        // Commit prior rows into the slab before the next update builds on it.
        cache.keys_view().eval().expect("eval keys after chunk");
        cache.values_view().eval().expect("eval values after chunk");
        // Mirror the per-element values into the expected vec.
        for row in 0..step_t {
            for h in 0..N_KV_HEADS {
                for d in 0..HEAD_DIM {
                    keys_expected.push(bf16::from_f32(value(
                        next_value,
                        chunks_pos + row,
                        h,
                        d,
                        1.0,
                    )));
                    values_expected.push(bf16::from_f32(value(
                        next_value,
                        chunks_pos + row,
                        h,
                        d,
                        100.0,
                    )));
                }
            }
        }
        chunks_pos += step_t;
        next_value += 1.0;
    }

    let total: usize = chunks.iter().sum();
    assert_eq!(cache.position(), total);

    let keys_view = cache.keys_view();
    let values_view = cache.values_view();
    assert_eq!(keys_view.shape().dims(), &[total, N_KV_HEADS, HEAD_DIM]);
    assert_eq!(values_view.shape().dims(), &[total, N_KV_HEADS, HEAD_DIM]);
    assert_eq!(keys_view.dtype(), DType::BF16);

    keys_view.eval().expect("eval keys_view");
    values_view.eval().expect("eval values_view");
    let got_keys: Vec<bf16> = keys_view.cpu_to_vec().expect("keys view bf16 vec");
    let got_values: Vec<bf16> = values_view.cpu_to_vec().expect("values view bf16 vec");
    assert_eq!(got_keys, keys_expected);
    assert_eq!(got_values, values_expected);
}

#[test]
fn views_at_position_zero_are_empty_but_valid() {
    let device = Device::system_default().expect("device");
    let cache =
        KvCache::new(&device, MAX_TOKENS, N_KV_HEADS, HEAD_DIM, DType::BF16).expect("alloc");
    let k = cache.keys_view();
    assert_eq!(k.shape().dims(), &[0, N_KV_HEADS, HEAD_DIM]);
    assert_eq!(k.cpu_to_vec::<bf16>(), Some(Vec::new()));
}

#[test]
fn reset_rewinds_position_and_subsequent_updates_overwrite() {
    let device = Device::system_default().expect("device");
    let mut cache =
        KvCache::new(&device, MAX_TOKENS, N_KV_HEADS, HEAD_DIM, DType::BF16).expect("alloc");

    let first = make_chunk_tensor(&device, 3, |row, h, d| (row * 100 + h * 10 + d) as f32);
    cache.update(&first, &first).expect("first update");
    assert_eq!(cache.position(), 3);

    // reset() requires the last update to be evaluated (so no outstanding
    // view can replay a stale write into the reused slab).
    cache.keys_view().eval().expect("eval keys before reset");
    cache
        .values_view()
        .eval()
        .expect("eval values before reset");
    cache.reset().expect("reset after eval");
    assert_eq!(cache.position(), 0);

    let second = make_chunk_tensor(&device, 2, |row, h, d| -((row * 7 + h * 3 + d + 1) as f32));
    cache.update(&second, &second).expect("second update");
    assert_eq!(cache.position(), 2);

    let kview = cache.keys_view();
    kview.eval().expect("eval keys after reset");
    let got: Vec<bf16> = kview.cpu_to_vec().expect("keys after reset");
    let expected: Vec<bf16> = (0..2)
        .flat_map(|row| {
            (0..N_KV_HEADS).flat_map(move |h| {
                (0..HEAD_DIM).map(move |d| bf16::from_f32(-((row * 7 + h * 3 + d + 1) as f32)))
            })
        })
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn overflowing_update_is_rejected() {
    let device = Device::system_default().expect("device");
    let mut cache =
        KvCache::new(&device, MAX_TOKENS, N_KV_HEADS, HEAD_DIM, DType::BF16).expect("alloc");
    let big = make_chunk_tensor(&device, MAX_TOKENS - 1, |_, _, _| 0.0);
    cache.update(&big, &big).expect("first fits");
    let one_too_many = make_chunk_tensor(&device, 2, |_, _, _| 0.0);
    let err = cache
        .update(&one_too_many, &one_too_many)
        .expect_err("must reject");
    let msg = format!("{err}");
    assert!(msg.contains("max_tokens"), "unexpected error: {msg}");
    // Position must NOT have advanced after a rejected write.
    assert_eq!(cache.position(), MAX_TOKENS - 1);
}

#[test]
fn shape_mismatch_is_rejected() {
    let device = Device::system_default().expect("device");
    let mut cache =
        KvCache::new(&device, MAX_TOKENS, N_KV_HEADS, HEAD_DIM, DType::BF16).expect("alloc");
    // Wrong n_kv_heads.
    let wrong = Tensor::zeros(&device, [1, N_KV_HEADS + 1, HEAD_DIM], DType::BF16).expect("zeros");
    let err = cache.update(&wrong, &wrong).expect_err("must reject");
    assert!(format!("{err}").contains("n_kv_heads"));
}

#[test]
fn dtype_mismatch_is_rejected() {
    let device = Device::system_default().expect("device");
    let mut cache =
        KvCache::new(&device, MAX_TOKENS, N_KV_HEADS, HEAD_DIM, DType::BF16).expect("alloc");
    let wrong = Tensor::zeros(&device, [1, N_KV_HEADS, HEAD_DIM], DType::F32).expect("zeros");
    let err = cache.update(&wrong, &wrong).expect_err("must reject");
    assert!(format!("{err}").contains("dtype"));
}

#[test]
fn update_with_outstanding_view_succeeds_under_lazy_contract() {
    // A previously-returned view that is still live must not block a
    // subsequent update: there is no host memcpy (the write is an in-graph
    // GPU op) and the cache is append-only, so a live view reads a still-
    // valid prefix. The prefix is evaluated between the two updates to
    // satisfy the slab-base eval-between-updates contract; the held view
    // is what stays outstanding across the second update.
    let device = Device::system_default().expect("device");
    let mut cache =
        KvCache::new(&device, MAX_TOKENS, N_KV_HEADS, HEAD_DIM, DType::BF16).expect("alloc");
    let first = make_chunk_tensor(&device, 1, |_, _, _| 1.0);
    cache.update(&first, &first).expect("first update");

    // Hold a view across the second update; commit the first write first.
    let _view = cache.keys_view();
    cache.keys_view().eval().expect("eval keys");
    cache.values_view().eval().expect("eval values");
    cache
        .update(&first, &first)
        .expect("update with outstanding view is allowed");
    assert_eq!(cache.position(), 2);
}

#[test]
fn update_with_derived_op_tensor_succeeds_under_lazy_contract() {
    // An (unevaluated) op tensor derived from a view must not block a
    // subsequent update — the held derived op stays outstanding across the
    // second update. The prefix is evaluated between updates to satisfy the
    // slab-base eval-between-updates contract.
    let device = Device::system_default().expect("device");
    let mut cache =
        KvCache::new(&device, MAX_TOKENS, N_KV_HEADS, HEAD_DIM, DType::BF16).expect("alloc");
    let first = make_chunk_tensor(&device, 1, |_, _, _| 1.0);
    cache.update(&first, &first).expect("first update");

    let other = make_chunk_tensor(&device, 1, |_, _, _| 2.0);
    let _derived = cache.keys_view().add(&other).expect("add");
    cache.keys_view().eval().expect("eval keys");
    cache.values_view().eval().expect("eval values");
    cache
        .update(&first, &first)
        .expect("update with derived op tensor is allowed");
    assert_eq!(cache.position(), 2);
}

#[test]
fn new_rejects_integer_dtype() {
    let device = Device::system_default().expect("device");
    let err = KvCache::new(&device, MAX_TOKENS, N_KV_HEADS, HEAD_DIM, DType::U32)
        .expect_err("must reject u32");
    assert!(format!("{err}").contains("U32"));
    let err = KvCache::new(&device, MAX_TOKENS, N_KV_HEADS, HEAD_DIM, DType::I32)
        .expect_err("must reject i32");
    assert!(format!("{err}").contains("I32"));
}

#[test]
fn kv_cache_update_from_strided_source_matches_contiguous() {
    // The real attention layout: a dense [1,H,T,D] tensor permuted to a
    // strided [T,H,D,1] view (D unit-stride, trailing size-1 axis). The
    // cache write must accept that strided source and produce bytes
    // bit-identical to writing the .copy()-contiguous equivalent.
    let device = Device::system_default().expect("device");
    let (h, t, d) = (2usize, 5usize, 64usize);

    // Build dense [1,H,T,D] with distinguishable values.
    let mut data: Vec<bf16> = Vec::with_capacity(h * t * d);
    for head in 0..h {
        for row in 0..t {
            for dd in 0..d {
                data.push(bf16::from_f32(
                    (head * 100_000 + row * 1000 + dd) as f32,
                ));
            }
        }
    }
    let dense = Tensor::from_slice(&device, &data, [1, h, t, d]).expect("from_slice");

    let strided = dense.permute(&[2, 1, 3, 0]).expect("permute"); // [T,H,D,1]
    assert_eq!(strided.shape().dims(), &[t, h, d, 1]);
    let contiguous = strided.copy().expect("copy");

    let mut cache_a =
        KvCache::new(&device, t, h, d, DType::BF16).expect("alloc a");
    let mut cache_b =
        KvCache::new(&device, t, h, d, DType::BF16).expect("alloc b");

    cache_a.update(&strided, &strided).expect("update strided");
    cache_b
        .update(&contiguous, &contiguous)
        .expect("update contig");

    let a_keys = cache_a.keys_view();
    let a_vals = cache_a.values_view();
    let b_keys = cache_b.keys_view();
    let b_vals = cache_b.values_view();
    a_keys.eval().expect("eval a keys");
    a_vals.eval().expect("eval a values");
    b_keys.eval().expect("eval b keys");
    b_vals.eval().expect("eval b values");

    let a_k: Vec<bf16> = a_keys.cpu_to_vec().expect("a keys vec");
    let b_k: Vec<bf16> = b_keys.cpu_to_vec().expect("b keys vec");
    let a_v: Vec<bf16> = a_vals.cpu_to_vec().expect("a values vec");
    let b_v: Vec<bf16> = b_vals.cpu_to_vec().expect("b values vec");

    assert_eq!(a_k, b_k, "strided vs contiguous keys mismatch");
    assert_eq!(a_v, b_v, "strided vs contiguous values mismatch");
}

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

fn make_chunk_tensor<F: Fn(usize, usize, usize) -> f32>(
    device: &Device,
    step_t: usize,
    f: F,
) -> Tensor {
    let mut data: Vec<bf16> = Vec::with_capacity(step_t * N_KV_HEADS * HEAD_DIM);
    for row in 0..step_t {
        for h in 0..N_KV_HEADS {
            for d in 0..HEAD_DIM {
                data.push(bf16::from_f32(f(row, h, d)));
            }
        }
    }
    Tensor::from_slice(device, &data, [step_t, N_KV_HEADS, HEAD_DIM]).expect("from_slice")
}

#[allow(clippy::cast_precision_loss)]
fn value(chunk_base: f32, row: usize, h: usize, d: usize, multiplier: f32) -> f32 {
    chunk_base * multiplier + (row * 1000 + h * 100 + d) as f32
}
