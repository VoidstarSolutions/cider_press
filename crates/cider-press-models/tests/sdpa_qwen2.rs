//! Parity test driving `qwen2::attention::sdpa` at Qwen2.5-0.5B
//! attention shapes against `mx.fast.scaled_dot_product_attention`.
//!
//! Covers two regimes:
//! - **Decode** (`T=1`, `T_cache>1`, no mask): GQA fan-out from 2 KV
//!   heads to 14 Q heads, single-row Q, full attention to the cache.
//! - **Prefill with causal mask** (`T=T_cache>1`): same GQA but the
//!   additive `-inf` causal mask folds into the binary-add path
//!   before softmax. Drives the `mask` argument too.
//!
//! Tolerance bar: bf16-compose precision (~0.02 abs / 0.05 rel for
//! the longer prefill chain; tighter for decode). MLX's fused SDPA
//! kernel does the same `Q @ K^T → scale → softmax → @ V` math but
//! with a single accumulator path, so the bf16 round-trips between
//! our composed ops shed an extra ULP or two relative to MLX's
//! fused execution.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use cider_press_models::qwen2::{Qwen2Config, attention};
use cider_press_runtime::{Device, Tensor};
use cider_press_test_utils::{dump_mlx_op, read_bf16, tempdir};
use half::bf16;
use safetensors::SafeTensors;

const PINNED_CONFIG_JSON: &str = r#"{
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "num_hidden_layers": 24,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-06,
    "vocab_size": 151936,
    "tie_word_embeddings": true,
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

const B: usize = 1;

#[test]
fn qwen2_sdpa_decode_matches_mlx() {
    let config = Qwen2Config::from_json_bytes(PINNED_CONFIG_JSON.as_bytes()).expect("parse config");
    let seq_q = 1;
    let seq_kv = 8;
    let fixture = sdpa_fixture(&config, seq_q, seq_kv, false);

    let device = Device::system_default().expect("device");
    let q = Tensor::from_slice(
        &device,
        &fixture.q,
        [
            B,
            config.num_attention_heads,
            seq_q,
            config.head_dim().unwrap(),
        ],
    )
    .expect("q");
    let k = Tensor::from_slice(
        &device,
        &fixture.k,
        [
            B,
            config.num_key_value_heads,
            seq_kv,
            config.head_dim().unwrap(),
        ],
    )
    .expect("k");
    let v = Tensor::from_slice(
        &device,
        &fixture.v,
        [
            B,
            config.num_key_value_heads,
            seq_kv,
            config.head_dim().unwrap(),
        ],
    )
    .expect("v");

    let out = attention::sdpa(&q, &k, &v, None, &config).expect("schedule sdpa");
    out.eval().expect("eval");
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");
    assert_within_tolerance("Qwen2 SDPA decode", &got, &fixture.out, 0.02, 0.05);
}

#[test]
fn qwen2_sdpa_prefill_causal_matches_mlx() {
    let config = Qwen2Config::from_json_bytes(PINNED_CONFIG_JSON.as_bytes()).expect("parse config");
    let seq_q = 8;
    let seq_kv = 8;
    let fixture = sdpa_fixture(&config, seq_q, seq_kv, true);

    let device = Device::system_default().expect("device");
    let q = Tensor::from_slice(
        &device,
        &fixture.q,
        [
            B,
            config.num_attention_heads,
            seq_q,
            config.head_dim().unwrap(),
        ],
    )
    .expect("q");
    let k = Tensor::from_slice(
        &device,
        &fixture.k,
        [
            B,
            config.num_key_value_heads,
            seq_kv,
            config.head_dim().unwrap(),
        ],
    )
    .expect("k");
    let v = Tensor::from_slice(
        &device,
        &fixture.v,
        [
            B,
            config.num_key_value_heads,
            seq_kv,
            config.head_dim().unwrap(),
        ],
    )
    .expect("v");
    let mask_host = fixture.mask.as_ref().expect("causal fixture has mask");
    let mask = Tensor::from_slice(&device, mask_host, [seq_q, seq_kv]).expect("mask");

    let out = attention::sdpa(&q, &k, &v, Some(&mask), &config).expect("schedule sdpa");
    out.eval().expect("eval");
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");
    assert_within_tolerance("Qwen2 SDPA prefill causal", &got, &fixture.out, 0.03, 0.08);
}

struct SdpaFixture {
    q: Vec<bf16>,
    k: Vec<bf16>,
    v: Vec<bf16>,
    mask: Option<Vec<bf16>>,
    out: Vec<bf16>,
}

fn sdpa_fixture(config: &Qwen2Config, seq_q: usize, seq_kv: usize, causal: bool) -> SdpaFixture {
    let tmp = tempdir("models-sdpa-qwen2");
    let tag = if causal { "causal" } else { "nomask" };
    let path = tmp.join(format!("sdpa-{tag}-{seq_q}x{seq_kv}.safetensors"));
    let h_q = config.num_attention_heads.to_string();
    let h_kv = config.num_key_value_heads.to_string();
    let head_dim = config.head_dim().unwrap().to_string();
    let batch = B.to_string();
    let seq_q_s = seq_q.to_string();
    let seq_kv_s = seq_kv.to_string();
    let mut args: Vec<&str> = vec![
        "sdpa",
        "--batch",
        &batch,
        "--h-q",
        &h_q,
        "--h-kv",
        &h_kv,
        "--seq-q",
        &seq_q_s,
        "--seq-kv",
        &seq_kv_s,
        "--head-dim",
        &head_dim,
        "--dtype",
        "bf16",
    ];
    if causal {
        args.push("--causal");
    }
    dump_mlx_op(&path, &args);
    let bytes = std::fs::read(&path).expect("read fixture");
    let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
    SdpaFixture {
        q: read_bf16(&st, "q"),
        k: read_bf16(&st, "k"),
        v: read_bf16(&st, "v"),
        mask: if causal {
            Some(read_bf16(&st, "mask"))
        } else {
            None
        },
        out: read_bf16(&st, "out"),
    }
}

fn assert_within_tolerance(
    label: &str,
    got: &[bf16],
    expected: &[bf16],
    abs_tol: f32,
    rel_tol: f32,
) {
    // np.allclose-style combined bound: `|a - b| <= abs_tol + rel_tol *
    // |b|`. The composed-SDPA chain produces 1-ULP drift on individual
    // bf16 values; near zero a 1-ULP absolute diff is huge relatively,
    // which the pure AND form (used by earlier branches) flags
    // spuriously. The combined form is what numpy / mlx use for
    // tolerance comparisons.
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    let mut worst_excess = 0.0f32;
    let mut worst_at = 0usize;
    let mut worst_pair = (0.0f32, 0.0f32);
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        if a == b {
            continue;
        }
        let af = a.to_f32();
        let bf = b.to_f32();
        let abs = (af - bf).abs();
        let bound = abs_tol + rel_tol * bf.abs();
        let excess = abs - bound;
        if excess > worst_excess {
            worst_excess = excess;
            worst_at = i;
            worst_pair = (af, bf);
        }
    }
    assert!(
        worst_excess <= 0.0,
        "{label}: tolerance exceeded — worst element [{worst_at}] got={} expected={} \
         |diff|={} > abs_tol={abs_tol} + rel_tol={rel_tol} * |expected|",
        worst_pair.0,
        worst_pair.1,
        (worst_pair.0 - worst_pair.1).abs(),
    );
}
