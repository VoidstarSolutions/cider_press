//! Model-level fused-prefill parity test.
//!
//! Drives `qwen2::attention::sdpa` at the real Qwen2.5-0.5B prefill
//! shape (`T = 39`, causal self-attention) with `None` mask, asserting
//! the model now dispatches the same fused steel `sdpa_full` kernel MLX
//! uses (causal via the kernel's `do_causal` function constant — no
//! materialized mask). Compares to `mx.fast.scaled_dot_product_attention`
//! with a causal mask.

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
fn qwen2_sdpa_prefill_fused_matches_mlx() {
    let config = Qwen2Config::from_json_bytes(PINNED_CONFIG_JSON.as_bytes()).expect("parse config");
    // T = 39: the real Qwen2 prompt length used elsewhere in the repo.
    let seq_q = 39;
    let seq_kv = 39;
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

    // None mask: the fused path is causal via the kernel's do_causal; an
    // explicit mask is now rejected at the runtime layer.
    let out = attention::sdpa(&q, &k, &v, None, &config).expect("schedule sdpa");
    out.eval().expect("eval");
    let got: Vec<bf16> = out.cpu_to_vec().expect("dense out");
    assert_within_tolerance("Qwen2 SDPA prefill fused", &got, &fixture.out, 0.03, 0.08);
}

struct SdpaFixture {
    q: Vec<bf16>,
    k: Vec<bf16>,
    v: Vec<bf16>,
    out: Vec<bf16>,
}

fn sdpa_fixture(config: &Qwen2Config, seq_q: usize, seq_kv: usize, causal: bool) -> SdpaFixture {
    let tmp = tempdir("models-sdpa-prefill-fused");
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
    // np.allclose-style combined bound: `|a - b| <= abs_tol + rel_tol * |b|`.
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
