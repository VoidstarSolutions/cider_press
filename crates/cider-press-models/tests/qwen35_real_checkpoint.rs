//! Phase-1 loader faithfulness: every tensor the loader maps must byte-match the
//! safetensors archive. Gated on `CIDER_QWEN35_CHECKPOINT_PATH` (skips in CI).

#![cfg(target_os = "macos")]

use std::{fs, path::PathBuf};

use cider_press_models::qwen3_5::{
    Qwen35Config, Qwen35FullAttnWeights, Qwen35LinearAttnWeights, Qwen35MixerWeights,
    load_qwen35_weights,
};
use cider_press_runtime::Device;
use safetensors::SafeTensors;

fn checkpoint_path() -> Option<PathBuf> {
    std::env::var_os("CIDER_QWEN35_CHECKPOINT_PATH").map(PathBuf::from)
}

#[test]
fn real_checkpoint_loads_and_byte_round_trips() {
    const P: &str = "language_model.model";

    let Some(dir) = checkpoint_path() else {
        eprintln!("skipping: CIDER_QWEN35_CHECKPOINT_PATH unset");
        return;
    };
    let config =
        Qwen35Config::from_json_bytes(&fs::read(dir.join("config.json")).unwrap()).unwrap();
    let bytes = fs::read(dir.join("model.safetensors")).unwrap();
    let archive = SafeTensors::deserialize(&bytes).unwrap();
    let device = Device::shared().unwrap();

    let weights = load_qwen35_weights(&archive, &config, &device).unwrap();

    let check_q = |w: &cider_press_runtime::QuantizedWeight, base: &str| {
        let (lw, ls, lb) = w.component_bytes();
        assert_eq!(
            lw,
            archive.tensor(&format!("{base}.weight")).unwrap().data(),
            "{base}.weight"
        );
        assert_eq!(
            ls,
            archive.tensor(&format!("{base}.scales")).unwrap().data(),
            "{base}.scales"
        );
        assert_eq!(
            lb,
            archive.tensor(&format!("{base}.biases")).unwrap().data(),
            "{base}.biases"
        );
    };
    let check_dense = |t: &cider_press_runtime::Tensor, key: &str| {
        assert_eq!(
            t.cpu_bytes().unwrap(),
            archive.tensor(key).unwrap().data(),
            "{key}"
        );
    };

    check_q(&weights.embed, &format!("{P}.embed_tokens"));
    check_dense(&weights.norm, &format!("{P}.norm.weight"));

    let l0 = &weights.layers[0];
    check_dense(
        &l0.input_layernorm,
        &format!("{P}.layers.0.input_layernorm.weight"),
    );
    let Qwen35MixerWeights::LinearAttn(Qwen35LinearAttnWeights {
        in_proj_qkv,
        a_log,
        conv1d_weight,
        out_proj,
        ..
    }) = &l0.mixer
    else {
        panic!("layer 0 should be linear");
    };
    check_q(
        in_proj_qkv,
        &format!("{P}.layers.0.linear_attn.in_proj_qkv"),
    );
    check_q(out_proj, &format!("{P}.layers.0.linear_attn.out_proj"));
    check_dense(a_log, &format!("{P}.layers.0.linear_attn.A_log"));
    check_dense(
        conv1d_weight,
        &format!("{P}.layers.0.linear_attn.conv1d.weight"),
    );

    let l3 = &weights.layers[3];
    let Qwen35MixerWeights::FullAttn(Qwen35FullAttnWeights {
        q_proj,
        q_norm,
        o_proj,
        ..
    }) = &l3.mixer
    else {
        panic!("layer 3 should be full");
    };
    check_q(q_proj, &format!("{P}.layers.3.self_attn.q_proj"));
    check_q(o_proj, &format!("{P}.layers.3.self_attn.o_proj"));
    check_dense(q_norm, &format!("{P}.layers.3.self_attn.q_norm.weight"));
}
