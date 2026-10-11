//! EOS stop set on the real Qwen3.5/3.6 checkpoints: the `tokenizer_config.json`
//! ids unioned with the `config.json` / `generation_config.json` ids, matching
//! `mlx_lm`. `#[ignore]`d; needs the checkpoint env vars.

#![cfg(target_os = "macos")]

use std::collections::HashSet;
use std::path::Path;

use cider_press_models::chat_template::ChatTemplate;
use cider_press_models::config_eos_ids;
use cider_press_models::tokenizer::Tokenizer;
use cider_press_test_utils::{qwen35_4b_checkpoint_dir, qwen35_27b_checkpoint_dir};

fn eos_set(dir: &Path) -> HashSet<u32> {
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let template = ChatTemplate::from_file(&dir.join("tokenizer_config.json"), &tokenizer).unwrap();
    let mut ids: HashSet<u32> = template.eos_ids().collect();
    ids.extend(config_eos_ids(dir).unwrap());
    ids
}

#[test]
#[ignore = "requires a real checkpoint via CIDER_QWEN35_CHECKPOINT_PATH"]
fn qwen35_4b_eos_is_im_end_only() {
    assert_eq!(
        eos_set(&qwen35_4b_checkpoint_dir()),
        HashSet::from([248_046])
    );
}

#[test]
#[ignore = "requires a real checkpoint via CIDER_QWEN35_27B_CHECKPOINT_PATH"]
fn qwen36_27b_eos_includes_endoftext() {
    assert_eq!(
        eos_set(&qwen35_27b_checkpoint_dir()),
        HashSet::from([248_044, 248_046])
    );
}
