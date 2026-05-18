//! Build script — flattens vendored MLX Metal kernel sources into self-
//! contained MSL strings consumable by this crate's integration tests
//! (and, eventually, by the runtime layer).
//!
//! Metal's runtime compiler (`newLibraryWithSource:`) has no notion of a
//! filesystem, so any `#include "..."` directives must be resolved
//! offline. For each entry-point kernel we recursively inline every
//! project-relative include and leave system includes (`<metal_stdlib>`,
//! `<metal_simdgroup>`, etc.) for the Metal compiler. Each output lands
//! at `$OUT_DIR/<name>_inlined.metal`.
//!
//! Sources live in `kernels-mlx/` at the workspace root; see
//! `kernels-mlx/VENDORED.md` for the file list and sync procedure.

use std::collections::HashSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// (entry-relative-to-MLX-root, output-file-stem). The output filename
/// will be `<stem>_inlined.metal` in `OUT_DIR`.
const ENTRY_POINTS: &[(&str, &str)] = &[
    ("mlx/backend/metal/kernels/copy.metal", "copy"),
    ("mlx/backend/metal/kernels/quantized.metal", "quantized"),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // kernels-mlx/ lives at the workspace root, two levels up from this
    // crate's manifest directory.
    let mlx_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("kernels-mlx");

    for (rel, stem) in ENTRY_POINTS {
        let entry = mlx_dir.join(rel);
        assert!(
            entry.exists(),
            "vendored MLX source missing: {}\n\
             Did you run `scripts/sync_mlx_kernels.sh`? See kernels-mlx/VENDORED.md.",
            entry.display()
        );

        let mut seen: HashSet<PathBuf> = HashSet::new();
        let mut flat = String::new();
        flatten(&entry, &mlx_dir, &mut seen, &mut flat);
        write_output(&format!("{stem}_inlined.metal"), &flat);
    }
}

fn flatten(path: &Path, root: &Path, seen: &mut HashSet<PathBuf>, out: &mut String) {
    let canon = path
        .canonicalize()
        .unwrap_or_else(|_| panic!("cannot canonicalize {}", path.display()));
    if !seen.insert(canon.clone()) {
        return;
    }
    println!("cargo:rerun-if-changed={}", canon.display());

    let content = fs::read_to_string(&canon)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", canon.display()));

    writeln!(out, "// ===== begin {} =====", canon.display()).unwrap();
    for line in content.lines() {
        let trimmed = line.trim_start();

        // #pragma once is implicit (the `seen` set guarantees idempotence).
        if trimmed.starts_with("#pragma once") {
            continue;
        }

        // Project-relative include — inline it. We deliberately leave the
        // angle-bracket form (#include <metal_stdlib> etc.) for the Metal
        // compiler to resolve from its built-in headers.
        if let Some(rest) = trimmed.strip_prefix("#include \"") {
            if let Some(end) = rest.find('"') {
                let inc = &rest[..end];
                let resolved = root.join(inc);
                flatten(&resolved, root, seen, out);
                continue;
            }
        }

        out.push_str(line);
        out.push('\n');
    }
    writeln!(out, "// ===== end {} =====", canon.display()).unwrap();
}

fn write_output(name: &str, content: &str) {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest = Path::new(&out_dir).join(name);
    fs::write(&dest, content).unwrap_or_else(|e| panic!("cannot write {}: {e}", dest.display()));
}
