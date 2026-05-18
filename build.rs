//! Spike build script — flattens MLX Metal kernel sources into self-
//! contained MSL strings consumable by `tests/stage*` integration tests.
//!
//! Metal's runtime compiler (`newLibraryWithSource:`) has no notion of a
//! filesystem, so any `#include "..."` directives must be resolved
//! offline. For each MLX entry-point kernel, we recursively inline every
//! project-relative include and leave system includes (`<metal_stdlib>`
//! etc.) for the Metal compiler. Each output lands at
//! `$OUT_DIR/<name>_inlined.metal`.
//!
//! MLX checkout location must be supplied via the `CIDER_MLX_DIR`
//! environment variable. If it is unset or points at a path with no
//! MLX kernel sources, the script writes a stub for each entry point
//! and the dependent integration tests skip with a clear message.
//!
//! Vendoring MLX into the repo (see the follow-up workspace
//! restructure) removes this build-time requirement entirely; this
//! script is the spike-era arrangement.

use std::collections::HashSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// (entry-relative-to-MLX, output-file-stem). The output filename will be
/// `<stem>_inlined.metal` in `OUT_DIR`.
const ENTRY_POINTS: &[(&str, &str)] = &[
    ("mlx/backend/metal/kernels/copy.metal", "copy"),
    ("mlx/backend/metal/kernels/quantized.metal", "quantized"),
];

fn main() {
    println!("cargo:rerun-if-env-changed=CIDER_MLX_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    let mlx_dir = env::var_os("CIDER_MLX_DIR").map(PathBuf::from);

    for (rel, stem) in ENTRY_POINTS {
        let dest_name = format!("{stem}_inlined.metal");

        let Some(dir) = mlx_dir.as_ref() else {
            skip(&dest_name, "CIDER_MLX_DIR is not set");
            continue;
        };
        let entry = dir.join(rel);
        if !entry.exists() {
            skip(&dest_name, &format!("entry not found: {}", entry.display()));
            continue;
        }

        let mut seen: HashSet<PathBuf> = HashSet::new();
        let mut flat = String::new();
        flatten(&entry, dir, &mut seen, &mut flat);
        write_output(&dest_name, &flat);
    }
}

fn skip(dest_name: &str, detail: &str) {
    let stub = format!(
        "// MLX kernel sources unavailable at build time.\n\
         // {detail}\n\
         // Set CIDER_MLX_DIR to a local MLX checkout to enable.\n"
    );
    write_output(dest_name, &stub);
    println!(
        "cargo:warning=cider-press: MLX sources unavailable ({detail}); \
         dependent tests will skip. Set CIDER_MLX_DIR to enable."
    );
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
