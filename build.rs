//! Spike build script — flattens MLX Metal kernel sources into a single
//! self-contained MSL string for `tests/stage2_copy.rs`.
//!
//! Metal's runtime compiler (`newLibraryWithSource:`) has no notion of a
//! filesystem, so any `#include "..."` directives must be resolved offline.
//! This script reads `mlx/backend/metal/kernels/copy.metal`, recursively
//! inlines every project-relative include (anything with a quoted path),
//! and leaves system includes (`<metal_stdlib>` etc.) for the Metal
//! compiler. The result lands at `$OUT_DIR/copy_inlined.metal`.
//!
//! MLX checkout location is resolved from `CIDER_MLX_DIR` if set, otherwise
//! the documented default in CLAUDE.md.

use std::collections::HashSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_MLX_DIR: &str = "/Users/zacharyheylmun/dev/llm/mlx";
const ENTRY_POINT: &str = "mlx/backend/metal/kernels/copy.metal";

fn main() {
    println!("cargo:rerun-if-env-changed=CIDER_MLX_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    let mlx_dir =
        env::var("CIDER_MLX_DIR").map_or_else(|_| PathBuf::from(DEFAULT_MLX_DIR), PathBuf::from);

    let entry = mlx_dir.join(ENTRY_POINT);
    if !entry.exists() {
        // Don't hard-fail the whole build — Stage 1 doesn't need this. The
        // Stage 2 test will skip with a clear message when the inlined file
        // is missing or empty.
        let stub = "// MLX checkout not found at build time. Set CIDER_MLX_DIR.\n";
        write_output(stub);
        println!(
            "cargo:warning=cider-press: MLX checkout not found at {}; \
             stage 2 test will skip. Set CIDER_MLX_DIR to override.",
            entry.display()
        );
        return;
    }

    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut flat = String::new();
    flatten(&entry, &mlx_dir, &mut seen, &mut flat);
    write_output(&flat);
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

fn write_output(content: &str) {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest = Path::new(&out_dir).join("copy_inlined.metal");
    fs::write(&dest, content).unwrap_or_else(|e| panic!("cannot write {}: {e}", dest.display()));
}
