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
//! `HEADER_BUNDLES` is the same flattener pointed at a *header-only*
//! entry — useful for JIT-only kernels like gather where MLX has no
//! precompiled `.metal` to vendor and the dispatch-time source string
//! is assembled from a header prefix plus a per-instantiation wrapper.
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
    ("mlx/backend/metal/kernels/binary.metal", "binary"),
    ("mlx/backend/metal/kernels/unary.metal", "unary"),
    ("mlx/backend/metal/kernels/reduce.metal", "reduce"),
    ("mlx/backend/metal/kernels/rope.metal", "rope"),
    ("mlx/backend/metal/kernels/quantized.metal", "quantized"),
];

/// Header-only bundles for JIT-assembled kernels.
///
/// Each entry flattens `(headers...)` into one `$OUT_DIR/<stem>_inlined.metal`
/// string. The result is *not* directly compileable — it has no kernel
/// entry points — but is meant to be prepended to a per-instantiation
/// wrapper source at dispatch time and then compiled together. The
/// transitive include set for each listed header is followed by the
/// same flattener used for full `.metal` entry points.
const HEADER_BUNDLES: &[(&[&str], &str)] = &[(
    &[
        // `utils.h` mirrors what MLX prepends via `metal::utils()`.
        "mlx/backend/metal/kernels/utils.h",
        // `gather.h` transitively pulls `indexing.h` and uses
        // `elem_to_loc` + `Indices` + `offset_neg_idx` from utils.
        "mlx/backend/metal/kernels/indexing/gather.h",
    ],
    "gather_headers",
)];

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

    for (rels, stem) in HEADER_BUNDLES {
        let mut seen: HashSet<PathBuf> = HashSet::new();
        let mut flat = String::new();
        for rel in *rels {
            let entry = mlx_dir.join(rel);
            assert!(
                entry.exists(),
                "vendored MLX header missing: {}\n\
                 Did you run `scripts/sync_mlx_kernels.sh`? See kernels-mlx/VENDORED.md.",
                entry.display()
            );
            flatten(&entry, &mlx_dir, &mut seen, &mut flat);
        }
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
