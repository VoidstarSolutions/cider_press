//! Metal kernel libraries with per-name pipeline-state caching.
//!
//! A [`KernelLibrary`] is the result of JIT-compiling one MSL source
//! string (via `newLibraryWithSource:`) plus a memoizing lookup table
//! from kernel name to [`Pipeline`]. Compile cost is one-time per
//! library; pipeline-build cost is one-time per kernel name. Subsequent
//! lookups for the same name hit the in-memory cache.
//!
//! The crate ships three pre-built libraries from vendored MLX sources:
//! [`KernelLibrary::copy`], [`KernelLibrary::binary`], and
//! [`KernelLibrary::quantized`]. All three wrap
//! [`KernelLibrary::from_source`], which is also public for callers
//! that want to compile their own inline MSL (e.g. one-off kernels).
//!
//! The Metal driver itself caches compiled libraries across processes
//! to disk (we measured ~29 s cold, sub-100 ms warm for the flattened
//! `quantized.metal`), so re-creating a [`KernelLibrary`] across runs
//! is cheap after the first.

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::Write as _;
use std::ptr::NonNull;
use std::sync::Mutex;

use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
};

use crate::device::Device;
use crate::error::{Error, Result};

/// Pre-flattened MLX `copy.metal`, produced by `build.rs`.
const COPY_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/copy_inlined.metal"));

/// Pre-flattened MLX `binary.metal`, produced by `build.rs`.
const BINARY_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/binary_inlined.metal"));

/// Pre-flattened MLX `unary.metal`, produced by `build.rs`.
const UNARY_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/unary_inlined.metal"));

/// Pre-flattened MLX `reduce.metal`, produced by `build.rs`.
const REDUCE_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/reduce_inlined.metal"));

/// Pre-flattened MLX `rope.metal`, produced by `build.rs`.
const ROPE_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/rope_inlined.metal"));

/// Pre-flattened MLX `quantized.metal`, produced by `build.rs`.
const QUANTIZED_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/quantized_inlined.metal"));

/// One specialization value bound to a kernel's `[[function_constant(N)]]`
/// slot at pipeline build time.
///
/// MLX kernels parameterize cheap branches (e.g. `forward` /
/// `traditional` in rope, the activation flag in SDPA) as Metal
/// function constants so each specialization compiles to its own
/// branchless pipeline. We don't need a typed Rust constants surface
/// yet — `bool` covers every consumer through branch 9 — so the enum
/// is minimal. Extend with `U32` / `F32` when softmax / SDPA land.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionConstant {
    /// `constant bool x [[function_constant(index)]]` in MSL.
    Bool { index: u32, value: bool },
}

impl FunctionConstant {
    fn write_cache_token(self, out: &mut String) {
        match self {
            Self::Bool { index, value } => {
                let _ = write!(out, "b{index}={}", u8::from(value));
            }
        }
    }

    fn apply(self, values: &MTLFunctionConstantValues) {
        match self {
            Self::Bool { index, value } => {
                // SAFETY: bool is one byte on every Apple Silicon
                // target; MTLDataType::Bool reads one byte. The
                // pointer borrows the stack-local for the duration of
                // the immediate call.
                let ptr: NonNull<c_void> = NonNull::from(&value).cast();
                unsafe {
                    values.setConstantValue_type_atIndex(ptr, MTLDataType::Bool, index as usize);
                }
            }
        }
    }
}

/// A compiled Metal compute pipeline ready for dispatch.
///
/// Cheap to [`Clone`] — wraps a reference-counted Metal handle. The
/// underlying `MTLComputePipelineState` lives as long as any clone
/// (or its origin [`KernelLibrary`]) is alive.
#[derive(Clone)]
pub struct Pipeline(Retained<ProtocolObject<dyn MTLComputePipelineState>>);

impl Pipeline {
    /// Escape hatch to the underlying [`MTLComputePipelineState`] for
    /// dispatch encoding.
    #[must_use]
    pub fn metal_pipeline_state(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.0
    }
}

/// A compiled Metal library plus a pipeline-state cache keyed by
/// kernel name.
pub struct KernelLibrary {
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pipelines: Mutex<HashMap<String, Pipeline>>,
}

impl KernelLibrary {
    /// Compile MSL source into a library on `device`.
    ///
    /// For the MLX-vendored kernels, prefer [`KernelLibrary::copy`] /
    /// [`KernelLibrary::quantized`] — they wrap this with the
    /// pre-flattened source already embedded.
    pub fn from_source(device: &Device, source: &str) -> Result<Self> {
        let ns_source = NSString::from_str(source);
        let library = device
            .metal_device()
            .newLibraryWithSource_options_error(&ns_source, None)
            .map_err(|err| Error::Compile(err.localizedDescription().to_string()))?;
        Ok(Self {
            library,
            device: device.metal_device().retain(),
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    /// JIT-compile MLX's `copy.metal` (vendored under `kernels-mlx/`).
    pub fn copy(device: &Device) -> Result<Self> {
        Self::from_source(device, COPY_SOURCE)
    }

    /// JIT-compile MLX's `binary.metal` (vendored under
    /// `kernels-mlx/`). Hosts the element-wise binary op family
    /// (`Add`, `Multiply`, etc.) in contiguous (`vv_/vs_/sv_/ss_`)
    /// and strided (`g1_/g2_/g3_/gn2_`) variants.
    pub fn binary(device: &Device) -> Result<Self> {
        Self::from_source(device, BINARY_SOURCE)
    }

    /// JIT-compile MLX's `unary.metal` (vendored under
    /// `kernels-mlx/`). Hosts the element-wise unary op family
    /// (`Square`, `Rsqrt`, `Exp`, etc.) in contiguous (`v_/v2_`) and
    /// strided (`g{n}_`) variants.
    pub fn unary(device: &Device) -> Result<Self> {
        Self::from_source(device, UNARY_SOURCE)
    }

    /// JIT-compile MLX's `reduce.metal` (vendored under
    /// `kernels-mlx/`). Hosts the reduction op family
    /// (`Sum`, `Mean`, `Max`, `Min`, `Prod`) for `all`, `row`, and
    /// `column` reductions.
    pub fn reduce(device: &Device) -> Result<Self> {
        Self::from_source(device, REDUCE_SOURCE)
    }

    /// JIT-compile MLX's `rope.metal` (vendored under `kernels-mlx/`).
    /// Hosts the rotary positional-embedding family. Each instantiation
    /// requires three `[[function_constant]]` booleans bound at
    /// pipeline build time — use [`KernelLibrary::pipeline_specialized`]
    /// rather than [`KernelLibrary::pipeline`] for these.
    pub fn rope(device: &Device) -> Result<Self> {
        Self::from_source(device, ROPE_SOURCE)
    }

    /// JIT-compile MLX's `quantized.metal` (vendored under
    /// `kernels-mlx/`).
    pub fn quantized(device: &Device) -> Result<Self> {
        Self::from_source(device, QUANTIZED_SOURCE)
    }

    /// Look up a kernel function by its `[[host_name]]` and return a
    /// dispatchable [`Pipeline`], building + caching it on first call.
    ///
    /// The cache hit path returns a cloned Metal handle (cheap, no
    /// pipeline rebuild). The cache miss path builds the pipeline via
    /// `newComputePipelineStateWithFunction:error:`.
    pub fn pipeline(&self, name: &str) -> Result<Pipeline> {
        let mut cache = self.pipelines.lock().expect("pipeline cache poisoned");
        if let Some(p) = cache.get(name) {
            return Ok(p.clone());
        }

        let ns_name = NSString::from_str(name);
        let function = self
            .library
            .newFunctionWithName(&ns_name)
            .ok_or_else(|| Error::KernelNotFound(name.to_string()))?;
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| Error::Pipeline {
                name: name.to_string(),
                message: err.localizedDescription().to_string(),
            })?;

        let pipeline = Pipeline(state);
        cache.insert(name.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    /// Look up a kernel that uses `[[function_constant(N)]]`
    /// specialization, building + caching the specialized pipeline on
    /// first call. The cache shares storage with [`Self::pipeline`];
    /// the key is `kernel_name` plus a canonical, index-sorted token
    /// for each constant, so the same `(name, constants)` combination
    /// always hits the same entry, and an unspecialized
    /// [`Self::pipeline`] call doesn't collide with any specialization.
    ///
    /// Internally calls
    /// `newFunctionWithName:constantValues:error:` — see Apple's docs:
    /// constants whose `[[function_constant(N)]]` slots are not bound
    /// here and have no in-source default produce a compile error at
    /// this point, so missing constants surface as
    /// [`Error::Compile`].
    pub fn pipeline_specialized(
        &self,
        kernel_name: &str,
        constants: &[FunctionConstant],
    ) -> Result<Pipeline> {
        let mut sorted: Vec<FunctionConstant> = constants.to_vec();
        sorted.sort_by_key(|c| match c {
            FunctionConstant::Bool { index, .. } => *index,
        });

        let mut cache_key = String::with_capacity(kernel_name.len() + 8 * sorted.len());
        cache_key.push_str(kernel_name);
        for c in &sorted {
            cache_key.push('#');
            c.write_cache_token(&mut cache_key);
        }

        let mut cache = self.pipelines.lock().expect("pipeline cache poisoned");
        if let Some(p) = cache.get(&cache_key) {
            return Ok(p.clone());
        }

        let values = MTLFunctionConstantValues::new();
        for c in &sorted {
            c.apply(&values);
        }

        let ns_name = NSString::from_str(kernel_name);
        let function = self
            .library
            .newFunctionWithName_constantValues_error(&ns_name, &values)
            .map_err(|err| Error::Compile(err.localizedDescription().to_string()))?;
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| Error::Pipeline {
                name: kernel_name.to_string(),
                message: err.localizedDescription().to_string(),
            })?;

        let pipeline = Pipeline(state);
        cache.insert(cache_key, pipeline.clone());
        Ok(pipeline)
    }

    /// Escape hatch to the underlying [`MTLLibrary`].
    #[must_use]
    pub fn metal_library(&self) -> &ProtocolObject<dyn MTLLibrary> {
        &self.library
    }
}
