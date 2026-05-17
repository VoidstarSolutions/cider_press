//! Metal kernel libraries with per-name pipeline-state caching.
//!
//! A [`KernelLibrary`] is the result of JIT-compiling one MSL source
//! string (via `newLibraryWithSource:`) plus a memoizing lookup table
//! from kernel name to [`Pipeline`]. Compile cost is one-time per
//! library; pipeline-build cost is one-time per kernel name. Subsequent
//! lookups for the same name hit the in-memory cache.
//!
//! The crate ships two pre-built libraries from vendored MLX sources:
//! [`KernelLibrary::copy`] and [`KernelLibrary::quantized`]. Both wrap
//! [`KernelLibrary::from_source`], which is also public for callers
//! that want to compile their own inline MSL (e.g. one-off kernels).
//!
//! The Metal driver itself caches compiled libraries across processes
//! to disk (we measured ~29 s cold, sub-100 ms warm for the flattened
//! `quantized.metal`), so re-creating a [`KernelLibrary`] across runs
//! is cheap after the first.

use std::collections::HashMap;
use std::sync::Mutex;

use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

use crate::device::Device;
use crate::error::{Error, Result};

/// Pre-flattened MLX `copy.metal`, produced by `build.rs`.
const COPY_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/copy_inlined.metal"));

/// Pre-flattened MLX `quantized.metal`, produced by `build.rs`.
const QUANTIZED_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/quantized_inlined.metal"));

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

    /// Escape hatch to the underlying [`MTLLibrary`].
    #[must_use]
    pub fn metal_library(&self) -> &ProtocolObject<dyn MTLLibrary> {
        &self.library
    }
}
