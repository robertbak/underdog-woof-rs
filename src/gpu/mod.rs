//! wgpu compute back-end (feature `gpu`).
//!
//! Scope, deliberately: this provides a verified `A · B` kernel plus the small epilogues the
//! decoder needs, and it keeps weights and activations on the device between calls. It does *not*
//! know about the model.
//!
//! Measured on a Galaxy Tab S9 (Adreno 740, Vulkan) and an RTX 5090 (Vulkan), the naive
//! one-dispatch-per-op integration is slower than the threaded CPU path — each call uploads
//! operands, dispatches, copies back and blocks on `map_async`, so ~10 sequential stages pay ~10
//! GPU↔CPU round trips to do ~20 µs of work each. Use it for the batched stages only, and prefer
//! keeping a whole chain resident (see `resident`) if you need the win.

#[cfg(feature = "gpu-resident")]
mod resident;

#[cfg(feature = "gpu-resident")]
pub use resident::ResidentDecoder;

use std::collections::HashMap;
use std::sync::Mutex;

use ndarray::Array2;

use crate::backend::MatMul;
use crate::Result;

const SHADERS: &str = include_str!("shaders.wgsl");

/// Shape of a GEMM dispatch: `target[rows, cols] [+]= a[rows, inner] · b[inner, cols]`.
#[derive(Clone, Copy)]
pub struct Gemm<'a> {
    pub rows: usize,
    pub inner: usize,
    pub cols: usize,
    /// row offset into `a` (convolution taps slide down one buffer)
    pub row_offset: usize,
    pub accumulate: bool,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl Gemm<'_> {
    pub fn new(rows: usize, inner: usize, cols: usize) -> Self {
        Gemm {
            rows,
            inner,
            cols,
            row_offset: 0,
            accumulate: false,
            _marker: std::marker::PhantomData,
        }
    }
    pub fn offset(mut self, row_offset: usize) -> Self {
        self.row_offset = row_offset;
        self
    }
    pub fn accumulate(mut self) -> Self {
        self.accumulate = true;
        self
    }
}

/// Shape of a transposed-convolution epilogue: `z[frames, 2*stride*cout] -> out[frames*stride, cout]`.
#[derive(Clone, Copy)]
pub struct ConvT<'a> {
    pub frames: usize,
    pub cout: usize,
    pub stride: usize,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl ConvT<'_> {
    pub fn new(frames: usize, cout: usize, stride: usize) -> Self {
        ConvT {
            frames,
            cout,
            stride,
            _marker: std::marker::PhantomData,
        }
    }
}

/// Uniform block shared by all kernels; see `shaders.wgsl` for the per-kernel meaning.
fn dims(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
    aux: u32,
) -> wgpu::Buffer {
    let values = [m, n, k, flags, aux, 0, 0, 0];
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dims"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, as_bytes(&values));
    buffer
}

fn as_bytes<T>(slice: &[T]) -> &[u8] {
    // SAFETY: `f32`/`u32` are plain-old-data with no padding; the slice is read-only for the
    // lifetime of the borrow.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    // read_write everywhere: the module shares these globals between kernels
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Device, queue and the pipelines this crate uses.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout4: wgpu::BindGroupLayout,
    layout5: wgpu::BindGroupLayout,
    gemm: wgpu::ComputePipeline,
    seed: wgpu::ComputePipeline,
    elu: wgpu::ComputePipeline,
    convt: wgpu::ComputePipeline,
    /// weights uploaded once, keyed by tensor name — f32 on the device, int8 kept on the host
    cache: Mutex<HashMap<String, wgpu::Buffer>>,
    /// When open, ops record into this encoder instead of submitting one each. Android's Vulkan
    /// driver charges several ms per submit, and a decode issues ~100 of them.
    recorder: Mutex<Option<wgpu::CommandEncoder>>,
    dispatches: std::sync::atomic::AtomicUsize,
    info: String,
}

impl Gpu {
    /// Pick a high-performance adapter and build the pipelines.
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all()),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("woof"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("device request failed: {e}"))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("woof-decoder"),
            source: wgpu::ShaderSource::Wgsl(SHADERS.into()),
        });
        let layout4 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("4-binding"),
            entries: &[
                uniform_entry(0),
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
            ],
        });
        let layout5 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("5-binding"),
            entries: &[
                uniform_entry(0),
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
                storage_entry(4),
            ],
        });
        let pipeline = |entry: &str, layout: &wgpu::BindGroupLayout| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Gpu {
            gemm: pipeline("gemm", &layout4),
            seed: pipeline("seed", &layout4),
            elu: pipeline("elu_kernel", &layout4),
            convt: pipeline("convt_epilogue", &layout5),
            layout4,
            layout5,
            device,
            queue,
            cache: Mutex::new(HashMap::new()),
            recorder: Mutex::new(None),
            dispatches: std::sync::atomic::AtomicUsize::new(0),
            info: format!("{} ({:?})", info.name, info.backend),
        })
    }

    pub fn name(&self) -> &str {
        &self.info
    }

    fn buffer(&self, len: usize) -> wgpu::Buffer {
        self.device_create_buffer(len.max(1) * 4)
    }

    fn device_create_buffer(&self, bytes: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("f32"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Upload a plain vector (bias, scales, …), cached by key.
    pub fn upload_cached_slice(&self, key: &str, values: &[f32]) -> wgpu::Buffer {
        let mut cache = self.cache.lock().expect("weight cache poisoned");
        if let Some(existing) = cache.get(key) {
            return existing.clone();
        }
        let buffer = self.buffer(values.len());
        self.write(&buffer, values);
        cache.insert(key.to_string(), buffer.clone());
        buffer
    }

    /// Zero (or fill) a device buffer — used to reset convolution state between utterances.
    pub fn fill(&self, target: &wgpu::Buffer, value: f32) {
        self.flush_recording();
        let len = (target.size() / 4) as usize;
        let values = vec![value; len];
        self.queue.write_buffer(target, 0, as_bytes(&values));
    }

    pub fn write(&self, target: &wgpu::Buffer, values: &[f32]) {
        self.flush_recording();
        self.queue.write_buffer(target, 0, as_bytes(values));
    }

    /// `queue.write_buffer` is ordered against the next *submit*, not against an encoder created
    /// earlier — recording across a host write would let the dispatches read uninitialised memory.
    /// Submitting first is the cheap, safe answer.
    fn flush_recording(&self) {
        let open = self.recorder.lock().expect("recorder poisoned").is_some();
        if open {
            self.end_batch();
        }
    }

    /// Upload a matrix; the device buffer stays alive as long as the caller holds it.
    pub fn upload(&self, values: &Array2<f32>) -> wgpu::Buffer {
        let dense = values.as_standard_layout();
        let buffer = self.buffer(values.len());
        self.write(&buffer, dense.as_slice().expect("dense matrix"));
        buffer
    }

    /// Upload a matrix once and reuse it, keyed by name.
    pub fn upload_cached(&self, key: &str, values: &Array2<f32>) -> wgpu::Buffer {
        let mut cache = self.cache.lock().expect("weight cache poisoned");
        if let Some(existing) = cache.get(key) {
            return existing.clone();
        }
        let buffer = self.upload(values);
        cache.insert(key.to_string(), buffer.clone());
        buffer
    }

    pub fn download(&self, source: &wgpu::Buffer, rows: usize, cols: usize) -> Array2<f32> {
        // anything recorded but not submitted has to land first
        self.end_batch();
        let bytes = rows * cols * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let data = slice.get_mapped_range().expect("map failed");
        let (words, _) = data.as_chunks::<4>();
        let values: Vec<f32> = words.iter().map(|w| f32::from_le_bytes(*w)).collect();
        drop(data);
        staging.unmap();
        Array2::from_shape_vec((rows, cols), values).expect("shape checked by caller")
    }

    /// Open a recording scope: every op appends to one encoder, submitted together by
    /// [`Gpu::end_batch`]. Nested calls are reference-counted by the caller, not here.
    pub fn begin_batch(&self) {
        let mut slot = self.recorder.lock().expect("recorder poisoned");
        if slot.is_none() {
            *slot = Some(
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("batch"),
                    }),
            );
        }
    }

    /// Close the scope and submit once. No-op if no batch is open.
    pub fn end_batch(&self) {
        let encoder = self.recorder.lock().expect("recorder poisoned").take();
        if let Some(encoder) = encoder {
            self.queue.submit(Some(encoder.finish()));
        }
    }

    /// Number of dispatches recorded since startup (diagnostic).
    pub fn dispatch_count(&self) -> usize {
        self.dispatches.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Run `record` against an encoder, submitting immediately when no batch is open.
    fn with_encoder<R>(&self, record: impl FnOnce(&mut wgpu::CommandEncoder) -> R) -> R {
        let mut slot = self.recorder.lock().expect("recorder poisoned");
        match slot.as_mut() {
            Some(encoder) => record(encoder),
            None => {
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("immediate"),
                        });
                let out = record(&mut encoder);
                self.queue.submit(Some(encoder.finish()));
                out
            }
        }
    }

    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        dims: &wgpu::Buffer,
        bindings: &[(&wgpu::Buffer, usize)],
        groups: (u32, u32, u32),
    ) {
        let entries: Vec<wgpu::BindGroupEntry> = std::iter::once(wgpu::BindGroupEntry {
            binding: 0,
            resource: dims.as_entire_binding(),
        })
        .chain(
            bindings
                .iter()
                .enumerate()
                .map(|(i, (buffer, _))| wgpu::BindGroupEntry {
                    binding: i as u32 + 1,
                    resource: buffer.as_entire_binding(),
                }),
        )
        .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        });
        self.dispatches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.with_encoder(|encoder| {
            // separate passes: wgpu inserts the memory barrier between them
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        });
    }

    /// `target [+]= a(row_offset..) · b`, `a` is `(rows, k)`, `b` is `(k, cols)`.
    pub fn gemm(&self, a: &wgpu::Buffer, b: &wgpu::Buffer, target: &wgpu::Buffer, shape: Gemm<'_>) {
        let Gemm {
            rows,
            inner: k,
            cols,
            row_offset,
            accumulate,
            ..
        } = shape;
        let dims = dims(
            &self.device,
            &self.queue,
            rows as u32,
            cols as u32,
            k as u32,
            u32::from(accumulate),
            row_offset as u32,
        );
        self.dispatch(
            &self.gemm,
            &self.layout4,
            &dims,
            &[(a, 0), (b, 0), (target, 0)],
            ((cols as u32).div_ceil(16), (rows as u32).div_ceil(16), 1),
        );
    }

    /// `target = bias [+ residual]`, seeding a convolution's accumulator.
    pub fn seed(
        &self,
        target: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        residual: Option<&wgpu::Buffer>,
        rows: usize,
        cols: usize,
    ) {
        let has_residual = residual.is_some();
        let dims = dims(
            &self.device,
            &self.queue,
            rows as u32,
            cols as u32,
            0,
            u32::from(has_residual),
            0,
        );
        let empty = self.buffer(rows * cols);
        let residual = residual.unwrap_or(&empty);
        self.dispatch(
            &self.seed,
            &self.layout4,
            &dims,
            &[(target, 0), (bias, 0), (residual, 0)],
            ((cols as u32).div_ceil(16), (rows as u32).div_ceil(16), 1),
        );
    }

    /// Elementwise ELU, `src -> dst`.
    pub fn elu(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, count: usize) {
        let dims = dims(&self.device, &self.queue, count as u32, 0, 0, 0, 0);
        let unused = self.buffer(1);
        self.dispatch(
            &self.elu,
            &self.layout4,
            &dims,
            &[(src, 0), (&unused, 0), (dst, 0)],
            // a dispatch dimension caps at 65535 workgroups; the shaders grid-stride past that
            ((count as u32).div_ceil(64).min(65535), 1, 1),
        );
    }

    /// Transposed-conv epilogue, and refresh the carry from the last frame's second half.
    pub fn convt(
        &self,
        z: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        carry: &wgpu::Buffer,
        out: &wgpu::Buffer,
        shape: ConvT<'_>,
    ) {
        let ConvT {
            frames,
            cout,
            stride,
            ..
        } = shape;
        let width = 2 * stride * cout;
        let dims = dims(
            &self.device,
            &self.queue,
            frames as u32,
            cout as u32,
            stride as u32,
            0,
            width as u32,
        );
        self.dispatch(
            &self.convt,
            &self.layout5,
            &dims,
            &[(z, 0), (bias, 0), (carry, 0), (out, 0)],
            (
                ((frames * stride * cout) as u32).div_ceil(64).min(65535),
                1,
                1,
            ),
        );

        let half = (stride * cout) as u64 * 4;
        let offset = (frames - 1) as u64 * (width as u64 * 4) + half;
        self.with_encoder(|encoder| {
            encoder.copy_buffer_to_buffer(z, offset, carry, 0, half);
        });
    }

    /// Device-to-device copy of `count` f32 values.
    pub fn copy(
        &self,
        src: &wgpu::Buffer,
        src_offset: usize,
        dst: &wgpu::Buffer,
        dst_offset: usize,
        count: usize,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            src,
            (src_offset * 4) as u64,
            dst,
            (dst_offset * 4) as u64,
            (count * 4) as u64,
        );
        self.queue.submit(Some(encoder.finish()));
    }

    /// Allocate a device buffer of `count` f32 values.
    pub fn alloc(&self, count: usize) -> wgpu::Buffer {
        self.buffer(count)
    }
}

impl MatMul for Gpu {
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (rows, k) = a.dim();
        assert_eq!(b.nrows(), k, "matmul shape mismatch");
        let cols = b.ncols();
        let a_buf = self.upload(a);
        let b_buf = self.upload(b);
        let out = self.buffer(rows * cols);
        self.gemm(&a_buf, &b_buf, &out, Gemm::new(rows, k, cols));
        self.download(&out, rows, cols)
    }

    fn name(&self) -> &str {
        &self.info
    }
}
