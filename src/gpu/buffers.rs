//! Device buffers and recording: creation, upload, readback, and the batched encoder.
//!
//! The ordering rule that matters lives here: one `queue.write_buffer` is ordered against the *next*
//! submit, not against an encoder that is already open. Getting that wrong shows up as constant-sample
//! audio rather than as an error, which is why `write`/`fill` flush the batch first.

use super::*;

pub(super) fn dims(
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

pub(super) fn as_bytes<T>(slice: &[T]) -> &[u8] {
    // SAFETY: `f32`/`u32` are plain-old-data with no padding; the slice is read-only for the
    // lifetime of the borrow.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

pub(super) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub(super) fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

impl Gpu {
    pub(super) fn buffer(&self, len: usize) -> wgpu::Buffer {
        self.device_create_buffer(len.max(1) * 4)
    }

    pub(super) fn device_create_buffer(&self, bytes: usize) -> wgpu::Buffer {
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
    pub(super) fn flush_recording(&self) {
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
