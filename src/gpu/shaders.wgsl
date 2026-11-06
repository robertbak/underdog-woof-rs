// Compute kernels for the decoder's batched stages.
//
// One module serves every entry point. WGSL globals belong to the module, not to the entry point,
// so `a`/`b`/`c`/`d` are all declared `read_write`: wgpu validates each pipeline layout against
// the resources the *entry point* touches, and a kernel that only reads must still see the
// declaration it shares. Getting this wrong produces
// "Storage class Storage { access: LOAD } doesn't match the shader Storage { access: LOAD | STORE }".
//
// The uniform block is shared; `aux` and `flags` mean different things per kernel:
//   gemm            flags bit0 = accumulate into C, aux = row offset into A (conv taps slide
//                   down one buffer instead of using bind offsets, which would have to satisfy
//                   the 256-byte min_storage_buffer_offset_alignment)
//   seed            flags bit0 = add the residual from C
//   convt_epilogue  aux = Z row width (k*Cout, where k = 2*stride)

struct Dims {
    m: u32,
    n: u32,
    k: u32,
    flags: u32,
    aux: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<storage, read_write> d: array<f32>;

var<workgroup> tile_a: array<f32, 256>; // 16x16 tile of A
var<workgroup> tile_b: array<f32, 256>; // 16x16 tile of B

// C = A · B (row-major), optionally + C.
//
// One output per thread, 16x16 tiles. A 16x64 tile with four outputs per thread was measured on a
// Tab S9 (Adreno 740) and is *slower* here (0.80 s -> ~1.0 s decoder), because the decoder's `n`
// runs from 768 down to 1: a 64-wide tile wastes most of its work on the narrow stages and shrinks
// the grid until the GPU starves. Keep this shape until the tiling is chosen per stage.
@compute @workgroup_size(16, 16)
fn gemm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.y * 16u + lid.y;
    let col = wid.x * 16u + lid.x;
    var acc = 0.0;
    let tiles = (dims.k + 15u) / 16u;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let k0 = t * 16u;
        let ac = k0 + lid.x;
        // conv taps read a sliding window of one buffer: `aux` is the row offset
        tile_a[lid.y * 16u + lid.x] =
            select(0.0, a[(row + dims.aux) * dims.k + ac], row < dims.m && ac < dims.k);
        let br = k0 + lid.y;
        tile_b[lid.y * 16u + lid.x] = select(0.0, b[br * dims.n + col], br < dims.k && col < dims.n);
        workgroupBarrier();
        for (var i = 0u; i < 16u; i = i + 1u) {
            acc = acc + tile_a[lid.y * 16u + i] * tile_b[i * 16u + lid.x];
        }
        workgroupBarrier();
    }
    if (row < dims.m && col < dims.n) {
        if ((dims.flags & 1u) == 1u) {
            acc = acc + c[row * dims.n + col];
        }
        c[row * dims.n + col] = acc;
    }
}

// A = bias (+ residual in C): the seed a convolution accumulates into.
@compute @workgroup_size(16, 16)
fn seed(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.y * 16u + lid.y;
    let col = wid.x * 16u + lid.x;
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    var value = b[col];
    if ((dims.flags & 1u) == 1u) {
        value = value + c[row * dims.n + col];
    }
    a[row * dims.n + col] = value;
}

// Elementwise ELU, src = a, dst = c, count = dims.m.
// Grid-stride: a single dispatch dimension caps at 65535 workgroups, and the high-rate stages
// exceed that (e.g. 5040 rows * 4 * 192 / 64).
@compute @workgroup_size(64)
fn elu_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = groups.x * 64u;
    var i = gid.x;
    loop {
        if (i >= dims.m) {
            break;
        }
        let v = a[i];
        c[i] = select(v, exp(v) - 1.0, v < 0.0);
        i = i + stride;
    }
}

// Transposed-conv epilogue: Z (T, s*Cout) in `a` -> out (T*s, Cout) in `d`.
//   out[i*s+sub][c] = Z[i][sub*Cout + c] + bias[sub*Cout + c]
//                     + carry[...]                (i == 0)
//                     + Z[i-1][s*Cout + sub*Cout + c]   (i > 0)
// dims: m = T (latent frames), n = Cout, k = stride, aux = Z row width.
@compute @workgroup_size(64)
fn convt_epilogue(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let total = dims.m * dims.k * dims.n;
    let stride = groups.x * 64u;
    var idx = gid.x;
    loop {
        if (idx >= total) {
            break;
        }
        epilogue_element(idx);
        idx = idx + stride;
    }
}

fn epilogue_element(idx: u32) {
    let col = idx % dims.n;
    let rest = idx / dims.n;
    let frame = rest / dims.k;
    let sub = rest % dims.k;
    let j = sub * dims.n + col;
    let half = dims.k * dims.n;
    let width = dims.aux;
    var value = a[frame * width + j] + b[j];
    if (frame == 0u) {
        value = value + c[j];
    }
    if (frame > 0u) {
        value = value + a[(frame - 1u) * width + half + j];
    }
    d[rest * dims.n + col] = value;
}
