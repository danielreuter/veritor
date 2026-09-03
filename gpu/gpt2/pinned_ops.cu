// The pinned CUDA-core operations of GPT-2 as CUDA kernels.
//
// Every floating-point operation is an explicitly rounded IEEE binary32
// operation (`__fadd_rn`, `__fsub_rn`, `__fmul_rn`, `__fdiv_rn`,
// `__fsqrt_rn`: correctly rounded, never contracted into an FMA), so each
// kernel is the same operation sequence as the numpy functions of
// `veritor.core.silicon` and the gate semantics of `make_pinned_gate_set`.
// Reductions use the fixed pairwise tree of `GPT2.reduce`: level i combines
// elements (2j, 2j + 1), an odd last element is carried, carries are folded
// into the result in the order they arose.  No libdevice transcendental is
// called: exp and tanh are the range-reduction + polynomial sequences below.
//
// Build (the flags are belt and braces: the _rn intrinsics fix the semantics
// on their own):
//   nvcc -O3 -arch=sm_89 -fmad=false -prec-div=true -prec-sqrt=true -ftz=false \
//        -Xcompiler -fPIC -shared -o libpinned_ops.so pinned_ops.cu
//
// Host entry points take host pointers, copy in, launch, copy out; the
// forward pass of gpu/gpt2/run_gpt2.py drives them through ctypes.

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// -- scalar semantics ------------------------------------------------------------------------

__device__ __forceinline__ float f32_exp(float x) {
    const float LOG2E = __int_as_float(0x3FB8AA3B);
    const float MAGIC = __int_as_float(0x4B400000);  // 1.5 * 2^23
    const float LN2_HI = __int_as_float(0x3F317200);
    const float LN2_LO = __int_as_float(0x35BFBE8E);
    float t = __fmul_rn(x, LOG2E);
    float kf = __fsub_rn(__fadd_rn(t, MAGIC), MAGIC);
    float r = __fsub_rn(x, __fmul_rn(kf, LN2_HI));
    r = __fsub_rn(r, __fmul_rn(kf, LN2_LO));
    float p = __int_as_float(0x39506967);
    p = __fadd_rn(__fmul_rn(p, r), __int_as_float(0x3AB743CE));
    p = __fadd_rn(__fmul_rn(p, r), __int_as_float(0x3C088908));
    p = __fadd_rn(__fmul_rn(p, r), __int_as_float(0x3D2AA9C1));
    p = __fadd_rn(__fmul_rn(p, r), __int_as_float(0x3E2AAAAA));
    p = __fadd_rn(__fmul_rn(p, r), __int_as_float(0x3F000000));
    float y = __fmul_rn(p, r);
    y = __fmul_rn(y, r);
    y = __fadd_rn(y, r);
    y = __fadd_rn(y, 1.0f);
    float kc = fminf(fmaxf(kf, -126.0f), 127.0f);
    int ki = (int)kc;
    float scale = __int_as_float((ki + 127) << 23);
    y = __fmul_rn(y, scale);
    if (x < -86.5f) y = 0.0f;
    if (x > 88.0f) y = __int_as_float(0x7F800000);
    if (x != x) y = x;
    return y;
}

__device__ __forceinline__ float f32_tanh(float x) {
    float a = fabsf(x);
    float e = f32_exp(__fadd_rn(a, a));
    float r = __fsub_rn(1.0f, __fdiv_rn(2.0f, __fadd_rn(e, 1.0f)));
    if (a >= 9.0f) r = 1.0f;
    r = copysignf(r, x);
    if (x != x) r = x;
    return r;
}

__device__ __forceinline__ float gelu_tanh(float x) {
    const float C0 = __int_as_float(0x3F4C422A);  // sqrt(2 / pi)
    const float C1 = __int_as_float(0x3D372713);  // 0.044715
    float x2 = __fmul_rn(x, x);
    float x3 = __fmul_rn(x2, x);
    float inner = __fadd_rn(x, __fmul_rn(C1, x3));
    float t = f32_tanh(__fmul_rn(C0, inner));
    return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, t));
}

__device__ __forceinline__ float ln_rstd(float variance) {
    const float EPS = __int_as_float(0x3727C5AC);  // 1e-5
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(variance, EPS)));
}

__device__ __forceinline__ float f32_max(float a, float b) { return b > a ? b : a; }

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t bits = __float_as_uint(f);
    uint32_t rounded = (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
    bool nan = ((bits & 0x7F800000u) == 0x7F800000u) && (bits & 0x007FFFFFu) != 0u;
    return nan ? (uint16_t)0x7FC0 : (uint16_t)rounded;
}

__device__ __forceinline__ float bf16_to_f32(uint16_t w) { return __uint_as_float(((uint32_t)w) << 16); }

// -- the fixed tree ---------------------------------------------------------------------------

struct Pair {
    float v;
    uint16_t i;
};

struct AddOp {
    __device__ float operator()(float a, float b) const { return __fadd_rn(a, b); }
};
struct MaxOp {
    __device__ float operator()(float a, float b) const { return f32_max(a, b); }
};
struct SelectOp {  // one tournament node: the later entry wins only when strictly greater
    __device__ Pair operator()(Pair a, Pair b) const { return b.v > a.v ? b : a; }
};

constexpr int MAX_N = 1024;  // one block of MAX_N threads reduces up to MAX_N values
constexpr int MAX_CARRY = 16;

// buf[0..n) in shared memory, all threads of the block participate; returns the reduction.
template <typename T, typename Op>
__device__ T tree_reduce(T* buf, int n, Op op) {
    __shared__ T carry[MAX_CARRY];
    __shared__ int ncarry;
    if (threadIdx.x == 0) ncarry = 0;
    __syncthreads();
    int len = n;
    while (len > 1) {
        if ((len & 1) && threadIdx.x == 0) carry[ncarry++] = buf[len - 1];
        const int half = len >> 1;
        const bool active = threadIdx.x < half;
        T v = buf[0];
        if (active) v = op(buf[2 * threadIdx.x], buf[2 * threadIdx.x + 1]);
        __syncthreads();
        if (active) buf[threadIdx.x] = v;
        __syncthreads();
        len = half;
    }
    T result = buf[0];
    for (int c = 0; c < ncarry; ++c) result = op(result, carry[c]);
    return result;
}

// -- kernels -----------------------------------------------------------------------------------

enum Unary { U_EXP = 0, U_TANH = 1, U_GELU = 2, U_RSTD = 3 };
enum Binary { B_ADD = 0, B_SUB = 1, B_MUL = 2, B_DIV = 3, B_MAX = 4 };

__global__ void unary_kernel(int op, const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y;
    switch (op) {
        case U_EXP: y = f32_exp(x); break;
        case U_TANH: y = f32_tanh(x); break;
        case U_GELU: y = gelu_tanh(x); break;
        default: y = ln_rstd(x); break;
    }
    out[i] = y;
}

__global__ void binary_kernel(int op, const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i], z;
    switch (op) {
        case B_ADD: z = __fadd_rn(x, y); break;
        case B_SUB: z = __fsub_rn(x, y); break;
        case B_MUL: z = __fmul_rn(x, y); break;
        case B_DIV: z = __fdiv_rn(x, y); break;
        default: z = f32_max(x, y); break;
    }
    out[i] = z;
}

__global__ void scale_kernel(const float* a, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fmul_rn(a[i], s);
}

__global__ void round_kernel(const float* a, uint16_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f32_to_bf16(a[i]);
}

__global__ void select_kernel(const float* la, const float* lb, const uint16_t* ia, const uint16_t* ib,
                              uint16_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = lb[i] > la[i] ? ib[i] : ia[i];
}

__global__ void token_eq_kernel(const uint16_t* t, const uint16_t* j, uint16_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = t[i] == j[i] ? (uint16_t)0x3F80 : (uint16_t)0;
}

// one block per row of x[rows][d]
__global__ void ln_stats_kernel(const float* x, int d, float n, float* mean, float* center, float* rstd) {
    __shared__ float buf[MAX_N];
    const float* row = x + (size_t)blockIdx.x * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) buf[i] = row[i];
    __syncthreads();
    float total = tree_reduce(buf, d, AddOp());
    float mu = __fdiv_rn(total, n);
    __syncthreads();  // everyone has read buf[0] and the carries
    float* crow = center + (size_t)blockIdx.x * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float c = __fsub_rn(row[i], mu);
        crow[i] = c;
        buf[i] = __fmul_rn(c, c);
    }
    __syncthreads();
    float squares = tree_reduce(buf, d, AddOp());
    if (threadIdx.x == 0) {
        mean[blockIdx.x] = mu;
        rstd[blockIdx.x] = ln_rstd(__fdiv_rn(squares, n));
    }
}

__global__ void ln_out_kernel(const float* center, const float* rstd, const uint16_t* g, const uint16_t* b,
                              uint16_t* out, int rows, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * d) return;
    int r = idx / d, j = idx - r * d;
    float y = __fmul_rn(center[idx], rstd[r]);
    y = __fmul_rn(y, bf16_to_f32(g[j]));
    y = __fadd_rn(y, bf16_to_f32(b[j]));
    out[idx] = f32_to_bf16(y);
}

template <typename Op>
__global__ void row_reduce_kernel(const float* u, int c, float* out, Op op) {
    __shared__ float buf[MAX_N];
    const float* row = u + (size_t)blockIdx.x * c;
    for (int i = threadIdx.x; i < c; i += blockDim.x) buf[i] = row[i];
    __syncthreads();
    float r = tree_reduce(buf, c, op);
    if (threadIdx.x == 0) out[blockIdx.x] = r;
}

__global__ void exp_shift_kernel(const float* u, const float* m, float* out, int rows, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * c) return;
    int r = idx / c;
    out[idx] = f32_exp(__fsub_rn(u[idx], m[r]));
}

__global__ void div_round_kernel(const float* e, const float* s, uint16_t* out, int rows, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * c) return;
    int r = idx / c;
    out[idx] = f32_to_bf16(__fdiv_rn(e[idx], s[r]));
}

// one block per argmax block: the tournament over (logit, token) pairs of that block
__global__ void tournament_kernel(const float* logits, const uint16_t* tokens, const int* starts, const int* sizes,
                                  float* best, uint16_t* index) {
    __shared__ Pair buf[MAX_N];
    const int start = starts[blockIdx.x], size = sizes[blockIdx.x];
    for (int i = threadIdx.x; i < size; i += blockDim.x) buf[i] = Pair{logits[start + i], tokens[start + i]};
    __syncthreads();
    Pair r = tree_reduce(buf, size, SelectOp());
    if (threadIdx.x == 0) {
        best[blockIdx.x] = r.v;
        index[blockIdx.x] = r.i;
    }
}

// -- host helpers -------------------------------------------------------------------------------

struct Device {
    cudaError_t err = cudaSuccess;
    template <typename T>
    T* in(const T* host, size_t count) {
        T* d = nullptr;
        if (err != cudaSuccess || host == nullptr) return nullptr;
        if ((err = cudaMalloc(&d, count * sizeof(T))) != cudaSuccess) return nullptr;
        err = cudaMemcpy(d, host, count * sizeof(T), cudaMemcpyHostToDevice);
        return d;
    }
    template <typename T>
    T* out(size_t count) {
        T* d = nullptr;
        if (err != cudaSuccess) return nullptr;
        err = cudaMalloc(&d, count * sizeof(T));
        return d;
    }
    template <typename T>
    void back(T* host, const T* d, size_t count) {
        if (err != cudaSuccess) err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        if (err == cudaSuccess) err = cudaMemcpy(host, d, count * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

inline int blocks_for(int n) { return (n + 255) / 256; }

}  // namespace

extern "C" {

int pinned_unary(int op, const float* a, float* out, int n) {
    Device dev;
    float* da = dev.in(a, n);
    float* dout = dev.out<float>(n);
    if (dev.err == cudaSuccess) unary_kernel<<<blocks_for(n), 256>>>(op, da, dout, n);
    dev.back(out, dout, n);
    cudaFree(da);
    cudaFree(dout);
    return (int)dev.err;
}

int pinned_binary(int op, const float* a, const float* b, float* out, int n) {
    Device dev;
    float* da = dev.in(a, n);
    float* db = dev.in(b, n);
    float* dout = dev.out<float>(n);
    if (dev.err == cudaSuccess) binary_kernel<<<blocks_for(n), 256>>>(op, da, db, dout, n);
    dev.back(out, dout, n);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    return (int)dev.err;
}

int pinned_scale(const float* a, float s, float* out, int n) {
    Device dev;
    float* da = dev.in(a, n);
    float* dout = dev.out<float>(n);
    if (dev.err == cudaSuccess) scale_kernel<<<blocks_for(n), 256>>>(da, s, dout, n);
    dev.back(out, dout, n);
    cudaFree(da);
    cudaFree(dout);
    return (int)dev.err;
}

int pinned_round(const float* a, uint16_t* out, int n) {
    Device dev;
    float* da = dev.in(a, n);
    uint16_t* dout = dev.out<uint16_t>(n);
    if (dev.err == cudaSuccess) round_kernel<<<blocks_for(n), 256>>>(da, dout, n);
    dev.back(out, dout, n);
    cudaFree(da);
    cudaFree(dout);
    return (int)dev.err;
}

int pinned_select(const float* la, const float* lb, const uint16_t* ia, const uint16_t* ib, uint16_t* out, int n) {
    Device dev;
    float* dla = dev.in(la, n);
    float* dlb = dev.in(lb, n);
    uint16_t* dia = dev.in(ia, n);
    uint16_t* dib = dev.in(ib, n);
    uint16_t* dout = dev.out<uint16_t>(n);
    if (dev.err == cudaSuccess) select_kernel<<<blocks_for(n), 256>>>(dla, dlb, dia, dib, dout, n);
    dev.back(out, dout, n);
    cudaFree(dla);
    cudaFree(dlb);
    cudaFree(dia);
    cudaFree(dib);
    cudaFree(dout);
    return (int)dev.err;
}

int pinned_token_eq(const uint16_t* t, const uint16_t* j, uint16_t* out, int n) {
    Device dev;
    uint16_t* dt = dev.in(t, n);
    uint16_t* dj = dev.in(j, n);
    uint16_t* dout = dev.out<uint16_t>(n);
    if (dev.err == cudaSuccess) token_eq_kernel<<<blocks_for(n), 256>>>(dt, dj, dout, n);
    dev.back(out, dout, n);
    cudaFree(dt);
    cudaFree(dj);
    cudaFree(dout);
    return (int)dev.err;
}

int pinned_ln_stats(const float* x, int rows, int d, float n, float* mean, float* center, float* rstd) {
    if (d > MAX_N || d < 1) return -1;
    Device dev;
    float* dx = dev.in(x, (size_t)rows * d);
    float* dmean = dev.out<float>(rows);
    float* dcenter = dev.out<float>((size_t)rows * d);
    float* drstd = dev.out<float>(rows);
    if (dev.err == cudaSuccess) ln_stats_kernel<<<rows, MAX_N>>>(dx, d, n, dmean, dcenter, drstd);
    dev.back(mean, dmean, rows);
    dev.back(center, dcenter, (size_t)rows * d);
    dev.back(rstd, drstd, rows);
    cudaFree(dx);
    cudaFree(dmean);
    cudaFree(dcenter);
    cudaFree(drstd);
    return (int)dev.err;
}

int pinned_ln_out(const float* center, const float* rstd, const uint16_t* g, const uint16_t* b, uint16_t* out,
                  int rows, int d) {
    Device dev;
    const size_t n = (size_t)rows * d;
    float* dc = dev.in(center, n);
    float* dr = dev.in(rstd, rows);
    uint16_t* dg = dev.in(g, d);
    uint16_t* db = dev.in(b, d);
    uint16_t* dout = dev.out<uint16_t>(n);
    if (dev.err == cudaSuccess) ln_out_kernel<<<blocks_for((int)n), 256>>>(dc, dr, dg, db, dout, rows, d);
    dev.back(out, dout, n);
    cudaFree(dc);
    cudaFree(dr);
    cudaFree(dg);
    cudaFree(db);
    cudaFree(dout);
    return (int)dev.err;
}

// op: 0 = tree sum, 1 = tree max, over each row of u[rows][c]
int pinned_row_reduce(int op, const float* u, int rows, int c, float* out) {
    if (c > MAX_N || c < 1) return -1;
    Device dev;
    float* du = dev.in(u, (size_t)rows * c);
    float* dout = dev.out<float>(rows);
    if (dev.err == cudaSuccess) {
        if (op == 0)
            row_reduce_kernel<<<rows, MAX_N>>>(du, c, dout, AddOp());
        else
            row_reduce_kernel<<<rows, MAX_N>>>(du, c, dout, MaxOp());
    }
    dev.back(out, dout, rows);
    cudaFree(du);
    cudaFree(dout);
    return (int)dev.err;
}

int pinned_exp_shift(const float* u, const float* m, float* out, int rows, int c) {
    Device dev;
    const size_t n = (size_t)rows * c;
    float* du = dev.in(u, n);
    float* dm = dev.in(m, rows);
    float* dout = dev.out<float>(n);
    if (dev.err == cudaSuccess) exp_shift_kernel<<<blocks_for((int)n), 256>>>(du, dm, dout, rows, c);
    dev.back(out, dout, n);
    cudaFree(du);
    cudaFree(dm);
    cudaFree(dout);
    return (int)dev.err;
}

int pinned_div_round(const float* e, const float* s, uint16_t* out, int rows, int c) {
    Device dev;
    const size_t n = (size_t)rows * c;
    float* de = dev.in(e, n);
    float* ds = dev.in(s, rows);
    uint16_t* dout = dev.out<uint16_t>(n);
    if (dev.err == cudaSuccess) div_round_kernel<<<blocks_for((int)n), 256>>>(de, ds, dout, rows, c);
    dev.back(out, dout, n);
    cudaFree(de);
    cudaFree(ds);
    cudaFree(dout);
    return (int)dev.err;
}

// the tournament over `nblocks` blocks (starts[b], sizes[b]) of (logits, tokens); each size <= MAX_N
int pinned_tournament(const float* logits, const uint16_t* tokens, int n, const int* starts, const int* sizes,
                      int nblocks, float* best, uint16_t* index) {
    for (int b = 0; b < nblocks; ++b)
        if (sizes[b] > MAX_N || sizes[b] < 1) return -1;
    Device dev;
    float* dl = dev.in(logits, n);
    uint16_t* dt = dev.in(tokens, n);
    int* dstarts = dev.in(starts, nblocks);
    int* dsizes = dev.in(sizes, nblocks);
    float* dbest = dev.out<float>(nblocks);
    uint16_t* dindex = dev.out<uint16_t>(nblocks);
    if (dev.err == cudaSuccess) tournament_kernel<<<nblocks, MAX_N>>>(dl, dt, dstarts, dsizes, dbest, dindex);
    dev.back(best, dbest, nblocks);
    dev.back(index, dindex, nblocks);
    cudaFree(dl);
    cudaFree(dt);
    cudaFree(dstarts);
    cudaFree(dsizes);
    cudaFree(dbest);
    cudaFree(dindex);
    return (int)dev.err;
}

}  // extern "C"
