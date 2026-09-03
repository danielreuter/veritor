// Fixed-order tensor-core GEMM: every output element is one chain of
// `mma.sync` steps in k order.
//
//   D[i][j] = tc_dot(...tc_dot(tc_dot(C[i][j], A[i][0:KS], B[0:KS][j]), A[i][KS:2KS], ...)...)
//
// with KS = 16 (BF16, m16n8k16) or 32 (E4M3, m16n8k32).  There is no split-K,
// no software reduction and no reordering: the k-chunks of one output element
// are consumed by one warp, in one thread's accumulator registers, in
// increasing k.  The tiling only decides which warp owns which outputs:
//
//   grid  = (ceil(N/64), ceil(M/128)), block = 128 threads = 4 warps
//   warp w owns rows [32w, 32w+32) of the block's 128 rows and all 64 columns,
//   i.e. 2 (m16) x 8 (n8) mma tiles = 16 mma per k-step, 64 f32 accumulators.
//
// Operands are read straight from global memory into fragments (L1/L2
// cached); this is deliberately the simplest correct kernel, so the measured
// throughput is a lower bound for a fixed-order kernel, not an upper bound.
//
// Layouts: A is [M][K] row-major, Bt is [N][K] row-major (B in the ".col"
// operand layout), C/D are [M][N] f32 row-major.  K % KS == 0 is required;
// rows >= M and columns >= N are computed on zero operands and not stored.
//
// Build: nvcc -O3 -arch=sm_89 -Xcompiler -fPIC -shared -o libgemm_chain.so gemm_chain.cu

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

template <int K>
__device__ __forceinline__ void mma_step(float d[4], const uint32_t a[4], const uint32_t b[2]);

template <>
__device__ __forceinline__ void mma_step<16>(float d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <>
__device__ __forceinline__ void mma_step<32>(float d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

constexpr int BM = 128, BN = 64, WM = 32, THREADS = 128;

template <int KS, int BYTES>
__global__ void __launch_bounds__(THREADS) gemm_chain_kernel(const uint8_t* __restrict__ A,
                                                            const uint8_t* __restrict__ Bt,
                                                            const float* __restrict__ C,
                                                            float* __restrict__ D, int M, int N,
                                                            int K) {
    constexpr int E = 4 / BYTES;  // operand elements per 32-bit register
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.y * BM + warp * WM;  // this warp's first row
    const int col0 = blockIdx.x * BN;              // this block's first column
    const size_t rowbytes = (size_t)K * BYTES;

    // accumulators: acc[mt][nt][4]  (mt: rows row0+16mt.., nt: cols col0+8nt..)
    float acc[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int i = row0 + mt * 16 + g + (r >> 1) * 8;
                const int j = col0 + nt * 8 + 2 * t + (r & 1);
                acc[mt][nt][r] = (C != nullptr && i < M && j < N) ? C[(size_t)i * N + j] : 0.0f;
            }

    // operand row pointers (clamped rows read zeros through the `valid` flags)
    const uint8_t* arow[2][2];
    bool avalid[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int i = row0 + mt * 16 + g + h * 8;
            avalid[mt][h] = i < M;
            arow[mt][h] = A + (size_t)(avalid[mt][h] ? i : 0) * rowbytes;
        }
    const uint8_t* brow[8];
    bool bvalid[8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int j = col0 + nt * 8 + g;
        bvalid[nt] = j < N;
        brow[nt] = Bt + (size_t)(bvalid[nt] ? j : 0) * rowbytes;
    }

    for (int k = 0; k < K; k += KS) {
        uint32_t af[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            af[mt][0] = avalid[mt][0] ? *reinterpret_cast<const uint32_t*>(arow[mt][0] + (k + t * E) * BYTES) : 0u;
            af[mt][1] = avalid[mt][1] ? *reinterpret_cast<const uint32_t*>(arow[mt][1] + (k + t * E) * BYTES) : 0u;
            af[mt][2] = avalid[mt][0] ? *reinterpret_cast<const uint32_t*>(arow[mt][0] + (k + t * E + KS / 2) * BYTES) : 0u;
            af[mt][3] = avalid[mt][1] ? *reinterpret_cast<const uint32_t*>(arow[mt][1] + (k + t * E + KS / 2) * BYTES) : 0u;
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            uint32_t bf[2];
            bf[0] = bvalid[nt] ? *reinterpret_cast<const uint32_t*>(brow[nt] + (k + t * E) * BYTES) : 0u;
            bf[1] = bvalid[nt] ? *reinterpret_cast<const uint32_t*>(brow[nt] + (k + t * E + KS / 2) * BYTES) : 0u;
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) mma_step<KS>(acc[mt][nt], af[mt], bf);
        }
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int i = row0 + mt * 16 + g + (r >> 1) * 8;
                const int j = col0 + nt * 8 + 2 * t + (r & 1);
                if (i < M && j < N) D[(size_t)i * N + j] = acc[mt][nt][r];
            }
}

template <int KS, int BYTES>
void launch(const void* dA, const void* dB, const float* dC, float* dD, int M, int N, int K) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_chain_kernel<KS, BYTES><<<grid, THREADS>>>((const uint8_t*)dA, (const uint8_t*)dB, dC, dD, M, N, K);
}

template <int KS, int BYTES>
int run(const void* A, const void* Bt, const float* C, float* D, int M, int N, int K, int bench_iters,
        float* ms_out) {
    if (K % KS) return -1;
    const size_t a_bytes = (size_t)M * K * BYTES, b_bytes = (size_t)N * K * BYTES;
    const size_t c_bytes = (size_t)M * N * sizeof(float);
    void *dA = nullptr, *dB = nullptr;
    float *dC = nullptr, *dD = nullptr;
    cudaError_t err;
    if ((err = cudaMalloc(&dA, a_bytes)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dB, b_bytes)) != cudaSuccess) return (int)err;
    if (C != nullptr && (err = cudaMalloc(&dC, c_bytes)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dD, c_bytes)) != cudaSuccess) return (int)err;
    cudaMemcpy(dA, A, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bt, b_bytes, cudaMemcpyHostToDevice);
    if (C != nullptr) cudaMemcpy(dC, C, c_bytes, cudaMemcpyHostToDevice);
    launch<KS, BYTES>(dA, dB, dC, dD, M, N, K);
    err = cudaDeviceSynchronize();
    if (err == cudaSuccess) cudaMemcpy(D, dD, c_bytes, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess && bench_iters > 0) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int i = 0; i < 3; ++i) launch<KS, BYTES>(dA, dB, dC, dD, M, N, K);  // warm-up
        cudaEventRecord(e0);
        for (int i = 0; i < bench_iters; ++i) launch<KS, BYTES>(dA, dB, dC, dD, M, N, K);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms_out) *ms_out = ms / bench_iters;
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        err = cudaGetLastError();
    }
    cudaFree(dA);
    cudaFree(dB);
    if (dC) cudaFree(dC);
    cudaFree(dD);
    return (int)err;
}

}  // namespace

extern "C" {

// A: M*K bf16 words, Bt: N*K bf16 words, C: M*N f32 or NULL (zeros), D: M*N f32.
int gemm_chain_bf16(const uint16_t* A, const uint16_t* Bt, const float* C, float* D, int M, int N, int K,
                    int bench_iters, float* ms_out) {
    return run<16, 2>(A, Bt, C, D, M, N, K, bench_iters, ms_out);
}

int gemm_chain_e4m3(const uint8_t* A, const uint8_t* Bt, const float* C, float* D, int M, int N, int K,
                    int bench_iters, float* ms_out) {
    return run<32, 1>(A, Bt, C, D, M, N, K, bench_iters, ms_out);
}

}  // extern "C"
