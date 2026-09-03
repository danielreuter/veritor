// Single-instruction tensor-core probes.
//
// Every warp executes exactly one `mma.sync` on one tile, so each output
// element D[i][j] is the hardware's own function of (C[i][j], A[i][:], B[:][j])
// with no software reduction in between.  Operands and results move as raw
// bits; nothing is converted or rounded on the way in or out.
//
//   BF16: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32   (sm_80+)
//         A: 16x16 bf16 row-major,  Bt: 8x16 bf16 (B stored n-major, k
//         contiguous, i.e. the ".col" B operand), C/D: 16x8 f32.
//   E4M3: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32   (sm_89+)
//         A: 16x32 e4m3 row-major,  Bt: 8x32 e4m3, C/D: 16x8 f32.
//
// Fragment layouts follow the PTX ISA "Matrix Fragments for mma.m16n8k16"
// (floating point) and "mma.m16n8k32" (8-bit) figures: with g = lane>>2 and
// t = lane&3, register i of A holds row g (+8 for odd i) columns
// t*E + (i>=2)*K/2 ... where E is the number of elements per 32-bit register.
// The layout is verified empirically by the harness (exact small-integer
// tiles) before any semantic probe runs.
//
// Build:  nvcc -O2 -arch=sm_89 -Xcompiler -fPIC -shared -o libmma_tiles.so mma_tiles.cu

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ void mma_bf16_m16n8k16(float d[4], const uint32_t a[4],
                                                  const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void mma_e4m3_m16n8k32(float d[4], const uint32_t a[4],
                                                  const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// One warp per tile.  A tile is A[16][K], Bt[8][K], C[16][8] (u32 bits).
template <int K, int BYTES>
__global__ void tiles_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,
                             const uint32_t* __restrict__ C, uint32_t* __restrict__ D,
                             int ntiles) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= ntiles) return;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    constexpr int E = 4 / BYTES;          // elements per 32-bit register
    constexpr int ROW = K * BYTES;        // bytes per row of A / Bt

    const uint8_t* a = A + (size_t)warp * 16 * ROW;
    const uint8_t* b = Bt + (size_t)warp * 8 * ROW;
    const uint32_t* c = C + (size_t)warp * 128;
    uint32_t* d = D + (size_t)warp * 128;

    uint32_t af[4], bf[2];
    af[0] = *reinterpret_cast<const uint32_t*>(a + (g) * ROW + (t * E) * BYTES);
    af[1] = *reinterpret_cast<const uint32_t*>(a + (g + 8) * ROW + (t * E) * BYTES);
    af[2] = *reinterpret_cast<const uint32_t*>(a + (g) * ROW + (t * E + K / 2) * BYTES);
    af[3] = *reinterpret_cast<const uint32_t*>(a + (g + 8) * ROW + (t * E + K / 2) * BYTES);
    bf[0] = *reinterpret_cast<const uint32_t*>(b + g * ROW + (t * E) * BYTES);
    bf[1] = *reinterpret_cast<const uint32_t*>(b + g * ROW + (t * E + K / 2) * BYTES);

    float cf[4], df[4];
    cf[0] = __uint_as_float(c[g * 8 + 2 * t]);
    cf[1] = __uint_as_float(c[g * 8 + 2 * t + 1]);
    cf[2] = __uint_as_float(c[(g + 8) * 8 + 2 * t]);
    cf[3] = __uint_as_float(c[(g + 8) * 8 + 2 * t + 1]);

    if (K == 16) {
        mma_bf16_m16n8k16(df, af, bf, cf);
    } else {
        mma_e4m3_m16n8k32(df, af, bf, cf);
    }

    d[g * 8 + 2 * t] = __float_as_uint(df[0]);
    d[g * 8 + 2 * t + 1] = __float_as_uint(df[1]);
    d[(g + 8) * 8 + 2 * t] = __float_as_uint(df[2]);
    d[(g + 8) * 8 + 2 * t + 1] = __float_as_uint(df[3]);
}

template <int K, int BYTES>
int run_tiles(const void* A, const void* Bt, const uint32_t* C, uint32_t* D, int ntiles) {
    const size_t a_bytes = (size_t)ntiles * 16 * K * BYTES;
    const size_t b_bytes = (size_t)ntiles * 8 * K * BYTES;
    const size_t c_bytes = (size_t)ntiles * 128 * 4;
    uint8_t *dA = nullptr, *dB = nullptr;
    uint32_t *dC = nullptr, *dD = nullptr;
    cudaError_t err = cudaSuccess;
    if ((err = cudaMalloc(&dA, a_bytes)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dB, b_bytes)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dC, c_bytes)) != cudaSuccess) return (int)err;
    if ((err = cudaMalloc(&dD, c_bytes)) != cudaSuccess) return (int)err;
    cudaMemcpy(dA, A, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bt, b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, c_bytes, cudaMemcpyHostToDevice);
    const int warps_per_block = 4;
    const int blocks = (ntiles + warps_per_block - 1) / warps_per_block;
    tiles_kernel<K, BYTES><<<blocks, 32 * warps_per_block>>>(dA, dB, dC, dD, ntiles);
    err = cudaDeviceSynchronize();
    if (err == cudaSuccess) cudaMemcpy(D, dD, c_bytes, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dD);
    return (int)err;
}

}  // namespace

extern "C" {

// A: ntiles*16*16 bf16 bits, Bt: ntiles*8*16 bf16 bits, C/D: ntiles*16*8 f32 bits.
int mma_bf16_tiles(const uint16_t* A, const uint16_t* Bt, const uint32_t* C, uint32_t* D,
                   int ntiles) {
    return run_tiles<16, 2>(A, Bt, C, D, ntiles);
}

// A: ntiles*16*32 e4m3 bytes, Bt: ntiles*8*32 e4m3 bytes, C/D: ntiles*16*8 f32 bits.
int mma_e4m3_tiles(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D,
                   int ntiles) {
    return run_tiles<32, 1>(A, Bt, C, D, ntiles);
}

}  // extern "C"
