// CPU reference for the fixed-order tensor-core GEMM chain.
//
// A line-by-line C++ twin of veritor.core.silicon (itself a port of the
// Hawkeye simulator's group_sum), parametrised by the group structure,
// internal width and exponent floor, evaluating every output element of a
// GEMM as the chain  acc = tc_dot(acc, A[i][k..k+KS], Bt[j][k..k+KS])  in k
// order.  Pure integer arithmetic on raw bit patterns; OpenMP over outputs.
//
// Build: g++ -O2 -std=c++17 -fopenmp -fPIC -shared -o libref_chain.so ref_chain.cpp

#include <cstdint>
#include <cstdlib>

namespace {

struct Term {
    bool neg;
    int exp;
    int64_t mag;  // value = (-1)^neg * mag * 2^(exp-23); mag == 0 is zero
};

inline Term bf16_product(uint32_t a, uint32_t b) {
    int ea = (a >> 7) & 0xFF, eb = (b >> 7) & 0xFF;
    int64_t sa = (a & 0x7F) | (ea ? 0x80 : 0), sb = (b & 0x7F) | (eb ? 0x80 : 0);
    int64_t mag = (sa * sb) << 9;
    if (!mag) return Term{false, 0, 0};
    return Term{((a ^ b) >> 15 & 1) != 0, (ea ? ea : 1) + (eb ? eb : 1) - 254, mag};
}

inline Term e4m3_product(uint32_t a, uint32_t b) {
    int ea = (a >> 3) & 0xF, eb = (b >> 3) & 0xF;
    int64_t sa = (a & 0x7) | (ea ? 0x8 : 0), sb = (b & 0x7) | (eb ? 0x8 : 0);
    int64_t mag = (sa * sb) << 17;
    if (!mag) return Term{false, 0, 0};
    return Term{((a ^ b) >> 7 & 1) != 0, (ea ? ea : 1) + (eb ? eb : 1) - 14, mag};
}

inline Term fp32_term(uint32_t bits) {
    int e = (bits >> 23) & 0xFF;
    int64_t m = bits & 0x7FFFFF;
    bool neg = (bits >> 31) != 0;
    if (e == 0) return m ? Term{neg, -126, m} : Term{false, 0, 0};
    return Term{neg, e - 127, (int64_t(1) << 23) | m};
}

inline uint32_t pack_fp32(const Term& t) {
    if (!t.mag) return 0;
    if (t.exp > 127) return (uint32_t(t.neg) << 31) | 0x7F800000u;
    int e = t.exp + 127;
    if (!(t.mag & (int64_t(1) << 23))) e -= 1;
    return (uint32_t(t.neg) << 31) | (uint32_t(e) << 23) | (uint32_t(t.mag) & 0x7FFFFF);
}

inline int bit_length(int64_t x) { return x ? 64 - __builtin_clzll((unsigned long long)x) : 0; }

inline Term group_sum(const Term* terms, int n, int width, int zero_exp) {
    const int rescale = width - 24;
    int max_exp = zero_exp;
    for (int i = 0; i < n; ++i)
        if (terms[i].mag && terms[i].exp > max_exp) max_exp = terms[i].exp;
    int64_t total = 0;
    for (int i = 0; i < n; ++i) {
        if (!terms[i].mag) continue;
        int64_t scaled = rescale >= 0 ? terms[i].mag << rescale : terms[i].mag >> (-rescale);
        int shift = max_exp - terms[i].exp;
        int64_t aligned = shift >= 63 ? 0 : scaled >> shift;
        total += terms[i].neg ? -aligned : aligned;
    }
    if (!total) return Term{false, 0, 0};
    bool neg = total < 0;
    int64_t mag = neg ? -total : total;
    int bl = bit_length(mag);
    int exp = max_exp + bl - width;
    if (bl > width) mag >>= (bl - width);
    else mag <<= (width - bl);
    if (exp < -126) {
        int s = -126 - exp;
        mag = s >= 63 ? 0 : mag >> s;
        exp = -126;
    }
    mag = rescale >= 0 ? mag >> rescale : mag << (-rescale);
    if (!mag) return Term{false, 0, 0};
    return Term{neg, exp, mag};
}

template <typename W, bool BF16>
void gemm(const W* A, const W* Bt, const uint32_t* C, uint32_t* D, int M, int N, int K, const int* groups,
          int ngroups, int width, int zero_exp) {
    int KS = 0;
    for (int i = 0; i < ngroups; ++i) KS += groups[i];
#pragma omp parallel for schedule(dynamic, 64)
    for (long idx = 0; idx < (long)M * N; ++idx) {
        int i = idx / N, j = idx % N;
        const W* a = A + (size_t)i * K;
        const W* b = Bt + (size_t)j * K;
        Term acc = fp32_term(C ? C[idx] : 0u);
        Term terms[65];
        for (int k = 0; k < K; k += KS) {
            int off = k;
            for (int gi = 0; gi < ngroups; ++gi) {
                terms[0] = acc;
                for (int p = 0; p < groups[gi]; ++p)
                    terms[1 + p] = BF16 ? bf16_product(a[off + p], b[off + p]) : e4m3_product(a[off + p], b[off + p]);
                acc = group_sum(terms, 1 + groups[gi], width, zero_exp);
                off += groups[gi];
                if (acc.mag && acc.exp > 127) goto saturated;  // sticky per-group overflow to +-inf
            }
        }
    saturated:
        D[idx] = pack_fp32(acc);
    }
}

}  // namespace

extern "C" {

// groups: ngroups ints summing to the instruction K.  C may be NULL (zeros).
void ref_chain_bf16(const uint16_t* A, const uint16_t* Bt, const uint32_t* C, uint32_t* D, int M, int N, int K,
                    const int* groups, int ngroups, int width, int zero_exp) {
    gemm<uint16_t, true>(A, Bt, C, D, M, N, K, groups, ngroups, width, zero_exp);
}

void ref_chain_e4m3(const uint8_t* A, const uint8_t* Bt, const uint32_t* C, uint32_t* D, int M, int N, int K,
                    const int* groups, int ngroups, int width, int zero_exp) {
    gemm<uint8_t, false>(A, Bt, C, D, M, N, K, groups, ngroups, width, zero_exp);
}

}  // extern "C"
