/* 
    Copyright (C) 2010,2011 Wei Dong <wdong.pku@gmail.com>. All Rights Reserved.

    DISTRIBUTION OF THIS PROGRAM IN EITHER BINARY OR SOURCE CODE FORM MUST BE
    PERMITTED BY THE AUTHOR.
*/

#ifndef __NNDES_DATA_AVX__
#define __NNDES_DATA_AVX__
#include <immintrin.h>

namespace nndes {
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_load_ps(addr1);\
    tmp2 = _mm256_load_ps(addr2);\
    tmp1 = _mm256_sub_ps(tmp1, tmp2); \
    tmp1 = _mm256_mul_ps(tmp1, tmp1); \
    dest = _mm256_add_ps(dest, tmp1); 

template <>
float OracleL2<Dataset<float, 32> >::operator () (int i, int j) const 
{
    __attribute__ ((aligned (32))) __m256 sum;
    __attribute__ ((aligned (32))) __m256 l0, l1, l2, l3;
    __attribute__ ((aligned (32))) __m256 r0, r1, r2, r3;
    int D = (m.getDim() + 7) & ~7U; // # dim aligned up to 256 bits, or 8 floats
    int DR = D % 32;
    int DD = D - DR;
    const float *l = m[i];
    const float *r = m[j];
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float ret = 0.0;
    sum = _mm256_loadu_ps(unpack);
    switch (DR) {
        case 24:
            AVX_L2SQR(e_l+16, e_r+16, sum, l2, r2);
        case 16:
            AVX_L2SQR(e_l+8, e_r+8, sum, l1, r1);
        case 8:
            AVX_L2SQR(e_l, e_r, sum, l0, r0);
    }
    for (int i = 0; i < DD; i += 32, l += 32, r += 32) {
        AVX_L2SQR(l, r, sum, l0, r0);
        AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
        AVX_L2SQR(l + 16, r + 16, sum, l2, r2);
        AVX_L2SQR(l + 24, r + 24, sum, l3, r3);
    }
    _mm256_storeu_ps(unpack, sum);
    ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    return sqrt(ret);
}

}
#endif
