/* 
    Copyright (C) 2010,2011 Wei Dong <wdong.pku@gmail.com>. All Rights Reserved.

    DISTRIBUTION OF THIS PROGRAM IN EITHER BINARY OR SOURCE CODE FORM MUST BE
    PERMITTED BY THE AUTHOR.
*/

#ifndef __NNDES_DATA_SSE2__
#define __NNDES_DATA_SSE2__
#include <xmmintrin.h>

namespace nndes {
#define SSE_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm_load_ps(addr1);\
    tmp2 = _mm_load_ps(addr2);\
    tmp1 = _mm_sub_ps(tmp1, tmp2); \
    tmp1 = _mm_mul_ps(tmp1, tmp1); \
    dest = _mm_add_ps(dest, tmp1); 

    template <>
    float OracleL2<Dataset<float, 16> >::operator () (int i, int j) const 
    {
        __m128 sum;
        __m128 l0, l1, l2, l3;
        __m128 r0, r1, r2, r3;
        int D = (m.getDim() + 3) & ~3U;
        int DR = D % 16;
        int DD = D - DR;
        const float *l = m[i];
        const float *r = m[j];
        const float *e_l = l + DD;
        const float *e_r = r + DD;
        float unpack[4] = {0, 0, 0, 0};
        float ret = 0.0;
        sum = _mm_loadu_ps(unpack);
        switch (DR) {
            case 12:
                SSE_L2SQR(e_l+8, e_r+8, sum, l2, r2);
            case 8:
                SSE_L2SQR(e_l+4, e_r+4, sum, l1, r1);
            case 4:
                SSE_L2SQR(e_l, e_r, sum, l0, r0);
        }
        for (int i = 0; i < DD; i += 16, l += 16, r += 16) {
            SSE_L2SQR(l, r, sum, l0, r0);
            SSE_L2SQR(l + 4, r + 4, sum, l1, r1);
            SSE_L2SQR(l + 8, r + 8, sum, l2, r2);
            SSE_L2SQR(l + 12, r + 12, sum, l3, r3);
        }
        _mm_storeu_ps(unpack, sum);
        ret = unpack[0] + unpack[1] + unpack[2] + unpack[3];
        return sqrt(ret);
    }

}
#endif
