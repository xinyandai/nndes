/* 
    Copyright (C) 2010,2011 Wei Dong <wdong.pku@gmail.com>. All Rights Reserved.

    DISTRIBUTION OF THIS PROGRAM IN EITHER BINARY OR SOURCE CODE FORM MUST BE
    PERMITTED BY THE AUTHOR.
*/

#ifndef __NNDES_DATA__
#define __NNDES_DATA__
#include <malloc.h>
#include <fstream>
#include <cmath>

namespace nndes {

template <typename T>
T sqr (T a) {
    return a * a;
}

// Dataset
// T: element type
// A: alignment, default = 128 bits
#ifdef __GNUC__
#ifdef __AVX__
#define NNDES_MATRIX_ALIGN 32
#else
#ifdef __SSE2__
#define NNDES_MATRIX_ALIGN 16
#else
#define NNDES_MATRIX_ALIGN 1
#endif
#endif
#endif

template <typename T, int A = NNDES_MATRIX_ALIGN> 
class Dataset
{
    int dim;
    int N;
    size_t stride;
    char *dims;
public:
    typedef T value_type;
    static const int ALIGN = A;

    void reset (int _dim, int _N)
    {
        BOOST_ASSERT((ALIGN % sizeof(T)) == 0);
        dim = _dim;
        N = _N;
        stride = dim * sizeof(T) + ALIGN - 1;
        stride = stride / ALIGN * ALIGN;
        if (dims != NULL) delete[] dims;
        dims = (char *)memalign(ALIGN, N * stride); // SSE instruction needs data to be aligned
        std::fill(dims, dims + N * stride, 0);
    }

    void free (void) {
        dim = N = stride = 0;
        if (dims != NULL) free(dims);
        dims = NULL;
    }
    
    Dataset () :dim(0), N(0), dims(NULL) {}
    Dataset (int _dim, int _N) : dims(NULL) { reset(_dim, _N); }
    ~Dataset () { if (dims != NULL) delete[] dims; }

    /// Access the ith vector.
    const T *operator [] (int i) const {
        return (const T *)(dims + i * stride);
    }

    /// Access the ith vector.
    T *operator [] (int i) {
        return (T *)(dims + i * stride);
    }

    int getDim () const {return dim; }
    int size () const {return N; }

    void load (const std::string &path) {
        std::ifstream is(path.c_str(), std::ios::binary);
        int header[3]; /* entry size, row, col */
        assert(sizeof header == 3*4);
        is.read((char *)header, sizeof header);
        BOOST_VERIFY(is);
        BOOST_VERIFY(header[0] == sizeof(T));
        reset(header[2], header[1]);
        char *off = dims;
        for (int i = 0; i < N; ++i) {
            is.read(off, sizeof(T) * dim);
            off += stride;
        }
        BOOST_VERIFY(is);
    }

    void loadFvecs(const std::string &path) {


   	std::ifstream is(path.c_str(), std::ios::binary);
	assert(dims!=NULL);

	char* off = dims;
	int dimension;

	for(int i=0; i<this->N; i++) {
		is.read((char*)&dimension, sizeof(dimension));
		assert(dimension == dim);
		is.read(off, sizeof(T)*dim);

		off += stride;
	}
	std::cout << stride << std::endl;
	BOOST_VERIFY(is);
    }

    // initialize from a file
    void load (const std::string &path, int _dim, int skip = 0, int gap = 0) {
        std::ifstream is(path.c_str(), std::ios::binary);
        BOOST_VERIFY(is);
        is.seekg(0, std::ios::end);
        size_t size = is.tellg();
        size -= skip;
        int line = sizeof(float) * _dim + gap;
        BOOST_VERIFY(size % line == 0);
        int _N =  size / line;
        reset(_dim, _N);
        is.seekg(skip, std::ios::beg);
        char *off = dims;
        for (int i = 0; i < N; ++i) {
            is.read(off, sizeof(T) * dim);
            is.seekg(gap, std::ios::cur);
            off += stride;
        }
        BOOST_VERIFY(is);
    }

    Dataset (const std::string &path, int _dim, int skip = 0, int gap = 0): dims(NULL) {
        load(path, _dim, skip, gap);
    }

    /*
    void save (std::ostream &os) {
        int header[3];
        assert(sizeof header == 3*4);
        header[0] = sizeof(T);
        header[1] = N;
        header[2] = dim;
        os.write((const char *)header, sizeof(header));
        const char *off = dims;
        for (int i = 0; i < N; ++i) {
            os.write(off, sizeof(T) * dim);
            off += stride;
        }
        BOOST_VERIFY(os);
    }

    void save (const std::string &path) {
        std::ofstream os(path.c_str(), std::ios::binary);
        save(os);
    }

    void print (std::ostream &v, int id) const {
        const T *p = operator[](id);
        for (int i = 0; i < std::min(5, dim); ++i) {
            if (i) {
                v << " ";
            }
            v << p[i];
        }
    }
    */

    float operator () (int i, int j) const __attribute__ ((noinline));
};

// L1 distance oracle on a dense dataset
class OracleDirect {
    const Dataset<float> &m;
public:
    OracleDirect (const Dataset<float> &m_): m(m_) {
    }
    float operator () (int i, int j) const {
        return m[i][j];
    }
};

// L1 distance oracle on a dense dataset
template <typename M>
class OracleL1 {
    const M &m;
public:
    OracleL1 (const M &m_): m(m_) {
    }
    float operator () (int i, int j) const {
        const typename M::value_type *first1 = m[i];
        const typename M::value_type *first2 = m[j];
        float r = 0.0;
        for (int i = 0; i < m.getDim(); ++i)
        {
            r += fabs(first1[i] - first2[i]);
        }
        return r;
    }
};

// L2 distance oracle on a dense dataset
// special SSE optimization is implemented for float data
template <typename M>
class OracleL2 {
    const M &m;
public:
    OracleL2 (const M &m_): m(m_) {}
    float operator () (int i, int j) const __attribute__ ((noinline));
};

template <typename M>
float OracleL2<M>::operator () (int i, int j) const {
    const typename M::value_type *first1 = m[i];
    const typename M::value_type *first2 = m[j];
    float r = 0.0;
    for (int i = 0; i < m.getDim(); ++i)
    {
        float v = first1[i] - first2[i];
        r += v * v;
    }
    return sqrt(r);
}


template <typename M>
class OracleAngular {
    const M &m;
public:
    OracleAngular (const M &m_): m(m_) {}
    float operator () (int i, int j) const __attribute__ ((noinline));
};

template <typename M>
float OracleAngular<M>::operator () (int i, int j) const {
    const typename M::value_type *first1 = m[i];
    const typename M::value_type *first2 = m[j];
    float dist_  = 0.0;
    float norm_1 = 0.0;
    float norm_2 = 0.0;
    for (int i = 0; i < m.getDim(); ++i)
    {
	dist_  += first1[i] * first2[i];
	norm_1 += first1[i] * first1[i];
	norm_2 += first2[i] * first2[i];

    }
    float angle =  acos( dist_ / std::sqrt(norm_1*norm_2) );
    return angle;
}


typedef Dataset<float> FloatDataset;

}

// !!! If problems happen here, the following lines can be removed
// without affecting the correctness of the library.
#ifdef __GNUC__
#ifdef __AVX__
#include <nndes-data-avx.h>
#else
#ifdef __SSE2__
#include <nndes-data-sse2.h>
#endif
#endif
#endif

#endif
