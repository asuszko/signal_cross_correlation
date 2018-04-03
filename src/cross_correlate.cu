#include "cu_errchk.h"
#include "cross_correlate.h"

#define IX2D(i,j,nx) (((j)*(nx))+(i))

#define BLOCKSIZE 256

__constant__ char d_signal_mf[49152];


/**
*  Does a normalized cross correlation of signal y against the
*  signal stored within the constant memory.
*  @param y [T *] : Signal y with length signal_len.
*  @param signal_len [int] : Length of signal y.
*  @return r [T] : Normalized cross correlation result.
*/
template <typename T>
__device__ T correlate_xy(const T* __restrict__ y, int signal_len)
{
    T zero = static_cast<T>(0);
    T s_xisq = zero;
    T s_yisq = zero;
    T s_xiyi = zero;
    T s_xbar = zero;
    T s_ybar = zero;
    int counter = 0;
    
    T *ref_ptr = reinterpret_cast<T*>(d_signal_mf);
    
    for (int i = 0; i < signal_len; ++i) {
        T ref_pix_val = ref_ptr[i];
        T jth_pix_val = y[i];
        
        s_xbar += ref_pix_val;
        s_ybar += jth_pix_val;
        s_xiyi += ref_pix_val * jth_pix_val;
        s_xisq += ref_pix_val * ref_pix_val;
        s_yisq += jth_pix_val * jth_pix_val;
        counter += 1;
    }
    
    T Tcounter = static_cast<T>(counter);
    s_xbar /= Tcounter;
    s_ybar /= Tcounter;
    
    T r = (s_xiyi - Tcounter*s_xbar*s_ybar)/
            (static_cast<T>(sqrt(s_xisq - Tcounter*s_xbar*s_xbar))*
             static_cast<T>(sqrt(s_yisq - Tcounter*s_ybar*s_ybar)));
    
    return r;
}


/**
*  Wrapper for the correlate_xy device function.
*  @param corrcoef [T *] : Device pointer to correlation coefficient matrix.
*  @param data [T *] : Device pointer to correlation return signal matrix.
*  @param batch_x [int] : i-dimension of the data matrix.
*  @param batch_y [int] : j-dimension of the data matrix.
*  @param signal_len [int] : Length of each signal in the data matrix.
*/
template <typename T>
__global__ void cross_correlate_kernel(T* __restrict__ corrcoef,
                                       const T* __restrict__ data,
                                       int batch_y,
                                       int signal_len)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    
    for(; index < batch_y; index+=stride) {
        corrcoef[index] = correlate_xy(&data[index*signal_len], signal_len);
    }
}


/**
*  Copies the signal to cross correlate against to constant memory.
*  @param y [void *] : Host pointer to signal.
*  @param size [size_t] : Size of the signal in bytes.
*/
void copy_signal_mf(void *y,
                    size_t size)
{
    gpuErrchk(cudaMemcpyToSymbol(d_signal_mf, y, size));
    return;
}


/**
*  Extern C compatible wrapper to export in shared library.
*  @param d_corrcoef [void *] : Device pointer to correlation coefficient matrix.
*  @param d_data [void *] : Device pointer to correlation return signal matrix.
*  @param dims [dim3] : Dimensions of d_data as (nx,ny,nz)
*  @param dtype_id [int] : Data type identifier to convert void to proper type.
*  @param stream [cudaStream_t *] : CUDA stream.
*/
void cross_correlate(void *d_corrcoef,
                     void *d_data,
                     dim3 dims,
                     int dtype_id,
                     cudaStream_t *stream)
{
    /* Some useful constants. */
    int signal_len = dims.x;
    int batch_y = dims.y;

    /* Get the grid size for this stream. */
    dim3 blockSize(BLOCKSIZE);
    dim3 gridSize((((batch_y-1)/blockSize.x+1)-1)/blockSize.x+1);
    
    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype_id) {
        case 0:
            cross_correlate_kernel<<<gridSize, blockSize, 0, stream_id>>>(static_cast<float*>(d_corrcoef),
                                                                          static_cast<const float*>(d_data),
                                                                          batch_y, signal_len);
            break;

        case 1:
            cross_correlate_kernel<<<gridSize, blockSize, 0, stream_id>>>(static_cast<double*>(d_corrcoef),
                                                                          static_cast<const double*>(d_data),
                                                                          batch_y, signal_len);
            break;
    }
    return;
}
