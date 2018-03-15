#ifndef CROSS_CORRELATE_H
#define CROSS_CORRELATE_H

#ifdef _WIN32
  #include <windows.h>
  #define DLL_EXPORT __declspec(dllexport)
#else
  #define DLL_EXPORT
#endif

extern "C" {

    void DLL_EXPORT copy_signal_mf(void *y,
                                   size_t size);

    void DLL_EXPORT cross_correlate(void *d_corrcoef,
                                    void *d_data,
                                    dim3 dims,
                                    int dtype_id,
                                    cudaStream_t *stream=NULL);
                                    
}

#endif /* ifndef CROSS_CORRELATE_H */
