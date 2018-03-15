#ifndef CUDA_ERRCHK_H
#define CUDA_ERRCHK_H


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cufft.h>
#include <iostream>
#include <stdio.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %i %s %s %d\n", code, cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


#define gpuContextErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(CUresult code, const char *file, int line, bool abort = true)
{
	if (code != CUDA_SUCCESS)
	{
		const char* errstr;
		cuGetErrorString(code, &errstr);
		fprintf(stderr, "GPUassert: %i %s %s %d\n", code, errstr, file, line);
		if (abort) exit(code);
	}
}



static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

#define gpuBlasErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cublasStatus_t code, const char *file, int line, bool abort = true)
{
	if (code != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "GPUassert: %i %s %s %d\n", code, _cublasGetErrorEnum(code), file, line);
		if (abort) exit(code);
	}
}


static const char *_cusolverGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }

    return "<unknown>";
}

#define gpuSolverErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cusolverStatus_t code, const char *file, int line, bool abort = true)
{
	if (code != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "GPUassert: %i %s %s %d\n", code, _cusolverGetErrorEnum(code), file, line);
		if (abort) exit(code);
	}
}



static const char *_cufftGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

#define gpuFFTErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cufftResult_t code, const char *file, int line, bool abort = true)
{
	if (code != CUFFT_SUCCESS)
	{
		fprintf(stderr, "GPUassert: %i %s %s %d\n", code, _cufftGetErrorEnum(code), file, line);
		if (abort) exit(code);
	}
}


#endif/* ifndef CUDA_ERRCHK_H */
