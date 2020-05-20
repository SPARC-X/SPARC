#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

static void cudaCheckCore(cudaError_t code, const char *file, int line) 
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDA Error %d : %s, at %s:%d\n", code, cudaGetErrorString(code), file, line);
        exit(code);
    }
}
 
#define cudaCheck(test)      { cudaCheckCore((test), __FILE__, __LINE__); }
#define cudaCheckAfterCall() { cudaCheckCore((cudaGetLastError()), __FILE__, __LINE__); }

static const char *cusparseGetErrorString(cusparseStatus_t error)
{
    // From: http://berenger.eu/blog/cusparse-cccuda-sparse-matrix-examples-csr-bcsr-spmv-and-conversions/
    // Read more at: http://docs.nvidia.com/cuda/cusparse/index.html#ixzz3f79JxRar
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "The operation completed successfully.";
            
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "The cuSPARSE library was not initialized.";
            
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "Resource allocation failed inside the cuSPARSE library.";
            
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "An unsupported value or parameter was passed to the function.";
            
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "The function requires a feature absent from the device architecture.";
     
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "An access to GPU memory space failed.";
     
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "The GPU program failed to execute.";
     
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "An internal cuSPARSE operation failed.";
     
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "The matrix type is not supported by this function.";
    }
 
    return "<unknown>";
}

static void cudaSparseCheckCore(cusparseStatus_t code, const char *file, int line) 
{
    if (code != CUSPARSE_STATUS_SUCCESS) 
    {
        fprintf(stderr,"Cuda Error %d : %s, at %s:%d\n", code, cusparseGetErrorString(code), file, line);
        exit(code);
    }
}
 
#define cudaSparseCheck(test) { cudaSparseCheckCore((test), __FILE__, __LINE__); }

#endif
 