#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <stdio.h>

namespace cuda {

#define THROW(fmt, ...)                                            \
  do {                                                             \
    std::string msg;                                               \
    char errMsg[2048];                                             \
    snprintf(errMsg, 2048, "Exception occured! file=%s line=%d: ", \
             __FILE__, __LINE__);                                  \
    msg += errMsg;                                                 \
    sprintf(errMsg, fmt, ##__VA_ARGS__);                           \
    msg += errMsg;                                                 \
    throw std::runtime_error(msg);                                 \
  } while (0)

#define ASSERT(check, fmt, ...)                                 \
  do { if (!(check))  THROW(fmt, ##__VA_ARGS__); } while (0)

#define CUDA_CHECK(call)                                           \
  do {                                                             \
    cudaError_t status = call;                                     \
    ASSERT(status == cudaSuccess, "FAIL: call='%s'. Reason:%s\n",  \
           #call, cudaGetErrorString(status));                     \
    } while (0)

template <typename T>
void copy(T* out, const T* in, size_t len, cudaMemcpyKind kind) {
  CUDA_CHECK(cudaMemcpy(out, in, sizeof(T) * len, kind));
}

template <typename T>
void copy_async(T* out, const T* in, size_t len, cudaMemcpyKind kind,
                cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(out, in, sizeof(T) * len, kind, stream));
}

inline void sync(cudaStream_t stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace cuda
