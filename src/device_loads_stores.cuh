
#pragma once

#include <cuda_fp16.h>

namespace cuda {

__device__ inline float half2_as_float(__half2& x) {
  return *static_cast<float*>(&x);
}
__device__ inline __half2 float_as_half2(float& x) {
  return *static_cast<__half2*>(&x);
}

/**
 * @defgroup Stores Shared/Global memory store operations
 * @{
 * @brief Stores to shared/global memory (both vectorized and non-vectorized
 *        forms)
 *
 * @param[out] addr shared/global memory address
 * @param[in]  x    data to be stored at this address
 */
__device__ inline void store(__half2* addr, const __half2& x) { *addr = x; }
__device__ inline void store(__half2* addr, const __half2 (&x)[1]) {
  *addr = x[0];
}
__device__ inline void store(__half2* addr, const __half2 (&x)[2]) {
  float2 v2 = make_float2(half2_as_float(x[0]), half2_as_float(x[1]));
  auto* s2 = reinterpret_cast<float2*>(addr);
  *s2 = v2;
}
__device__ inline void store(__half2* addr, const __half2 (&x)[4]) {
  float4 v4 = make_float4(half2_as_float(x[0]), half2_as_float(x[1]),
                          half2_as_float(x[2]), half2_as_float(x[3]));
  auto* s4 = reinterpret_cast<float4*>(addr);
  *s4 = v4;
}

__device__ inline void store(float* addr, const float& x) { *addr = x; }
__device__ inline void store(float* addr, const float (&x)[1]) { *addr = x[0]; }
__device__ inline void store(float* addr, const float (&x)[2]) {
  float2 v2 = make_float2(x[0], x[1]);
  auto* s2 = reinterpret_cast<float2*>(addr);
  *s2 = v2;
}
__device__ inline void store(float* addr, const float (&x)[4]) {
  float4 v4 = make_float4(x[0], x[1], x[2], x[3]);
  auto* s4 = reinterpret_cast<float4*>(addr);
  *s4 = v4;
}

__device__ inline void store(double* addr, const double& x) { *addr = x; }
__device__ inline void store(double* addr, const double (&x)[1]) {
  *addr = x[0];
}
__device__ inline void store(double* addr, const double (&x)[2]) {
  double2 v2 = make_double2(x[0], x[1]);
  auto* s2 = reinterpret_cast<double2*>(addr);
  *s2 = v2;
}
/** @} */

/**
 * @defgroup Loads Shared/Global memory load operations
 * @{
 * @brief Loads from shared/global memory (both vectorized and non-vectorized
 *        forms)
 *
 * @param[out] x    the data to be loaded
 * @param[in]  addr shared/global memory address from where to load
 */
__device__ inline void load(__half2& x, __half2* addr) { x = *addr; }
__device__ inline void load(__half2 (&x)[1], __half2* addr) { x[0] = *addr; }
__device__ inline void load(__half2 (&x)[2], __half2* addr) {
  auto* s2 = reinterpret_cast<float2*>(addr);
  auto v2 = *s2;
  x[0] = float_as_half2(v2.x);
  x[1] = float_as_half2(v2.y);
}
__device__ inline void load(float (&x)[4], float* addr) {
  auto* s4 = reinterpret_cast<float4*>(addr);
  auto v4 = *s4;
  x[0] = float_as_half2(v4.x);
  x[1] = float_as_half2(v4.y);
  x[2] = float_as_half2(v4.z);
  x[3] = float_as_half2(v4.w);
}

__device__ inline void load(float& x, float* addr) { x = *addr; }
__device__ inline void load(float (&x)[1], float* addr) { x[0] = *addr; }
__device__ inline void load(float (&x)[2], float* addr) {
  auto* s2 = reinterpret_cast<float2*>(addr);
  auto v2 = *s2;
  x[0] = v2.x;
  x[1] = v2.y;
}
__device__ inline void load(float (&x)[4], float* addr) {
  auto* s4 = reinterpret_cast<float4*>(addr);
  auto v4 = *s4;
  x[0] = v4.x;
  x[1] = v4.y;
  x[2] = v4.z;
  x[3] = v4.w;
}

__device__ inline void load(double& x, double* addr) { x = *addr; }
__device__ inline void load(double (&x)[1], double* addr) { x[0] = *addr; }
__device__ inline void load(double (&x)[2], double* addr) {
  auto* s2 = reinterpret_cast<double2*>(addr);
  auto v2 = *s2;
  x[0] = v2.x;
  x[1] = v2.y;
}
/** @} */

/**
 * @defgroup GlobalLoads Global cached load operations
 * @{
 * @brief Load from global memory with caching at L1 level
 *
 * @param[out] x    data to be loaded from global memory
 * @param[in]  addr address in global memory from where to load
 */
__device__ inline void ldg(__half2& x, const __half2* addr) {
  auto f = half2_as_float(x);
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(f) : "l"(addr));
}
__device__ inline void ldg(__half2 (&x)[1], const __half2* addr) {
  ldg(x[0], addr);
}
__device__ inline void ldg(__half2 (&x)[2], const __half2* addr) {
  auto f0 = half2_as_float(x[0]);
  auto f1 = half2_as_float(x[1]);
  asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];"
               : "=f"(f0), "=f"(f1)
               : "l"(addr));
}
__device__ inline void ldg(__half2 (&x)[4], const __half2* addr) {
  auto f0 = half2_as_float(x[0]);
  auto f1 = half2_as_float(x[1]);
  auto f2 = half2_as_float(x[2]);
  auto f3 = half2_as_float(x[3]);
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(f0), "=f"(f1), "=f"(f2), "=f"(f3)
               : "l"(addr));
}

__device__ inline void ldg(float& x, const float* addr) {
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x) : "l"(addr));
}
__device__ inline void ldg(float (&x)[1], const float* addr) {
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x[0]) : "l"(addr));
}
__device__ inline void ldg(float (&x)[2], const float* addr) {
  asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];"
               : "=f"(x[0]), "=f"(x[1])
               : "l"(addr));
}
__device__ inline void ldg(float (&x)[4], const float* addr) {
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
               : "l"(addr));
}

__device__ inline void ldg(double& x, const double* addr) {
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(x) : "l"(addr));
}
__device__ inline void ldg(double (&x)[1], const double* addr) {
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(x[0]) : "l"(addr));
}
__device__ inline void ldg(double (&x)[2], const double* addr) {
  asm volatile("ld.global.cg.v2.f64 {%0, %1}, [%2];"
               : "=d"(x[0]), "=d"(x[1])
               : "l"(addr));
}
/** @} */

}  // namespace cuda
