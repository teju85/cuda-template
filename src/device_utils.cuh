#pragma once

namespace cuda {

/** number of threads per warp */
static const int WarpSize = 32;

/**
 * @brief Provide a ceiling division operation ie. ceil(a / b)
 *
 * @tparam int_t supposed to be only integers for now!
 *
 * @param[in] a dividend
 * @param[in] b divisor
 */
template <typename int_t>
constexpr __host__ __device__ inline int_t ceil_div(int_t a, int_t b) {
  return (a + b - 1) / b;
}

/**
 * @brief Provide an alignment function ie. ceil(a / b) * b
 *
 * @tparam int_t supposed to be only integers for now!
 *
 * @param[in] a dividend
 * @param[in] b divisor
 */
template <typename int_t>
constexpr __host__ __device__ inline int_t align_to(int_t a, int_t b) {
  return ceildiv(a, b) * b;
}

/**
 * @brief Provide an alignment function ie. (a / b) * b
 *
 * @tparam int_t supposed to be only integers for now!
 *
 * @param[in] a dividend
 * @param[in] b divisor
 */
template <typename int_t>
constexpr __host__ __device__ inline int_t align_down(int_t a, int_t b) {
  return (a / b) * b;
}

/**
 * @brief Check if the input is a power of 2
 *
 * @tparam int_t data type (checked only for integers)
 *
 * @param[in] num input
 */
template <typename int_t>
constexpr __host__ __device__ inline bool is_po2(int_t num) {
  return (num && !(num & (num - 1)));
}

/**
 * @brief Give logarithm of the number to base-2
 *
 * @tparam int_t data type (checked only for integers)
 *
 * @param[in] num input number
 * @param[in] ret value returned during each recursion
 */
template <typename int_t>
constexpr __host__ __device__ inline int_t log2(int_t num,
                                                int_t ret = int_t(0)) {
  return num <= int_t(1) ? ret : log2(num >> int_t(1), ++ret);
}

/** apply a warp-wide sync (useful from Volta+ archs) */
__device__ inline void warp_sync() {
#if __CUDA_ARCH__ >= 700
  __syncwarp();
#endif
}

/** get the laneId of the current thread */
__device__ inline int lane_id() {
  int id;
  asm("mov.s32 %0, %laneid;" : "=r"(id));
  return id;
}

/**
 * @brief warp-wide any boolean aggregator
 *
 * @param[in] in_flag flag to be checked across threads in the warp
 * @param[in] mask    set of threads to be checked in this warp
 *
 * @return true if any one of the threads have their `in_flag` set to true
 */
__device__ inline bool any(bool in_flag, uint32_t mask = 0xffffffffu) {
#if CUDART_VERSION >= 9000
  in_flag = __any_sync(mask, in_flag);
#else
  in_flag = __any(in_flag);
#endif
  return in_flag;
}

/**
 * @brief warp-wide all boolean aggregator
 *
 * @param[in] in_flag flag to be checked across threads in the warp
 * @param[in] mask    set of threads to be checked in this warp
 *
 * @return true if all of the threads have their `in_flag` set to true
 */
__device__ inline bool all(bool in_flag, uint32_t mask = 0xffffffffu) {
#if CUDART_VERSION >= 9000
  in_flag = __all_sync(mask, in_flag);
#else
  in_flag = __all(in_flag);
#endif
  return in_flag;
}

/**
 * @brief Shuffle the data inside a warp
 *
 * @tparam T the data type (currently assumed to be 4B)
 *
 * @param[in] val      value to be shuffled
 * @param[in] src_lane lane from where to shuffle
 * @param[in] width    lane width
 * @param[in] mask     mask of participating threads (Volta+)
 *
 * @return the shuffled data
 */
template <typename T>
__device__ inline T shfl(T val, int src_lane, int width = WarpSize,
                         uint32_t mask = 0xffffffffu) {
#if CUDART_VERSION >= 9000
  return __shfl_sync(mask, val, src_lane, width);
#else
  return __shfl(val, src_lane, width);
#endif
}

/**
 * @brief Shuffle the data inside a warp
 *
 * @tparam T the data type (currently assumed to be 4B)
 *
 * @param[in] val       value to be shuffled
 * @param[in] lane_mask mask to be applied in order to perform xor shuffle
 * @param[in] width     lane width
 * @param[in] mask      mask of participating threads (Volta+)
 *
 * @return the shuffled data
 */
template <typename T>
__device__ inline T shfl_xor(T val, int lane_mask, int width = WarpSize,
                             uint32_t mask = 0xffffffffu) {
#if CUDART_VERSION >= 9000
  return __shfl_xor_sync(mask, val, lane_mask, width);
#else
  return __shfl_xor(val, lane_mask, width);
#endif
}

}  // namespace cuda
