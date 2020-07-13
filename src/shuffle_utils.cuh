#pragma once

namespace cuda {

/**
 * @brief Shuffle the data inside a warp with a specified thread
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
 * @brief Shuffle the data inside a warp with the upper thread-id
 *
 * @tparam T the data type (currently assumed to be 4B)
 *
 * @param[in] val   value to be shuffled
 * @param[in] delta other thread is this much value higher
 * @param[in] width lane width
 * @param[in] mask  mask of participating threads (Volta+)
 *
 * @return the shuffled data
 */
template <typename T>
__device__ inline T shfl_up(T val, int delta, int width = WarpSize,
                            uint32_t mask = 0xffffffffu) {
#if CUDART_VERSION >= 9000
  return __shfl_up_sync(mask, val, delta, width);
#else
  return __shfl_up(val, delta, width);
#endif
}

/**
 * @brief Shuffle the data inside a warp with the lower thread-id
 *
 * @tparam T the data type (currently assumed to be 4B)
 *
 * @param[in] val   value to be shuffled
 * @param[in] delta other thread is this much value lower
 * @param[in] width lane width
 * @param[in] mask  mask of participating threads (Volta+)
 *
 * @return the shuffled data
 */
template <typename T>
__device__ inline T shfl_down(T val, int delta, int width = WarpSize,
                              uint32_t mask = 0xffffffffu) {
#if CUDART_VERSION >= 9000
  return __shfl_down_sync(mask, val, delta, width);
#else
  return __shfl_down(val, delta, width);
#endif
}

/**
 * @brief Shuffle the data inside a warp between the 2 participating threads
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
