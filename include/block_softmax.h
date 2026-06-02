#ifndef BLOCK_SOFTMAX_H
#define BLOCK_SOFTMAX_H

#include <cuda_runtime.h>

// Online-softmax update helpers for tiled softmax / flash attention.
// Each thread maintains its own running max and sum (denom).
// When processing a new tile, update max and rescale.

__inline__ __device__ void online_softmax_update(float &old_max,
                                                  float &old_sum,
                                                  const float *tile_scores,
                                                  int tile_size) {
  float new_max = old_max;
  for (int i = 0; i < tile_size; i++) {
    new_max = fmaxf(new_max, tile_scores[i]);
  }
  float rescale = expf(old_max - new_max);
  float local_sum = 0.0f;
  for (int i = 0; i < tile_size; i++) {
    local_sum += expf(tile_scores[i] - new_max);
  }
  old_sum = old_sum * rescale + local_sum;
  old_max = new_max;
}

#endif // BLOCK_SOFTMAX_H