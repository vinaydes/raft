/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <sys/timeb.h>

#include "../test_utils.h"
#include <cub/cub.cuh>
#include <gtest/gtest.h>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/stddev.cuh>

using namespace raft::random;

uint64_t get_seed() {
  FILE* fp;
  fp = fopen("/dev/urandom", "r+b");
  uint64_t seed;
  auto _ = fread(&seed, sizeof(uint64_t), 1, fp);
  fclose(fp);
  return seed;
}

void uniform_cpu(float* buff, size_t len, uint64_t seed) {
    std::mt19937 rng(seed);
    for (size_t i = 0 ; i < len; i++) {
      buff[i] = float(rng() >> 8) / (1u << 24);
    }
}

template <uint64_t TRIALS, uint64_t BINS, int METHOD>
size_t chi_experiment(uint64_t seed) {
  raft::handle_t handle;

  float* data;
  float* h_data;

  RAFT_CUDA_TRY(cudaMalloc(&data, TRIALS * sizeof(float)));
  h_data = (float*) malloc(TRIALS * sizeof(float));

  RngState r_pc(seed, GenPC);
  RngState r_philox(seed, GenPhilox);

  switch (METHOD) {
    case 0: // Use GenPC
      uniform(handle, r_pc, data, TRIALS, 0.0f, 1.0f);
      RAFT_CUDA_TRY(cudaMemcpy(h_data, data, TRIALS*sizeof(float), cudaMemcpyDeviceToHost));
    break;

    case 1: // Use Philox
      uniform(handle, r_philox, data, TRIALS, 0.0f, 1.0f);
      RAFT_CUDA_TRY(cudaMemcpy(h_data, data, TRIALS*sizeof(float), cudaMemcpyDeviceToHost));
    break;

    case 2: // Use CPU
      uniform_cpu(h_data, TRIALS, seed);
    break;
  }

  uint64_t* histogram;
  histogram = (uint64_t*) malloc(BINS * sizeof(uint64_t));
  memset(histogram, 0, BINS * sizeof(uint64_t));

  for (size_t i = 0; i < TRIALS; i++) {
    uint64_t bin = uint64_t(h_data[i] * (1u << 24));
    histogram[bin]++;
  }

  uint64_t scaled_chisq_stat = 0;
  uint64_t E_i = (TRIALS/BINS); // Expected count

  for (uint64_t i = 0; i < BINS; i++) {
    scaled_chisq_stat += ((histogram[i] - E_i) * (histogram[i] - E_i));
  }

  printf("[Method: %d] chisq_stat * E_i = %lu, E_i = %lu, chisq_stat = %.8e\n", METHOD,
         scaled_chisq_stat, E_i, double(scaled_chisq_stat) / E_i);

  RAFT_CUDA_TRY(cudaFree(data));
  free(h_data);
  free(histogram);
  return 0;
}

TEST(Rng, chisquare)
{
  constexpr uint64_t TRIALS = uint64_t(1) << 31;
  constexpr uint64_t BINS = uint64_t(1) << 24;

  uint64_t seed = get_seed();
  printf("Using seed = %lu\n", seed);
  chi_experiment<TRIALS, BINS, 0>(seed); // GenPC
  chi_experiment<TRIALS, BINS, 1>(seed); // GenPhilox
  chi_experiment<TRIALS, BINS, 2>(seed); // MT generator on CPU
}
