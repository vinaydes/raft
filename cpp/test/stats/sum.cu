/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include "../test_utils.h"
#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/eltwise.cuh>
#include <raft/random/rng.hpp>
#include <raft/stats/sum.hpp>

namespace raft {
namespace stats {

template <typename T>
struct SumInputs {
  T tolerance;
  int rows, cols;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SumInputs<T>& dims)
{
  return os;
}

template <typename T>
class SumTest : public ::testing::TestWithParam<SumInputs<T>> {
 public:
  SumTest()
    : params(::testing::TestWithParam<SumInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      rows(params.rows),
      cols(params.cols),
      data(rows * cols, stream),
      sum_act(cols, stream)
  {
  }

 protected:
  void SetUp() override
  {
    int len = rows * cols;

    T data_h[len];
    for (int i = 0; i < len; i++) {
      data_h[i] = T(1);
    }

    raft::update_device(data.data(), data_h, len, stream);
    sum(sum_act.data(), data.data(), cols, rows, false, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  SumInputs<T> params;
  int rows, cols;
  rmm::device_uvector<T> data, sum_act;
};

const std::vector<SumInputs<float>> inputsf = {{0.05f, 1024, 32, 1234ULL},
                                               {0.05f, 1024, 256, 1234ULL}};

const std::vector<SumInputs<double>> inputsd = {{0.05, 1024, 32, 1234ULL},
                                                {0.05, 1024, 256, 1234ULL}};

typedef SumTest<float> SumTestF;
TEST_P(SumTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    float(params.rows), sum_act.data(), params.cols, raft::CompareApprox<float>(params.tolerance)));
}

typedef SumTest<double> SumTestD;
TEST_P(SumTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(double(params.rows),
                                sum_act.data(),
                                params.cols,
                                raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(SumTests, SumTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(SumTests, SumTestD, ::testing::ValuesIn(inputsd));

}  // end namespace stats
}  // end namespace raft
