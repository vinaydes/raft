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
#include "matrix_vector_op.cuh"
#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/random/rng.hpp>

namespace raft {
namespace linalg {

template <typename T, typename IdxType = int>
struct MatVecOpInputs {
  T tolerance;
  IdxType rows, cols;
  bool rowMajor, bcastAlongRows, useTwoVectors;
  unsigned long long int seed;
};

template <typename T, typename IdxType>
::std::ostream& operator<<(::std::ostream& os, const MatVecOpInputs<T, IdxType>& dims)
{
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T, typename IdxType>
void matrixVectorOpLaunch(T* out,
                          const T* in,
                          const T* vec1,
                          const T* vec2,
                          IdxType D,
                          IdxType N,
                          bool rowMajor,
                          bool bcastAlongRows,
                          bool useTwoVectors,
                          cudaStream_t stream)
{
  if (useTwoVectors) {
    matrixVectorOp(
      out,
      in,
      vec1,
      vec2,
      D,
      N,
      rowMajor,
      bcastAlongRows,
      [] __device__(T a, T b, T c) { return a + b + c; },
      stream);
  } else {
    matrixVectorOp(
      out,
      in,
      vec1,
      D,
      N,
      rowMajor,
      bcastAlongRows,
      [] __device__(T a, T b) { return a + b; },
      stream);
  }
}

template <typename T, typename IdxType>
class MatVecOpTest : public ::testing::TestWithParam<MatVecOpInputs<T, IdxType>> {
 public:
  MatVecOpTest()
    : params(::testing::TestWithParam<MatVecOpInputs<T, IdxType>>::GetParam()),
      stream(handle.get_stream()),
      in(params.rows * params.cols, stream),
      out_ref(params.rows * params.cols, stream),
      out(params.rows * params.cols, stream),
      vec1(params.bcastAlongRows ? params.cols : params.rows, stream),
      vec2(params.bcastAlongRows ? params.cols : params.rows, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::Rng r(params.seed);
    IdxType N = params.rows, D = params.cols;
    IdxType len    = N * D;
    IdxType vecLen = params.bcastAlongRows ? D : N;
    r.uniform(in.data(), len, (T)-1.0, (T)1.0, stream);
    r.uniform(vec1.data(), vecLen, (T)-1.0, (T)1.0, stream);
    r.uniform(vec2.data(), vecLen, (T)-1.0, (T)1.0, stream);
    if (params.useTwoVectors) {
      naiveMatVec(out_ref.data(),
                  in.data(),
                  vec1.data(),
                  vec2.data(),
                  D,
                  N,
                  params.rowMajor,
                  params.bcastAlongRows,
                  (T)1.0,
                  stream);
    } else {
      naiveMatVec(out_ref.data(),
                  in.data(),
                  vec1.data(),
                  D,
                  N,
                  params.rowMajor,
                  params.bcastAlongRows,
                  (T)1.0,
                  stream);
    }
    matrixVectorOpLaunch(out.data(),
                         in.data(),
                         vec1.data(),
                         vec2.data(),
                         D,
                         N,
                         params.rowMajor,
                         params.bcastAlongRows,
                         params.useTwoVectors,
                         stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  MatVecOpInputs<T, IdxType> params;
  rmm::device_uvector<T> in, out, out_ref, vec1, vec2;
};

const std::vector<MatVecOpInputs<float, int>> inputsf_i32 = {
  {0.00001f, 1024, 32, true, true, false, 1234ULL},
  {0.00001f, 1024, 64, true, true, false, 1234ULL},
  {0.00001f, 1024, 32, true, false, false, 1234ULL},
  {0.00001f, 1024, 64, true, false, false, 1234ULL},
  {0.00001f, 1024, 32, false, true, false, 1234ULL},
  {0.00001f, 1024, 64, false, true, false, 1234ULL},
  {0.00001f, 1024, 32, false, false, false, 1234ULL},
  {0.00001f, 1024, 64, false, false, false, 1234ULL},

  {0.00001f, 1024, 32, true, true, true, 1234ULL},
  {0.00001f, 1024, 64, true, true, true, 1234ULL},
  {0.00001f, 1024, 32, true, false, true, 1234ULL},
  {0.00001f, 1024, 64, true, false, true, 1234ULL},
  {0.00001f, 1024, 32, false, true, true, 1234ULL},
  {0.00001f, 1024, 64, false, true, true, 1234ULL},
  {0.00001f, 1024, 32, false, false, true, 1234ULL},
  {0.00001f, 1024, 64, false, false, true, 1234ULL}};
typedef MatVecOpTest<float, int> MatVecOpTestF_i32;
TEST_P(MatVecOpTestF_i32, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.rows * params.cols, CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MatVecOpTests, MatVecOpTestF_i32, ::testing::ValuesIn(inputsf_i32));

const std::vector<MatVecOpInputs<float, size_t>> inputsf_i64 = {
  {0.00001f, 2500, 250, false, false, false, 1234ULL},
  {0.00001f, 2500, 250, false, false, true, 1234ULL}};
typedef MatVecOpTest<float, size_t> MatVecOpTestF_i64;
TEST_P(MatVecOpTestF_i64, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.rows * params.cols, CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MatVecOpTests, MatVecOpTestF_i64, ::testing::ValuesIn(inputsf_i64));

const std::vector<MatVecOpInputs<double, int>> inputsd_i32 = {
  {0.0000001, 1024, 32, true, true, false, 1234ULL},
  {0.0000001, 1024, 64, true, true, false, 1234ULL},
  {0.0000001, 1024, 32, true, false, false, 1234ULL},
  {0.0000001, 1024, 64, true, false, false, 1234ULL},
  {0.0000001, 1024, 32, false, true, false, 1234ULL},
  {0.0000001, 1024, 64, false, true, false, 1234ULL},
  {0.0000001, 1024, 32, false, false, false, 1234ULL},
  {0.0000001, 1024, 64, false, false, false, 1234ULL},

  {0.0000001, 1024, 32, true, true, true, 1234ULL},
  {0.0000001, 1024, 64, true, true, true, 1234ULL},
  {0.0000001, 1024, 32, true, false, true, 1234ULL},
  {0.0000001, 1024, 64, true, false, true, 1234ULL},
  {0.0000001, 1024, 32, false, true, true, 1234ULL},
  {0.0000001, 1024, 64, false, true, true, 1234ULL},
  {0.0000001, 1024, 32, false, false, true, 1234ULL},
  {0.0000001, 1024, 64, false, false, true, 1234ULL}};
typedef MatVecOpTest<double, int> MatVecOpTestD_i32;
TEST_P(MatVecOpTestD_i32, Result)
{
  ASSERT_TRUE(devArrMatch(out_ref.data(),
                          out.data(),
                          params.rows * params.cols,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MatVecOpTests, MatVecOpTestD_i32, ::testing::ValuesIn(inputsd_i32));

const std::vector<MatVecOpInputs<double, size_t>> inputsd_i64 = {
  {0.0000001, 2500, 250, false, false, false, 1234ULL},
  {0.0000001, 2500, 250, false, false, true, 1234ULL}};
typedef MatVecOpTest<double, size_t> MatVecOpTestD_i64;
TEST_P(MatVecOpTestD_i64, Result)
{
  ASSERT_TRUE(devArrMatch(out_ref.data(),
                          out.data(),
                          params.rows * params.cols,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MatVecOpTests, MatVecOpTestD_i64, ::testing::ValuesIn(inputsd_i64));

}  // end namespace linalg
}  // end namespace raft
