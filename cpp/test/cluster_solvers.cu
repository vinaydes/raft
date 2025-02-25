/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <raft/handle.hpp>

#include <raft/spectral/modularity_maximization.hpp>

namespace raft {

TEST(Raft, ClusterSolvers)
{
  using namespace matrix;
  using index_type = int;
  using value_type = double;

  handle_t h;

  index_type maxiter{100};
  value_type tol{1.0e-10};
  unsigned long long seed{100110021003};

  auto stream = h.get_stream();

  index_type n{100};
  index_type d{10};
  index_type k{5};

  // nullptr expected to trigger exceptions:
  //
  value_type* eigvecs{nullptr};
  index_type* codes{nullptr};

  cluster_solver_config_t<index_type, value_type> cfg{k, maxiter, tol, seed};

  kmeans_solver_t<index_type, value_type> cluster_solver{cfg};

  EXPECT_ANY_THROW(cluster_solver.solve(h, n, d, eigvecs, codes));
}

TEST(Raft, ModularitySolvers)
{
  using namespace matrix;
  using index_type = int;
  using value_type = double;

  handle_t h;
  ASSERT_EQ(0, h.get_device());

  index_type neigvs{10};
  index_type maxiter{100};
  index_type restart_iter{10};
  value_type tol{1.0e-10};
  bool reorthog{true};

  // nullptr expected to trigger exceptions:
  //
  index_type* clusters{nullptr};
  value_type* eigvals{nullptr};
  value_type* eigvecs{nullptr};

  unsigned long long seed{100110021003};

  eigen_solver_config_t<index_type, value_type> eig_cfg{
    neigvs, maxiter, restart_iter, tol, reorthog, seed};
  lanczos_solver_t<index_type, value_type> eig_solver{eig_cfg};

  index_type k{5};

  cluster_solver_config_t<index_type, value_type> clust_cfg{k, maxiter, tol, seed};
  kmeans_solver_t<index_type, value_type> cluster_solver{clust_cfg};

  auto stream = h.get_stream();
  sparse_matrix_t<index_type, value_type> sm{h, nullptr, nullptr, nullptr, 0, 0};

  EXPECT_ANY_THROW(spectral::modularity_maximization(
    h, sm, eig_solver, cluster_solver, clusters, eigvals, eigvecs));

  value_type modularity{0};
  EXPECT_ANY_THROW(spectral::analyzeModularity(h, sm, k, clusters, modularity));
}

}  // namespace raft
