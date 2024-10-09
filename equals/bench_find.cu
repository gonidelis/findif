/******************************************************************************
 * Copyright (c) 2024, Giannis Gonidelis. All rights reserved.
 ******************************************************************************/

#include "thrust/distance.h"
#include "thrust/functional.h"
#include "thrust/iterator/detail/iterator_traits.inl"
#include "thrust/iterator/zip_iterator.h"
#include <chrono>
#include <cub/device/device_find_if.cuh>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

template <typename T> struct equals {
  T val;
  equals(T _val) : val(_val) {}

  __device__ __host__ bool operator()(T i) { return i == val; }
};

namespace giannis {

template <class InputIt1> int find(InputIt1 first1, InputIt1 last1, int val) {

  auto range_size = thrust::distance(first1, last1);
  thrust::device_vector<std::size_t> d_result(1);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes{};

  cub::DeviceFind::FindIf(d_temp_storage, temp_storage_bytes, first1,
                          thrust::raw_pointer_cast(d_result.data()),
                          equals{val}, range_size, 0);

  thrust::device_vector<int32_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cub::DeviceFind::FindIf(d_temp_storage, temp_storage_bytes, first1,
                          thrust::raw_pointer_cast(d_result.data()),
                          equals{val}, range_size, 0);

  return d_result[0];
}

} // namespace giannis

int main() {

  printf("mismatchAt, cub::FindIf, thrust::find_if, thrust::count_if\n");
  for (float mismatch_at = 0; mismatch_at < 0.1; mismatch_at += 0.005) {
    std::printf("%.3f", mismatch_at);

    // data set up
    static constexpr std::size_t elements = 2 << 28;
    int val = 1;
    // set up input
    const auto mismatch_point = elements * mismatch_at;

    thrust::device_vector<int32_t> dinput1(elements, 0);
    thrust::fill(dinput1.begin() + mismatch_point, dinput1.end(), val);
    thrust::device_vector<int32_t> d_result(1);
    ///

    // warm up the clock
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);

    /// cub::FindIf
    start = std::chrono::high_resolution_clock::now();
    auto x = giannis::find(dinput1.begin(), dinput1.end(), val);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - start);

    std::printf(", %.4f ", duration.count());

    /// thrust::find_if backend
    start = std::chrono::high_resolution_clock::now();
    auto y = thrust::find_if(dinput1.begin(), dinput1.end(), equals{val});
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - start);
    std::printf(", %.4f", duration.count());

    /// thrust::count_if backend
    start = std::chrono::high_resolution_clock::now();
    auto z = thrust::count_if(dinput1.begin(), dinput1.end(), equals{val});
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - start);
    std::printf(", %.4f \n", duration.count());

    std::cout << x << *y << z << std::endl;
  }
}