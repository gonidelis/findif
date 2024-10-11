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

namespace giannis {

template <typename T> struct not_equal_pair {
  __host__ __device__ bool operator()(T i) {
    return !(thrust::get<0>(i) == thrust::get<1>(i));
  }
};

template <class InputIt1, class InputIt2>
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2) {

  auto range_size = thrust::distance(first1, last1);
  auto zipit = thrust::make_zip_iterator(first1, first2);
  using value_type = thrust::iterator_value_t<decltype(zipit)>;

  thrust::device_vector<std::size_t> d_result(1);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes{};

  cub::DeviceFind::FindIf(d_temp_storage, temp_storage_bytes, zipit,
                          thrust::raw_pointer_cast(d_result.data()),
                          giannis::not_equal_pair<value_type>{}, range_size, 0);

  thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cub::DeviceFind::FindIf(d_temp_storage, temp_storage_bytes, zipit,
                          thrust::raw_pointer_cast(d_result.data()),
                          giannis::not_equal_pair<value_type>{}, range_size, 0);

  return d_result[0] == range_size; // device_vector operator[] copies to host
                                    // when invoked from host
}

} // namespace giannis

namespace bernard {

template <class InputIt1, class InputIt2>
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2) {

  using InputType1 = typename thrust::iterator_value<InputIt1>::type;
  const auto n = distance(first1, last1);

  using transform_t = thrust::cuda_cub::transform_pair_of_input_iterators_t<
      bool, InputIt1, InputIt2, thrust::not_equal_to<InputType1>>;

  transform_t transformed_first =
      transform_t(first1, first2, thrust::not_equal_to<InputType1>{});

  return thrust::count_if(thrust::device, transformed_first,
                          transformed_first + n, thrust::identity{}) == 0;
}

} // namespace bernard

int main() {

  printf("mismatchAt, cub::FindIf, thrust::find_if, thrust::count_if\n");
  for (float mismatch_at = 0; mismatch_at < 1; mismatch_at += 0.01) {
    std::printf("%.3f", mismatch_at * 100);

    // data set up
    static constexpr std::size_t elements = 2 << 28;
    int val = 1;
    // set up input
    const auto mismatch_point = elements * mismatch_at;

    thrust::device_vector<int32_t> dinput1(elements, 0);
    thrust::device_vector<int32_t> dinput2(elements, 0);
    thrust::fill(dinput1.begin() + mismatch_point, dinput1.end(), val);
    thrust::device_vector<int32_t> d_result(1);
    ///

    // warm up the clock
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);

    /// cub::FindIf
    start = std::chrono::high_resolution_clock::now();
    giannis::equal(dinput1.begin(), dinput1.end(), dinput2.begin());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - start);

    std::printf(", %.4f ", duration.count());

    /// thrust::find_if backend
    start = std::chrono::high_resolution_clock::now();
    thrust::equal(dinput1.begin(), dinput1.end(), dinput2.begin());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - start);
    std::printf(", %.4f", duration.count());

    /// thrust::count_if backend
    start = std::chrono::high_resolution_clock::now();
    bernard::equal(dinput1.begin(), dinput1.end(), dinput2.begin());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - start);
    std::printf(", %.4f \n", duration.count());
  }
}