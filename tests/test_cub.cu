#include <cub/cub.cuh>
#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

int main() {
  int N = 20;
  size_t mem = 0;
  long long *key = nullptr;
  void *d_temp_storage = NULL;
  cub::DeviceRadixSort::SortKeys(nullptr, mem, key, key, N);
  cudaMalloc(&d_temp_storage, mem);
  thrust::device_vector<long long> d_vec(N);
  thrust::sequence(d_vec.begin(), d_vec.end());
  mem = 9999;
  cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, mem, d_vec.data().get(), d_vec.data().get(), N);
  cudaDeviceSynchronize();
  thrust::host_vector<long long> h_vec = d_vec;
  for (int i = 0; i < N; i++) std::cout << h_vec[i] << std::endl;
}
