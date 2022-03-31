#include <ATen/ATen.h>
#include <torch/torch.h>

#include "count_get_kernel.cuh"

int count_get_cuda_forward(
    at::Tensor& descriptor,
    at::Tensor& r_array_q,
    at::Tensor& theta_array_q,
    at::Tensor& sum_points) {
      count_get_kernel_forward(descriptor,r_array_q,theta_array_q,sum_points);
    return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &count_get_cuda_forward, "GetCount forward (CUDA)");
}

