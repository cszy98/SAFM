#pragma once

#include <ATen/ATen.h>

void count_get_kernel_forward(
    at::Tensor& descriptor,
    at::Tensor& r_array_q,
    at::Tensor& theta_array_q,
    at::Tensor& sum_points
    );