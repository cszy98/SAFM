#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>
#include <iostream>

#define CUDA_NUM_THREADS 512
#define CUDA_MAX_THREADS 256

// #define THREADS_PER_BLOCK 64

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])
#define EPS 1e-8
#define SAFE_DIV(a, b)  ( (b==0)? ( (a)/(EPS) ): ( (a)/(b) )  )
#define CHECK_LEGALITY(x, min, max) ((x>=min && x<=max)? (true):(false))

#define GETDIS(x1,y1,x2,y2) (sqrt(pow(x1-x2,2)*1.0+pow(y1-y2,2)*1.0))


template <typename scalar_t>
__global__ void kernel_count_get_output(
            scalar_t* __restrict__ descriptor,
            const long4 descriptor_size,
            const long4 descriptor_stride,
            const scalar_t* __restrict__ r_array_q,
            const long4 r_array_q_size,
            const long4 r_array_q_stride,
            const scalar_t* __restrict__ theta_array_q,
            const long4 theta_array_q_size,
            const long4 theta_array_q_stride,
            const scalar_t* __restrict__ sum_points,
            const long4 sum_points_size,
            const long4 sum_points_stride
            )
        {

        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index>=DIM2(r_array_q_size)){
            return;
        }

        int cnt = 0;
        for(cnt=0;cnt<DIM3(r_array_q_size);cnt+=1){
            if(DIM3_INDEX(r_array_q,0,0,index,cnt)>0){
                int cn = (DIM3_INDEX(r_array_q,0,0,index,cnt)-1)*DIM1(descriptor_size)/6+DIM3_INDEX(theta_array_q,0,0,index,cnt)-1;
                int h =  DIM3_INDEX(sum_points,0,0,index,0);
                int w = DIM3_INDEX(sum_points,0,0,index,1);
                DIM3_INDEX(descriptor,0, cn,h,w) += 1.0;
            }

        }

}


void count_get_kernel_forward(
    at::Tensor& descriptor, //(b,60,h,w)
    at::Tensor& r_array_q , //(b,1,m,n)
    at::Tensor& theta_array_q, //(b,1,m,n)
    at::Tensor& sum_points //(b,1,m,2)
    ) {

    const long4 descriptor_size = make_long4(descriptor.size(0), descriptor.size(1), descriptor.size(2), descriptor.size(3));
    const long4 descriptor_stride = make_long4(descriptor.stride(0), descriptor.stride(1), descriptor.stride(2), descriptor.stride(3));

    const long4 r_array_q_size = make_long4(r_array_q.size(0), r_array_q.size(1), r_array_q.size(2), r_array_q.size(3));
    const long4 r_array_q_stride = make_long4(r_array_q.stride(0), r_array_q.stride(1), r_array_q.stride(2), r_array_q.stride(3));

    const long4 theta_array_q_size = make_long4(theta_array_q.size(0),theta_array_q.size(1),theta_array_q.size(2),theta_array_q.size(3));
    const long4 theta_array_q_stride = make_long4(theta_array_q.stride(0),theta_array_q.stride(1),theta_array_q.stride(2),theta_array_q.stride(3));

    const long4 sum_points_size = make_long4(sum_points.size(0),sum_points.size(1),sum_points.size(2),sum_points.size(3));
    const long4 sum_points_stride = make_long4(sum_points.stride(0),sum_points.stride(1),sum_points.stride(2),sum_points.stride(3));


    int Threads = CUDA_NUM_THREADS;

    const dim3 threads(Threads);
    const dim3 blocks(Threads);

    AT_DISPATCH_FLOATING_TYPES(r_array_q.type(), "count_get_forward_kernel", ([&] {
        kernel_count_get_output<scalar_t><<< blocks, threads, 0, at::cuda::getCurrentCUDAStream() >>>(
            descriptor.data<scalar_t>(),
            descriptor_size,
            descriptor_stride,
            r_array_q.data<scalar_t>(),
            r_array_q_size,
            r_array_q_stride,
            theta_array_q.data<scalar_t>(),
            theta_array_q_size,
            theta_array_q_stride,
            sum_points.data<scalar_t>(),
            sum_points_size,
            sum_points_stride
            );
    }));
}
