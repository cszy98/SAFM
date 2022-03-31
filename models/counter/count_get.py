from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import count_get_cuda
import numpy as np
import torch

class GetCountFunction(Function):

    @staticmethod
    def forward(ctx, descriptor,r_array_q,theta_array_q,sum_points):
        assert descriptor.is_contiguous()
        assert r_array_q.is_contiguous()

        ctx.save_for_backward(descriptor)
        # dismap = torch.from_numpy(np.zeros((6,8,256,256)))

        # dismap = label.new_zeros((6, 8,256,256))
        #print(dismap.dtype)
        #print(dismap.device)
        #print(label.dtype)
        #print(label.device)

        if not descriptor.is_cuda:
            raise NotImplementedError
        else:

            count_get_cuda.forward(descriptor,r_array_q,theta_array_q,sum_points)
        return descriptor

    @staticmethod
    def backward(ctx, grad_dismap):
        # if not grad_dismap.is_contiguous():
        #     grad_dismap.contiguous()
        # label, = ctx.saved_tensors
        # grad_label = Variable(label.new(label.size()).one_())

        # local_attn_reshape_cuda.backward(inputs, grad_output.data,
        #                          grad_inputs.data, ctx.kernel_size)

        return None, None

class GetCount(Module):
    """
    将(bs,ks*ks,h,w)的tensor变成(bs,1,ks*w,ks*w)的tensor,从而可以与extractor的结果直接相乘
    """
    def __init__(self):
        super(GetCount, self).__init__()

    def forward(self, descriptor,r_array_q,theta_array_q,sum_points):
        descriptor = descriptor.contiguous()
        r_array_q = r_array_q.contiguous()
        theta_array_q = theta_array_q.contiguous()
        sum_points = sum_points.contiguous()
        # print(descriptor.shape)
        # print(r_array_q.shape)
        # print(theta_array_q.shape)
        # print(sum_points.shape)
        return GetCountFunction.apply(descriptor,r_array_q,theta_array_q,sum_points)
