from models.counter.count_get import GetCount

import cv2
import math
import torch
import numpy as np

class ShapeContext(object):
    def __init__(self, nbins_r=6, nbins_theta=12, r_inner=0.1250, r_outer=2.0):
        self.nbins_r = nbins_r
        self.nbins_theta = nbins_theta
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.gc = GetCount()

    def mycompute(self, contours, points, ans):
        contours = contours.cuda()
        points = points.cuda()

        xgap = points[:, :, 0].reshape(1, -1, 1) - contours[:, :, 0].reshape(1, 1, -1)
        ygap = points[:, :, 1].reshape(1, -1, 1) - contours[:, :, 1].reshape(1, 1, -1)

        r_array = torch.sqrt(xgap**2+ygap**2)

        (b, m, n) = r_array.shape

        r_array_n = r_array / (torch.max(r_array)/2)

        r_bin_edges = np.logspace(np.log10(self.r_inner), np.log10(self.r_outer), self.nbins_r)
        r_array_q = torch.zeros((b, m, n)).cuda()

        for ct in range(self.nbins_r):
            r_array_q += (r_array_n <= r_bin_edges[ct]).float()

        ygap = -ygap
        theta_array = torch.atan2(xgap,ygap)

        theta_array_2 = theta_array + 2 * math.pi * (theta_array < 0).float()
        theta_array_q = (1 + torch.floor(theta_array_2 / (2 * math.pi / self.nbins_theta)))

        self.gc(ans,r_array_q.reshape(1,1,m,n),theta_array_q.reshape(1,1,m,n),points.reshape(1,1,m,2))

    def spd(self, instance):
        bsz,_,dh,dw = instance.shape
        torch.cuda.set_device(torch.device('cuda:0'))
        discriptor = torch.zeros((bsz,self.nbins_theta*self.nbins_r,dh,dw),dtype=torch.float32).cuda()

        ins = np.array(instance)
        
        ins = ins[:,0,:,:]

        (b,h,w) = ins.shape

        for b_num in range(b):

            idxes = np.unique(ins[b_num])

            for idx in idxes:
                if idx == 0:
                    continue

                tmp = np.where(ins[b_num, :, :] == idx, 255, 0)
                tmp = np.array(tmp, dtype=np.uint8)

                contours = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                real_cont = np.zeros_like(tmp)

                for cntr in contours:
                    real_cont = cv2.drawContours(real_cont, [cntr], 0, 255, 1)

                counters = np.where(real_cont == 255)
                counters = np.vstack((counters[0], counters[1])).T

                mask = real_cont == 255
                tmp[mask] = 0
                highlights = np.where(tmp == 255)
                highlights = np.vstack((highlights[0], highlights[1])).T

                sump = np.vstack((counters, highlights))

                sump = np.array(sump, dtype=np.float32)
                counters = np.array(counters, dtype=np.float32)
                counters = torch.tensor(counters.reshape((1, -1, 2)))
                sump = torch.tensor(sump.reshape((1, -1, 2)))

                self.mycompute(counters, sump, discriptor[b_num:b_num + 1])
        
        sum_for_norm = torch.sum(discriptor, dim = 1).unsqueeze(1)
        sum_for_norm[sum_for_norm == 0] = 1
        discriptor = discriptor / sum_for_norm
        return discriptor