import torch
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        # total0 = total.unsqueeze(0).expand(int(total.size(0)), \
        #                                 int(total.size(0)), \
        #                                 int(total.size(1)))
        # total1 = total.unsqueeze(1).expand(int(total.size(0)), \
        #                                 int(total.size(0)), \
        #                                 int(total.size(1)))
        L2_distance = torch.cdist(total, total, p=2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                    bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

def mmd_loss(source, target):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target)
    # kernels = self.imq_kernel(source, target)
    XX = kernels[:n, :n] 
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss