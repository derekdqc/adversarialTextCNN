import torch
import torch.nn.functional as F

from torch.autograd import Variable


# class FreeAT():
#     def __init__(self, model):
#         self.model = model
#         self.backup = {}
#
#     def fgsm(self, gradz, step_size):
#         return step_size * torch.sign(gradz)
#
#     def attack(self, global_noise_data, emb_name='emb.'):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and emb_name in name:
#                 print('param before: ', param)
#                 self.backup[name] = param.data.clone()
#                 noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
#                 param.data = param.data + noise_batch
#                 param.data.clamp_(0, 1.0)
#
#
#         output = model(in1)
#         loss = criterion(output, target)
#
#         prec1, prec5 = accuracy(output, target, topk=(1, 5))
#         losses.update(loss.item(), input.size(0))
#         top1.update(prec1[0], input.size(0))
#         top5.update(prec5[0], input.size(0))
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#
#         # Update the noise for the next iteration
#         pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
#         global_noise_data[0:input.size(0)] += pert.data
#         global_noise_data.clamp_(-configs.ADV.clip_eps, configs.ADV.clip_eps)
#
#         optimizer.step()


class FreeAT():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # print('param before: ', param)
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
