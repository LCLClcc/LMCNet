#### This file is based on code of cross attention network
from resnet import resnet12
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.scale_cls = self.args.scale_cls
        self.base = resnet12()
        self.nFeat = self.base.nFeat
        self.num_classes = self.args.num_classes
        self.clasifier = nn.Conv2d(self.nFeat, self.num_classes, kernel_size=1, bias=False)

    def test(self, ftrain, ftest):
        ftest = ftest.mean(4)
        ftest = ftest.mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim() - 1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim() - 1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def reform(self, f1, f2):
        """
        f: batch, num, c, h, w
        """
        b, n1, c, h, w = f1.size()
        n2 = f2.size(1)
        f1 = f1.view(b, n1, c, -1)
        f2 = f2.view(b, n2, c, -1)
        f1 = f1.unsqueeze(2)
        f1 = f1.repeat(1, 1, n2, 1, 1)
        f1 = f1.view(b, n1, n2, c, h, w)
        f2 = f2.unsqueeze(1)
        f2 = f2.repeat(1, 1, n1, 1, 1)
        f2 = f2.view(b, n1, n2, c, h, w)
        return f1.transpose(1, 2), f2.transpose(1, 2)

    def forward(self, xtrain, xtest, ytrain, ytest, ptrain):
        """
        xtest: after patchmix
        """
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)
        xtrain = xtrain.contiguous().view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))

        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)
        f = torch.relu(f).pow(0.7)
        global_cls = self.clasifier(f)

        # ProtoType
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1)
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        num_classes = ftrain.size(1)

        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])

        if self.training:
            h, w = ftrain.shape[3], ftrain.shape[4]
            protos = ftrain.view(batch_size * num_classes, ftrain.shape[2], -1)
            protos_l = protos.transpose(1, 2)

            # global as true prototype
            gprotos = self.clasifier.weight.data.clone().detach().squeeze(-1).squeeze(-1)
            ptrain = ptrain.view(-1, ptrain.shape[2])
            protos_g = torch.mm(ptrain, gprotos)
            protos_g = torch.bmm(ytrain, protos_g.view(batch_size, num_train, -1))
            protos_g = protos_g.div(ytrain.sum(dim=2, keepdim=True).expand_as(protos_g)).view(-1, protos_g.size()[-1]).unsqueeze(1)
            proto_local_logits_slice = F.cosine_similarity(protos_l, protos_g, dim=-1).view(batch_size, num_classes, 1, h, w)
            masks = generate_masks(proto_local_logits_slice)
            masks = masks.view(batch_size, num_classes, *masks.size()[1:])
            ftrain = ftrain.unsqueeze(2) * masks.unsqueeze(3)
            ftrain = ftrain.transpose(1, 2).contiguous().view(batch_size, -1, *ftrain.size()[3:])

        ftrain, ftest = self.reform(ftrain, ftest)
        ftrain = ftrain.mean(dim=[-1, -2])
        if not self.training:
            return self.test(ftrain, ftest)
        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)  # (b, n2, n1, c, 1, 1)

        similarity_matrix = torch.sum(ftest_norm * ftrain_norm, dim=3)
        cls_scores = self.scale_cls * similarity_matrix
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])

        return global_cls, cls_scores

