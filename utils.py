import os
import sys

import torch
import errno
import json
import shutil
import numpy as np
import os.path as osp

import torch.nn.functional as F


def patch_loss_global(ytest, pids_test_patch, pids_train):
    criterion = torch.nn.CrossEntropyLoss()
    ytest = ytest.view(ytest.size()[0], ytest.size()[1], -1).transpose(1, 2).reshape(-1,
                                                                                     ytest.size()[1])
    pids_test_patch = pids_test_patch.view(-1)
    pids_train_patch = pids_train.unsqueeze(2).unsqueeze(2).repeat(1, 1, 11, 11).view(-1)
    loss = criterion(ytest, torch.cat([pids_train_patch, pids_test_patch.view(-1)]))
    return loss


def patch_loss_local(labels_test_patch, cls_scores):
    labels_test_patch = labels_test_patch.view(-1)
    cls_scores = cls_scores.view(cls_scores.size()[0], cls_scores.size()[1], -1).transpose(1, 2).reshape(-1,
                                                                                                         cls_scores.size()[
                                                                                                             1])
    cls_scores = SoftSort()(cls_scores)[:, :3, :].sum(1)
    labels_test_patch_1hot = (one_hot(labels_test_patch).unsqueeze(1).repeat(1, 4, 1).view(labels_test_patch.shape[0],
                                                                                           -1) / 4.).cuda()  # label smoothing
    loss = cross_entropy(cls_scores, labels_test_patch_1hot)
    return loss


def cross_entropy(logits, lables_1hot):
    log_logits = F.log_softmax(logits, dim=-1)
    loss = - (lables_1hot * log_logits).sum(1).mean()

    return loss


def one_hot_pids(labels_train):
    """
    Turn the pids_labels_train to one-hot encoding.
    """
    labels_train = labels_train.cpu()
    nKnovel = 64
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel, ]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1,
                                                                     labels_train_unsqueeze, 1)
    return labels_train_1hot


def generate_masks(similarity_matrix):
    h1, w1 = [9, 9]
    kernel_weights = torch.ones([1, 1, h1, w1], requires_grad=False).cuda()
    stride = 1
    padding = 0
    local_similarity = F.conv2d(similarity_matrix.view(-1, 1, 11, 11), kernel_weights, bias=None, padding=padding, stride=stride)
    h2, w2 = [int(((11 - h1) + 2 * padding) / stride + 1), int(((11 - h1) + 2 * padding) / stride + 1)]
    local_similarity_scores = local_similarity.view(-1, h2 * w2)
    _, local_similarity_scores_rank = torch.topk(local_similarity_scores, k=4, dim=-1, sorted=True)
    masks = []
    for i in range(local_similarity_scores_rank.shape[0]):
        local_similarity_scores_rank_slice = local_similarity_scores_rank[i]
        for j in local_similarity_scores_rank_slice:
            mask = torch.zeros(121)
            if int(j / h2) == 0:
                x = 0
            else:
                x = int(j / h2) * 11 + (stride - 1) * 11
            y = (j % h2) * stride
            sp = x + y
            mask[sp] = 1
            mask = mask.view(11, 11)
            index = torch.where(mask == 1)
            mask[index[0]:index[0] + h1, index[1]:index[1] + w1] = 1
            masks.append(mask)
    masks = torch.stack(masks, dim=0).view(*local_similarity_scores_rank.size(), 11, 11).cuda()
    return masks


def random_block_distangle(x):
    identity = x
    # ablation study
    zeros = torch.ones(9, 9)
    length = zeros.size()[0]
    width = zeros.size()[1]
    masks = torch.zeros(4, 11, 11)  # hyperparameter numbers
    h = [0, 0, 2, 2]
    w = [0, 2, 0, 2]

    for i in range(masks.size()[0]):
        for j in range(length):
            for k in range(width):
                masks[i, h[i] + j, w[i] + k] = zeros[j, k]
    masks = masks.float().cuda()
    x = x.unsqueeze(2) * masks.unsqueeze(0).unsqueeze(0).unsqueeze(3)
    return x


def generate_matrix():
    xd = np.random.randint(1, 2)
    yd = np.random.randint(1, 2)
    index = list(range(11))
    x0 = np.random.choice(index, size=xd, replace=False)
    y0 = np.random.choice(index, size=yd, replace=False)
    return x0, y0


def random_block(x):
    x0, y0 = generate_matrix()
    mask = torch.zeros([1, 1, 11, 11], requires_grad=False) + 1
    for i in x0:
        for j in y0:
            mask[:, :, i, j] = 0
    mask = mask.float()
    x = x * mask.cuda()
    return x


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def patchmix(test_img, test_label, global_label):
    """
    img: batch_size, num_qry
    label: batch_size, num_qry
    """
    test_label_patch = test_label.unsqueeze(2).unsqueeze(2).repeat(1, 1, 11, 11)
    global_label_patch = global_label.unsqueeze(2).unsqueeze(2).repeat(1, 1, 11, 11)

    batch_size = test_img.size()[0]
    for i in range(batch_size):
        test_label_patch_slice = test_label_patch[i]
        global_label_patch_slice = global_label_patch[i]
        input = test_img[i]  # (num, 3, 84, 84)
        lam = np.random.beta(1, 1)
        rand_index = torch.randperm(input.size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2,
                                            bby1:bby2]
        test_img[i] = input

        #### calculate patch label
        bbx1, bby1, bbx2, bby2 = float(bbx1), float(bby1), float(bbx2), float(bby2)
        bbx1, bby1, bbx2, bby2 = round(bbx1 * 11.0 / 84.0), round(bby1 * 11.0 / 84.0), round(bbx2 * 11.0 / 84.0), round(
            bby2 * 11.0 / 84.0)
        test_label_patch_slice[:, bbx1:bbx2, bby1:bby2] = test_label_patch_slice[rand_index, bbx1:bbx2, bby1:bby2]
        global_label_patch_slice[:, bbx1:bbx2, bby1:bby2] = global_label_patch_slice[rand_index, bbx1:bbx2, bby1:bby2]
        ### #############
        test_label_patch[i] = test_label_patch_slice
        global_label_patch[i] = global_label_patch_slice

    return test_img, test_label_patch, global_label_patch


class SoftSort(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False, pow=1.0):
        super(SoftSort, self).__init__()
        self.hard = hard
        self.tau = tau
        self.pow = pow

    def forward(self, scores):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(self.pow).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat


def get_mask(input, coordinate):
    """
    input: num, c, h, w
    """
    bbx1, bby1, bbx2, bby2 = coordinate
    shape = input.shape
    mask = torch.zeros(shape).cuda()
    for i in range(bbx1, bbx2):
        for j in range(bby1, bby2):
            mask[:, :, i, j] = 1

    return mask


#### from cross attention
def adjust_learning_rate(optimizer, iters, LUT):
    # decay learning rate by 'gamma' for every 'stepsize'
    for (stepvalue, base_lr) in LUT:
        if iters < stepvalue:
            lr = base_lr
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best=False, fpath='checkpoint.pth.tar'):
    if len(osp.dirname(fpath)) != 0:
        mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def one_hot(labels_train):
    """
    Turn the labels_train to one-hot encoding.
    Args:
        labels_train: [batch_size, num_train_examples]
    Return:
        labels_train_1hot: [batch_size, num_train_examples, K]
    """
    labels_train = labels_train.cpu()
    nKnovel = 1 + labels_train.max()
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel, ]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1,
                                                                     labels_train_unsqueeze, 1)
    return labels_train_1hot


def load_model(model, dir):
    model_dict = model.state_dict()
    # pretrained_dict = torch.load(dir)['params']
    pretrained_dict = torch.load(dir)['state_dict']

    if pretrained_dict.keys() == model_dict.keys():  # load from a parallel meta-trained model and all keys match
        print('all state_dict keys match, loading model from :', dir)
        model.load_state_dict(pretrained_dict)
    else:
        print('loading model from :', dir)
        if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
            if 'module' in list(pretrained_dict.keys())[0]:
                pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {'base.' + k: v for k, v in pretrained_dict.items()}  # load from a pretrained model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
        model.load_state_dict(model_dict)

    return model

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None, mode='a'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
