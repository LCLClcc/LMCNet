import datetime
import random
import time

import torch.backends.cudnn as cudnn
import torch.nn
from tqdm import tqdm

from args_mini import argument_parser
from full_model import Model
from mini_dataloader import DataManager
from utils import *

parser = argument_parser()
args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    print("==========\nArgs:{}\n==========".format(args))
    mkdir_if_missing(args.save_dir)

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()
    model = Model(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                nesterov=True)
    if use_gpu:
        model = model.cuda()
    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0
    sys.stdout = Logger(os.path.join(args.resume, 'log_train_mini_5shot.txt'))
    print("==> Start training")
    for epoch in range(args.start_epoch, args.max_epoch):
        learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)
        start_train_time = time.time()
        train(epoch, model, optimizer, trainloader, learning_rate, use_gpu)
        train_time += round(time.time() - start_train_time)
        if epoch == 0 or epoch > (args.stepsize[0] - 1) or (epoch + 1) % 10 == 0:
            acc = test(model, testloader, use_gpu)
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                best_epoch = epoch + 1
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch),
                           os.path.join(args.save_dir, 'max_acc.pth'))
                torch.save(optimizer.state_dict(), os.path.join(args.save_dir, 'optimizer_max_acc.pth'))

            print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))


def train(epoch, model, optimizer, trainloader, learning_rate, use_gpu):
    model.train()

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.train()
    end = time.time()
    tqdm_gen = tqdm(trainloader)
    for batch_idx, (images_train, labels_train, images_test, labels_test, pids_train, pids_test) in enumerate(tqdm_gen):
        data_time.update(time.time() - end)
        if use_gpu:
            images_train, labels_train = images_train.cuda(), labels_train.cuda()
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
            pids_train, pids_test = pids_train.cuda(), pids_test.cuda()
        batch_size, num_train_examples, channels, height, width = images_train.size()
        num_test_examples = images_test.size(1)
        pids = torch.cat([pids_train.view(-1), pids_test.view(-1)], 0).view(-1)
        labels_train_1hot = one_hot(labels_train).cuda()
        labels_test_1hot = one_hot(labels_test).cuda()
        images_test, labels_test_patch, pids_test_patch = patchmix(images_test, labels_test, pids_test)
        pids_train_1hot = one_hot_pids(pids_train).cuda()

        ytest, cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot, pids_train_1hot)
        loss1 = patch_loss_global(ytest, pids_test_patch, pids_train)
        loss2 = patch_loss_local(labels_test_patch, cls_scores)
        loss = loss1 + 0.5 * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), pids_test.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'lr: {1} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '.format(
        epoch + 1, learning_rate, batch_time=batch_time,
        data_time=data_time, loss=losses))


def test(model, testloader, use_gpu):
    model.eval()

    accs = AverageMeter()
    test_accuracies = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (images_train, labels_train, images_test, labels_test) in enumerate(tqdm(testloader)):
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()
            end = time.time()
            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)
            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()

            cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot, None)
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)
            _, preds = torch.max(cls_scores.detach().cpu(), 1)
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))
            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy()
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)
    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(accuracy, ci95))
    return accuracy


if __name__ == '__main__':
    main()
