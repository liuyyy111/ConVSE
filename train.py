# -*- coding:UTF-8 -*-

"""Training script"""
import os
import time
import shutil

import torch

from torch.nn.utils.clip_grad import clip_grad_norm_
import logging
import argparse
import numpy as np
import random

from data import get_loaders
from evaluation import encode_data, compute_sim, i2t, t2i
from model import VSE, ContrastiveLoss
from vocab import deserialize_vocab


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed(seed) #gpu
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def logging_func(log_file, message):
    with open(log_file,'a') as f:
        f.write(message)
    f.close()


def main():
    setup_seed(1024)
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='D:/data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--grad_clip', default=2.0, type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_epochs', default=35, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--learning_rate', default=.0005, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=25, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--model_name', default='./runs/',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')

    parser.add_argument('--txt_dropout', action='store_true',
                        help='Use text branch contrastive learning')
    parser.add_argument('--txt_dropout_rate', default=.2, type=float,
                        help='text branch dropout rate')

    parser.add_argument('--img_dropout', action='store_true',
                        help='Use image branch contrastive learning')
    parser.add_argument('--img_dropout_rate', default=.2, type=float,
                        help='image branch dropout rate')

    parser.add_argument('--use_contrastive', action='store_true',
                        help='Use contrastive learning')
    opt = parser.parse_known_args()[0]
    print(opt)
    # torch.autograd.set_detect_anomaly(True)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info('train')
    # tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    word2idx = vocab.word2idx
    opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader, val_loader = get_loaders(
        opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = VSE(opt, word2idx)
    model.cuda()

    criterion = ContrastiveLoss(opt)
    # criterion = nn.CosineSimilarity(dim=1).cuda()
    # criterion1 = ContrastiveLoss(opt)
    decay_factor = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=decay_factor)

    best_rsum = 0
    start_epoch = 0
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        print(opt.model_name)
        if not os.path.exists(opt.model_name):
            os.makedirs(opt.model_name)
        message = "epoch: %d, model name: %s\n" % (epoch, opt.model_name)
        log_file = os.path.join(opt.model_name, "performance.log")
        logging_func(log_file, message)

        adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        train(opt, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        rsum = validate(opt, val_loader, model)
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename='checkpoint.pth.tar', prefix=opt.model_name + '/')


class DataPrefetcher():
    def __init__(self, loader, opt):
        self.use_contrastive = opt.use_contrastive

        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()

        self.preload()

    def preload(self):
        try:
            # if not self.use_contrastive:
            self.images, self.img_length, self.captions, self.length, self.index = next(self.loader)
            # else:
            #     self.q_images, self.q_img_length, self.q_captions, self.q_length, self.k_images, self.k_img_length, self.k_captions, self.k_length, self.index = next(
            #         self.loader)
        except StopIteration:
            # if not self.use_contrastive:
            self.images, self.img_length, self.captions, self.length, self.index = None, None, None, None, None
            # else:
            #     self.q_images, self.q_img_length, self.q_captions, self.q_length, self.k_images, self.k_img_length, self.k_captions, self.k_length, self.index = None, None, None, None, None, None, None, None, None
            return
        with torch.cuda.stream(self.stream):
            # if not self.use_contrastive:
            self.images = self.images.cuda()
            self.img_length = self.img_length.cuda()
            self.captions = self.captions.cuda()
            self.length = self.length.cuda()


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        self.preload()
        return self.images, self.img_length, self.captions, self.length, self.index


def train(opt, train_loader, model, criterion, optimizer, epoch):
    # average meters to record the training statistics
    run_time = 0
    start_time = time.time()
    prefetcher = DataPrefetcher(train_loader, opt)
    # if not opt.use_contrastive:
    images, img_lengths, captions, lengths, index = prefetcher.next()
    # else:
    #     images, img_lengths, captions, lengths, k_images, k_img_lengths, k_captions, k_lengths, index = prefetcher.next()
    i = 0
    while images is not None:
        # switch to train mode
        model.train()
        # measure data loading time

        # Update the model
        if not opt.use_contrastive:
            imgs, caps = model(images, captions, lengths, img_lengths=img_lengths)

            loss = criterion(imgs, caps)
            loss.backward()
        else:
            imgs, caps = model(images, captions, lengths, img_lengths=img_lengths)
            # pi1, pi2, zi1, zi2, pc1, pc2, zc1, zc2 = model(images, captions, lengths, img_lengths=img_lengths, k_images=k_images, k_captions=k_captions, k_lengths=k_lengths, k_img_lengths=k_img_lengths)
            loss = criterion(imgs, caps)
            loss.backward()

        if opt.grad_clip > 0:
            clip_grad_norm_(model.parameters(), opt.grad_clip)
            # clip_grad_norm_(amp.master_params(optimizer), opt.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % opt.log_step == 0:
            run_time += time.time() - start_time
            log = "epoch: %d; batch: %d/%d; loss: %.6f; time: %.4f" % (epoch, i, len(train_loader), loss.data.item(), run_time)
            print(log)
            start_time = time.time()
            run_time = 0
        # validate at every val_step
        # if not opt.use_contrastive:
        images, img_lengths, captions, lengths, index = prefetcher.next()
        # else:
        #     images, img_lengths, captions, lengths, k_images, k_img_lengths, k_captions, k_lengths, index = prefetcher.next()
        i += 1

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(
        model, val_loader, opt.log_step, logging.info)
    print(img_embs.shape, cap_embs.shape)

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    start = time.time()
    sims = compute_sim(img_embs, cap_embs)
    end = time.time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, sims)
    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, sims)
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                message = "--------save best model at epoch %d---------\n" % (state["epoch"])
                print(message)
                log_file = os.path.join(prefix, "performance.log")
                logging_func(log_file, message)
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    # lr = opt.learning_rate * 0.5 * (1. + math.cos(math.pi * epoch / opt.num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
