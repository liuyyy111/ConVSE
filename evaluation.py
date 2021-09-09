"""Evaluation"""

from __future__ import print_function
import os

import numpy as np
import torch
from collections import OrderedDict
import time

from data import get_test_loader
from model import VSE
from vocab import deserialize_vocab


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    with torch.no_grad():
        for i, data_i in enumerate(data_loader):
            images, img_lengths, captions, lengths, ids = data_i

            # make sure val logger is used
            model.logger = val_logger
            images = images.cuda()
            captions = captions.cuda()
            lengths = lengths.cuda()
            img_lengths = img_lengths.cuda()
            # compute the embeddings
            img_emb, cap_emb = model.forward_emb(images, captions, lengths, img_lengths=img_lengths)
            # img_emb, cap_emb, cap_len = model.forward_emb(images, captions, pos, lengths)
            # print(img_emb)
            if img_embs is None:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

            # cache embeddings
            img_embs[ids] = img_emb.data.cpu().numpy().copy()
            cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                logging('Test: [{0}/{1}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                    i, len(data_loader), batch_time=batch_time,
                    e_log=str(model.logger)))
            del images, captions
    return img_embs, cap_embs


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print(opt)
    if data_path is not None:
        opt.data_path = data_path


    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    word2idx = vocab.word2idx
    opt.vocab_size = len(vocab)

    model = VSE(opt, word2idx)
    model.cuda()
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, 0, opt)

    print('Computing results...')
    img_embs, cap_embs,= encode_data(model, data_loader)

    # np.save("img_embs", img_embs)
    # np.save("cap_embs", cap_embs)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        start = time.time()

        sims = compute_sim(img_embs, cap_embs)

        # np.save('f30k_sims_single', sims)
        end = time.time()
        print("calculate similarity time:", end - start)

        r, rt = i2t(img_embs, cap_embs, sims, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            start = time.time()
            sims = compute_sim(img_embs_shard, cap_embs_shard)

            end = time.time()
            print("calculate similarity time:", end - start)

            r, rt0 = i2t(img_embs_shard, cap_embs_shard, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs_shard, cap_embs_shard, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[12]))
        print("Average i2t Recall: %.1f" % mean_metrics[10])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[11])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def ensemble_evalrank(model_path1, model_path2, data_path=None, split='dev', fold5=False):
    checkpoint1 = torch.load(model_path1)
    opt1 = checkpoint1['opt']
    print(opt1)

    checkpoint2 = torch.load(model_path2)
    opt2 = checkpoint2['opt']
    print(opt2)

    if data_path is not None:
        opt1.data_path = data_path
        opt2.data_path = data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt1.vocab_path, '%s_vocab.json' % opt1.data_name))
    word2idx = vocab.word2idx
    # opt2.word2idx = vocab.word2idx

    opt1.vocab_size = len(vocab)
    opt2.vocab_size = len(vocab)
    opt1.txt_enc_type='rnn'
    opt2.txt_enc_type = 'rnn'

    model1 = VSE(opt1, word2idx)
    model1.cuda()
    model1.load_state_dict(checkpoint1['model'])

    model2 = VSE(opt2, word2idx)
    model2.cuda()
    model2.load_state_dict(checkpoint2['model'])

    print('Loading dataset')
    # data_loader = get_test_loader(split, opt1.data_name, vocab, None,
    #                               opt1.batch_size, 0, opt1)
    data_loader = get_test_loader(split, opt1.data_name, vocab,
                                  opt1.batch_size, 0, opt1)
    print('Computing results...')
    img_embs_1, cap_embs_1,= encode_data(model1, data_loader)
    img_embs_2, cap_embs_2, = encode_data(model2, data_loader)

    if not fold5:
        # no cross-validation, full evaluation
        img_embs_1 = np.array([img_embs_1[i] for i in range(0, len(img_embs_1), 5)])
        img_embs_2 = np.array([img_embs_2[i] for i in range(0, len(img_embs_2), 5)])
        start = time.time()

        sims1 = compute_sim(img_embs_1, cap_embs_1)
        sims2 = compute_sim(img_embs_2, cap_embs_2)

        sims = (sims1 + sims2)/2
        np.save('ConVSE_f30k_test', sims)
        end = time.time()
        print("calculate similarity time:", end - start)

        r, rt = i2t(img_embs_1, cap_embs_1, sims, return_ranks=True)
        ri, rti = t2i(img_embs_1, cap_embs_1, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard_1 = img_embs_1[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard_1 = cap_embs_1[i * 5000:(i + 1) * 5000]

            img_embs_shard_2 = img_embs_2[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard_2 = cap_embs_2[i * 5000:(i + 1) * 5000]
            start = time.time()

            sims1 = compute_sim(img_embs_shard_1, cap_embs_shard_1)
            sims2 = compute_sim(img_embs_shard_2, cap_embs_shard_2)

            sims = (sims1 + sims2) / 2
            end = time.time()
            print("calculate similarity time:", end - start)

            r, rt0 = i2t(img_embs_shard_1, cap_embs_shard_1, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs_shard_1, cap_embs_shard_1, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[12]))
        print("Average i2t Recall: %.1f" % mean_metrics[10])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[11])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities


def i2t(images, captions, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            # print(inds, i, index, npts)
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
