import random

import nltk
import torch
import torch.utils.data as data
import os
import numpy as np
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, opt):
        print('word txt encoder')
        self.vocab = vocab
        self.use_contrastive = opt.use_contrastive
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc + '%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip().decode('utf-8'))

        self.data_split = data_split

        # Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # self.length = 10000
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index / self.im_div)
        image = self.images[img_id]
        caption = self.captions[index]
        vocab = self.vocab
        train = self.data_split == 'train'
        # train = False
        # if not train or not self.use_contrastive:
        target = process_caption(vocab, caption, train)
            # new_tags = torch.Tensor(new_tags)
        image = process_image1(image, train)

        return image, target, index, img_id

    def __len__(self):
        return self.length


def process_image1(image, train=False):
    if train:
        num_features = image.shape[0]
        rand_list = np.random.rand(num_features)
        image = image[np.where(rand_list > 0.20)]
        np.random.shuffle(image)

    image = torch.Tensor(image)

    return image


def process_caption(vocab, caption, drop=False):
    if not drop:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = list()
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return target
    else:
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        deleted_idx = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.20:
                prob /= 0.20
                # 50% randomly change token to mask token
                if prob < 0.5:
                    tokens[i] = vocab.word2idx['<mask>']
                # 10% randomly change token to random token
                elif prob < 0.6:
                    tokens[i] = random.randrange(len(vocab))
                # 40% randomly remove the token
                else:
                    tokens[i] = vocab(token)
                    deleted_idx.append(i)
            else:
                tokens[i] = vocab(token)
        if len(deleted_idx) != 0:
            tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]
        tokens = [vocab('<start>')] + tokens + [vocab('<end>')]
        target = torch.Tensor(tokens)
        return target


class Collate:
    def __init__(self, opt, train):
        self.train = train

    def __call__(self, data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions, ids, img_ids = zip(*data)

        # Merge images (convert tuple of 3D tensor to 4D tensor)
        img_lengths = torch.LongTensor([len(img) for img in images])
        all_images = torch.zeros([len(images), max(img_lengths), images[0].size(-1)])
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = torch.LongTensor([len(cap) for cap in captions])
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return all_images, img_lengths, targets, lengths, ids


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    train = data_split == 'train'
    collate_fn = Collate(opt, train)
    dset = PrecompDataset(data_path, data_split, vocab, opt)
    data_loader = DataLoaderX(dataset=dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn, drop_last=train)

    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'test', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
