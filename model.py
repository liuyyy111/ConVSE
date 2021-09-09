# coding=utf-8
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import l2norm, init_weight


def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)

        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features, pool_weights

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)

        return x


class ImageEncoder(nn.Module):

    def __init__(self, opt):
        super(ImageEncoder, self).__init__()
        self.img_drop = opt.img_dropout
        self.no_imgnorm = opt.no_imgnorm
        self.fc = nn.Linear(opt.img_dim, opt.embed_size)
        self.fc.apply(init_weight)
        self.dropout = nn.Dropout(opt.img_dropout_rate)
        self.mlp = MLP(opt.img_dim, opt.embed_size // 2, opt.embed_size, 2)
        self.gpool = GPO(32, 32)
        # self.use_bn = opt.data_name == 'coco_precomp'
        # if self.use_bn:
        #         # self.bn = nn.BatchNorm1d(opt.embed_size)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        if self.img_drop:
            images = self.dropout(images)
        features = self.fc(images)
        features = self.mlp(images) + features
        # normalize in the joint embedding space
        features, pool_weights = self.gpool(features, image_lengths)
        # if self.use_bn:
        # features = self.bn(features)
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features


class TextEncoder(nn.Module):
    def __init__(self, opt, word2idx):
        super(TextEncoder, self).__init__()
        self.txt_drop = opt.txt_dropout
        self.data_path = opt.data_path
        # word embedding
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.dropout = nn.Dropout(opt.txt_dropout_rate)
        self.rnn = nn.GRU(opt.word_dim, opt.embed_size, batch_first=True, bidirectional=True)
        self.init_weights(word2idx)
        self.gpool = GPO(32, 32)

    def init_weights(self, word2idx):
        # self.embed.weight.data.uniform_(-0.1, 0.1)
        path = os.path.join(self.data_path, 'vector_cache')
        print(path)
        wemb = torchtext.vocab.GloVe(cache=path)

        # quick-and-dirty trick to improve word-hit rate
        missing_words = []
        for word, idx in word2idx.items():
            if word not in wemb.stoi:
                word = word.replace('-', '').replace('.', '').replace("'", '')
                if '/' in word:
                    word = word.split('/')[0]
            if word in wemb.stoi:
                self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
            else:
                missing_words.append(word)
        print('Words: {}/{} found in vocabulary; {} words missing'.format(
            len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        if self.txt_drop:
            x = self.dropout(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Forward propagate RNN
        out, _ = self.rnn(packed)
        # print(lengths)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2
        # normalization in the joint embedding space
        cap_emb, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))
        cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb


def get_img_encoder(opt):
    return ImageEncoder(opt)


def get_txt_encoder(opt, word2idx):
    return TextEncoder(opt, word2idx)


class VSE(nn.Module):
    def __init__(self, opt, word2idx):
        super(VSE, self).__init__()
        self.use_contrastive = opt.use_contrastive
        self.img_enc_q = get_img_encoder(opt)
        self.txt_enc_q = get_txt_encoder(opt, word2idx)
        use_mlp = True

    def forward_emb(self, images, captions, lengths, img_lengths=None, k_images=None, k_captions=None, k_lengths=None,
                    k_img_lengths=None):
        if not self.training or not self.use_contrastive:

            img_emb = self.img_enc_q(images, img_lengths)
            cap_emb = self.txt_enc_q(captions, lengths)

            return img_emb, cap_emb
        else:
            img_view1 = self.img_enc_q(images, img_lengths)
            img_view2 = self.img_enc_q(images, img_lengths)

            cap_view1 = self.txt_enc_q(captions, lengths)
            cap_view2 = self.txt_enc_q(captions, lengths)

            img_embs = torch.stack([img_view1, img_view2], dim=1)
            cap_embs = torch.stack([cap_view1, cap_view2], dim=1)

            return img_embs, cap_embs

    def forward(self, images, captions, lengths, img_lengths=None):

        return self.forward_emb(images, captions, lengths, img_lengths=img_lengths)


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt):
        super(ContrastiveLoss, self).__init__()
        self.margin = opt.margin
        self.use_contrastive = opt.use_contrastive
        self.criterion = nn.CosineSimilarity(dim=1).cuda()

    def _compute_triplet_loss(self, img, cap, margin=0.2):
        scores = img.mm(cap.t())
        diagonal = scores.diag().view(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

    def _compute_contrastive_loss(self, view1, view2, temperature=0.1):

        sims = view1.mm(view2.t())
        mask = torch.eye(sims.size(0)).bool().cuda()

        l_pos = sims[mask].unsqueeze(-1)
        l_neg = sims[~mask].view(sims.shape[0], -1)

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss

    def forward(self, img, cap):
        # compute image-sentence score matrix
        if self.use_contrastive:

            img_view1 = img[:, 0, :]
            img_view2 = img[:, 1, :]
            cap_view1 = cap[:, 0, :]
            cap_view2 = cap[:, 1, :]
            loss1 = self._compute_triplet_loss(img_view1, cap_view1)
            loss2 = self._compute_triplet_loss(img_view1, img_view2, 0.3)
            loss3 = self._compute_triplet_loss(cap_view1, cap_view2, 0.3)

            loss = (loss1 + loss2 + loss3)

            return loss
        else:
            loss1 = self._compute_triplet_loss(img, cap)
            return loss1