# coding:utf8
import os  # ,ipdb
import torch as t
from torch.autograd import Variable
import torchvision as tv
# from torchnet import meter
import tqdm
import time
from torch.nn.utils.rnn import pack_padded_sequence
from .model import Encoder, DecoderWithAttention
# from config import Config
# from utils import Visualizer
# from data import get_dataloader
from PIL import Image
# import ipdb

# Model parameters
epochs = 120
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
grad_clip = 5.  # clip gradients at an absolute value of
fine_tune_encoder = False
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
print_freq = 100  # print training/validation stats every __ batches
device = t.device("cuda" if t.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


# device = t.device("cpu")  # sets device for model and PyTorch tensors
# t.cuda.set_device(3)


# cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
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


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    ):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             #  'bleu-4': bleu4,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = data_name + '.pth.tar'
    t.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    # if is_best:
    #     t.save(state, 'BEST_' + filename)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train_one_epoch(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    # ipdb.set_trace()
    # Batches
    for i, (imgs, (caps, caplens), indexes) in tqdm.tqdm(enumerate(train_loader)):
        # for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caps = caps.t()
        caplens = t.tensor(caplens).to(device)
        caplens = caplens.reshape([caplens.shape[0], 1])

        ipdb.set_trace()

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch %d: loss = %.4f, losses avg = %.4f, top-5 acc = %.4f' % (
                epoch, loss.cpu(), losses.avg, top5accs.avg))


def train():
    opt = Config()
    opt.caption_data_path = 'data/caption.pth'  # 原始数据
    opt.test_img = ''  # 输入图片
    start_epoch = 0
    # opt.model_ckpt='checkpoints/caption_1119_1903.pth.tar' # 预训练的模型
    # opt.model_ckpt=None # 预训练的模型
    dataloader = get_dataloader(opt)
    _data = dataloader.dataset._data
    word2ix, ix2word = _data['word2ix'], _data['ix2word']

    # Initialize / load checkpoint
    if opt.model_ckpt is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word2ix),
                                       dropout=dropout)
        decoder_optimizer = t.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = t.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr) if fine_tune_encoder else None
    else:
        checkpoint = t.load(opt.model_ckpt)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        # best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = t.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = t.nn.CrossEntropyLoss().to(device)
    ipdb.set_trace()
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU

    for epoch in range(start_epoch, epochs):
        # loss_meter.reset()
        # 数据
        # vis = Visualizer(env = opt.env)
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
        print("Epoch is:%d" % epoch)

        train_one_epoch(train_loader=dataloader,
                        encoder=encoder,
                        decoder=decoder,
                        criterion=criterion,
                        encoder_optimizer=encoder_optimizer,
                        decoder_optimizer=decoder_optimizer,
                        epoch=epoch)
        # Save checkpoint
        # ipdb.set_trace()
        data_name = '{prefix}_{time}'.format(prefix=opt.prefix,
                                             time=time.strftime('%m%d_%H%M'))
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer)


def extract_weights():
    device = t.device("cpu")
    opt = Config()
    opt.model_ckpt = 'checkpoints/caption_1119_1903.pth.tar'  # 预训练的模型
    checkpoint = t.load(opt.model_ckpt, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']
    encoder = checkpoint['encoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    epoch = checkpoint['epoch']
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    # state =
    filename = 'ch_weights' + '.pth.tar'
    t.save({'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}, filename)


if __name__ == "__main__":
    # train()
    extract_weights()
