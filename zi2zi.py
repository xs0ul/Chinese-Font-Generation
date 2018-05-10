import argparse
import os
import numpy as np
import math
import itertools
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('saved_models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="font", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=128, help='size of image width')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=2000, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--generator_type', type=str, default='unet', help="'resnet' or 'unet'")
parser.add_argument('--n_residual_blocks', type=int, default=6, help='number of residual blocks in resnet generator')
parser.add_argument('--log', type=str, default='', help='filename of log file')
parser.add_argument('--train_size', type=int, default=100, help='number of training samples')
parser.add_argument('--augmentation', type=str, default='', help='different methods for data augmentation')
parser.add_argument('--aug_flr', action='store_true', help='flip left-right')
parser.add_argument('--aug_fud', action='store_true', help='flip up-down')
parser.add_argument('--aug_gb', action='store_true', help='gaussian blur')
parser.add_argument('--aug_crop', action='store_true', help='random crop')
parser.add_argument('--aug_rot', action='store_true', help='rotate')
parser.add_argument('--aug_sft', action='store_true', help='shift')
opt = parser.parse_args()
print(opt)

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_translation = torch.nn.L1Loss()

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.img_height / 2**4), int(opt.img_width / 2**4)
patch = (opt.batch_size, 1, patch_h, patch_w)

# Initialize generator and discriminator
generator = GeneratorResNet(in_channels=1, out_channels=1, resblocks=opt.n_residual_blocks) if opt.generator_type == 'resnet' else GeneratorUNet(in_channels=1, out_channels=1)
#generator = GeneratorUNet(in_channels=1, out_channels=1)
discriminator = Discriminator(in_channels=1)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_translation.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/generator_{}.pth').format(opt.epoch))
    discriminator.load_state_dict(torch.load('saved_models/discriminator_{}.pth').format(opt.epoch))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_trans = 100
lambda_const = 15

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.channels, opt.img_height, opt.img_width)
input_B = Tensor(opt.batch_size, opt.channels, opt.img_height, opt.img_width)

# Adversarial ground truths
valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)

# Dataset loader
transforms_ = [transforms.ToTensor()]

TRAIN_SIZE = opt.train_size

source_font_raw = np.fromfile('../data/kai_128.np', dtype=np.int64).reshape(-1, 1, 128, 128).astype(np.float32) * 2. - 1.
target_font_raw = np.fromfile('../data/hwxw_128.np', dtype=np.int64).reshape(-1, 1, 128, 128).astype(np.float32) * 2. - 1.

# shuffle
shuffled_indices = np.random.RandomState(seed=0).permutation(len(source_font_raw))
#shuffled_indices = np.random.permutation(len(source_font_raw), seed=0)
source_font_raw = source_font_raw[shuffled_indices]
target_font_raw = target_font_raw[shuffled_indices]

source_val_sample = torch.FloatTensor(source_font_raw[2000:2005].copy())
target_val_sample = torch.FloatTensor(target_font_raw[2000:2005].copy())

source_font_val = torch.FloatTensor(source_font_raw[2000:3000].copy())
target_font_val = torch.FloatTensor(target_font_raw[2000:3000].copy())

train_indices = np.random.permutation(2000)[:TRAIN_SIZE]
source_font = source_font_raw[train_indices]
target_font = target_font_raw[train_indices]

aug_list = []
if opt.aug_flr:
    aug_list.append('flipleftright')
if opt.aug_fud:
    aug_list.append('flipupdown')
if opt.aug_gb:
    aug_list.append('GaussianBlur')
if opt.aug_crop:
    aug_list.append('crop')
if opt.aug_rot:
    aug_list.append('rotation')
if opt.aug_sft:
    aug_list.append('shift')


dataloader_val = DataLoader(FontDataset(x=source_font_val, y=target_font_val),
                            batch_size=opt.batch_size, shuffle=False,
                            drop_last=True)

image_path_name = '{}_{}_{}'.format(opt.generator_type, opt.train_size, opt.augmentation)
os.makedirs(image_path_name, exist_ok=True)

if not opt.log:
    LOG_NAME = '{}_{}_{}.log'.format(opt.generator_type, opt.train_size, opt.augmentation)
else:
    LOG_NAME = opt.log

# Progress logger
logger = Logger(opt.n_epochs, len(source_font), opt.sample_interval, generator, target_val_sample, source_val_sample, 
         'logs/'+LOG_NAME, image_path_name)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    # process for data augmentation
    if opt.augmentation == '':
        source_font_tmp, target_font_tmp = source_font.copy(), target_font.copy()
    else:
        aug = np.random.choice(aug_list)
        source_font_tmp, target_font_tmp = data_augmentation(aug, source_font.copy(), target_font.copy())
    # assert the length is double or whatever, if no randomness here
    # TODO: get the seed within the function, cuz I need it when apply random operation
    # TODO: add parameters here to control if apply randomness augmentation

    source_font_tmp = torch.FloatTensor(source_font_tmp)
    target_font_tmp = torch.FloatTensor(target_font_tmp)

    dataloader = DataLoader(FontDataset(x=source_font_tmp, y=target_font_tmp),
                            batch_size=opt.batch_size, shuffle=True,
                            drop_last=True)

    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        if opt.generator_type == 'unet':
            fake_A, encoded_real_B = generator(real_B, return_encoded=True)
            _, encoded_fake_A = generator(fake_A, return_encoded=True)
            loss_const = criterion_translation(encoded_fake_A, repackage(encoded_real_B))
        else:
            fake_A = generator(real_B)
        pred_fake = discriminator(fake_A, real_B)
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_trans = criterion_translation(fake_A, real_A)

        # Total loss
        loss_G = loss_GAN + lambda_trans * loss_trans

        if opt.generator_type == 'unet':
            loss_G += lambda_const * loss_const

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_A, real_B)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_A.detach(), real_B)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        logger.log({'loss_G': loss_G, 'loss_G_trans': loss_trans, 'loss_D': loss_D},
                   images={'real_B': real_B,
                           'fake_A': fake_A, 'real_A': real_A},
                   epoch=epoch, batch=i)

    loss_val = eval(generator, dataloader_val, criterion_translation)
    logger.log_val(loss_val)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
        torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)
