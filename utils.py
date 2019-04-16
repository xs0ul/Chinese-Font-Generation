import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image
from scipy import ndimage


def tensor2image(tensor):
    image = 255*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)


class Logger():
    def __init__(self, n_epochs, batches_epoch, sample_interval, generator, real_A, real_B, out_file, 
                 image_path_name, n_samples=5):
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.sample_interval = sample_interval
        self.batches_done = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.n_samples = n_samples
        self.past_images = []

        self.out_file = out_file
        self.generator = generator
        self.image_path_name = image_path_name
        self.real_A = real_A
        self.real_B = real_B

    def log(self, losses=None, images=None, epoch=0, batch=0):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        epoch += 1
        batch += 1

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (epoch, self.n_epochs, batch, self.batches_epoch))
        f = open(self.out_file, 'a')
        f.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (epoch, self.n_epochs, batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = 0

            self.losses[loss_name] += losses[loss_name].data[0]

            sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batches_done))
            f.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batches_done))

        f.write('\n')
        f.close()
        batches_left = self.batches_epoch*(self.n_epochs - epoch) + self.batches_epoch - batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/self.batches_done)))

        # Save image sample
        image_sample = torch.cat((images['real_B'].data, images['fake_A'].data, images['real_A'].data), -2)
        self.past_images.append(image_sample)
        if len(self.past_images) > self.n_samples:
            self.past_images.pop(0)

        # If at sample interval save past samples
        if self.batches_done % self.sample_interval == 0 and images is not None:
            train_images = torch.cat(self.past_images, 0).cpu()[:5]
            val_images = generate_and_save_sample(self.generator, self.real_A, self.real_B)
            save_image(torch.cat([train_images, val_images], -2) + 0.5,
                        './'+str(self.image_path_name)+'/%d.png' % self.batches_done,
                        normalize=True)

        self.batches_done += 1

    def log_val(self, loss):
        with open(self.out_file, 'a') as f:
            print('\nvalidation loss: {}'.format(loss))
            f.write('validation loss: {}\n'.format(loss))


def generate_and_save_sample(generator, real_A, real_B):
    # Save image sample
    fake_A = generator(Variable(real_B).cuda()).cpu().data.view(-1, 1, 128, 128)
    image_sample = torch.cat((real_B, fake_A, real_A), -2)

    return image_sample
    # If at sample interval save past samples

    # save_image(image_sample,
    #             './images/{}.png'.format('test'),
    #             normalize=True)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def repackage(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage(v) for v in h)


def eval(generator, dataloader_val, criterion_translation):
    generator.eval()

    losses = []
    for i, batch in enumerate(dataloader_val):
        real_A = Variable(batch['A'].cuda())
        real_B = Variable(batch['B'].cuda())

        fake_A = generator(real_B)

        loss_trans = criterion_translation(fake_A, real_A)

        losses.append(loss_trans.cpu().data[0])

    generator.train()

    return np.mean(losses)


##############################
#           Data Augmentation
##############################

# TODO: when apply randomness, pay attention that some functions are for images some are for single image

def flip_leftright(img):
    """flip the image and exchange left and right"""
    return np.flip(img, 3)

def flip_updown(img):
    """flip the image and exchange left and right"""
    return np.flip(img, 2)

def GaussianBlur(img, sigma):
    """Blur the image"""
    img_new = img.copy()
    num_images = img_new.shape[0]
    for i in range(num_images):
        img_new[i] = ndimage.gaussian_filter(img_new[i], sigma)  # here doesn't differentiate chanels
    return img_new

def multiply(img, constant):
    """this is to multiply pixels in image with a constant number, this will make the photo brighter or darker"""
    return img*constant


def crop(img, x_lr=None, x_up=None, y_lr=None, y_up=None):
    img_new = img.copy()
    num_images = img.shape[0]
    x_width = img.shape[2]  # as the data we have has four dimensions
    y_height = img.shape[3]
    background = np.ones((x_width, y_height), dtype=np.float32)

    if None in [x_lr, x_up, y_lr, y_up]:
        #TODO: change the seed or it wouldn't work here
        x_lr, y_lr = np.random.randint(0, x_width), np.random.randint(0, y_height)
        x_up, y_up = np.random.randint(x_lr, x_width + 1), np.random.randint(y_lr, y_height + 1)

  # TODO: or change it to another color, or some random color, i.e. not use np.zeros
  # if change to colorful picture, need to remove the [0] after [i], slice three dimensions at the same time
    for i in range(num_images):
        img_temp = background.copy()
        img_temp[x_lr: x_up+1, y_lr: y_up+1] = img_new[i][0][x_lr: x_up+1,y_lr: y_up+1]
        img_new[i][0] = img_temp
    return img_new


def shift(source, target):
    num_images, _, x_width, y_height = source.shape
    shifted_source = np.ones((num_images, 1, x_width, y_height), dtype=np.float32)
    shifted_target = np.ones((num_images, 1, x_width, y_height), dtype=np.float32)

    shift_xs, shift_ys = np.random.randint(-8, 8, size=num_images), np.random.randint(-8, 8, size=num_images)
    for i, (shift_x, shift_y) in enumerate(zip(shift_xs, shift_ys)):
        shifted_source[i, 0] = np.roll(source[i, 0], (shift_x, shift_y), axis=(0, 1))
        shifted_target[i, 0] = np.roll(target[i, 0], (shift_x, shift_y), axis=(0, 1))

    return shifted_source, shifted_target


def expand(img):
    pass
    """zoom"""

def rotate(img):
    pass

# TODO: change numpy function to tensors, should try like tensor stack, or change the position of tensors
def data_augmentation(mode, source_font, target_font, randomness=False):
    if mode == 'flipleftright':
        source_font_temp = flip_leftright(source_font)
        target_font_temp = flip_leftright(target_font)

    elif mode == 'flipupdown':
        source_font_temp = flip_updown(source_font)
        target_font_temp = flip_updown(target_font)

    elif mode == 'GaussianBlur':
        source_font_temp = GaussianBlur(source_font, 2)
        target_font_temp = GaussianBlur(target_font, 2)

    elif mode == 'multiply':
        source_font_temp = multiply(source_font, 2)
        target_font_temp = multiply(target_font, 2)

    elif mode == 'crop':
        source_font_temp = crop(source_font)
        target_font_temp = crop(target_font)

    elif mode == 'rotation':
        source_font_temp = rotate(source_font)
        target_font_temp = rotate(target_font)

    elif mode == 'shift':
        source_font_temp, target_font_temp = shift(source_font, target_font)

    else:
        pass

    source_font = np.vstack([source_font, source_font_temp])
    target_font = np.vstack([target_font, target_font_temp])

    return source_font, target_font
