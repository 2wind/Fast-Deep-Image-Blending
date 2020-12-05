import pdb
from random import random
import numpy as np
from L_BFGS import LBFGS
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imsave
from utils import compute_gt_gradient, make_canvas_mask, numpy2tensor, laplacian_filter_tensor, \
                  MeanShift, Vgg16, gram_matrix
import argparse
import torch
from torch.optim import Adam
from model import LaplacianFilter, Transfer
from dataset import get_source_loader


# Default weights for loss functions in the first pass
grad_weight = 1e4
style_weight = 1e4
content_weight = 1
tv_weight = 1e-6

parser = argparse.ArgumentParser()
parser.add_argument('--trainset', type=str, default='/content/drive/MyDrive/datasets/MSRA-B', help='path to the source image dataset')
parser.add_argument('--target_file', type=str, default='data/1_target.png', help='path to the target image')
parser.add_argument('--batchsize', type=int, default=1, help='')
parser.add_argument('--worker_num', type=int, default=0, help='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--c_up', type=int, default=32, help='')
parser.add_argument('--down', type=int, default=2, help='')
parser.add_argument('--ss', type=int, default=300, help='source image size')
parser.add_argument('--ts', type=int, default=512, help='target image size')
parser.add_argument('--x', type=int, default=200, help='vertical location')
parser.add_argument('--y', type=int, default=235, help='vertical location')
parser.add_argument('--device', type=int, default=0, help='GPU ID')
parser.add_argument('--epoch', type=int, default=10, help='Number of epoch')
args = parser.parse_args()

# Dataset
loader = get_source_loader(args)

###################################
########### First Pass ###########
###################################

# Inputs
target_file = args.target_file

# Hyperparameter Inputs
gpu_id = args.gpu_id
num_steps = args.num_steps
ss = args.ss  # source image size
ts = args.ts  # target image size

# Model
transfer = Transfer(args).to(gpu_id)
lf = LaplacianFilter().to(gpu_id)

# Load target images
target_img = np.array(Image.open(target_file).convert('RGB').resize((ts, ts)))
target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)

# optimizer
optimizer = Adam(transfer.parameters(), lr=args.lr)
mse = torch.nn.MSELoss()

# Import VGG network for computing style and content loss
mean_shift = MeanShift(gpu_id)
vgg = Vgg16().to(gpu_id)


""""
[shape]

source_img torch.Size([1, 3, 300, 300])
target_img torch.Size([1, 3, 512, 512])
input_img torch.Size([1, 3, 512, 512])
blend_img torch.Size([1, 3, 512, 512])
mask_img torch.Size([1, 3, 300, 300])
canvas_mask torch.Size([1, 3, 512, 512])
"""

# random blending location is needed


epoch = 0
while epoch <= args.epoch:
    for itr, (x, mask) in enumerate(loader):
        # x: tensor
        # mask: ndarray
        x = x.to(gpu_id)
        x_start = np.random.randint((ss+1)//2, ts-((ss+1)//2))
        y_start = np.random.randint((ss+1)//2, ts-((ss+1)//2))

        # Make Canvas Mask
        canvas_mask = make_canvas_mask(x_start, y_start, target_img, mask)
        canvas_mask = numpy2tensor(canvas_mask, gpu_id)
        canvas_mask = canvas_mask.squeeze(0).repeat(3, 1).view(3, ts, ts).unsqueeze(0)

        # Compute gt_gradient
        gt_gradient = compute_gt_gradient(x_start, y_start, x, target_img, mask, gpu_id)

        x_ts = torch.zeros((x.shape[0], x.shape[1], ts, ts)).to(gpu_id)
        x_ts[:, :, x_start - ss//2:x_start + ss//2, y_start - ss//2:y_start + ss//2] = x
        input_img = transfer(x_ts)

        # Composite Foreground and Background to Make Blended Image
        blend_img = torch.zeros(target_img.shape).to(gpu_id)
        blend_img = input_img * canvas_mask + target_img * (canvas_mask - 1) * (-1)  # I_B

        # Compute Laplacian Gradient of Blended Image
        pred_gradient = lf(blend_img)

        # Compute Gradient Loss
        grad_loss = 0
        for c in range(len(pred_gradient)):
            grad_loss += mse(pred_gradient[c], gt_gradient[c])
        grad_loss /= len(pred_gradient)
        grad_loss *= grad_weight

        # Compute Style Loss
        target_features_style = vgg(mean_shift(target_img))
        target_gram_style = [gram_matrix(y) for y in target_features_style]

        blend_features_style = vgg(mean_shift(input_img))
        blend_gram_style = [gram_matrix(y) for y in blend_features_style]

        style_loss = 0
        for layer in range(len(blend_gram_style)):
            style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
        style_loss /= len(blend_gram_style)
        style_loss *= style_weight

        # Compute Content Loss
        mask = numpy2tensor(mask, gpu_id)
        mask = mask.squeeze(0).repeat(3, 1).view(3, ss, ss).unsqueeze(0)
        blend_obj = blend_img[:, :, int(x_start - x.shape[2] * 0.5):int(x_start + x.shape[2] * 0.5),
                    int(y_start - x.shape[3] * 0.5):int(y_start + x.shape[3] * 0.5)]
        source_object_features = vgg(mean_shift(x * mask))
        blend_object_features = vgg(mean_shift(blend_obj * mask))
        content_loss = content_weight * mse(blend_object_features.relu2_2, source_object_features.relu2_2)
        content_loss *= content_weight

        # Compute TV Reg Loss
        tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
                  torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))
        tv_loss *= tv_weight

        # Compute Total Loss and Update Image
        loss = grad_loss + style_loss + content_loss + tv_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print Loss
        if itr % 1 == 0:
            print(f'[{epoch:>02} epoch {itr:>05} itr]'
                  'grad : {grad_loss.item():4f}, style : {style_loss.item():4f}, '
                  'content: {content_loss.item():4f}, tv: {tv_loss.item():4f}')
    epoch += 1
