import pdb
import os
import numpy as np
from PIL import Image
from datetime import datetime
from utils import compute_gt_gradient, make_canvas_mask, numpy2tensor, laplacian_filter_tensor, \
                  MeanShift, Vgg16, gram_matrix, save_grid
import argparse
import torch
from torch.optim import Adam
from model import LaplacianFilter, Transfer
from dataset import get_source_loader  # , mean, std, denormalize
from torchvision import transforms
from torchvision.utils import save_image


# Default weights for loss functions in the first pass
grad_weight = 1e4  # 1e4
style_weight = 1e4  # 1e4
content_weight = 1  # 1
tv_weight = 1e-6

parser = argparse.ArgumentParser()
parser.add_argument('--source_file', type=str, default='data/1_source.png', help='path to the source image')
parser.add_argument('--mask_file', type=str, default='data/1_mask.png', help='path to the source mask image')
parser.add_argument('--target_file', type=str, default='data/1_target.png', help='path to the target image')
parser.add_argument('--snapshot', type=str, default='checkpoints/6_target_8.pt', help='path to the snapshot')
parser.add_argument('--batchsize', type=int, default=1, help='')
parser.add_argument('--worker_num', type=int, default=0, help='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--c_up', type=int, default=32, help='')
parser.add_argument('--down', type=int, default=2, help='')
parser.add_argument('--ss', type=int, default=300, help='source image size')
parser.add_argument('--ts', type=int, default=512, help='target image size')
parser.add_argument('--x', type=int, default=200, help='vertical location')
parser.add_argument('--y', type=int, default=235, help='vertical location')
parser.add_argument('--gpu_id', type=int, default=7, help='GPU ID')
parser.add_argument('--epoch', type=int, default=20, help='Number of epoch')
args = parser.parse_args()
now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

###################################
########### First Pass ###########
###################################

# Inputs
target_file = args.target_file
source_file = args.source_file
mask_file = args.mask_file

# Hyperparameter Inputs
gpu_id = args.gpu_id
max_epoch = args.epoch
ss = args.ss  # source image size
ts = args.ts  # target image size

# Model
transfer = Transfer(args).to(gpu_id)
transfer.load_state_dict(torch.load(args.snapshot))
lf = LaplacianFilter().to(gpu_id)

# Load target images
t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255)),
    # transforms.Normalize(mean, std)
])
target_img = Image.open(target_file).convert('RGB').resize((ts, ts))
target_img = t(target_img).to(gpu_id).unsqueeze(0).expand((args.batchsize, -1, -1, -1))
source_img = Image.open(source_file).convert('RGB').resize((ss, ss))
source_img = t(source_img).to(gpu_id).unsqueeze(0).expand((args.batchsize, -1, -1, -1))
mask = np.array(Image.open(mask_file).convert('L').resize((ss, ss)))
mask[mask > 0] = 1
mask = torch.from_numpy(mask).unsqueeze(0)

# optimizer
mse = torch.nn.MSELoss()

# Import VGG network for computing style and content loss
vgg = Vgg16().to(gpu_id)
mean_shift = MeanShift(gpu_id)
target_features_style = vgg(mean_shift(target_img))
target_gram_style = [gram_matrix(y) for y in target_features_style]

x = source_img
x_start = np.random.randint((ss+1)//2, ts-((ss+1)//2))
y_start = np.random.randint((ss+1)//2, ts-((ss+1)//2))

# Make Canvas Mask
canvas_mask = make_canvas_mask(x_start, y_start, target_img, mask)  # (B, ts, ts)
canvas_mask = torch.from_numpy(canvas_mask).unsqueeze(1).float().to(gpu_id)  # (B, 1, ts, ts)

# Compute gt_gradient
gt_gradient = compute_gt_gradient(lf, canvas_mask, x_start, y_start, x, target_img, mask, gpu_id)  # list of (B, ts, ts)

x_ts = torch.zeros((x.shape[0], x.shape[1], ts, ts)).to(gpu_id)  # (B, 3, ts, ts)
x_ts[:, :, x_start - ss//2:x_start + ss//2, y_start - ss//2:y_start + ss//2] = x
input_img = transfer(torch.cat([canvas_mask, x_ts], dim=1))


# Composite Foreground and Background to Make Blended Image
canvas_mask = canvas_mask.expand((-1, 3, -1, -1))  # (B, 3, ts, ts)
blend_img = torch.zeros(target_img.shape).to(gpu_id)
blend_img = input_img * canvas_mask + target_img * (canvas_mask - 1) * (-1)  # I_B

# Compute Laplacian Gradient of Blended Image
pred_gradient = lf(blend_img)  # list of (B, ts, ts)

# Compute Gradient Loss
grad_loss = 0
for c in range(len(pred_gradient)):
    grad_loss += mse(pred_gradient[c], gt_gradient[c])
grad_loss /= len(pred_gradient)
grad_loss *= grad_weight

# Compute Style Loss
blend_features_style = vgg(mean_shift(input_img))
blend_gram_style = [gram_matrix(y) for y in blend_features_style]

style_loss = 0
for layer in range(len(blend_gram_style)):
    style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
style_loss /= len(blend_gram_style)
style_loss *= style_weight

# Compute Content Loss
blend_obj = blend_img[:, :, int(x_start - x.shape[2] * 0.5):int(x_start + x.shape[2] * 0.5),
            int(y_start - x.shape[3] * 0.5):int(y_start + x.shape[3] * 0.5)]
m = mask.unsqueeze(1).to(gpu_id)
source_object_features = vgg(mean_shift(x * m))
blend_object_features = vgg(mean_shift(blend_obj * m))
content_loss = content_weight * mse(blend_object_features.relu2_2, source_object_features.relu2_2)


# Compute TV Reg Loss
tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
          torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))
tv_loss *= tv_weight

print(f'grad : {grad_loss.item():.4f}, style : {style_loss.item():.4f}, '
      f'content: {content_loss.item():.4f}, tv: {tv_loss.item():.4f}')

blend_img = blend_img.detach()
input_img = input_img.detach()
out = torch.cat([x_ts, x_ts*canvas_mask, blend_img, input_img], dim=0)
save_grid(out, f'results/{now}.png', nrow=args.batchsize)
style_name = os.path.basename(target_file).split('.')[0]
