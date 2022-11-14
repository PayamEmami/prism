import torch
import numpy as np
import skimage.transform
import math
from PIL import Image
import torch.nn.functional as F
import torch

def gram_matrix(input, feature_norm):
    b, c, h, w = input.size()
    feature_maps = input.view(b * c, h * w)
    with torch.cuda.amp.autocast(enabled=False):
        matrix = feature_maps.float() @ feature_maps.t().float()
    norm = b * c * h * w if feature_norm else h * w
    return matrix / norm

def match_color(content, style, eps=1e-5):
    content_pixels, style_pixels = content.view(3, -1), style.view(3, -1)
    mean_content, mean_style = content_pixels.mean(1), style_pixels.mean(1)
    dif_content = content_pixels - mean_content.view(3, 1)
    dif_style = style_pixels - mean_style.view(3, 1)
    eps = eps * torch.eye(3, device=content.device)
    cov_content = dif_content @ dif_content.t() / content_pixels.size(1) + eps
    cov_style = dif_style @ dif_style.t() / style_pixels.size(1) + eps
    eval_content, evec_content = torch.linalg.eig(cov_content)
    eval_content = eval_content.real.diag()
    evec_content = evec_content.real
    eval_style, evec_style = torch.linalg.eig(cov_style)
    eval_style = eval_style.real.diag()
    evec_style = evec_style.real
    cov_content_sqrt = evec_content @ eval_content.sqrt() @ evec_content.t()
    cov_style_sqrt = evec_style @ eval_style.sqrt() @ evec_style.t()
    weights = cov_style_sqrt @ cov_content_sqrt.inverse()
    bias = mean_style - weights @ mean_content
    content_pixels = weights @ content_pixels + bias.view(3, 1)
    return content_pixels.clamp(0,1).view_as(content)





mean_channel_values = np.array([103.939, 116.779, 123.68]).reshape((3, 1, 1))

def compute_shape(img_shape, max_size):
    # initial shape is (h, w, c)
    h, w, c = img_shape
    if w < h:
        # new size param is (h, w), so keep h, scale down w
        return (max_size, int(float(w) / h * max_size), c)
    else:
        # new size param is (h, w), so keep w, scale down h
        return (int(float(h) / w * max_size), max_size, c)


def resize_image_to_max_size(img, max_size):
    h, w, _ = compute_shape(img.shape, max_size)
    return skimage.transform.resize(img, (h, w), preserve_range=True)


def resize_image_to_vgg_input(img, max_size):
    if max_size is not None:
        img = resize_image_to_max_size(img, max_size)

    return image_to_vgg_input(img)


def image_to_vgg_input(img):
    # shuffle axes from hwc to c01
    img = img.transpose((2, 0, 1))
    # convert RGB to BGR
    img = img[::-1, :, :]
    # normalize the values for the VGG net
    img = img - mean_channel_values
    # add a batch dimension for the VGG net
    img = img.reshape([1] + list(img.shape))
    return float(img)


def vgg_input_to_image(x):
    # x in bc01 layout
    # get the first element out of the batch dimension
    x = np.copy(x[0])

    # adjust the mean to image range [0-255]
    x += mean_channel_values

    # convert BGR to RGB
    x = x[::-1]

    # shuffle axes from c01 to 01c
    x = x.transpose((1, 2, 0))

    # ensure limits and type of an image
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# make the match dimensions match the desired shape by cropping and padding w/reflection
def image_to_shape(image, desired_shape):
    if image.shape != desired_shape:
        # crop if a dimension is larger
        if image.shape[2] > desired_shape[2] or image.shape[3] > desired_shape[3]:
            # TODO: get a crop that has the most "stuff"
            # one idea is to compare 9 crops (4 corners, 4 center edges, full center) by total variation and choose the one with largest t.v.
            #image = image[:, :, :desired_shape[2], :desired_shape[3]]
            h = image.shape[2]
            w = image.shape[3]
            image = image[:, :, h // 2 - desired_shape[2] // 2:h // 2 + desired_shape[2] // 2, w // 2 - desired_shape[3] // 2:w // 2 + desired_shape[3] // 2]
        # pad if a dimension is smaller
        if image.shape[2] < desired_shape[2] or image.shape[3] < desired_shape[3]:
            r_diff = desired_shape[2] - image.shape[2]
            c_diff = desired_shape[3] - image.shape[3]
            image = np.pad(image, ((0, 0), (0, 0), (r_diff // 2, r_diff - (r_diff // 2)), (c_diff // 2, c_diff - (c_diff // 2))), mode='reflect')
    return image


def remove_reflect_padding(image):
    return image[32:image.shape[0]-32, 32:image.shape[1]-32, :]

def add_reflect_padding(image):
    return np.pad(image, ((32, 32), (32, 32), (0, 0)), mode='reflect')

def remove_reflect_padding_vgg(image_tensor):
    # bc01
    s = image_tensor.shape
    return image_tensor[:, :, 32:s[2]-32, 32:s[3]-32]


def unpadding(image, padding):
    b, c, h ,w = image.shape
    image = image[...,padding:h-padding, padding:w-padding]
    return image

def preprocess(image:Image, padding=32, patch_size=1024, transform=None, cuda=True, square=False):
    W, H = image.size
    N = math.ceil(math.sqrt((W * H) / (patch_size ** 2)))
    W_ = math.ceil(W / N) * N + 2 * padding
    H_ = math.ceil(H / N) * N + 2 * padding
    w = math.ceil(W / N) + 2 * padding
    h = math.ceil(H / N) + 2 * padding
    if square:
        w = patch_size + 2 * padding
        h = patch_size + 2 * padding
    if transform is not None:
        image = transform(image)
    image = image.unsqueeze(0)
    
    if cuda:
        image = image.cuda()
    p_left = (W_ - W) // 2
    p_right = (W_ - W) - p_left
    p_top = (H_ - H) // 2
    p_bottom = (H_ - H) - p_top
    image = F.pad(image, [p_left, p_right, p_top, p_bottom], mode="reflect")

    b, c, _, _ = image.shape
    images = F.unfold(image, kernel_size=(h, w), stride=(h-2*padding, w-2*padding))
    B, C_kw_kw, L = images.shape
    images = images.permute(0, 2, 1).contiguous()
    images = images.view(B, L, c, h, w).squeeze(dim=0)
    return images

def image_process(image):
    image = image.permute(1, 2, 0).mul_(255.0).add_(0.5).clamp_(0, 255)
    image = image.to(torch.uint8).cpu().data.numpy()
    return image


