import os
import time
import sys
import math
import torch
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)

    print(f"Total number of parameters: {total_params}")


def rgb_to_hsv_mine(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.
    The image data is assumed to be in the range of (0, 1).
    Args:
        image (torch.Tensor): RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps (float, optional): scalar to enforce numarical stability. Default: 1e-6.
    Returns:
        torch.Tensor: HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.
    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    # The first or last occurance is not guarenteed before 1.6.0
    # https://github.com/pytorch/pytorch/issues/20414
    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    # s: torch.Tensor = deltac
    s: torch.Tensor = deltac / (maxc + eps)

    # avoid division by zero
    deltac = torch.where(deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: torch.Tensor = maxc_tmp[..., 0, :, :]
    gc: torch.Tensor = maxc_tmp[..., 1, :, :]
    bc: torch.Tensor = maxc_tmp[..., 2, :, :]

    h = torch.stack([bc - gc, 2.0 * deltac + rc - bc, 4.0 * deltac + gc - rc], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h

    return torch.stack([h, s, v], dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.
    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.
    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.
    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)

    return out



# Function for pytorch transforms 

class ToHSV(object):
    
    def __call__(self, pic):
        """RGB image to HSV image"""
        # return rgb_to_hsv_mine(pic)
        return rgb_to_hsv_mine(pic)

    def __repr__(self):
        return self.__class__.__name__+'()'

class ToRGB(object):
    def __call__(self, img):
        """HSV image to RGB image"""
        return hsv_to_rgb(img)

    def __repr__(self) -> str:
        return self.__class__.__name__+'()'

class ToiRGB(object):
    def __call__(self, img):
        if not img.is_complex():
            raise ValueError(f"Input should be a complex tensor")

        real, imag = img.real, img.imag

        return (hsv_to_rgb(real)).type(torch.complex64) + 1j * (hsv_to_rgb(imag)).type(torch.complex64)


    def __repr__(self):
        return self.__class__.__name__+'()'

# Function for complex conversion
class ToComplex(object):
    def __call__(self, img):
        hue = img[..., 0, : , :]
        sat = img[..., 1, : , :]
        val = img[..., 2, : , :]


        # tmp = 2*math.pi - hue

        # hue_mod = torch.where(hue<=math.pi, hue, tmp)

        
        real_1 = sat * hue
        # real_1 = sat * hue_mod
        real_2 = sat * torch.cos(hue)
        real_3 = val


        imag_1 = val
        imag_2 = sat * torch.sin(hue)
        imag_3 = sat

        real = torch.stack([real_1, real_2, real_3], dim= -3)
        imag = torch.stack([imag_1, imag_2, imag_3], dim= -3)

        comp_tensor = torch.complex(real, imag)

        assert comp_tensor.dtype == torch.complex64
        return comp_tensor

    def __repr__(self):
        return self.__class__.__name__+'()'

class ToComplex2(object):
    def __call__(self, img):
        hue = img[..., 0, :, :]
        sat = img[..., 1, :, :]
        val = img[..., 2, :, :]

        real1 = val
        real2 = torch.cos(hue)

        imag1 = sat
        imag2 = torch.sin(hue)

        real = torch.stack([real1, real2], dim=-3)
        imag = torch.stack([imag1, imag2], dim=-3)

        return torch.complex(real, imag)

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def get_mean_and_std(dataset):
    """ Compute the mean abd std value of the dataset. """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size= 1, shuffle= True, num_workers= 4)
    mean = torch.zeros(6)
    std = torch.zeros(6)
    print("==> Computing mean and std..")

    for inputs, targets in dataloader:
        for i in range(6):
            mean[i] += inputs[:, i, : , :].mean()
            std[i] += inputs[:, i, : , :].std()

    mean.div_(len(dataset))
    std.div_(len(dataset))

    return mean, std


