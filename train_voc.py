import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2 as cv
import time
import os
import segmentation_models_pytorch.utils as smp_utils 
from networks import resnet50
from complexnn import ComplexBatchNorm2d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def give_encoder(model):
    # takingin resnet50 give_encoder
    model = resnet50(num_classes=1000)
    checkpoint = torch.load("resnet50_best.tar", map_location=device)
    pretrained_dict = checkpoint["state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}

    model.load_state_dict(pretrained_dict)
    
    model = nn.Sequential(*list(model.children())[:-4])
    # for param in model.parameters():
    #     param.requires_grad = False
        
    return model

class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))
    
class ComplexSoftMax(nn.Module):
    def forward(self, x):
        return torch.complex(F.softmax(x.real), F.softmax(x.imag))

class ComplexSemantic(nn.Module):
    def __init__(self, in_channels=1024, bilinear=False):
        super().__init__()
        self.encoder = give_encoder(resnet50())
        # factor = 2 if bilinear else 1
        self.up1 = nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2, dtype=torch.complex64)
        # self.up1 = (Up(in_channels, 512 // factor, bilinear))
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, dtype=torch.complex64)
        self.bn1 = ComplexBatchNorm2d(256)
        # self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, dtype=torch.complex64)
        # self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, dtype=torch.complex64)
        self.bn2 = ComplexBatchNorm2d(64)
        # self.up4 = (Up(128, 64 // factor, bilinear))
        self.outc = nn.Conv2d(64, 21, kernel_size=1, dtype=torch.complex64)
        self.relu = ComplexReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x_feat = self.encoder(x)
        # print(f"Shape of x features: {x_feat.shape}")
        # x_feat = torch.cat([x_feat.real, x_feat.imag], dim=1)
        x_up = self.up1(x_feat)
        x_up = self.relu(x_up)
        x_up = self.up2(x_up)
        x_up = self.bn1(x_up)
        x_up = self.relu(x_up)
        x_up = self.up3(x_up)
        x_up = self.relu(x_up)
        x_up = self.up4(x_up)
        x_up = self.bn2(x_up)
        x_up = self.relu(x_up)
        logits = self.outc(x_up)
        logits = logits.abs()
        logits = self.softmax(logits)
        # print(f"Shape of output: {logits.shape}")

        return logits

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

class VocDataset(Dataset):
  def __init__(self,dir,color_map):
    self.root=os.path.join(dir,'VOC2012')
    self.target_dir=os.path.join(self.root,'SegmentationClass')
    self.images_dir=os.path.join(self.root,'JPEGImages')
    file_list=os.path.join(self.root,'ImageSets/Segmentation/trainval.txt')
    self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
    self.color_map=color_map

  def convert_to_segmentation_mask(self,mask):
  # This function converts color channels of semgentation masks to number of classes (21 in this case)
  # Semantic Segmentation requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
  # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
  # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
    height, width = mask.shape[:2]
    segmentation_mask = np.zeros((height, width, len(self.color_map)), dtype=np.float32)
    for label_index, label in enumerate(self.color_map):
          segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
    return segmentation_mask

  def __getitem__(self,index):
    image_id=self.files[index]
    image_path=os.path.join(self.images_dir,f"{image_id}.jpg")
    label_path=os.path.join(self.target_dir,f"{image_id}.png")
    image=cv.imread(image_path)
    image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
    image=cv.resize(image,(256,256))
    image=torch.tensor(image).float()
    label=cv.imread(label_path)
    label=cv.cvtColor(label,cv.COLOR_BGR2RGB)
    label=cv.resize(label,(256,256))
    label = self.convert_to_segmentation_mask(label)
    label=torch.tensor(label).float()
    
    return image,label


  
  def __len__(self):
    return len(self.files)

data=VocDataset('voc_dataset',VOC_COLORMAP)

batch_size = 10

train_set,val_set=torch.utils.data.random_split(data,[int(len(data)*0.9),round(len(data)*0.1)+1])
train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=False)

print('Train Size   : ', len(train_set))
print('Val Size     : ', len(val_set))

model = ComplexSemantic()
   
# criterion = smp.utils.losses.DiceLoss(eps=1.)
metrics = smp_utils.metrics.IoU(eps=1.)
    
import math

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
    
class ToiRGB(object):
    def __call__(self, img):
        if not img.is_complex():
            raise ValueError(f"Input should be a complex tensor")

        real, imag = img.real, img.imag

        return (hsv_to_rgb(real)).type(torch.complex64) + 1j * (hsv_to_rgb(imag)).type(torch.complex64)


    def __repr__(self):
        return self.__class__.__name__+'()'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def convert_complex_to_real(complex_img):
    real = complex_img.real
    imag = complex_img.imag
    s = imag[..., 0, :, :]
    v = real[..., 0, :, :]
    h = real[...,1,:,:]/(s+ 1e-6)
    return torch.stack([h, s, v], dim=-3)


def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []; val_acc = []
    train_iou = []; train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1 ; not_improve=0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # Training loop
        model.train()
        for i, data in enumerate(train_loader):
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()
                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            image = image.permute(0, 3, 1, 2)
            mask = mask.permute(0, 3, 1, 2)
            
            image = rgb_to_hsv_mine(image)
            cmplx = ToComplex()
            irgb = ToiRGB()
            image = cmplx(image)
            # image = irgb(image)

            output = model(image)
            loss = criterion(output, mask)

            # Evaluation metrics
            iou_score += metrics(output, mask)
            # accuracy += pixel_accuracy(output, mask)

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        model.eval()
        test_loss = 0
        test_accuracy = 0
        val_iou_score = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                image_tiles, mask_tiles = data
                if patch:
                    bs, n_tiles, c, h, w = image_tiles.size()
                    image_tiles = image_tiles.view(-1, c, h, w)
                    mask_tiles = mask_tiles.view(-1, h, w)

                image = image_tiles.to(device)
                mask = mask_tiles.to(device)
                image = image.permute(0, 3, 1, 2)
                mask = mask.permute(0, 3, 1, 2)
                image = rgb_to_hsv_mine(image)
                cmplx = ToComplex()
                irgb = ToiRGB()
                image = cmplx(image)
                # image = irgb(image)
                output = model(image)
                val_iou_score += metrics(output, mask)
                # test_accuracy += pixel_accuracy(output, mask)
                loss = criterion(output, mask)
                test_loss += loss.item()

            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss / len(val_loader))))
                min_loss = (test_loss / len(val_loader))
                decrease += 1
                if decrease % 5 == 0:
                    print('saving model...')
                    torch.save(model, 'CVSaliency_mIoU-{:.3f}.pt'.format(val_iou_score / len(val_loader)))

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss / len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 41:
                    print('Loss not decrease for 41 times, Stop Training')
                    break

            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            # train_acc.append(accuracy / len(train_loader))
            # val_acc.append(test_accuracy / len(val_loader))
            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                #   "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                #   "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'lrs': lrs}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


max_lr = 1e-3
epoch = 80
weight_decay = 1e-4

# model = torch.load(r'CVSaliency_full.pt', map_location=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))

history = fit(epoch, model, train_loader, val_loader, criterion, optimizer, sched)

import pickle
geeky_file = open('history.pickle', 'wb') 
pickle.dump(history, geeky_file) 
geeky_file.close() 

torch.save(model, 'CVSaliency_full_VOC2012.pt')