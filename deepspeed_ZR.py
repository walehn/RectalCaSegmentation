import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed

import glob, os ,time, math
from random import *
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

import torchio as tio

from monai.config import print_config
from monai.data import CacheDataset, DataLoader, partition_dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_meandice
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    LoadNiftid,
    ToTensord, LabelToContour,
)
from monai.networks.blocks import Convolution, Upsample
from monai.networks.layers.factories import Pool, Act
from monai.networks.layers import split_args
from monai.utils import set_determinism


def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args

bs = 4
epoch_num = 350
Height = 144
Width = 144
Depth = 16
KIUNET = True
Zero = False

os.environ["MONAI_DATA_DIRECTORY"] = "./data"
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

data_dir = os.path.join(root_dir, "nifti_data")
train_images = sorted(glob.glob(os.path.join(data_dir, "image", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "mask", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

set_determinism(seed=0)

### image augmentation transform with monai and torchio API

# # HistogramStandardization parameter calculation
# histogram_landmarks_path = 'landmarks.npy'
# landmarks = tio.HistogramStandardization.train(
#     train_images,
#     output_path=histogram_landmarks_path,
# )
# np.set_printoptions(suppress=True, precision=3)
# print('\nTrained landmarks:', landmarks)

train_transforms_monai = [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
]

train_transforms_io = [
        tio.CropOrPad((Height, Width, Depth),mask_name='label', include=["image", "label"]),
        #tio.HistogramStandardization({'image': landmarks}, include=["image"]),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean, include=["image"]),
        tio.RandomNoise(p=0.1, include=["image"]),
        #tio.RandomMotion(num_transforms=1, image_interpolation='nearest',include=["image", "label"]),
        tio.RandomFlip(axes=(0,), include=["image", "label"]),
        #tio.RandomElasticDeformation(max_displacement=(40,20,0),include=["image", "label"]),
]

validation_transforms_monai = [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
]

validation_transforms_io = [
    tio.CropOrPad((Height, Width, Depth), include=["image", "label"], mask_name='label'),
    #tio.HistogramStandardization({'image': landmarks}, include=["image"]),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean, include=["image"]),
]

train_transforms = Compose(train_transforms_monai + train_transforms_io)
val_transforms = Compose(validation_transforms_monai + validation_transforms_io )


train_data, val_data, test_data = partition_dataset(data_dicts, ratios = [0.8, 0.1, 0.1], shuffle = True)

train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0, num_workers=4)
val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=1.0, num_workers=4)
test_ds = CacheDataset(data=test_data, transform=val_transforms, cache_rate=1.0, num_workers=4)

print('\n'+'Training set:', len(train_data), 'subjects')
print('Validation set:', len(val_data), 'subjects')
print('Validation set:', len(test_data), 'subjects')

#train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=4, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=4, num_workers=4)


## KI-UNet implementation using MONAI API, by H.Kim, MD.

class kiunet3dwcrfb(nn.Module):
    def __init__(self, c=1, n=1, activation = "PReLU", normalization = "BATCH", num_classes=2, drop_out = None):
        super(kiunet3dwcrfb, self).__init__()

        # Entry flow
        
        #U-NET part
        self.encoder1 = Convolution(dimensions = 3, in_channels = c, out_channels = n, dropout = drop_out, dropout_dim = 3,
                                     act = activation, norm = normalization, kernel_size = 3, padding = 1, strides = 1, bias = False)   
        self.encoder2 = Convolution(dimensions = 3, in_channels = n, out_channels = 2*n, dropout = drop_out, dropout_dim = 3,
                                     act = activation, norm = normalization, kernel_size = 3, padding = 1, strides = 1, bias = False) 
        self.encoder3 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = 4*n, dropout = drop_out, dropout_dim = 3,
                                     act = activation, norm = normalization, kernel_size = 3, padding = 1, strides = 1, bias = False) 

        #Ki-NET part
        self.kencoder1 = Convolution(dimensions = 3, in_channels = c, out_channels = n, dropout = drop_out, dropout_dim = 3,
                                     act = activation, norm = normalization, kernel_size = 3, padding = 1, strides = 1, bias = False)
        self.kencoder2 = Convolution(dimensions = 3, in_channels = n, out_channels = 2*n, dropout = drop_out, dropout_dim = 3,
                                     act = activation, norm = normalization, kernel_size = 3, padding = 1, strides = 1, bias = False)
        self.kencoder3 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = 2*n, dropout = drop_out, dropout_dim = 3,
                                     act = activation, norm = normalization, kernel_size = 3, padding = 1, strides = 1, bias = False)

        dim = 3
        pool_type, args = split_args(("MAX", {"kernel_size": 2, "stride": 2}))
        self.maxpooling3d = Pool[pool_type, dim](**args)
        
        self.upsample1 = Upsample(dimensions = dim, scale_factor=2, mode = "nontrainable", interp_mode ='trilinear', align_corners=False)         
        self.upsample2 = Upsample(dimensions = dim, scale_factor=4, mode = "nontrainable", interp_mode ='trilinear', align_corners=False) 
        self.upsample3 = Upsample(dimensions = dim, scale_factor=16, mode = "nontrainable", interp_mode ='trilinear', align_corners=False) 
        self.upsample4 = Upsample(dimensions = dim, scale_factor=64, mode = "nontrainable", interp_mode ='trilinear', align_corners=False) 

        self.CRFBupsample1 = Upsample(dimensions = dim, scale_factor=0.25, mode = "nontrainable", interp_mode ='trilinear', align_corners=False)        
        self.CRFBupsample2 = Upsample(dimensions = dim, scale_factor=0.0625, mode = "nontrainable", interp_mode ='trilinear', align_corners=False) 
        self.CRFBupsample3 = Upsample(dimensions = dim, scale_factor=0.015625, mode = "nontrainable", interp_mode ='trilinear',align_corners=False) 
        
        self.decoder1 = Convolution(dimensions = 3, in_channels = 4*n, out_channels = 2*n, dropout = drop_out, dropout_dim = 3,
                                     act = activation, norm = normalization, kernel_size = 3, padding = 1, strides = 1, bias = False)     
        self.decoder2 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = 2*n, dropout = drop_out, dropout_dim = 3,
                                     act = activation, norm = normalization, kernel_size = 3, padding = 1, strides = 1, bias = False)
        self.decoder3 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = c, dropout = drop_out, dropout_dim = 3,
                                     act = activation, norm = normalization, kernel_size = 3, padding = 1, strides = 1, bias = False)     

        self.kdecoder1 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = 2*n, dropout = drop_out, dropout_dim = 3,
                                     act = activation, norm = normalization, kernel_size = 3, padding = 1, strides = 1, bias = False)       
        self.kdecoder2 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = 2*n, dropout = drop_out, dropout_dim = 3,
                                     act = activation, norm = normalization, kernel_size = 3, padding = 1, strides = 1, bias = False)       
        self.kdecoder3 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = c, dropout = drop_out, dropout_dim = 3,
                                     act = activation, norm = normalization, kernel_size = 3, padding = 1, strides = 1, bias = False)      
        
        self.intere1_1 = Convolution(dimensions = 3, in_channels = n, out_channels = n, norm = normalization, act = activation,
                                     dropout = drop_out, dropout_dim = 3, kernel_size = 3, padding = 1, strides = 1) 
        self.intere2_1 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = 2*n, norm = normalization, act = activation,
                                     dropout = drop_out, dropout_dim = 3, kernel_size = 3, padding = 1, strides = 1) 
        self.intere3_1 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = 4*n, norm = normalization, act = activation,
                                     dropout = drop_out, dropout_dim = 3, kernel_size = 3, padding = 1, strides = 1) 

        self.intere1_2 = Convolution(dimensions = 3, in_channels = n, out_channels = n, norm = normalization, act = activation,
                                     dropout = drop_out, dropout_dim = 3, kernel_size = 3, padding = 1, strides = 1) 
        self.intere2_2 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = 2*n, norm = normalization, act = activation,
                                     dropout = drop_out, dropout_dim = 3, kernel_size = 3, padding = 1, strides = 1) 
        self.intere3_2 = Convolution(dimensions = 3, in_channels = 4*n, out_channels = 2*n, norm = normalization, act = activation,
                                     dropout = drop_out, dropout_dim = 3, kernel_size = 3, padding = 1, strides = 1) 

        self.interd1_1 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = 2*n, norm = normalization, act = activation,
                                     dropout = drop_out, dropout_dim = 3, kernel_size = 3, padding = 1, strides = 1) 
        self.interd2_1 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = 2*n, norm = normalization, act = activation, 
                                     dropout = drop_out, dropout_dim = 3, kernel_size = 3, padding = 1, strides = 1) 


        self.interd1_2 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = 2*n, norm = normalization, act = activation, 
                                     dropout = drop_out, dropout_dim = 3, kernel_size = 3, padding = 1, strides = 1) 
        self.interd2_2 = Convolution(dimensions = 3, in_channels = 2*n, out_channels = 2*n, norm = normalization, act = activation, 
                                     dropout = drop_out, dropout_dim = 3, kernel_size = 3, padding = 1, strides = 1) 


        self.seg = Convolution(dimensions = 3, in_channels = c, out_channels = num_classes, dropout = drop_out, dropout_dim = 3,
                               act = activation, norm = normalization, kernel_size = 1, padding = 0, strides = 1, bias = False)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight) #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        out =  self.maxpooling3d(self.encoder1(x))  #U-Net branch layer 1
        out1 = self.upsample1(self.kencoder1(x)) #Ki-Net branch layer 1
        tmp = out
        out = torch.add(out, self.CRFBupsample1(self.intere1_1(out1))) #CRFB1 down
        out1 = torch.add(out1, self.upsample2(self.intere1_2(tmp))) #CRFB1 up
        
        u1 = out  #skip conn
        o1 = out1  #skip conn

        out = self.maxpooling3d(self.encoder2(out)) #U-Net branch layer 2
        out1 = self.upsample1(self.kencoder2(out1)) #Ki-Net branch layer 2
        tmp = out
        out = torch.add(out, self.CRFBupsample2(self.intere2_1(out1))) #CRFB2 down
        out1 = torch.add(out1, self.upsample3(self.intere2_2(tmp))) #CRFB2 up
        
        u2 = out
        o2 = out1

        out = self.maxpooling3d(self.encoder3(out)) #U-Net branch layer 3
        out1 = self.upsample1(self.kencoder3(out1)) #Ki-Net branch layer 3
        tmp = out
        out = torch.add(out, self.CRFBupsample3(self.intere3_1(out1))) #CRFB3 down
        out1 = torch.add(out1, self.upsample4(self.intere3_2(tmp))) #CRFB3 up
        
        ### End of encoder block

        ### Start Decoder
        
        out = self.upsample1(self.decoder1(out)) #U-Net branch layer 1
        out1 = self.maxpooling3d(self.kdecoder1(out1)) #Ki-Net branch layer 1
        tmp = out
        out = torch.add(out, self.CRFBupsample2(self.interd1_1(out1))) #CRFB4 down
        out1 = torch.add(out1, self.upsample3(self.interd1_1(tmp))) #CRFB4 up 
        
        out = torch.add(out,u2)  #skip conn
        out1 = torch.add(out1,o2)  #skip conn

        out = self.upsample1(self.decoder2(out)) #U-Net branch layer 2
        out1 = self.maxpooling3d(self.kdecoder2(out1)) #Ki-Net branch layer 2
        tmp = out
        out = torch.add(out, self.CRFBupsample1(self.interd2_1(out1))) #CRFB5 down 
        out1 = torch.add(out1, self.upsample2(self.interd2_2(tmp))) #CRFB5 up
        
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = self.upsample1(self.decoder3(out)) #U-Net branch layer 3
        out1 = self.maxpooling3d(self.kdecoder3(out1)) #Ki-Net branch layer 3
 
        out = torch.add(out,out1) # fusion of both branches
        out = self.seg(out)  #1*1 conv
        return out


if KIUNET:
    model = kiunet3dwcrfb(c=1, n=1, num_classes=2, drop_out= 0.2)
    print("network is Ki-UNet")
    time.sleep(1)
else:
    model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=3,
    norm=Norm.BATCH,
    dropout=0.4
)

parameters = filter(lambda p: p.requires_grad, model.parameters())
args = add_argument()

model_engine, optimizer, train_loader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=parameters, training_data=train_ds)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)


## train start
chk_path = "./checkpoints"

val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
epoch_time = []
total_start = time.time()
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
post_label = AsDiscrete(to_onehot=True, n_classes=2)

for epoch in range(epoch_num):
    epoch_start = time.time()
    #print("-" * 50)
    #print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(model_engine.device),
            batch_data["label"].to(model_engine.device),
        )
        #optimizer.zero_grad()
        if Zero:
            outputs = model_engine(inputs.half())
            loss = loss_function(outputs.float(), labels)
        else:
            outputs = model_engine(inputs)
            loss = loss_function(outputs, labels)
        model_engine.backward(loss)
        model_engine.step()
        epoch_loss += loss.item()
        epoch_len = math.ceil(len(train_ds)/train_loader.batch_size)
        #print(
        #    f"{step}/{epoch_len}, train_loss: {loss.item():.4f}"
        #    f" step time: {(time.time() - step_start):.4f} seconds"
        #    )
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(model_engine.device),
                    val_data["label"].to(model_engine.device),
                )
                roi_size = (Height, Width, Depth)
                sw_batch_size = 1
                if Zero:
                    val_outputs = sliding_window_inference(
                        val_inputs.half(), roi_size, sw_batch_size, model
                        )
                    val_outputs = post_pred(val_outputs)
                    val_labels = post_label(val_labels)
                    value = compute_meandice(y_pred=val_outputs.float(), y=val_labels,
                    include_background=False)
                else:
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model
                        )
                    val_outputs = post_pred(val_outputs)
                    val_labels = post_label(val_labels)
                    value = compute_meandice(
                        y_pred=val_outputs,
                        y=val_labels,
                        include_background=False,
                    )
                metric_count += len(value)
                metric_sum += value.sum().item()
            metric = metric_sum / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                #if (best_metric>0.55):
                    #model_engine.save_checkpoint(chk_path)
                    #print("saved new best metric model")
                #torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))            
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )
    print(
        f"time consuming of epoch {epoch + 1} is:"
        f" {(time.time() - epoch_start):.4f} seconds"
        )

print(f"train completed, best_metric: {best_metric:.4f}  at epoch: {best_metric_epoch}")

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Average Training Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Validation Mean-Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()

_, client_sd = model_engine.load_checkpoint(chk_path)
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        roi_size = (Height, Width, Depth)
        sw_batch_size = 1
        test_image = test_data["image"].to(model_engine.device)
        if Zero:
            test_output = sliding_window_inference(
                test_image.half(), roi_size, sw_batch_size, model)
        else:
            test_output = sliding_window_inference(
                test_image, roi_size, sw_batch_size, model)

        # plot the slice [:, :, rand]
        j = randint(0, len(test_image[0,0,0,0,:])-1)
        plt.figure("check", (20, 4))

        plt.subplot(1, 5, 1)
        plt.title(f"image {i}")
        plt.imshow(test_image.detach().cpu()[0, 0, :, :, j], cmap="gray")

        plt.subplot(1, 5, 2)
        plt.title(f"Ground truth mask {i}")
        plt.imshow(test_data["label"][0, 0, :, :, j])

        plt.subplot(1, 5, 3)
        plt.title(f"AI predicted mask {i}")
        argmax = AsDiscrete(argmax=True)(test_output)
        plt.imshow(argmax.detach().cpu()[0, 0, :, :,j])

        plt.subplot(1, 5, 4)
        plt.title(f"contour {i}")
        contour = LabelToContour()(argmax)
        plt.imshow(contour.detach().cpu()[0, 0, :, :, j])

        plt.subplot(1, 5, 5)
        plt.title(f"overaying contour {i}")
        map_image = test_image.clone().detach()
        map_image[contour==1] = map_image.max()
        plt.imshow(map_image.detach().cpu()[0, 0, :, :, j], cmap="gray")
        plt.show()
print("Completed")