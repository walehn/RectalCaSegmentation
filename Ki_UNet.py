import torch
from torch import nn
from monai.networks.blocks import Convolution, Upsample
from monai.networks.layers.factories import Pool, Act
from monai.networks.layers import split_args

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