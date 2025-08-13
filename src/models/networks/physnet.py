import torch
import torch.nn as nn
from src.models.networks import Network

from typing import *
from torch import Tensor


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: List[int], stride: int, padding: Union[int, List[int]], *args, **kwargs) -> None:
        super(ConvolutionBlock, self).__init__()

        self.ConvBlock = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                *args, **kwargs
            ),
            nn.BatchNorm3d(
                num_features=out_channels
            ),
            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ConvBlock(x)


class PhysNet3DCNN(Network):
    """ Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks

    Source: https://arxiv.org/abs/1905.02419
    Code: https://github.com/ubicomplab/rPPG-Toolbox/tree/main

    We have adapted the PhysNet code from the rPPG-Toolbox.

    Args:
        Network (_type_): _description_
    """
    def __init__(self, 
        window: int, 
        img_size: int, 
        in_channels: Optional[int] = 3, 
        dropouts: Optional[List[int]] = [0.0, 0.0, 0.0, 0.0],
        *args, **kwargs) -> None:
        super(PhysNet3DCNN, self).__init__(*args, **kwargs)

        # Channels
        self.in_channels = in_channels
        self.img_size = img_size

        # Window
        self.window = window

        # Regularization
        self.dropouts = dropouts

        # Feature Extraction (g): Block 0
        self.ConvBlock1 = ConvolutionBlock(
            in_channels=self.in_channels,
            out_channels=16,
            kernel_size=[1, 5, 5],
            stride=1,
            padding=[0, 2, 2]
        )
        self.MaxpoolSpa = nn.MaxPool3d(
            kernel_size=(1, 2, 2), 
            stride=(1, 2, 2)
        )
        self.DropoutBlock0 = nn.Dropout(self.dropouts[0])

        # Feature Extraction (g): Block 1 [Repeat 0]
        self.ConvBlock2 = ConvolutionBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        self.ConvBlock3 = ConvolutionBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        self.MaxpoolSpaTem = nn.MaxPool3d(
            kernel_size=(2, 2, 2), 
            stride=2
        )
        self.DropoutBlock1 = nn.Dropout(self.dropouts[1])

        # Feature Extraction (g): Block 2 [Repeat 1]
        self.ConvBlock4 = ConvolutionBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        self.ConvBlock5 = ConvolutionBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        self.DropoutBlock2 = nn.Dropout(self.dropouts[2])

        # Feature Extraction (g): Block 3 [Repeat 2]
        self.ConvBlock6 = ConvolutionBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        self.ConvBlock7 = ConvolutionBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        self.DropoutBlock3 = nn.Dropout(self.dropouts[3])

        # Feature Extraction (g): Block 4 [Repeat 3]
        self.ConvBlock8 = ConvolutionBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        self.ConvBlock9 = ConvolutionBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )

        # Feature Projection (f): Block 5
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=64, 
                out_channels=64, 
                kernel_size=[4, 1, 1], 
                stride=[2, 1, 1], 
                padding=[1, 0, 0]
            ),  # [1, 128, 32]
            nn.BatchNorm3d(num_features=64),
            nn.ELU(alpha=1.0),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=64, 
                out_channels=64, 
                kernel_size=[4, 1, 1], 
                stride=[2, 1, 1], 
                padding=[1, 0, 0]
            ),  # [1, 128, 32]
            nn.BatchNorm3d(num_features=64),
            nn.ELU(alpha=1.0),
        )

        # Feature Projection (f): Block 5
        self.poolspa = nn.AdaptiveAvgPool3d(
            output_size=(self.window, 1, 1)
        )
        self.ConvBlock10 = nn.Conv3d(
            in_channels=64, 
            out_channels=1, 
            kernel_size=[1, 1, 1], 
            stride=1, 
            padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Forward-pass of the PhysNet-3DCNN architecture.

        Leverages a temporal encoder-decoder structure for the rPPG talk.

        After forward several convolution and pooling operations, the multi-channel
        manifolds are formed to represent the spatio-temporal features.

        Finally, the latent manifolds are projected into signal space using channel-wise
        convolution operation with a 1x1x1 kernel to generate the predicted PPG signal
        with length T.

        We can formulate this produce are follows:

            [y1, y2, ..., yT] = g(f([x1, x2, ..., xT]; theta); w)

        Args:
            inputs (Tensor): Batch of frames with size [B,C,T,H,W]

        Returns:
            Tensor: Batch of rPPG signals with size [B,T]
        """
        # Extract input structure
        # x_visual = x
        B, C, T, H, W = tuple(x.size())
        assert T == self.window, f"Require input temporal length T={self.window}, recieved {T}."

        # Block 1: Feature expansion/extraction and S-pooling
        x = self.ConvBlock1(x) # [3,T,H,W] to [16,T,H,W]
        x = self.MaxpoolSpa(x) # [16,T,H,W] to [16,T,H/2,W/2]
        x = self.DropoutBlock0(x)

        # Block 2: Feature expansion/extraction and ST-pooling
        x = self.ConvBlock2(x) # [16,T,H/2,W/2] to [32,T,H/2,W/2]
        x = self.ConvBlock3(x) # [32,T,H/2,W/2] to [64,T,H/2,W/2]
        # x_visual6464 = x
        x = self.MaxpoolSpaTem(x) # [64,T,H/2,W/2] to [64,T/2,H/4,W/4] (temporal pooling)
        x = x = self.DropoutBlock1(x)

        # Block 3: Feature extraction and ST-pooling
        x = self.ConvBlock4(x) # ...
        x = self.ConvBlock5(x) # ...
        # x_visual3232 = x
        x = self.MaxpoolSpaTem(x) # [64,T/2,H/4,W/4] to [64,T/4,H/8,W/8]
        x = self.DropoutBlock2(x)

        # Block 4: Feature extraction and S-pooling
        x = self.ConvBlock6(x) # ...
        x = self.ConvBlock7(x) # ...
        # x_visual1616 = x
        x = self.MaxpoolSpa(x) # [64,T/4,H/8,W/8] to [64,T/4,H/16,W/16]
        x = self.DropoutBlock3(x)

        # Block 5: Upsample features to length T
        x = self.ConvBlock8(x) # ...
        x = self.ConvBlock9(x) # ...
        x = self.upsample1(x)  # [64,T/4,H/16,W/16] to [64,T/2,H/16,W/16]
        x = self.upsample2(x)  # [64,T/2,H/16,W/16] to [64,T,H/16,W/16]

        # Block 5: Projection of features into signal space
        x = self.poolspa(x) # [64,T,H/16,W/16] to [64,T,1,1]
        x = self.ConvBlock10(x) # [64,T,1,1] to [1,T,1,1]
        
        # Output formatting
        x = x.view(-1, self.window)

        return x

    @property
    def dummy_input(self, batch_size: int) -> Tensor:
        return torch.rand(size=(batch_size, self.window, 3, self.img_size, self.img_size))
    

class PhysNet3DCNN_RGBA(Network):
    """ Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks

    Source: https://arxiv.org/abs/1905.02419
    Code: https://github.com/ubicomplab/rPPG-Toolbox/tree/main

    We have adapted the PhysNet code from the rPPG-Toolbox.

    Augments the baseline PhysNet3DCNN model to incorporate surface orientation information in the form
    of an additional input channel, use input concatenation as the form of feature fusion.

    Args:
        Network (_type_): _description_
    """
    def __init__(self, *args, **kwargs) -> None:
        super(PhysNet3DCNN, self).__init__(in_channels = 4, *args, **kwargs)


class SpatialAttentionBlock_MaxPool(nn.Module):
    def __init__(self, in_features: int, in_channels: int) -> None:
        super(SpatialAttentionBlock_MaxPool, self).__init__()

        self.max_pool = nn.MaxPool3d([in_channels, 1, 1], 1, 0)
        self.conv3d = nn.Conv3d(in_features,1,[1,3,3],1,[0,1,1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.max_pool(x)
        x = self.conv3d(x)
        x = self.sigmoid(x)
        return x
    

class SpatialAttentionBlock_MaxAvgPool(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(SpatialAttentionBlock_MaxAvgPool, self).__init__()
        
        self.max_pool = nn.MaxPool3d([3,3,3],1,[1,1,1])
        self.avg_pool = nn.AvgPool3d([3,3,3],1,[1,1,1])

        self.conv3d = nn.Conv3d(2*in_channels,1,[3,3,3],1,[1,1,1])

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)
        x = self.conv3d(x)
        x = self.sigmoid(x)
        return x


class SpatialAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super(SpatialAttentionBlock, self).__init__()
        # self.conv3d = nn.Conv3d(in_steps,in_steps,[in_channels,3,3],1,[0,1,1])
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.sum(x, dim=1, keepdim=True)
        # x = self.conv3d(x.permute(0,2,1,3,4)).permute(0,2,1,3,4) # [B,T,C,H,W]
        x = torch.softmax(x, dim=2)
        return x
        

        


class AttentionBranch(nn.Module):
    """_summary_

    Args:
        PhysNet3DCNN (_type_): _description_
    """
    def __init__(self, window: int, *args, **kwargs) -> None:
        super(AttentionBranch, self).__init__(*args, **kwargs)

        # Feature Extraction (g): Block 0
        self.ConvBlock1 = ConvolutionBlock(
            in_channels=1,
            out_channels=8,
            kernel_size=[1, 5, 5],
            stride=1,
            padding=[0, 2, 2]
        )
        self.MaxpoolSpa = nn.MaxPool3d(
            kernel_size=(1, 2, 2), 
            stride=(1, 2, 2)
        )
        self.SpaAtt0 = SpatialAttentionBlock() #in_channels=8

        # Feature Extraction (g): Block 1 [Repeat 0]
        self.ConvBlock2 = ConvolutionBlock(
            in_channels=8,
            out_channels=16,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        self.ConvBlock3 = ConvolutionBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        self.MaxpoolSpaTem = nn.MaxPool3d(
            kernel_size=(2, 2, 2), 
            stride=2
        )
        self.SpaAtt1 = SpatialAttentionBlock() #in_channels=32

        # Feature Extraction (g): Block 2 [Repeat 1]
        self.ConvBlock4 = ConvolutionBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        self.ConvBlock5 = ConvolutionBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        # MaxPoolSpaTem
        self.SpaAtt2 = SpatialAttentionBlock() #in_channels=32

        # Feature Extraction (g): Block 3 [Repeat 2]
        self.ConvBlock6 = ConvolutionBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        self.ConvBlock7 = ConvolutionBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1
        )
        # MaxPoolSpa
        self.SpaAtt3 = SpatialAttentionBlock() #in_channels=32

    def forward(self, x: Tensor) -> Tensor:
        # Block 0: Feature expansion/extraction and S-pooling
        x = self.ConvBlock1(x) # [3,T,H,W] to [16,T,H,W]
        x = self.MaxpoolSpa(x) # [16,T,H,W] to [16,T,H/2,W/2]
        a0 = self.SpaAtt0(x)

        # Block 1: Feature expansion/extraction and ST-pooling
        x = self.ConvBlock2(x) # [16,T,H/2,W/2] to [32,T,H/2,W/2]
        x = self.ConvBlock3(x) # [32,T,H/2,W/2] to [64,T,H/2,W/2]
        x = self.MaxpoolSpaTem(x) # [64,T,H/2,W/2] to [64,T/2,H/4,W/4] (temporal pooling)
        a1 = self.SpaAtt1(x)

        # Block 2: Feature extraction and ST-pooling
        x = self.ConvBlock4(x) # ...
        x = self.ConvBlock5(x) # ...
        x = self.MaxpoolSpaTem(x) # [64,T/2,H/4,W/4] to [64,T/4,H/8,W/8]
        a2 = self.SpaAtt2(x)

        # Block 3: Feature extraction and S-pooling
        x = self.ConvBlock6(x) # ...
        x = self.ConvBlock7(x) # ...
        x = self.MaxpoolSpa(x) # [64,T/4,H/8,W/8] to [64,T/4,H/16,W/16]
        a3 = self.SpaAtt3(x)

        return a0, a1, a2, a3
    

class PhysNet3DCNN_SpaAtt(PhysNet3DCNN):
    def __init__(self, *args, **kwargs) -> None:
        super(PhysNet3DCNN_SpaAtt, self).__init__(*args, **kwargs)

        self.attention_branch = AttentionBranch(self.window)

    def forward(self, x: Tensor) -> Tensor:
        """ Forward-pass of the PhysNet-3DCNN architecture with Spatial Attention

        Args:
            inputs (Tensor): Batch of frames with size [B,C,T,H,W]

        Returns:
            Tensor: Batch of rPPG signals with size [B,T]
        """
        # Extract input structure
        # x_visual = x
        B, C, T, H, W = tuple(x.size())
        assert T == self.window, f"Require input temporal length T={self.window}, recieved {T}."

        # Unpack
        x_rgb = x[:,:-1,:,:,:]
        x_ang = x[:,-1,:,:,:].unsqueeze(1)

        # Spatial-attention from angle map
        a0, a1, a2, a3 = self.attention_branch(x_ang)

        # Block 1: Feature expansion/extraction and S-pooling
        x = self.ConvBlock1(x_rgb) # [3,T,H,W] to [16,T,H,W]
        x = self.MaxpoolSpa(x) # [16,T,H,W] to [16,T,H/2,W/2]
        x = a0 * x # Spatial-attention

        # Block 2: Feature expansion/extraction and ST-pooling
        x = self.ConvBlock2(x) # [16,T,H/2,W/2] to [32,T,H/2,W/2]
        x = self.ConvBlock3(x) # [32,T,H/2,W/2] to [64,T,H/2,W/2]
        # x_visual6464 = x
        x = self.MaxpoolSpaTem(x) # [64,T,H/2,W/2] to [64,T/2,H/4,W/4] (temporal pooling)
        x = a1 * x

        # Block 3: Feature extraction and ST-pooling
        x = self.ConvBlock4(x) # ...
        x = self.ConvBlock5(x) # ...
        # x_visual3232 = x
        x = self.MaxpoolSpaTem(x) # [64,T/2,H/4,W/4] to [64,T/4,H/8,W/8]
        x = a2 * x

        # Block 4: Feature extraction and S-pooling
        x = self.ConvBlock6(x) # ...
        x = self.ConvBlock7(x) # ...
        # x_visual1616 = x
        x = self.MaxpoolSpa(x) # [64,T/4,H/8,W/8] to [64,T/4,H/16,W/16]
        x = a3 * x

        # Block 5: Upsample features to length T
        x = self.ConvBlock8(x) # ...
        x = self.ConvBlock9(x) # ...
        x = self.upsample1(x)  # [64,T/4,H/16,W/16] to [64,T/2,H/16,W/16]
        x = self.upsample2(x)  # [64,T/2,H/16,W/16] to [64,T,H/16,W/16]

        # Block 5: Projection of features into signal space
        x = self.poolspa(x) # [64,T,H/16,W/16] to [64,T,1,1]
        x = self.ConvBlock10(x) # [64,T,1,1] to [1,T,1,1]
        
        # Output formatting
        x = x.view(-1, self.window)

        return x
