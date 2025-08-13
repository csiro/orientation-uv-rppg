import torch
import torch.nn as nn
from src.models.networks import Network

from typing import *
from torch import Tensor


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class AppearanceBranch(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
        in_channels: int,
        nb_filters1: int,
        nb_filters2: int,
        kernel_size: int,
        pool_size: int,
        dropout_rate1: float,
        *args, **kwargs
    ) -> None:
        super(AppearanceBranch, self).__init__()

        # Apperance-branch Convolutions
        self.apperance_conv1 = nn.Conv2d(
            in_channels, nb_filters1, kernel_size=kernel_size,
            padding=(1, 1), bias=True
        )
        self.apperance_conv2 = nn.Conv2d(
            nb_filters1, nb_filters1, kernel_size=kernel_size, 
            bias=True
        )
        self.apperance_conv3 = nn.Conv2d(
            nb_filters1, nb_filters2, kernel_size=kernel_size,
            padding=(1, 1), bias=True
        )
        self.apperance_conv4 = nn.Conv2d(
            nb_filters2, nb_filters2, kernel_size=kernel_size, 
            bias=True
        )

        # Attention Layers
        self.apperance_att_conv1 = nn.Conv2d(
            nb_filters1, 1, kernel_size=1, 
            padding=(0, 0), bias=True
        )
        self.attn_mask_1 = AttentionMask()

        self.apperance_att_conv2 = nn.Conv2d(
            nb_filters2, 1, kernel_size=1, 
            padding=(0, 0), bias=True
        )
        self.attn_mask_2 = AttentionMask()

        # Average Pooling
        self.avg_pooling_2 = nn.AvgPool2d(pool_size)

        # Dropout Layers
        self.dropout_2 = nn.Dropout(dropout_rate1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """ Compute the attention mask based on the appearance frame.

        Args:
            x (Tensor): Appearance frame

        Returns:
            Tuple[Tensor, Tensor]: Gate attention masks
        """
        # Convolve over appearance frame [3:]
        r1 = torch.tanh(self.apperance_conv1(x)) # [N,3,H,W] -> [N,32,H,W]
        r2 = torch.tanh(self.apperance_conv2(r1)) # [N,32,H,W] -> [N,32,H,W]

        # Attention mask from appearance frame features : convolve, squash, calc
        g1 = torch.sigmoid(self.apperance_att_conv1(r2)) # [N,32,H,W] -> [N,1,H,W]
        g1 = self.attn_mask_1(g1)

        # Pool and regularized appearance features
        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        # Convolve over appearance features
        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        # Attention mask from appearance frame features : convolve, squash, and calculate
        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)

        return g1, g2


class AttentionMask(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
        *args, **kwargs
    ) -> None:
        super(AttentionMask, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): Output of sigmoid of convolved features

        Returns:
            Tensor: _description_
        """
        # Aggregate (summation) 
        xsum = torch.sum(x, dim=2, keepdim=True) # [N,1,H,W] -> [N,1,1,W]
        xsum = torch.sum(xsum, dim=3, keepdim=True) # [N,1,1,W] -> [N,1,1,1]

        # Normalize by (sum * num_pixels) * 0.5
        xshape = tuple(x.size())
        g = x / xsum * xshape[2] * xshape[3] * 0.5 # 

        return g

    def get_config(self):
        """May be generated manually. """
        config = super(AttentionMask, self).get_config()
        return config


class MotionBranch(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
        in_channels: int,
        nb_filters1: int,
        nb_filters2: int,
        kernel_size: int,
        pool_size: int,
        dropout_rate1: float
    ) -> None:
        super(MotionBranch, self).__init__()

        # Motion-branch Convolutions
        self.motion_conv1 = nn.Conv2d(
            in_channels, nb_filters1, kernel_size=kernel_size, padding=(1, 1),
            bias=True
        )
        self.motion_conv2 = nn.Conv2d(
            nb_filters1, nb_filters1, kernel_size=kernel_size, 
            bias=True
        )
        self.motion_conv3 = nn.Conv2d(
            nb_filters1,  nb_filters2, kernel_size=kernel_size, padding=(1, 1),
            bias=True
        )
        self.motion_conv4 = nn.Conv2d(
            nb_filters2, nb_filters2, kernel_size=kernel_size, 
            bias=True
        )

        # Average Pooling
        self.avg_pooling_1 = nn.AvgPool2d(pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(pool_size)

        # Dropout Layers
        self.dropout_1 = nn.Dropout(dropout_rate1)
        self.dropout_3 = nn.Dropout(dropout_rate1)

    def forward(self, x: Tensor, g1: Tensor, g2: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): _description_
            g1 (Tensor): _description_
            g2 (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        # Convolve over difference frame [:3]
        d1 = torch.tanh(self.motion_conv1(x))
        d2 = torch.tanh(self.motion_conv2(d1))

        # Gate difference frame from appearance
        gated1 = d2 * g1 

        # Pool and regularize difference features gated by appearance
        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        # Convolve over gated-difference features
        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(d5))

        # Gate difference features from appearance
        gated2 = d6 * g2 

        # Pool gated-diff features and regularize
        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)

        return d8
    

class DenseConnection(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, 
        image_size: int, 
        nb_dense: int, 
        dropout_rate2: float,
        *args, **kwargs
    ) -> None:
        super(DenseConnection, self).__init__()

        # Dense Layers
        if image_size == 36:
            self.final_dense_1 = nn.Linear(3136, nb_dense, bias=True)
        elif image_size == 72:
            self.final_dense_1 = nn.Linear(16384, nb_dense, bias=True)
        elif image_size == 96:
            self.final_dense_1 = nn.Linear(30976, nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        
        # Linear Output
        self.final_dense_2 = nn.Linear(nb_dense, 1, bias=True)

        # Dropout Layers
        self.dropout_4 = nn.Dropout(dropout_rate2)

    def forward(self, x: Tensor) -> Tensor:
        """ Compute non-linear mapping to output waveform
        """
        # Non-linear connection
        d9 = x.view(x.size(0), -1) # Reshape features
        d10 = torch.tanh(self.final_dense_1(d9)) # Dense connection

        # Dropout
        d11 = self.dropout_4(d10) 

        # Final prediction
        out = self.final_dense_2(d11)

        return out       


class DeepPhys(Network):
    """ DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks

    https://arxiv.org/abs/1805.07888

    Args:
        Network (_type_): _description_
    """
    def __init__(self, 
        in_channels: Optional[int] = 3, 
        nb_filters1: Optional[int] = 32, 
        nb_filters2: Optional[int] = 64,
        kernel_size: Optional[int] = 3, 
        dropout_rate1: Optional[float] = 0.25,
        dropout_rate2: Optional[float] = 0.50, 
        pool_size: Optional[Tuple[int]] = (2, 2), 
        nb_dense: Optional[int] = 128, 
        img_size: Optional[int] = 36,
        *args, **kwargs
    ) -> None:
        super(DeepPhys, self).__init__()

        # Appearance-branch
        self.appearance_branch = AppearanceBranch(
            in_channels,
            nb_filters1,
            nb_filters2,
            kernel_size,
            tuple(pool_size),
            dropout_rate1
        )

        # Motion-branch
        self.motion_branch = MotionBranch(
            in_channels,
            nb_filters1,
            nb_filters2,
            kernel_size,
            tuple(pool_size),
            dropout_rate1
        )

        # Dense Connections
        self.dense_connection = DenseConnection(
            img_size, 
            nb_dense, 
            dropout_rate2
        )

        # Attributes
        self.num_channels = 6
        self.image_size = img_size 

    def forward(self, inputs: Tensor) -> Tensor:
        """ Forward-pass for the `DeepPhys` network.

        Please see the `deepphys.yaml` configuration for further information on the pre-processing
        procedures.

        Args:
            inputs (Tensor): Concatenated appearance and difference frames of size [T,C,H,W] = [B,30,6,sz,sz]
            params (_type_, optional): _description_. Defaults to None.

        Returns:
            Tensor: Predicted difference of the BVP signal, p'(t) of size [T] = [B,30]
        """
        # Compute appearance-based attention masks
        g1, g2 = self.appearance_branch(inputs[:, 3:, :, :])

        # Compute motion-based features gated by appearance attention masks
        d8 = self.motion_branch(inputs[:, :3, :, :], g1, g2)

        # Compute dense connection to output (MLP w 1-layer)
        out = self.dense_connection(d8)
        
        return out
    
    def dummy_input(self, batch_size: int) -> None:
        return torch.rand(size=(
            batch_size, 
            self.num_channels, 
            self.image_size, 
            self.image_size
        ))
    

class DeepPhysOrientation(DeepPhys):
    """_summary_

    Args:
        DeepPhys (_type_): _description_
    """
    def __init__(self, 
        in_channels: Optional[int] = 3, 
        nb_filters1: Optional[int] = 32, 
        nb_filters2: Optional[int] = 64,
        kernel_size: Optional[int] = 3, 
        dropout_rate1: Optional[float] = 0.25,
        dropout_rate2: Optional[float] = 0.50, 
        pool_size: Optional[Tuple[int]] = (2, 2), 
        nb_dense: Optional[int] = 128, 
        img_size: Optional[int] = 36,
        *args, **kwargs
    ) -> None:
        super(DeepPhysOrientation, self).__init__(
            in_channels = in_channels,
            nb_filters1 = nb_filters1,
            nb_filters2 = nb_filters2,
            kernel_size = kernel_size,
            dropout_rate1 = dropout_rate1,
            dropout_rate2 = dropout_rate2,
            pool_size = tuple(pool_size),
            nb_dense = nb_dense,
            img_size= img_size,
            *args, **kwargs
        )

        # Orientation-branch
        self.angle_branch = AppearanceBranch(
            in_channels = 1, # Angle map contains the relative angle (1D) between surface and camera
            nb_filters1 = nb_filters1,
            nb_filters2 = nb_filters2,
            kernel_size = kernel_size,
            pool_size = tuple(pool_size),
            dropout_rate1 = dropout_rate1
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """ Forward-pass for the `DeepPhys` network.

        Please see the `deepphys_uv.yaml` configuration for further information on the pre-processing
        procedures.

        inputs[0:3] = motion-representation in UV-space
        inputs[3:7] = appearance-representation in UV-space
        inputs[7] = angle-representation in UV-space

        Args:
            inputs (Tensor): Concatenated appearance and difference frames of size [T,C,H,W] = [B,30,9,sz,sz]
            params (_type_, optional): _description_. Defaults to None.

        Returns:
            Tensor: Predicted difference of the BVP signal, p'(t) of size [T] = [B,30]
        """
        # Extract batch items
        appearance = inputs[:, 3:-1, :, :]
        motion = inputs[:, :3, :, :]
        orientation = inputs[:, -1, :, :].unsqueeze(1)

        # Compute appearance-based attention masks
        g1_a, g2_a = self.appearance_branch(appearance)

        # Compute angle-based attention masks
        g1_o, g2_o = self.angle_branch(orientation) 

        # Compute non-linearized attention mask
        g1 = g1_a * g1_o
        g2 = g2_a * g2_o
        
        # Compute motion-based features
        d8 = self.motion_branch(motion, g1, g2)

        # Compute dense connection to output (MLP w 1-layer)
        out = self.dense_connection(d8)
        
        return out


# class DeepPhysMaskedOrientation(DeepPhys):
#     def __init__(self, *args, **kwargs) -> None:
#         super(DeepPhysMaskedOrientation, self).__init__(*args, **kwargs)

#     def forward(self, inputs: Tensor) -> Tensor:
#         """ Forward-pass for the `DeepPhys` network.

#         Please see the `deepphys_orientation.yaml` configuration for further information on the pre-processing
#         procedures.

#         inputs[0:3] = motion-representation
#         inputs[3:7] = appearance-representation
#         inputs[7] = angle-representation
#         Args:
#             inputs (Tensor): Concatenated appearance and difference frames of size [T,C,H,W] = [B,30,9,sz,sz]
#             params (_type_, optional): _description_. Defaults to None.

#         Returns:
#             Tensor: Predicted difference of the BVP signal, p'(t) of size [T] = [B,30]
#         """
#         # Extract batch items
#         appearance = inputs[:, 3:-1, :, :]
#         motion = inputs[:, :3, :, :]
#         orientation = inputs[:, -1, :, :].unsqueeze(1)

#         # Compute appearance-based attention masks
#         g1_a, g2_a = self.appearance_branch(appearance)

#         # Compute angle-based attention masks
#         g1_o, g2_o = self.angle_branch(orientation) 

#         # Compute non-linearized attention mask
#         g1 = g1_a * g1_o
#         g2 = g2_a * g2_o
        
#         # Compute motion-based features
#         d8 = self.motion_branch(motion, g1, g2)

#         # Compute dense connection to output (MLP w 1-layer)
#         out = self.dense_connection(d8)
        
#         return out