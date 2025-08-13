import torch
import torch.nn as nn
from src.models.networks import Network

from typing import *
from torch import Tensor


class AttentionMaskModule(nn.Module):
    """
    """
    def __init__(self):
        super(AttentionMaskModule, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(AttentionMaskModule, self).get_config()
        return config
    

class TemporalShiftModule(nn.Module):
    """
    """
    def __init__(self, n_segment=10, fold_div=3):
        super(TemporalShiftModule, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)



class MTTS_CAN(Network):
    """ Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement

    Source: https://arxiv.org/abs/2006.03790
    Code: https://github.com/xliucs/MTTS-CAN

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
        frame_depth: Optional[int] = 20, 
        img_size: Optional[int] = 36,
        *,args, **kwargs
    ) -> None:
        super(MTTS_CAN, self).__init__(*args, **kwargs)

        # Attributes
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense

        # Temporal-shift Layers
        self.TSM_1 = TemporalShiftModule(n_segment=frame_depth)
        self.TSM_2 = TemporalShiftModule(n_segment=frame_depth)
        self.TSM_3 = TemporalShiftModule(n_segment=frame_depth)
        self.TSM_4 = TemporalShiftModule(n_segment=frame_depth)

        # Motion-branch Convolutions
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        
        # Appearance-branch Convolutions
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        
        # Attention-mask Layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = AttentionMaskModule()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = AttentionMaskModule()

        # Average Pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)

        # Dropout Layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)

        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        elif img_size == 128:
            self.final_dense_1 = nn.Linear(57600, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs: Tensor, params: Optional[Tensor] = None) -> Tensor:
        """ Forward-pass for the `MTTS-CAN` network.

        Please see the `mtts_can.yaml` configuration for further information on the pre-processing
        procedures.

        Args:
            inputs (Tensor): Concatenated appearance and difference frames of size [T,C,H,W] = [B,30,6,sz,sz]
            params (_type_, optional): _description_. Defaults to None.

        Returns:
            Tensor: Predicted difference of the BVP signal, p'(t) of size [T] = [B,30]
        """
        diff_input = inputs[:, :3, :, :] # difference
        raw_input = inputs[:, 3:, :, :] # appearance

        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return out