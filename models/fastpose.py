# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
import torch
import torch.nn as nn
from .builder import SPPE
from .layers.DUC import DUC
from .layers.SE_Resnet import SEResnet
from .layers.KnowledgeProjectionNet import KnowledgeProjectionNet
from .layers.Hough_Opt import compute_hog_features
import numpy as np


@SPPE.register_module
class FastPose(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(FastPose, self).__init__()
        self._preset_cfg = cfg['PRESET']
        if 'CONV_DIM' in cfg.keys():
            self.conv_dim = cfg['CONV_DIM']
        else:
            self.conv_dim = 128
        if 'DCN' in cfg.keys():
            stage_with_dcn = cfg['STAGE_WITH_DCN']
            dcn = cfg['DCN']
            self.preact = SEResnet(
                f"resnet{cfg['NUM_LAYERS']}", dcn=dcn, stage_with_dcn=stage_with_dcn)
        else:
            self.preact = SEResnet(f"resnet{cfg['NUM_LAYERS']}")

        # Imagenet pretrain model
        import torchvision.models as tm   # noqa: F401,F403
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2, norm_layer=norm_layer)
        if self.conv_dim == 256:
            self.duc2 = DUC(256, 1024, upscale_factor=2, norm_layer=norm_layer)
        else:
            self.duc2 = DUC(256, 512, upscale_factor=2, norm_layer=norm_layer)
        self.conv_out = nn.Conv2d(
            self.conv_dim, self._preset_cfg['NUM_JOINTS'], kernel_size=3, stride=1, padding=1)

        self.KP = KnowledgeProjectionNet(external_dim=9,feature_dim=128)
        self.linear_W = nn.Linear(128, 9, bias=False)
        self.beta = 0.01

# ---------------------------------------------when training------------------------------------------#
    # def forward(self, inputs):                          #

    #     x, prior = inputs                               # x = (3,256,192) 
    #     # print(type(x),type(prior))
    #     # print("-------------------------------",prior.shape,x.shape)
    #     out = self.preact(x)                            # (b,2048,8,6),
    #     out = self.suffle1(out)                         # (b,512,16,12)
    #     out = self.duc1(out)                            # (b,256,32,24)
    #     out = self.duc2(out)                            # (b,128,64,48)

    #     out, HJ = self.KP(out)                          # out=(b,128,64,48),HJ=(b,H*W,128)
    #     out = self.conv_out(out)                        # (b,6,64,48)

    #     # prior = torch.tensor(prior)
    #     K = compute_hog_features(prior)                 # shape = (b,3072,9)
    #     # print(K.shape,type(K))
    #     K = K / (K.sum(dim=-1, keepdims=True) + 1e-6) 

    #     HJ_projected = self.linear_W(HJ)                # shape=(b,64*48,9)
    #     # print("-----------------------------",HJ_projected.shape,K.shape)
    #     # print(type(HJ_projected),type(K))
    #     mse_loss = nn.MSELoss()(HJ_projected, K)       
    #     W = self.linear_W.weight                       
    #     l2_reg = self.beta * torch.norm(W, p=2) ** 2   
    #     loss_K = mse_loss + l2_reg                      

    #     return out, loss_K
    # --------------------------------------------------------------------------------------------#


    # ---------------------------------------------when predecting------------------------------------------#
    def forward(self,x):                               

        out = self.preact(x)                            # (b,2048,8,6),
        out = self.suffle1(out)                         # (b,512,16,12)
        out = self.duc1(out)                            # (b,256,32,24)
        out = self.duc2(out)                            # (b,128,64,48)

        out, HJ = self.KP(out)                          # out=(b,128,64,48),HJ=(b,H*W,128)
        out = self.conv_out(out)                        # (b,6,64,48)

        # prior = torch.tensor(prior)
        # K = compute_hog_features(prior)                 # shape = (b,3072,9)
        # print(K.shape,type(K))
        # K = K / (K.sum(dim=-1, keepdims=True) + 1e-6) 

        # HJ_projected = self.linear_W(HJ)               
        # print("-----------------------------",HJ_projected.shape,K.shape)
        # print(type(HJ_projected),type(K))
        # mse_loss = nn.MSELoss()(HJ, HJ)       
        W = self.linear_W.weight                        
        # l2_reg = self.beta * torch.norm(W, p=2) ** 2    
        # loss_K = mse_loss + l2_reg                      

        return out
    # --------------------------------------------------------------------------------------------#

    def _initialize(self):
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
