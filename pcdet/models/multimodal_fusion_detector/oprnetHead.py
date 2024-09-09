import torch
import torch.nn as nn
from ...ops.patch_ops import patch_ops_utils
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
from torch.nn import functional as F
import numpy as np
from ..model_utils import network_utils
from ..fusion_layers import PointSample
from ..auxiliary import CenterNet

class ShortcutLayer(nn.Module):
    def __init__(self, input_channels, hidden_channels=256, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_channels, input_channels, kernel_size=1)

        self.norm1 = nn.BatchNorm1d(input_channels)
        self.norm2 = nn.BatchNorm1d(input_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, x):
        """
        :param x: (B, C, N)
        :return:
            (B, C, N)
        """
        x = x + self.dropout1(x)
        x = self.norm1(x)
        x2 = self.conv2(self.dropout2(self.activation(self.conv1(x))))
        x = x + self.dropout3(x2)
        x = self.norm2(x)
        return x


class AttnGNNLayer(nn.Module):
    def __init__(self, input_channels, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.out_channel = model_cfg.OUT_DIM
        mlps = model_cfg.MLPS
        self.use_feats_dist = model_cfg.USE_FEATS_DIS
        self.k = model_cfg.K
        self.edge_layes = nn.ModuleList()
        in_channels = input_channels
        for i in range(len(mlps)):
            self.edge_layes.append(
                nn.Sequential(
                    nn.Conv2d(in_channels*2, mlps[i], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlps[i]),
                    nn.ReLU()
                )
            )
            in_channels = mlps[i]
        in_channels = sum(mlps)
        self.calib = nn.Sequential(
            nn.Conv1d(in_channels, model_cfg.CALIB_DIM, kernel_size=1, bias=False),
            nn.BatchNorm1d(model_cfg.CALIB_DIM),
            nn.ReLU(),
            nn.Conv1d(model_cfg.CALIB_DIM, in_channels, kernel_size=1)
        )
        self.expansion = network_utils.make_fc_layers(model_cfg.EXP_MLPS, in_channels, linear=False)
        in_channels = model_cfg.EXP_MLPS[-1]
        self.reduction = nn.Sequential(
            nn.Conv1d(in_channels, self.out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channel),
            nn.ReLU()
        ) if model_cfg.USE_REDUCTION else None
        self.shortcut = ShortcutLayer(
            input_channels=self.out_channel, hidden_channels=self.out_channel, dropout=0.1
        ) if model_cfg.USE_SHORT_CUT else None

    def knn(self, x, k=8):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
        idx = pairwise_distance.topk(k=k, dim=-1)[1] 
        return idx

    def get_graph_feature(self, x, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)

        if idx is None:
            idx = self.knn(x, self.k)
        k = idx.shape[-1]
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, xyz, feats):
        B, M, K, _ = xyz.shape
        xyz = xyz.view(B * M, K, -1).permute(0, 2, 1).contiguous()
        feats = feats.view(B * M, K, -1).permute(0, 2, 1).contiguous()
        idx = self.knn(xyz, self.k) if not self.use_feats_dist else None
        x = torch.cat([xyz, feats], dim=1)
        x_list = []
        for edge_layer in self.edge_layes:
            x = self.get_graph_feature(x, idx)
            x = edge_layer(x)
            x = x.max(dim=-1)[0]
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = torch.sigmoid(self.calib(x)) * x
        x = self.expansion(x).max(dim=-1)[0].view(B, M, -1).permute(0, 2, 1)
        if self.reduction is not None:
            x = self.reduction(x)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x



class oprnetHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        self.pc_range = point_cloud_range
        patch_range = np.round(np.concatenate([point_cloud_range[:3] - 1, point_cloud_range[3:] + 1]))
        patch_size = np.array([1.0, 1.0, -1.0], dtype=np.float32)

        dfvs_config = model_cfg.DFVS_CONFIG
        self.roilocal_dfvs_pool3d_layer = patch_ops_utils.RoILocalDFVSPool3dV2(
            pc_range=patch_range, 
            patch_size=patch_size, 
            num_dvs_points=dfvs_config.NUM_DVS_POINTS,
            num_fps_points=dfvs_config.NUM_FPS_POINTS, 
            hash_size=dfvs_config.HASH_SIZE,
            lambda_=dfvs_config.LAMBDA,
            delta=dfvs_config.DELTA,
            pool_extra_width=dfvs_config.POOL_EXTRA_WIDTH,
            num_boxes_per_patch=dfvs_config.NUM_BOXES_PER_PATCH
        )
        
 
        self.image_fusion = image_fusion_block()
        self.centernet = CenterNet(None)#.to('cuda:3')
        
        img_config = model_cfg.get('IMG_CONFIG', None)
        if img_config is not None:
            mlps = [img_config.IN_DIM] + img_config.MLPS
            img_convs = []
            for k in range(0, mlps.__len__() - 1):
                img_convs.extend([
                    nn.Conv2d(mlps[k], mlps[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlps[k + 1]),
                    nn.ReLU()
                ])
            self.img_conv = nn.Sequential(*img_convs)
            self.point_sample = PointSample()
            self.use_img = True
        else:
            self.use_img = False
            
        input_channels = model_cfg.ATTN_GNN_CONFIG.pop('IN_DIM')
        self.attn_gnn_layer = AttnGNNLayer(input_channels, model_cfg.ATTN_GNN_CONFIG)

        self.shared_fc_layer = nn.Sequential(
            nn.Conv1d(self.attn_gnn_layer.out_channel, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.cls_layers = nn.Conv1d(256, self.num_class, kernel_size=1, bias=True)
        self.reg_layers = nn.Conv1d(256, self.box_coder.code_size, kernel_size=1, bias=True)
        
        self.init_weights(weight_init='xavier')
        
        
    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers.weight, mean=0, std=0.001)

    def roipool3d_gpu(self, batch_dict):
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        rois = batch_dict['rois']

        pooled_feats_local_list = []
        pooled_pts_num_list = []
        for batch_idx in range(batch_size):
            cur_points = points[points[:, 0] == batch_idx][:, 1:]
            ndim = 2
            pc_range = cur_points.new_tensor(self.pc_range)
            keep = torch.all((cur_points[:, :ndim] >= pc_range[:ndim]) & (cur_points[:, :ndim] <= pc_range[3:3 + ndim]), dim=-1)
            cur_points = cur_points[keep, :]
            cur_points = F.pad(cur_points, (1, 0), mode='constant', value=0).contiguous()
            cur_rois = rois[batch_idx][:, :7].unsqueeze(0).contiguous()
            pooled_pts_idx, pooled_pts_num = self.roilocal_dfvs_pool3d_layer(
                cur_points[:, :4].contiguous(),
                cur_rois
            )
            pooled_feats_local = patch_ops_utils.gather_features(cur_points[:, 1:], pooled_pts_idx, pooled_pts_num)
            pooled_feats_local_list.append(pooled_feats_local)
            pooled_pts_num_list.append(pooled_pts_num)
            
        pooled_feats_local = torch.cat(pooled_feats_local_list, dim=0)
        pooled_feats_global = pooled_feats_local.clone()[..., :3].view(-1, pooled_feats_local.shape[-2], 3) 
        pooled_pts_num = torch.cat(pooled_pts_num_list, dim=0)

        pooled_feats_local[..., :3] -= rois[..., :3].unsqueeze(dim=2)
        pooled_feats_local = pooled_feats_local.view(-1, pooled_feats_local.shape[-2], pooled_feats_local.shape[-1])
        pooled_feats_local[..., :3] = common_utils.rotate_points_along_z(
            pooled_feats_local[..., :3], -rois.view(-1, rois.shape[-1])[:, 6]
        )
        # (B, M, 1, 6)
        local_corners = torch.stack([
            -rois[..., 3:4] / 2, -rois[..., 4:5] / 2, -rois[..., 5:6] / 2,
            rois[..., 3:4] / 2, rois[..., 4:5] / 2, rois[..., 5:6] / 2
        ], dim=-1)
        # (B*M, K, 3+C+6)
        pooled_feats_local = torch.cat([pooled_feats_local, local_corners.view(-1, 1, 6).repeat(1, pooled_feats_local.shape[-2], 1)], dim=-1)
        pooled_pts_num = pooled_pts_num.view(-1)  # (B*M)

        return pooled_feats_local, pooled_feats_global, pooled_pts_num

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:

        """
        from time import time
        start = time()
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        B, M, _ = batch_dict['rois'].shape
        roi_feats_local, roi_feats_global, roi_points_num  = self.roipool3d_gpu(batch_dict)
        
        if self.use_img:

            self.image_fusion(batch_dict)
            batch_dict['image_features'] = self.img_conv(batch_dict['image_features'])  # (B, 32, H/4, W/4)
            batch_dict['ct_results'] = self.centernet(batch_dict['image_features']) 
            batch_dict['sampled_points'] = roi_feats_global.view(B, -1, 3)  # (B, M*K, 3)
            roi_img_feats = self.point_sample(batch_dict)
            roi_img_feats = roi_img_feats.view(B * M, -1, roi_img_feats.shape[-1])  # (B*M, K, C)
            
            batch_dict['img_feat'] = roi_img_feats
            batch_dict['point_feat'] = roi_feats_local
            
            
            roi_feats_local = torch.cat([roi_feats_local, roi_img_feats], dim=-1)

        roi_feats_local = roi_feats_local * (roi_points_num > 0).unsqueeze(-1).unsqueeze(-1)
        roi_feats_local = roi_feats_local.view(B, M, -1, roi_feats_local.shape[-1])  # (B, M, K, C)
        roi_point_xyz = roi_feats_local[..., :3]
        roi_point_feats = roi_feats_local[..., 3:]


        pooled_features = self.attn_gnn_layer(roi_point_xyz, roi_point_feats)

        shared_features = self.shared_fc_layer(pooled_features)
        rcnn_cls = self.cls_layers(shared_features).permute(0, 2, 1).contiguous().view(B * M, -1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).permute(0, 2, 1).contiguous().view(B * M, -1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            # (B, N, 1)
            roi_scores_normalized = batch_dict['roi_scores'] if batch_dict.get('cls_preds_normalized', False) else torch.sigmoid(batch_dict['roi_scores'])
            batch_dict['batch_cls_preds'] = torch.pow(torch.sigmoid(batch_cls_preds), 0.5) * torch.pow(roi_scores_normalized.unsqueeze(-1), 0.5)
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = True
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['batch_size'] = batch_dict['batch_size']

            self.forward_ret_dict = targets_dict
        return batch_dict
    
class image_fusion_block(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(512), num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(4)])
        self.upsample_3 = nn.Upsample(size=None, scale_factor=2,mode='nearest')
        self.reduce_chan_level3 = nn.Conv2d(int(768), int(256), kernel_size=1, bias=False)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(256), num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(4)])

        self.reduce_chan_level2 = nn.Conv2d(int(384), int(128), kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(128), num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(4)])
        
        self.reduce_chan_level1 = nn.Conv2d(int(192), int(64), kernel_size=1, bias=False)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(64), num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(2)])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(64), num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(2)])
            
        self.output = nn.Conv2d(int(64), 3, kernel_size=3, stride=1, padding=1, bias=False) 
        self.img_feature_fusion1 = cross_AttnBlock(256)
        self.img_feature_fusion2 = cross_AttnBlock(512)
        self.img_feature_fusion3 = cross_AttnBlock(1024)
        self.fusion1conv = nn.Conv2d(384, 128, 1, bias=False)
        self.fusion2conv = nn.Conv2d(512, 256, 1, bias=False)
        self.fusion3conv = nn.Conv2d(512, 256, 1, bias=False)
        self.fusion1 = nn.Conv2d(256, 64, 3, 1, 1, bias=False)
        
        
        
    def forward(self, data_dict):
        from time import time
        start = time()
        y_recon4 = self.latent(data_dict['skip_feat4'])
        y_recon = self.upsample_3(y_recon4)
        y_recon = torch.cat([y_recon, data_dict['skip_feat3']], 1)
        y_recon = self.reduce_chan_level3(y_recon)
        y_recon3 = self.decoder_level3(y_recon)
        
        y_recon = self.upsample_3(y_recon3)
        y_recon = torch.cat([y_recon, data_dict['skip_feat2']], 1)
        y_recon = self.reduce_chan_level2(y_recon)
        y_recon2 = self.decoder_level2(y_recon)
        y_recon = self.upsample_3(y_recon2)
        y_recon = torch.cat([y_recon, data_dict['skip_feat1']], 1)
        y_recon = self.reduce_chan_level1(y_recon)
        y_recon = self.decoder_level1(y_recon)
        y_recon1 = self.refinement(y_recon)
        y_recon = self.output(y_recon1)
        data_dict['recon_image'] = y_recon

        img_f3 = self.img_feature_fusion3(torch.cat([data_dict['skip_feature'][3],y_recon4],dim=1))
        img_f3 = self.upsample_3(self.fusion3conv(img_f3))
        
        img_f2 = self.img_feature_fusion2(torch.cat([data_dict['skip_feature'][2],y_recon3],dim=1))
        img_f2 = self.upsample_3(self.fusion2conv(torch.cat([img_f2, img_f3],dim=1)))        
        
        img_f1 = self.img_feature_fusion1(torch.cat([data_dict['skip_feature'][1],y_recon2],dim=1))
        img_f1 = self.upsample_3(self.fusion1conv(torch.cat([img_f1,img_f2],dim=1)))        
        
        
        data_dict['image_features']= self.fusion1(torch.cat([img_f1, data_dict['skip_feature'][0], y_recon1], dim=1))

        return data_dict
    
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)
    
class cross_AttnBlock(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.c1 = c1
        channel = int(c1/2)
        self.norm = Normalize(in_channels= channel)
        self.q = torch.nn.Conv2d(channel, #hannel,
                                 channel,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(channel,
                                 channel,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(channel,
                                 channel,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(channel,
                                        channel,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, xy):
        x, y = torch.split(xy, int(xy.shape[1]/2), dim=1)
        h_ = x
        h_ = self.norm(h_)
        y = self.norm(y)
        q = self.q(y)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape


        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1) 
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        
        return x+h_
    

import torch.nn.functional as F
from einops import rearrange
import numbers
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)  
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)