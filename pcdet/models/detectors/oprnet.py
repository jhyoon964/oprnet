from .detector3d_template import Detector3DTemplate
import torch
from torchvision.transforms import Resize
from .perceptual_loss import VGGPerceptualLoss as p_loss
import numpy as np
from ..aux_2d_loss.loss import Loss


class oprnet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, logger):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger)
        self.module_list = self.build_networks()
        if self.model_cfg.get('FREEZE_LAYERS', None) is not None:
            self.freeze(self.model_cfg.FREEZE_LAYERS)
        self.loss_recon = torch.nn.L1Loss()
        self.losser = Loss()
        self.p_loss = p_loss().cuda()        
        self.k = 0
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        self.k += 1
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            h, w = int(batch_dict['gt_image'].shape[2]/4), int(batch_dict['gt_image'].shape[3]/4)
            torch_resize = Resize((h,w))   
            
            gt_ = self.transform_annotation(batch_dict)
            ctlosses = self.losser(batch_dict['ct_results'], gt_)
            ctloss = sum(ctlosses)            
            
            
            
            loss1 = self.loss_recon(batch_dict['recon_image'],torch_resize(batch_dict['gt_image']))  
            loss1 += self.p_loss(batch_dict['recon_image'],torch_resize(batch_dict['gt_image'])) * 0.01        
                 

            loss += loss1 *0.1 + ctloss * 0.1          
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict, batch_dict['img_feat'], batch_dict['point_feat']
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, batch_dict['img_feat'], batch_dict['point_feat']



    def get_training_loss(self):
        disp_dict, tb_dict = {}, {}
        loss = 0
        if self.model_cfg.get('FREEZE_LAYERS', None) is None:
            if self.dense_head is not None:
                loss_rpn, tb_dict = self.dense_head.get_loss(tb_dict)
            else:
                loss_rpn, tb_dict = self.point_head.get_loss(tb_dict)
            loss += loss_rpn

        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss += loss_rcnn
        
        return loss, tb_dict, disp_dict

    
    
    def transform_annotation(self, batch):
        down_stride = 4
        
        
        img = batch['image'].squeeze(0).permute(1,2,0).to('cpu').numpy()
        label = batch['gt_boxes2d'].to('cpu') 
        raw_h, raw_w, _ = img.shape
        info = {'raw_height': raw_h, 'raw_width': raw_w}
                
        boxes = np.array(label[:,:,:4])
        boxes = np.squeeze(boxes, axis=0)
        
        boxes_w, boxes_h = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]

        ct = np.array([(boxes[..., 0] + boxes[..., 2]) / 2,
                       (boxes[..., 1] + boxes[..., 3]) / 2], dtype=np.float32).T

        info['resize_height'], info['resize_width'] = img.shape[:2]

        classes = label[:,:,4].to(torch.int64)
        classes = np.squeeze(classes, axis=0)
        boxes = torch.from_numpy(boxes).float()
        classes = torch.LongTensor(classes)

        output_h, output_w = info['resize_height'] // down_stride, info['resize_width'] // down_stride  
        boxes_h, boxes_w, ct = boxes_h / down_stride, boxes_w / down_stride, ct / down_stride        
        hm = np.zeros((1, output_h, output_w), dtype=np.float32)

        ct[:, 0] = np.clip(ct[:, 0], 0, output_w - 1)
        ct[:, 1] = np.clip(ct[:, 1], 0, output_h - 1)
        info['gt_hm_height'], info['gt_hm_witdh'] = output_h, output_w
        obj_mask = torch.ones(len(classes))
        
        for i, cls_id in enumerate(classes):
            radius = gaussian_radius((np.ceil(boxes_h[i]), np.ceil(boxes_w[i])))
            radius = max(0, int(radius))
            ct_int = ct[i].astype(np.int32)
            if (hm[:, ct_int[1], ct_int[0]] == 1).sum() >= 1.:
                obj_mask[i] = 0
                continue

            draw_umich_gaussian(hm[cls_id - 1], ct_int, radius)
            if np.any(hm[cls_id - 1, ct_int[1], ct_int[0]] != 1):
                obj_mask[i] = 0
            
        hm = torch.from_numpy(hm)
        
        obj_mask = obj_mask.eq(1)
        boxes = boxes[obj_mask]
        classes = classes[obj_mask]
        info['ct'] = torch.tensor(ct)[obj_mask]

        assert hm.eq(1).sum().item() == len(classes) == len(info['ct']), \
            f"index: {index}, hm peer: {hm.eq(1).sum().item()}, object num: {len(classes)}"
        return batch['image'], boxes, classes, hm, info
        
def flip(img):
    return img[:, :, ::-1].copy()


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap
