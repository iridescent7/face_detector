import torch
import torch.nn as nn
import numpy as np
import os

from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from torchvision.ops.boxes import batched_nms

from models import PNet, RNet, ONet

class MTCNN(nn.Module):
    def __init__(
        self, image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], scale_factor=0.7,
        keep_all=True, select_method='largest', device=None
    ):
        super().__init__()

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.scale_factor = scale_factor
        self.keep_all = keep_all
        self.select_method = select_method

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        if True:
            train_path = 'train'
            self.pnet.load_state_dict(torch.load(os.path.join(train_path, 'pnet.pt')))
            self.rnet.load_state_dict(torch.load(os.path.join(train_path, 'rnet.pt')))
            self.onet.load_state_dict(torch.load(os.path.join(train_path, 'onet.pt')), strict=False)

        self.device = torch.device('cpu')

        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, img, prob=False):
        batch_boxes, batch_probs = self.detect(img)

        if not self.keep_all:
            batch_boxes, batch_probs = self.select_boxes(batch_boxes, batch_probs)

        batch_boxes = np.array(batch_boxes, dtype=np.int32)

        if prob:
            return batch_boxes, batch_probs
        else:
            return batch_boxes

    def select_boxes(self, batch_boxes, batch_probs):
        sel_boxes, sel_probs = [], []

        for boxes, probs in zip(batch_boxes, batch_probs):
            if boxes is None:
                sel_boxes.append(None)
                sel_probs.append([None])
                continue
            
            boxes = np.array(boxes)
            probs = np.array(probs)
                
            if self.select_method == 'largest':
                box_order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[::-1]
            elif self.select_method == 'probability':
                box_order = np.argsort(probs)[::-1]
           
            sel_boxes.append(boxes[box_order][[0]])
            sel_probs.append(probs[box_order][[0]])

        sel_boxes = np.array(sel_boxes)
        sel_probs = np.array(sel_probs)

        return sel_boxes, sel_probs

    def detect(self, img):
        with torch.no_grad():
            batch_boxes = self.detect_face(img)

        if len(batch_boxes) == 1:
            boxes = batch_boxes[0, :, :4]
            probs = batch_boxes[0, :, 4]
        else:
            boxes, probs = [], []
            for box in batch_boxes:
                box = np.array(box)

                if len(box) == 0:
                    boxes.append(None)
                    probs.append([None])
                else:
                    boxes.append(box[:, :4])
                    probs.append(box[:, 4])

            boxes = np.array(boxes)
            probs = np.array(probs)

        return boxes, probs

    def detect_face(self, imgs):
        if isinstance(imgs, (np.ndarray, torch.Tensor)):
            if isinstance(imgs,np.ndarray):
                imgs = torch.as_tensor(imgs.copy(), device=self.device)

            if isinstance(imgs,torch.Tensor):
                imgs = torch.as_tensor(imgs, device=self.device)

            if len(imgs.shape) == 3:
                imgs = imgs.unsqueeze(0)
        else:
            if not isinstance(imgs, (list, tuple)):
                imgs = [imgs]

            if any(img.size != imgs[0].size for img in imgs):
                raise Exception('mtcnn batch processing only works on equally sized images')

            imgs = np.stack([np.uint8(img) for img in imgs])
            imgs = torch.as_tensor(imgs.copy(), device=self.device)

        # imgs = torch.as_tensor(imgs, device=self.device)

        # # single image
        # if len(imgs.shape) == 3:
        #     imgs = imgs.unsqueeze(0)

        model_dtype = next(self.pnet.parameters()).dtype
        imgs = imgs.permute(0, 3, 1, 2).type(model_dtype) # _, C, H, W

        batch_size = len(imgs)
        h, w = imgs.shape[2:4]
        m = 12.0 / self.min_face_size
        min1 = min(h, w) * m

        # Scale pyramid
        scale_i = m
        scales = []

        while min1 >= 12:
            scales.append(scale_i)
            scale_i = scale_i * self.scale_factor
            min1 = min1 * self.scale_factor

        # First stage
        boxes = []
        image_inds = []

        scale_picks = []
        offset = 0

        for scale in scales:
            im_data = img_resize(imgs, (int(h*scale+1), int(w*scale+1)))
            im_data = (im_data - 127.5) * (1./128)

            prob, reg = self.pnet(im_data)

            boxes_scale, image_inds_scale = generatebbox(reg, prob[:, 1], scale, self.thresholds[0])
            boxes.append(boxes_scale)
            image_inds.append(image_inds_scale)

            pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5)
            scale_picks.append(pick + offset)
            offset += boxes_scale.shape[0]

        boxes = torch.cat(boxes, dim=0)
        image_inds = torch.cat(image_inds, dim=0)

        scale_picks = torch.cat(scale_picks, dim=0)

        # Filter according to picked NMS boxes (each separate scale, image)
        boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]

        # Perform NMS again on all scaled bboxes (each separate image)
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds = boxes[pick], image_inds[pick]

        regw = boxes[:, 2] - boxes[:, 0]
        regh = boxes[:, 3] - boxes[:, 1]

        q1 = boxes[:, 0] + boxes[:, 5] * regw
        q2 = boxes[:, 1] + boxes[:, 6] * regh
        q3 = boxes[:, 2] + boxes[:, 7] * regw
        q4 = boxes[:, 3] + boxes[:, 8] * regh

        boxes = torch.stack([q1, q2, q3, q4, boxes[:, 4]]).permute(1, 0)
        boxes = rerec(boxes)

        y, ey, x, ex = pad(boxes, w, h)

        # Second stage
        if len(boxes) > 0:
            # List of cropped PNet bbox results
            im_data = []
            for k in range(len(y)):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                    im_data.append(img_resize(img_k, (24, 24)))

            im_data = torch.cat(im_data, dim=0)
            im_data = (im_data - 127.5) * (1./128)

            # Equals to out = rnet(im_data) with batching
            prob, reg = feed_batch(im_data, self.rnet)

            # transpose
            prob = prob.permute(1, 0)
            reg = reg.permute(1, 0)

            score = prob[1, :]
            ipass = score > self.thresholds[1]
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)

            image_inds = image_inds[ipass]
            mv = reg[:, ipass].permute(1, 0)

            # NMS on each image
            pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
            boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]

            boxes = bbreg(boxes, mv)
            boxes = rerec(boxes)

        # Third stage
        if len(boxes) > 0:
            y, ey, x, ex = pad(boxes, w, h)

            im_data = []
            for k in range(len(y)):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                    im_data.append(img_resize(img_k, (48, 48)))

            im_data = torch.cat(im_data, dim=0)
            im_data = (im_data - 127.5) * (1./128)

            prob, reg = feed_batch(im_data, self.onet)

            prob = prob.permute(1, 0)
            reg = reg.permute(1, 0)

            score = prob[1, :]
            ipass = score > self.thresholds[2]

            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
            image_inds = image_inds[ipass]
            mv = reg[:, ipass].permute(1, 0)

            boxes = bbreg(boxes, mv)

            pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
            boxes, image_inds = boxes[pick], image_inds[pick]

        boxes = boxes.cpu().numpy()
        image_inds = image_inds.cpu().numpy()

        batch_boxes = []

        for b_i in range(batch_size):
            b_i_inds = np.where(image_inds == b_i)
            batch_boxes.append(boxes[b_i_inds].copy())

        batch_boxes = np.array(batch_boxes)

        return batch_boxes

def img_resize(img, sz):
    return interpolate(img, size=sz, mode='area')

def bbreg(bbox, reg):
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = bbox[:, 2] - bbox[:, 0] + 1
    h = bbox[:, 3] - bbox[:, 1] + 1

    b1 = bbox[:, 0] + reg[:, 0] * w
    b2 = bbox[:, 1] + reg[:, 1] * h
    b3 = bbox[:, 2] + reg[:, 2] * w
    b4 = bbox[:, 3] + reg[:, 3] * h

    bbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    return bbox

def generatebbox(reg, probs, scale, thresh):
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]

    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)

    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()

    bbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)

    return bbox, image_inds

def rerec(bbox):
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    
    l = torch.max(w, h)
    bbox[:, 0] = bbox[:, 0] + w * 0.5 - l * 0.5
    bbox[:, 1] = bbox[:, 1] + h * 0.5 - l * 0.5
    bbox[:, 2:4] = bbox[:, :2] + l.repeat(2, 1).permute(1, 0)

    return bbox

def pad(boxes, w, h):
    boxes = boxes.trunc().int().cpu().numpy()

    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y, ey, x, ex

def feed_batch(im_data, model):
    batch_size = 512
    out = []
    
    for i in range(0, len(im_data), batch_size):
        batch = im_data[i:(i+batch_size)]
        out.append(model(batch))

    return tuple(torch.cat(v, dim=0) for v in zip(*out))
