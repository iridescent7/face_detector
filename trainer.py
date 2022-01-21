import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from PIL import Image
from pathlib import Path
from timeit import default_timer as timer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms

from models import PNet, RNet, ONet

class FaceDataset(Dataset):
    def __init__(self, annotations, transforms):
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sp = self.annotations[idx].split(' ')

        img_path = sp[0]
        tag = int(sp[1])

        if tag == 0:
            offset = np.zeros(4, dtype=np.float32)
        else:
            offset = np.array(sp[2:6], dtype=np.float32)
        
        with Image.open(img_path) as pil_img:
            img = self.transforms(pil_img) # already transforms to (C, H, W)

        return img, tag, offset

def train(net, save_path, data_size, epochs, batch_size=512, use_cuda=True, summary_path='runs'):
    net_name = type(net).__name__

    # Glorot initialization
    # for p in net.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)

    if os.path.exists(save_path):
        while True:
            print(f'{net_name}: checkpoint already exist. continue training? [Y/n/a]: ', end='')

            ans = input().lower()

            if ans in ['', 'y']:
                net.load_state_dict(torch.load(save_path), strict=False)
                break

            if ans == 'n':
                break

            if ans == 'a':
                print('skipping..')
                return

    # Load dataset
    data_dir = Path(f'data/{data_size}')

    # ratio = 0.9
    train = []
    # val = []

    for name in ['pos', 'neg', 'part']:
        with (data_dir / (name + '.txt')).open('r') as file:
            train += file.readlines()

        # split = int(len(data) * ratio)
        # train += data[:split+1]
        # val += data[split+1:]

    # Shuffle pos, neg, part samples
    random.shuffle(train)

    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5)
    ])

    train_set = FaceDataset(train, train_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size)

    # Training
    os.makedirs(summary_path, exist_ok=True)

    index = len(os.listdir(summary_path))

    path = os.path.join(summary_path, f'{net_name}_{data_size}x{data_size}_exp{index}')
    summary_writer = SummaryWriter(path)

    if use_cuda:
        net.cuda()
    
    cls_loss_fn = nn.CrossEntropyLoss()
    offset_loss_fn = nn.SmoothL1Loss()

    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    best_loss = 0

    for epoch in range(1, epochs+1):
        total_loss = 0.

        start_time = timer()
        for img, tag, offset in train_loader:
            if use_cuda:
                img = img.cuda()
                tag = tag.cuda()
                offset = offset.cuda()

            out_tag, out_offset = net(img)

            out_tag = out_tag.view(-1, 2)
            out_offset = out_offset.view(-1, 4)

            # filter out part faces for classification
            tag_mask = torch.lt(tag, 2) # pos & neg samples
            tag_indices = torch.nonzero(tag_mask)[:, 0]

            cls_loss = None
            offset_loss = None

            if len(tag_indices) > 0:
                tag = tag[tag_indices]
                out_tag = out_tag[tag_indices]

                cls_loss = cls_loss_fn(out_tag, tag)

            # filter out negative samples for bbox regression
            offset_mask = torch.gt(tag, 0)
            offset_indices = torch.nonzero(offset_mask)[:, 0]

            if len(offset_indices) > 0:
                offset = offset[offset_indices]
                out_offset = out_offset[offset_indices]

                offset_loss = offset_loss_fn(out_offset, offset)

            # total loss
            if cls_loss is not None and offset_loss is not None:
                # put less weights in classification on R-Net & O-Net
                if net_name == 'PNet':
                    loss = cls_loss + offset_loss
                else:
                    loss = 0.5 * cls_loss + offset_loss
            elif cls_loss is not None:
                if net_name == 'PNet':
                    loss = cls_loss
                else:
                    loss = 0.5 * cls_loss                
            else:
                loss = offset_loss

            total_loss += loss.item()

            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = timer()
        epoch_time = end_time - start_time

        summary_writer.add_scalar('loss', total_loss)
        summary_writer.add_scalar('time', epoch_time)

        if epoch == 1 or total_loss < best_loss:
            best_loss = total_loss
            torch.save(net.state_dict(), save_path)

            print(f'{net_name}: saved at epoch: {epoch}, train loss: {total_loss}, time: {epoch_time:.2f}s')
        else:
            print(f'{net_name}: epoch: {epoch}, train loss: {total_loss}, time: {epoch_time:.2f}s')

if __name__ == '__main__':
    os.makedirs('train', exist_ok=True)

    # Retraining on wider dataset
    pnet = PNet()
    pnet.load_state_dict(torch.load(os.path.join('pretrain', 'pnet.pt')))
    train(pnet, 'train/pnet.pt', 12, 150, batch_size=8192)

    rnet = RNet()
    rnet.load_state_dict(torch.load(os.path.join('pretrain', 'rnet.pt')))
    train(rnet, 'train/rnet.pt', 24, 200, batch_size=4096)

    onet = ONet()
    onet.load_state_dict(torch.load(os.path.join('pretrain', 'onet.pt')), strict=False)
    train(onet, 'train/onet.pt', 48, 250, batch_size=1024)