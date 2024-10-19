import os
import time
import yaml
import json
import torch
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import datetime
import numpy as np
import pandas as pd
import cv2
import argparse
from setuptools._distutils.util import strtobool

from collections import OrderedDict

# headmap用データセット
class HeadmapYAMLDatasets(torch.utils.data.Dataset):
    def __init__(self, annotatefile_path, input_size=None, anno_size=5):
        if annotatefile_path.split('.')[-1] == 'yaml':
            with open(annotatefile_path, encoding="utf-8") as f:
                self.dat = yaml.safe_load(f)
        elif annotatefile_path.split('.')[-1] == 'json':
            with open(annotatefile_path, encoding="utf-8") as f:
                self.dat = json.load(f)
        self.dir = os.path.abspath(os.path.dirname(annotatefile_path))
        self.input_size = input_size
        self.anno_size = anno_size

    def __len__(self):
        return len(self.dat['annotations'])

    def __getitem__(self, idx):
        # print(self.dat['annotations'][idx]["imagefile"])
        image = cv2.imread(os.path.join(self.dir, self.dat['annotations'][idx]["imagefile"]))
        if self.input_size is not None:
            image = cv2.resize(image, self.input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        label = np.zeros(list(image.shape[0:2]) + [2], dtype=np.uint8)
        label_tmp = np.zeros(label.shape[0:2], dtype=np.uint8)

        for p in np.array(self.dat['annotations'][idx]['convex_vertex']).reshape((-1, 3)):
            if p[2] < 0.5:
                continue
            label_tmp = cv2.circle(label_tmp, (int(p[0]), int(p[1])), self.anno_size, 255, -1)
        label[:, :, 0][label_tmp != 0] = 1
        label_tmp = np.zeros(label.shape[0:2], dtype=np.uint8)
        for p in np.array(self.dat['annotations'][idx]['concave_vertex']).reshape((-1, 3)):
            if p[2] < 0.5:
                continue
            label_tmp = cv2.circle(label_tmp, (int(p[0]), int(p[1])), self.anno_size, 255, -1)
        label[:, :, 1][label_tmp != 0] = 1

        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))
        return image, label


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='',  # プログラム名
        usage='',  # プログラムの利用方法
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument('--trainyaml', type=str, default='/dataset/puzzle_block/train/annotation.yaml')
    parser.add_argument('--valyaml', type=str, default='')
    parser.add_argument('--batchsize', type=int, default=25)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--inputsize', type=int, nargs='*', default=None)
    parser.add_argument('--useamp', type=strtobool, default=0)
    parser.add_argument('--arch', type=str, default='Unet')
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--classnum', type=int, default=2)
    args = parser.parse_args()

    inputsize = args.inputsize
    if type(inputsize) is list:
        if len(inputsize) == 0:
            inputsize = None
        else:
            if len(inputsize) == 1:
                inputsize = inputsize + inputsize
            elif len(inputsize) > 2:
                inputsize = inputsize[:2]
            inputsize = tuple(inputsize)

    device = args.device
    date_string = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    outputdir = 'output-{}'.format(date_string) if args.output == '' else args.output
    use_amp = args.useamp == 1
    scaler = torch.amp.GradScaler(enabled=use_amp, init_scale=4096)

    model = smp.create_model(
        arch=args.arch,
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        classes=args.classnum
    )
    model.to(device)

    # データローダ準備
    trainset = HeadmapYAMLDatasets(args.trainyaml, input_size=inputsize)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=8)
    if args.valyaml != "":
        valset = HeadmapYAMLDatasets(args.valyaml, input_size=inputsize)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

    # 評価関数，最適化関数定義
    loss_func = smp.losses.FocalLoss(mode='binary' if args.classnum == 1 else 'multilabel')
    metric_mae_func = torch.nn.L1Loss()
    metric_mse_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=args.lr)])

    # 出力先ディレクトリ作成
    os.makedirs(outputdir, exist_ok=False)

    # loggerロガー準備
    writer = SummaryWriter(log_dir=outputdir)
    itr_num = 0

    with tqdm(range(args.epoch)) as pbar_epoch:
        for ep in pbar_epoch:
            pbar_epoch.set_description("[Epoch %d]" % (ep + 1))
            # training
            with tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False) as pbar_itr:
                for i, (images, gt_masks) in pbar_itr:
                    images = images.to(device)
                    gt_masks = gt_masks.to(device)
                    optimizer.zero_grad()
                    with torch.autocast(device_type=device, enabled=use_amp):
                        predicted_mask = model(images)
                        loss = loss_func(predicted_mask, gt_masks)
                    # loss.backward()
                    scaler.scale(loss).backward()
                    optimizer.step()
                    loss_value = loss.cpu().item()
                    # print(ep, i, loss.cpu().item())
                    writer.add_scalar("training_loss", loss_value, itr_num)
                    pbar_itr.set_postfix(OrderedDict(loss=loss_value))
                    if itr_num % 100 == 0:
                        predicted_mask_work = predicted_mask.sigmoid()
                        if args.classnum == 1:
                            predicted_mask_show = torch.cat([predicted_mask_work, predicted_mask_work, predicted_mask_work], dim=1)
                            gt_masks_work = torch.unsqueeze(gt_masks, 1)
                            gt_masks_show = torch.cat([gt_masks_work, gt_masks_work, gt_masks_work], dim=1)
                        else:
                            shape = (int(predicted_mask_work.shape[0]), 3, int(predicted_mask_work.shape[2]), int(predicted_mask_work.shape[3]))
                            predicted_mask_show = torch.zeros(*shape).to(predicted_mask_work.device).to(torch.float32)
                            predicted_mask_show[:, 0:2] = predicted_mask_work
                            shape = (int(gt_masks.shape[0]), 3, int(gt_masks.shape[2]), int(gt_masks.shape[3]))
                            gt_masks_show = torch.zeros(*shape).to(gt_masks.device).to(torch.float32)
                            gt_masks_show[:, 0:2] = gt_masks

                        # print(images.shape, gt_masks_show.shape, predicted_mask_show.shape)
                        img_sample = (torch.cat([images, gt_masks_show, predicted_mask_show], dim=3) * 255).to(torch.uint8)
                        writer.add_images("train_example", img_sample, itr_num, dataformats='NCHW')
                    itr_num += 1

            if args.valyaml != "":
                # val datasetを用いてloss, metricを確認
                model.eval()
                loss_list = []
                metric_mae_list = []
                metric_mse_list = []
                for i, (images, gt_masks) in enumerate(val_dataloader):
                    images = images.to(device)
                    gt_masks = gt_masks.to(device)
                    with torch.no_grad():
                        predicted_mask = model(images)
                        loss = loss_func(predicted_mask, gt_masks)
                        predicted_mask_sigmoid = predicted_mask.sigmoid()
                        metric_mae = metric_mae_func(torch.squeeze(predicted_mask_sigmoid, dim=1), gt_masks)
                        metric_mse = metric_mse_func(torch.squeeze(predicted_mask_sigmoid, dim=1), gt_masks)
                    loss_list.append(loss.item())
                    metric_mae_list.append(metric_mae.item())
                    metric_mse_list.append(metric_mse.item())
                # print(ep+1, np.mean(loss_list), np.mean(metric_list))

                writer.add_scalar("validation/loss", np.mean(loss_list), ep + 1)
                writer.add_scalar("validation/metric/mae", np.mean(metric_mae_list), ep + 1)
                writer.add_scalar("validation/metric/mse", np.mean(metric_mse_list), ep + 1)
                model.train()

            torch.save(model, os.path.join(outputdir, "model_{0:03d}.pth".format(ep + 1)))
    # end epoch loop
    writer.close()
