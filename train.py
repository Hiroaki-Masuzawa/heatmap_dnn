
import os
import time
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

# headmap用データセット
class HeadmapDatasets(torch.utils.data.Dataset):
    def __init__(self, csv_path, csv_dir, input_size=None):
        self.df = pd.read_csv(csv_path, header=None)
        self.dir = csv_dir
        self.input_size = input_size
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        image_path = os.path.join(self.dir, self.df[0][idx])
        label_path = os.path.join(self.dir, self.df[1][idx])
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if self.input_size is not None:
            image = cv2.resize(image, self.input_size)
            label = cv2.resize(label, self.input_size, cv2.INTER_NEAREST)
        image = image.astype(np.float32)/255
        image = np.transpose(image, (2,0,1))
        label[label!=0]=1
        label = label.astype(np.int64)
        return image, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='',  # プログラム名
        usage='',  # プログラムの利用方法
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument('--traincsv', type=str, default='train.csv')
    parser.add_argument('--valcsv', type=str, default='val.csv')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--inputsize', type=int, nargs='*', default=None)
    parser.add_argument('--useamp', type=strtobool, default=0)
    parser.add_argument('--arch', type=str, default='Unet')
    parser.add_argument('--encoder', type=str, default='resnet18')
    parser.add_argument('--classnum', type=int, default=1)
    args = parser.parse_args()

    inputsize = args.inputsize
    if type(inputsize) is list:
        if len(inputsize) == 0:
            inputsize = None
        else :
            if len(inputsize) == 1:
                inputsize = inputsize+inputsize
            elif len(inputsize) > 2:
                inputsize = inputsize[:2]
            inputsize = tuple(inputsize)

    device = args.device
    date_string = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    outputdir = 'output-{}'.format(date_string) if args.output == '' else args.output
    use_amp = args.useamp == 1
    scaler = torch.amp.GradScaler(enabled=use_amp, init_scale=4096)

    # モデル定義
    # model = smp.Unet (
    #     encoder_name="resnet18", 
    #     encoder_weights="imagenet", 
    #     classes=1, 
    # )
    model = smp.create_model(
        arch = args.arch,
        encoder_name = args.encoder,
        encoder_weights="imagenet",
        classes=args.classnum
    )
    model.to(device)
    
    # データローダ準備
    abs_traincsv_path = os.path.abspath(args.traincsv)
    abs_valcsv_path = os.path.abspath(args.valcsv)
    trainset = HeadmapDatasets(args.traincsv, os.path.dirname(abs_traincsv_path), input_size = inputsize)
    valset = HeadmapDatasets(args.valcsv, os.path.dirname(abs_valcsv_path), input_size = inputsize)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size = args.batchsize, shuffle = True, num_workers = 8)
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size = 1, shuffle = False, num_workers = 2)

    # 評価関数，最適化関数定義
    loss_func = smp.losses.FocalLoss(mode='binary')
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
            pbar_epoch.set_description("[Epoch %d]" % (ep))
            # training 
            for i, (images, gt_masks) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
                images = images.to(device)
                gt_masks = gt_masks.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type=device, enabled=use_amp):
                    predicted_mask = model(images)
                    loss = loss_func(predicted_mask, gt_masks)
                # loss.backward()
                scaler.scale(loss).backward()
                optimizer.step()
                # print(ep, i, loss.cpu().item())
                writer.add_scalar("training_loss", loss.cpu().item(), itr_num)
                if itr_num % 100 == 0:
                    predicted_mask_work = predicted_mask.sigmoid()
                    predicted_mask_show = torch.cat([predicted_mask_work,predicted_mask_work,predicted_mask_work], dim=1)
                    gt_masks_work = torch.unsqueeze(gt_masks,1)
                    gt_masks_show = torch.cat([gt_masks_work, gt_masks_work, gt_masks_work], dim=1)
                    img_sample = (torch.cat([images, gt_masks_show, predicted_mask_show], dim=3)*255).to(torch.uint8)
                    writer.add_images("train_example", img_sample, itr_num, dataformats='NCHW')
                itr_num += 1
                if False: # for debug
                    save_images = []
                    for j in range(3):
                        alpha = 0.5
                        input_image = (images.cpu().numpy()[j]*255).astype(np.uint8).transpose((1,2,0))
                        label_heatmap = (gt_masks.cpu().numpy()[j]*255).astype(np.uint8)
                        pred_heatmap = (predicted_mask.cpu().detach().sigmoid().numpy()[j][0]*255).astype(np.uint8)
                        pred_heatmap_color = cv2.applyColorMap(pred_heatmap, cv2.COLORMAP_JET)
                        blend = cv2.addWeighted(input_image, alpha, pred_heatmap_color, 1-alpha, 0)
                        # save_image = cv2.hconcat([input_image, cv2.cvtColor(label_heatmap, cv2.COLOR_GRAY2BGR), cv2.cvtColor(pred_heatmap, cv2.COLOR_GRAY2BGR)])
                        # save_image = cv2.hconcat([input_image, cv2.cvtColor(label_heatmap, cv2.COLOR_GRAY2BGR), pred_heatmap_color])
                        save_image = cv2.hconcat([input_image, cv2.cvtColor(label_heatmap, cv2.COLOR_GRAY2BGR), cv2.cvtColor(pred_heatmap, cv2.COLOR_GRAY2BGR), blend])
                        save_images.append(save_image)
                    save_images2 = cv2.vconcat(save_images)
                    cv2.imwrite("test.png", save_images2)
                    # time.sleep(0.05)

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
            
            writer.add_scalar("validation/loss", np.mean(loss_list), ep+1)
            writer.add_scalar("validation/metric/mae", np.mean(metric_mae_list), ep+1)
            writer.add_scalar("validation/metric/mse", np.mean(metric_mse_list), ep+1)
            model.train()

            torch.save(model, os.path.join(outputdir, "./model_{0:03d}.pth".format(ep+1)))
    # end epoch loop
    writer.close()