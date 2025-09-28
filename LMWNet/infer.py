import torch
import torch.nn.functional as F
from utils.caculate_mertric import *
import numpy as np
import pdb, os, argparse

import time
import imageio
from model.LMWNet import LMWNet
from utils.data import test_dataset


torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=288, help='testing size')
opt = parser.parse_args()

dataset_path = '/home/dell/HJL/remote/datasets/'

model = LMWNet()
model.load_state_dict(torch.load('./models/LMWNet/LFAMNet_ORSSD_tiao.pth.best0068'))

model.cuda()
model.eval()

test_datasets = ['ORSSD']


for dataset in test_datasets:
    save_path = './results/' + 'LMWNet-' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/test-images/'
    gt_root = dataset_path + dataset + '/test-labels/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()


        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        # res, s1_sig, s2, s3, s3_sig= model(image)
        sal, sal_sig, res, s1_sig, s2, s2_sig, s3, s3_sig, s4, s4_sig, s5, s5_sig = model(image)
        #res, s1_2 = torch.chunk(res, 2, dim=0)
        time_end = time.time()
        time_sum = time_sum + (time_end - time_start)
        res = F.interpolate(sal, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        threshold = 0.5

        precisions, recalls = calculate_precision_recall(gt, res)
        fbeta_list = calculate_fbeta(precisions, recalls, 0.3)
        fbeta_mean = sum(fbeta_list) / len(fbeta_list)
        fbeta_total = []
        fbeta_total.append(fbeta_mean)


        mae = calculate_mae(res, gt)
        mae_total.append(mae)

        # Convert the result to uint8 type before saving
        res = (res * 255).astype(np.uint8)
        imageio.imsave(save_path + name, res)
        if i == test_loader.size - 1:

            fbeta = sum(fbeta_total) / len(fbeta_total)
            print("fbeta:", fbeta)

            mae_average = sum(mae_total) / len(mae_total)
            print("Average MAE:", mae_average)