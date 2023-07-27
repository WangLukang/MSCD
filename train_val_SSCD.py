import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import pandas as pd

import numpy as np
import glob
import os

from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalCDDataset

# from model import BASNet
from model import MSCDNet_v1, MSCDNet_v2
import pytorch_ssim
import pytorch_iou

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def calMetric_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict==1)
    fn = np.sum(label == 1)
    return tp,fp+fn-tp

def main():
    # ------- 1. define loss function --------

    bce_loss = nn.BCELoss(size_average=True)
    ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
    iou_loss = pytorch_iou.IOU(size_average=True)

    def bce_ssim_loss(pred,target):

        bce_out = bce_loss(pred,target)
        ssim_out = 1 - ssim_loss(pred,target)
        iou_out = iou_loss(pred,target)

        loss = bce_out + ssim_out + iou_out

        return loss

    def m_bce_loss(pred,target):
        w1 = torch.abs(F.avg_pool2d(target, kernel_size=3, stride=1, padding=1) - target)

        bce_out = bce_loss(pred,target)
        # ssim_out = 1 - ssim_loss(pred,target)
        # iou_out = iou_loss(pred,target)

        # loss = bce_out + ssim_out + iou_out
        loss = bce_out
        return loss

    def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, labels_v):

        loss0 = bce_ssim_loss(d0,labels_v)
        loss1 = bce_ssim_loss(d1,labels_v)
        loss2 = bce_ssim_loss(d2,labels_v)
        loss3 = bce_ssim_loss(d3,labels_v)
        loss4 = bce_ssim_loss(d4,labels_v)
        loss5 = bce_ssim_loss(d5,labels_v)
        # loss6 = bce_ssim_loss(d6,labels_v)
        # loss7 = bce_ssim_loss(d7,labels_v)
        #ssim0 = 1 - ssim_loss(d0,labels_v)

        # iou0 = iou_loss(d0,labels_v)
        #loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
        # loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7#+ 5.0*lossa
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 #+ 5.0*lossa
        # loss = loss0 + 0.5*(loss1 + loss2 + loss3 + loss4) + loss5#+ 5.0*lossa
        # loss = loss0 + loss1#+ 5.0*lossa
        # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))
        # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item()))

        # print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))

        return loss0, loss

    # ------- 2. set the directory of training dataset --------
    # data_dir = 'C:/Users/11473/OneDrive/桌面/code/MSCDNet_20220403/DATA/landslide_paper/stage1cd/'
    data_dir = 'C:/Users/11473/OneDrive/桌面/code/MSCDNet_20220403/DATA/BCD_removeblank_split/'
  
    tra_image_dirA = 'trainAUG/A/'
    tra_image_dirB = 'trainAUG/B/'
    tra_label_dir = 'trainAUG/label/'

    val_image_dirA = 'val/A/'
    val_image_dirB = 'val/B/'
    val_label_dir = 'val/label/'
    # image_ext = '.tif'
    image_ext = '.png'
    label_ext = '.png'
    
  
    model_dir = "epochs/BCD/checkpoint_AUG/"
    sta_dir = "statistics/BCD_checkpoint_AUG.csv"

    os.makedirs(model_dir, exist_ok=True)


    
    epoch_num = 100
    batch_size_train = 8
    batch_size_val = 1


    tra_img_name_listA = glob.glob(data_dir + tra_image_dirA + '*' + image_ext)
    tra_img_name_listB = glob.glob(data_dir + tra_image_dirB + '*' + image_ext)
    tra_lbl_name_list = glob.glob(data_dir + tra_label_dir + '*' + image_ext)

    print("---")
    print("train imagesA: ", len(tra_img_name_listA))
    print("train imagesB: ", len(tra_img_name_listB))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    val_img_name_listA = glob.glob(data_dir + val_image_dirA + '*' + label_ext)
    val_img_name_listB = glob.glob(data_dir + val_image_dirB + '*' + label_ext)
    val_lbl_name_list = glob.glob(data_dir + val_label_dir + '*' + label_ext)

    salobj_dataset = SalCDDataset(
        img_name_listA=tra_img_name_listA,
        img_name_listB=tra_img_name_listB,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(256),
            RandomCrop(224),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

    salobj_dataset_val = SalCDDataset(
        img_name_listA=val_img_name_listA,
        img_name_listB=val_img_name_listB,
        lbl_name_list=val_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(256),
            # RandomCrop(224),
            ToTensorLab(flag=0)]))
    salobj_dataloader_val = DataLoader(salobj_dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=0)
    # ------- 3. define model --------
    # define the net
    net = MSCDNet_v2(3, 1)
    # net.load_state_dict(torch.load(model_dir_con))
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(3407)
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80 ,90], gamma=0.9)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    val_loss_max = 0.0
    results = {'train_loss': [], 'train_Tar':[],'val_IoU': []}

    for epoch in range(1, epoch_num+1):
        running_results = {'batch_sizes': 0, 'Tar_loss':0, 'CD_loss':0}
        net.train()
        train_bar = tqdm(salobj_dataloader)
        for data in train_bar:
            # ite_num = ite_num + 1
            # ite_num4val = ite_num4val + 1
            running_results['batch_sizes']+=batch_size_train

            inputsA, inputsB, labels = data['imageA'],data['imageB'], data['label']

            inputsA = inputsA.type(torch.FloatTensor)
            inputsB = inputsB.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_vA,inputs_vB, labels_v = Variable(inputsA.cuda(), requires_grad=False), Variable(inputsB.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
            else:
                inputs_vA,inputs_vB, labels_v = Variable(inputsA, requires_grad=False), Variable(inputsB, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5 = net(inputs_vA, inputs_vB)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, labels_v)

            loss.backward()
            optimizer.step()

            running_results['Tar_loss'] += loss2.item() * batch_size_train
            running_results['CD_loss'] += loss.item() * batch_size_train

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, loss2, loss

            train_bar.set_description(
                desc='[%d/%d] CD: %.4f  ' % (
                    epoch, epoch_num, running_results['CD_loss'] / running_results['batch_sizes'],
                    ))
        # scheduler.step()
        net.eval()
        with torch.no_grad():
            val_bar = tqdm(salobj_dataloader_val)
            inter, unin = 0, 0
            valing_results = {'CD_loss': 0, 'batch_sizes': 0, 'Tar_loss':0, 'IoU': 0}

            for data in val_bar:
                valing_results['batch_sizes'] += batch_size_val

                inputsA, inputsB, labels = data['imageA'],data['imageB'], data['label']

                inputsA = inputsA.type(torch.FloatTensor)
                inputsB = inputsB.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs_vA,inputs_vB, labels_v = Variable(inputsA.cuda(), requires_grad=False), Variable(inputsB.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
                else:
                    inputs_vA,inputs_vB, labels_v = Variable(inputsA, requires_grad=False), Variable(inputsB, requires_grad=False), Variable(labels, requires_grad=False)

                CD_final, d1, d2, d3, d4, d5 = net(inputs_vA, inputs_vB)
                pred_cdmap_ref = CD_final[:, 0, :, :]
                pred_cdmap_ref = normPRED(pred_cdmap_ref)
                pred_cdmap_ref = torch.ge(pred_cdmap_ref, 0.5).float()
                pred_cdmap_ref = pred_cdmap_ref.squeeze()
                pred_cdmap_ref = pred_cdmap_ref.cpu().data.numpy()
                gt_value = labels_v.squeeze().cpu().detach().numpy()
                intr, unn = calMetric_iou(gt_value, pred_cdmap_ref)
                inter = inter + intr
                unin = unin + unn

                val_bar.set_description(desc='IoU: %.4f' % (inter * 1.0 / unin))
            valing_results['IoU'] = inter * 1.0 / unin
            val_loss = valing_results['IoU']
            if val_loss > val_loss_max:
                val_loss_max = val_loss
                torch.save(net.state_dict(),  model_dir+'netCD_epoch_%d_val_iou_%.4f.pth' % (epoch, val_loss))
            results['train_loss'].append(running_results['CD_loss'] / running_results['batch_sizes'])
            results['train_Tar'].append(running_results['Tar_loss'] / running_results['batch_sizes'])
            results['val_IoU'].append(valing_results['IoU'])

            if epoch % 1 == 0 :
                data_frame = pd.DataFrame(
                    data={'train_loss': results['train_loss'],
                        'val_IoU': results['val_IoU']},
                    index=range(1, epoch + 1))
                data_frame.to_csv(sta_dir, index_label='Epoch')

    print('-------------Congratulations! Training Done!!!-------------')

if __name__ == '__main__':
    main()