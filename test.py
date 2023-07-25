import os
from skimage import io
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import Rescale
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset, SalCDDataset

# from net import LDF
# from net_v2 import LDF
# from net_v5 import LDF
from model import SS_CD, MSCDNet, PP_UNet, SS_CD_noBDFE, SSCD_noFE

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	# img_name = image_name.split("/")[-1]
	_, img_name = os.path.split(image_name)
	# print(image_name)
	# print(img_name)
	# print(img_nameC)
	# exit(-1)
	image = io.imread(image_name)
	# print(image.shape)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]
	# print(d_dir+imidx+'.png')
	# exit(-1)
	imo.save(d_dir+imidx+'.png')
	# imo.save(d_dir+imidx+'.jpg')


if __name__ == '__main__':
	# --------- 1. get image path and name ---------

	# Landslide
	
	# image_dirA = 'C:/Users/11473/OneDrive/桌面/code/MSCDNet_20220403/DATA/landslide_paper/stage1cd/val/A/'
	# image_dirB = 'C:/Users/11473/OneDrive/桌面/code/MSCDNet_20220403/DATA/landslide_paper/stage1cd/val/B/'	
	# prediction_dir_cdmap_ref = 'C:/Users/11473/OneDrive/桌面/segment_anything/SSCD_test_result/Landslide_iou0.6578/'


	#BCDDDDD

	image_dirA = 'C:/Users/11473/OneDrive/桌面/segment_anything/dataset/BCDDDDD/BCD_removeblank_split/val/A/'
	image_dirB = 'C:/Users/11473/OneDrive/桌面/segment_anything/dataset/BCDDDDD/BCD_removeblank_split/val/B/'	
	prediction_dir_cdmap_ref = 'C:/Users/11473/OneDrive/桌面/segment_anything/SSCD_result_Ablation Study/BCDDDDD_baseline+seg/'




	if not os.path.exists(prediction_dir_cdmap_ref):
		os.makedirs(prediction_dir_cdmap_ref, exist_ok=True)
	# landslide
	# model_dir = "C:/Users/11473/OneDrive/桌面/code/MSCDNet_20220403/epochs/landslide_paper/stage1cd/netCD_epoch_30_val_iou_0.6578.pth"

	#BCDDDDD
	model_dir = "C:/Users/11473/OneDrive/桌面/code/MSCDNet_20220403/epochs/BCDDDDD/SegANYTHING_baseline+mask_train_onlyCAM_seg/netCD_epoch_39_val_iou_0.5223.pth"
	
	# img_name_listA= glob.glob(image_dirA + '*.jpg')
	# img_name_listB= glob.glob(image_dirB + '*.jpg')
	# img_name_listA= glob.glob(image_dirA + '*.tif')
	# img_name_listB= glob.glob(image_dirB + '*.tif')
	img_name_listA= glob.glob(image_dirA + '*.png')
	img_name_listB= glob.glob(image_dirB + '*.png')


	
	# --------- 2. dataloader ---------
	#1. dataload
	test_salobj_dataset = SalCDDataset(
		img_name_listA = img_name_listA, 
		img_name_listB = img_name_listB, 
		lbl_name_list = [],
		transform=transforms.Compose([Rescale(256),ToTensor()]))
	test_salobj_dataloader = DataLoader(
		test_salobj_dataset, 
		batch_size=1,shuffle=False,num_workers=1)
	
	# --------- 3. model define ---------
	print("...load MSCDNet...")
	# net = MSCDNet(3, 1)
	net = SS_CD(3, 1)
	net.load_state_dict(torch.load(model_dir))
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	
	
	# --------- 4. inference for each image ---------
	for i_test, data_test in enumerate(test_salobj_dataloader):
		# print("inferencing:",img_name_listA[i_test].split("/")[-1])
	
		inputs_testA, inputs_testB = data_test['imageA'], data_test['imageB']
		inputs_testA = inputs_testA.type(torch.FloatTensor)
		inputs_testB = inputs_testB.type(torch.FloatTensor)
	
		if torch.cuda.is_available():
			inputs_testA = Variable(inputs_testA.cuda())
			inputs_testB = Variable(inputs_testB.cuda())
		else:
			inputs_testA = Variable(inputs_testA)
			inputs_testB = Variable(inputs_testB)
	
		refcd, out1, out2, out3, out4, out5 = net(inputs_testA, inputs_testB)
        # outb1, outd1, outb2, outd2, cdmap = net(inputs_vA, inputs_vB)

        # outb1, outd1, outb2, outd2 = net(inputs_vA, inputs_vB)
	
		# normalization
		pred_cdmap_ref = refcd[:,0,:,:]
		pred_cdmap_ref = normPRED(pred_cdmap_ref)
		pred_cdmap_ref = torch.ge(pred_cdmap_ref, 0.5).float()

		# pred_detail = outb2[:,0,:,:]
		# pred_detail = normPRED(pred_detail)
		# pred_cdmap_ou1 = out1[:,0,:,:]
		# pred_cdmap_ou1 = normPRED(pred_cdmap_ou1)
		# pred_cdmap_ou1 = torch.ge(pred_cdmap_ou1, 0.5).float()

		# pred_cdmap_ou2 = out2[:,0,:,:]
		# pred_cdmap_ou2 = normPRED(pred_cdmap_ou2)
		
		# pred_cdmap_ou3 = out3[:,0,:,:]
		# pred_cdmap_ou3 = normPRED(pred_cdmap_ou3)

		# pred_cdmap_ou4 = out4[:,0,:,:]
		# pred_cdmap_ou4 = normPRED(pred_cdmap_ou4)

		# pred_cdmap_ou5 = out5[:,0,:,:]
		# pred_cdmap_ou5 = normPRED(pred_cdmap_ou5)

		# save results to test_results folder
		# prediction_dir_final = os.path.join(prediction_dir, 'Refine/')
		# prediction_dir_Before = os.path.join(prediction_dir, 'Before/')
		save_output(img_name_listA[i_test],pred_cdmap_ref,prediction_dir_cdmap_ref)
		# save_output(img_name_listA[i_test],pred_cdmap_ou1,prediction_dir_cdmap_ou1)
		# save_output(img_name_listA[i_test],pred_cdmap_ou2,prediction_dir_cdmap_ou2)
		# save_output(img_name_listA[i_test],pred_cdmap_ou3,prediction_dir_cdmap_ou3)
		# save_output(img_name_listA[i_test],pred_cdmap_ou4,prediction_dir_cdmap_ou4)
		# save_output(img_name_listA[i_test],pred_detail,prediction_dir_detail)
		# save_output(img_name_listA[i_test],pred_cdmap_ou5,prediction_dir_cdmap_ou5)
	
		del refcd, out1, out2, out3, out4, out5