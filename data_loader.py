# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
#==========================dataset load==========================

class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imageA, imageB, label = sample['imageA'],sample['imageB'],sample['label']

		h, w = imageA.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		imgA = transform.resize(imageA,(self.output_size,self.output_size),mode='constant')
		imgB = transform.resize(imageB,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imageA':imgA,'imageB':imgB,'label':lbl}

class RescaleT_single(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imageB, label =sample['imageB'],sample['label']

		h, w = imageB.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		# imgA = transform.resize(imageA,(self.output_size,self.output_size),mode='constant')
		imgB = transform.resize(imageB,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imageB':imgB,'label':lbl}

class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		# image, label = sample['image'],sample['label']
		imageA, imageB, label = sample['imageA'],sample['imageB'],sample['label']

		h, w = imageA.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		imgA = transform.resize(imageA,(new_h,new_w),mode='constant')
		imgB = transform.resize(imageB,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		# return {'image':img,'label':lbl}
		return {'imageA':imgA,'imageB':imgB,'label':lbl}

class Rescale_single(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		# image, label = sample['image'],sample['label']
		imageB, label = sample['imageB'],sample['label']

		h, w = imageB.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# imgA = transform.resize(imageA,(new_h,new_w),mode='constant')
		imgB = transform.resize(imageB,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)
		# return {'image':img,'label':lbl}
		return {'imageB':imgB,'label':lbl}

class CenterCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		# image, label = sample['image'], sample['label']
		imageA, imageB, label = sample['imageA'],sample['imageB'],sample['label']

		h, w = imageA.shape[:2]
		new_h, new_w = self.output_size

		# print("h: %d, w: %d, new_h: %d, new_w: %d"%(h, w, new_h, new_w))
		assert((h >= new_h) and (w >= new_w))

		h_offset = int(math.floor((h - new_h)/2))
		w_offset = int(math.floor((w - new_w)/2))

		imageA = imageA[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
		imageB = imageB[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
		label = label[h_offset: h_offset + new_h, w_offset: w_offset + new_w]

		return {'imageA': imageA,'imageB': imageB, 'label': label}

class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		# image, label = sample['image'], sample['label']
		imageA, imageB, label = sample['imageA'],sample['imageB'],sample['label']

		h, w = imageA.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		imageA = imageA[top: top + new_h, left: left + new_w]
		imageB = imageB[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		# return {'image': image, 'label': label}
		return {'imageA': imageA,'imageB': imageB, 'label': label}

class RandomCrop_single(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		imageB, label = sample['imageB'], sample['label']
		# imageA, imageB, label = sample['imageA'],sample['imageB'],sample['label']

		h, w = imageB.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		# imageA = imageA[top: top + new_h, left: left + new_w]
		imageB = imageB[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		# return {'image': image, 'label': label}
		return {'imageB': imageB, 'label': label}

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		# image, label = sample['image'], sample['label']
		imageA, imageB, label = sample['imageA'],sample['imageB'],sample['label']

		tmpImg = np.zeros((imageA.shape[0],imageA.shape[1],3))
		tmpImgB = np.zeros((imageB.shape[0],imageB.shape[1],3))

		tmpLbl = np.zeros(label.shape)

		imageA = imageA/np.max(imageA)
		imageB = imageB/np.max(imageB)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if imageA.shape[2]==1:
			tmpImg[:,:,0] = (imageA[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (imageA[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (imageA[:,:,0]-0.485)/0.229
			tmpImgB[:,:,0] = (imageB[:,:,0]-0.485)/0.229
			tmpImgB[:,:,1] = (imageB[:,:,0]-0.485)/0.229
			tmpImgB[:,:,2] = (imageB[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (imageA[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (imageA[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (imageA[:,:,2]-0.406)/0.225
			tmpImgB[:,:,0] = (imageB[:,:,0]-0.485)/0.229
			tmpImgB[:,:,1] = (imageB[:,:,1]-0.456)/0.224
			tmpImgB[:,:,2] = (imageB[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpImgB = tmpImgB.transpose((2, 0, 1))

		tmpLbl = label.transpose((2, 0, 1))

		return {
			'imageA': torch.from_numpy(tmpImg),
			'imageB': torch.from_numpy(tmpImgB),
			'label': torch.from_numpy(tmpLbl)}

class ToTensor_single(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		# image, label = sample['image'], sample['label']
		imageB, label = sample['imageB'],sample['label']

		# tmpImg = np.zeros((imageA.shape[0],imageA.shape[1],3))
		tmpImgB = np.zeros((imageB.shape[0],imageB.shape[1],3))

		tmpLbl = np.zeros(label.shape)

		# imageA = imageA/np.max(imageA)
		imageB = imageB/np.max(imageB)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if imageB.shape[2]==1:
			# tmpImg[:,:,0] = (imageA[:,:,0]-0.485)/0.229
			# tmpImg[:,:,1] = (imageA[:,:,0]-0.485)/0.229
			# tmpImg[:,:,2] = (imageA[:,:,0]-0.485)/0.229
			tmpImgB[:,:,0] = (imageB[:,:,0]-0.485)/0.229
			tmpImgB[:,:,1] = (imageB[:,:,0]-0.485)/0.229
			tmpImgB[:,:,2] = (imageB[:,:,0]-0.485)/0.229
		else:
			# tmpImg[:,:,0] = (imageA[:,:,0]-0.485)/0.229
			# tmpImg[:,:,1] = (imageA[:,:,1]-0.456)/0.224
			# tmpImg[:,:,2] = (imageA[:,:,2]-0.406)/0.225
			tmpImgB[:,:,0] = (imageB[:,:,0]-0.485)/0.229
			tmpImgB[:,:,1] = (imageB[:,:,1]-0.456)/0.224
			tmpImgB[:,:,2] = (imageB[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		# tmpImg = tmpImg.transpose((2, 0, 1))
		tmpImgB = tmpImgB.transpose((2, 0, 1))

		tmpLbl = label.transpose((2, 0, 1))

		return {
			# 'imageA': torch.from_numpy(tmpImg),
			'imageB': torch.from_numpy(tmpImgB),
			'label': torch.from_numpy(tmpLbl)}

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		# image, label = sample['image'], sample['label']
		imageA, imageB, label = sample['imageA'],sample['imageB'],sample['label']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((imageA.shape[0],imageA.shape[1],6))
			tmpImgt = np.zeros((imageA.shape[0],imageA.shape[1],3))
			tmpImgB = np.zeros((imageB.shape[0],imageB.shape[1],6))
			tmpImgtB = np.zeros((imageB.shape[0],imageB.shape[1],3))
			if imageA.shape[2]==1:
				tmpImgt[:,:,0] = imageA[:,:,0]
				tmpImgt[:,:,1] = imageA[:,:,0]
				tmpImgt[:,:,2] = imageA[:,:,0]
				tmpImgtB[:,:,0] = imageB[:,:,0]
				tmpImgtB[:,:,1] = imageB[:,:,0]
				tmpImgtB[:,:,2] = imageB[:,:,0]
			else:
				tmpImgt = imageA
				tmpImgtB = imageB

			tmpImgtl = color.rgb2lab(tmpImgt)
			tmpImgtlB = color.rgb2lab(tmpImgtB)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			tmpImgB[:,:,0] = (tmpImgtB[:,:,0]-np.min(tmpImgtB[:,:,0]))/(np.max(tmpImgtB[:,:,0])-np.min(tmpImgtB[:,:,0]))
			tmpImgB[:,:,1] = (tmpImgtB[:,:,1]-np.min(tmpImgtB[:,:,1]))/(np.max(tmpImgtB[:,:,1])-np.min(tmpImgtB[:,:,1]))
			tmpImgB[:,:,2] = (tmpImgtB[:,:,2]-np.min(tmpImgtB[:,:,2]))/(np.max(tmpImgtB[:,:,2])-np.min(tmpImgtB[:,:,2]))
			tmpImgB[:,:,3] = (tmpImgtlB[:,:,0]-np.min(tmpImgtlB[:,:,0]))/(np.max(tmpImgtlB[:,:,0])-np.min(tmpImgtlB[:,:,0]))
			tmpImgB[:,:,4] = (tmpImgtlB[:,:,1]-np.min(tmpImgtlB[:,:,1]))/(np.max(tmpImgtlB[:,:,1])-np.min(tmpImgtlB[:,:,1]))
			tmpImgB[:,:,5] = (tmpImgtlB[:,:,2]-np.min(tmpImgtlB[:,:,2]))/(np.max(tmpImgtlB[:,:,2])-np.min(tmpImgtlB[:,:,2]))
			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

			tmpImgB[:,:,0] = (tmpImgB[:,:,0]-np.mean(tmpImgB[:,:,0]))/np.std(tmpImgB[:,:,0])
			tmpImgB[:,:,1] = (tmpImgB[:,:,1]-np.mean(tmpImgB[:,:,1]))/np.std(tmpImgB[:,:,1])
			tmpImgB[:,:,2] = (tmpImgB[:,:,2]-np.mean(tmpImgB[:,:,2]))/np.std(tmpImgB[:,:,2])
			tmpImgB[:,:,3] = (tmpImgB[:,:,3]-np.mean(tmpImgB[:,:,3]))/np.std(tmpImgB[:,:,3])
			tmpImgB[:,:,4] = (tmpImgB[:,:,4]-np.mean(tmpImgB[:,:,4]))/np.std(tmpImgB[:,:,4])
			tmpImgB[:,:,5] = (tmpImgB[:,:,5]-np.mean(tmpImgB[:,:,5]))/np.std(tmpImgB[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((imageA.shape[0],imageA.shape[1],3))
			tmpImgB = np.zeros((imageB.shape[0],imageB.shape[1],3))


			if imageA.shape[2]==1:
				tmpImg[:,:,0] = imageA[:,:,0]
				tmpImg[:,:,1] = imageA[:,:,0]
				tmpImg[:,:,2] = imageA[:,:,0]

				tmpImgB[:,:,0] = imageB[:,:,0]
				tmpImgB[:,:,1] = imageB[:,:,0]
				tmpImgB[:,:,2] = imageB[:,:,0]
			else:
				tmpImg = imageA
				tmpImgB = imageB


			tmpImg = color.rgb2lab(tmpImg)
			tmpImgB = color.rgb2lab(tmpImgB)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

			tmpImgB[:,:,0] = (tmpImgB[:,:,0]-np.min(tmpImgB[:,:,0]))/(np.max(tmpImgB[:,:,0])-np.min(tmpImgB[:,:,0]))
			tmpImgB[:,:,1] = (tmpImgB[:,:,1]-np.min(tmpImgB[:,:,1]))/(np.max(tmpImgB[:,:,1])-np.min(tmpImgB[:,:,1]))
			tmpImgB[:,:,2] = (tmpImgB[:,:,2]-np.min(tmpImgB[:,:,2]))/(np.max(tmpImgB[:,:,2])-np.min(tmpImgB[:,:,2]))

			tmpImgB[:,:,0] = (tmpImgB[:,:,0]-np.mean(tmpImgB[:,:,0]))/np.std(tmpImgB[:,:,0])
			tmpImgB[:,:,1] = (tmpImgB[:,:,1]-np.mean(tmpImgB[:,:,1]))/np.std(tmpImgB[:,:,1])
			tmpImgB[:,:,2] = (tmpImgB[:,:,2]-np.mean(tmpImgB[:,:,2]))/np.std(tmpImgB[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((imageA.shape[0],imageA.shape[1],3))
			tmpImgB = np.zeros((imageB.shape[0],imageB.shape[1],3))

			imageA = imageA/np.max(imageA)
			imageB = imageB/np.max(imageB)

			if imageA.shape[2]==1:
				tmpImg[:,:,0] = (imageA[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (imageA[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (imageA[:,:,0]-0.485)/0.229

				tmpImgB[:,:,0] = (imageB[:,:,0]-0.485)/0.229
				tmpImgB[:,:,1] = (imageB[:,:,0]-0.485)/0.229
				tmpImgB[:,:,2] = (imageB[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (imageA[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (imageA[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (imageA[:,:,2]-0.406)/0.225

				tmpImgB[:,:,0] = (imageB[:,:,0]-0.485)/0.229
				tmpImgB[:,:,1] = (imageB[:,:,1]-0.456)/0.224
				tmpImgB[:,:,2] = (imageB[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpImgB = tmpImgB.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {
			'imageA': torch.from_numpy(tmpImg),
			'imageB': torch.from_numpy(tmpImgB),
			'label': torch.from_numpy(tmpLbl)}

class ToTensorLab_single(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imageB, label = sample['imageB'], sample['label']
		# imageA, imageB, label = sample['imageA'],sample['imageB'],sample['label']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			# tmpImg = np.zeros((imageA.shape[0],imageA.shape[1],6))
			# tmpImgt = np.zeros((imageA.shape[0],imageA.shape[1],3))
			tmpImgB = np.zeros((imageB.shape[0],imageB.shape[1],6))
			tmpImgtB = np.zeros((imageB.shape[0],imageB.shape[1],3))
			if imageB.shape[2]==1:
				# tmpImgt[:,:,0] = imageA[:,:,0]
				# tmpImgt[:,:,1] = imageA[:,:,0]
				# tmpImgt[:,:,2] = imageA[:,:,0]
				tmpImgtB[:,:,0] = imageB[:,:,0]
				tmpImgtB[:,:,1] = imageB[:,:,0]
				tmpImgtB[:,:,2] = imageB[:,:,0]
			else:
				# tmpImgt = imageA
				tmpImgtB = imageB

			# tmpImgtl = color.rgb2lab(tmpImgt)
			tmpImgtlB = color.rgb2lab(tmpImgtB)

			# nomalize image to range [0,1]
			# tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			# tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			# tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			# tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			# tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			# tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			tmpImgB[:,:,0] = (tmpImgtB[:,:,0]-np.min(tmpImgtB[:,:,0]))/(np.max(tmpImgtB[:,:,0])-np.min(tmpImgtB[:,:,0]))
			tmpImgB[:,:,1] = (tmpImgtB[:,:,1]-np.min(tmpImgtB[:,:,1]))/(np.max(tmpImgtB[:,:,1])-np.min(tmpImgtB[:,:,1]))
			tmpImgB[:,:,2] = (tmpImgtB[:,:,2]-np.min(tmpImgtB[:,:,2]))/(np.max(tmpImgtB[:,:,2])-np.min(tmpImgtB[:,:,2]))
			tmpImgB[:,:,3] = (tmpImgtlB[:,:,0]-np.min(tmpImgtlB[:,:,0]))/(np.max(tmpImgtlB[:,:,0])-np.min(tmpImgtlB[:,:,0]))
			tmpImgB[:,:,4] = (tmpImgtlB[:,:,1]-np.min(tmpImgtlB[:,:,1]))/(np.max(tmpImgtlB[:,:,1])-np.min(tmpImgtlB[:,:,1]))
			tmpImgB[:,:,5] = (tmpImgtlB[:,:,2]-np.min(tmpImgtlB[:,:,2]))/(np.max(tmpImgtlB[:,:,2])-np.min(tmpImgtlB[:,:,2]))
			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			# tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			# tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			# tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			# tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			# tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			# tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

			tmpImgB[:,:,0] = (tmpImgB[:,:,0]-np.mean(tmpImgB[:,:,0]))/np.std(tmpImgB[:,:,0])
			tmpImgB[:,:,1] = (tmpImgB[:,:,1]-np.mean(tmpImgB[:,:,1]))/np.std(tmpImgB[:,:,1])
			tmpImgB[:,:,2] = (tmpImgB[:,:,2]-np.mean(tmpImgB[:,:,2]))/np.std(tmpImgB[:,:,2])
			tmpImgB[:,:,3] = (tmpImgB[:,:,3]-np.mean(tmpImgB[:,:,3]))/np.std(tmpImgB[:,:,3])
			tmpImgB[:,:,4] = (tmpImgB[:,:,4]-np.mean(tmpImgB[:,:,4]))/np.std(tmpImgB[:,:,4])
			tmpImgB[:,:,5] = (tmpImgB[:,:,5]-np.mean(tmpImgB[:,:,5]))/np.std(tmpImgB[:,:,5])

		elif self.flag == 1: #with Lab color
			# tmpImg = np.zeros((imageA.shape[0],imageA.shape[1],3))
			tmpImgB = np.zeros((imageB.shape[0],imageB.shape[1],3))


			if imageB.shape[2]==1:
				# tmpImg[:,:,0] = imageA[:,:,0]
				# tmpImg[:,:,1] = imageA[:,:,0]
				# tmpImg[:,:,2] = imageA[:,:,0]

				tmpImgB[:,:,0] = imageB[:,:,0]
				tmpImgB[:,:,1] = imageB[:,:,0]
				tmpImgB[:,:,2] = imageB[:,:,0]
			else:
				# tmpImg = imageA
				tmpImgB = imageB


			# tmpImg = color.rgb2lab(tmpImg)
			tmpImgB = color.rgb2lab(tmpImgB)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			# tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			# tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			# tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			# tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			# tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			# tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

			tmpImgB[:,:,0] = (tmpImgB[:,:,0]-np.min(tmpImgB[:,:,0]))/(np.max(tmpImgB[:,:,0])-np.min(tmpImgB[:,:,0]))
			tmpImgB[:,:,1] = (tmpImgB[:,:,1]-np.min(tmpImgB[:,:,1]))/(np.max(tmpImgB[:,:,1])-np.min(tmpImgB[:,:,1]))
			tmpImgB[:,:,2] = (tmpImgB[:,:,2]-np.min(tmpImgB[:,:,2]))/(np.max(tmpImgB[:,:,2])-np.min(tmpImgB[:,:,2]))

			tmpImgB[:,:,0] = (tmpImgB[:,:,0]-np.mean(tmpImgB[:,:,0]))/np.std(tmpImgB[:,:,0])
			tmpImgB[:,:,1] = (tmpImgB[:,:,1]-np.mean(tmpImgB[:,:,1]))/np.std(tmpImgB[:,:,1])
			tmpImgB[:,:,2] = (tmpImgB[:,:,2]-np.mean(tmpImgB[:,:,2]))/np.std(tmpImgB[:,:,2])

		else: # with rgb color
			# tmpImg = np.zeros((imageA.shape[0],imageA.shape[1],3))
			tmpImgB = np.zeros((imageB.shape[0],imageB.shape[1],3))

			# imageA = imageA/np.max(imageA)
			imageB = imageB/np.max(imageB)

			if imageB.shape[2]==1:
				# tmpImg[:,:,0] = (imageA[:,:,0]-0.485)/0.229
				# tmpImg[:,:,1] = (imageA[:,:,0]-0.485)/0.229
				# tmpImg[:,:,2] = (imageA[:,:,0]-0.485)/0.229

				tmpImgB[:,:,0] = (imageB[:,:,0]-0.485)/0.229
				tmpImgB[:,:,1] = (imageB[:,:,0]-0.485)/0.229
				tmpImgB[:,:,2] = (imageB[:,:,0]-0.485)/0.229
			else:
				# tmpImg[:,:,0] = (imageA[:,:,0]-0.485)/0.229
				# tmpImg[:,:,1] = (imageA[:,:,1]-0.456)/0.224
				# tmpImg[:,:,2] = (imageA[:,:,2]-0.406)/0.225

				tmpImgB[:,:,0] = (imageB[:,:,0]-0.485)/0.229
				tmpImgB[:,:,1] = (imageB[:,:,1]-0.456)/0.224
				tmpImgB[:,:,2] = (imageB[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		# tmpImg = tmpImg.transpose((2, 0, 1))
		tmpImgB = tmpImgB.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {
			# 'imageA': torch.from_numpy(tmpImg),
			'imageB': torch.from_numpy(tmpImgB),
			'label': torch.from_numpy(tmpLbl)}

class SalObjDataset(Dataset):
	def __init__(self,img_name_listB,lbl_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_listB = img_name_listB
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_listB)

	def __getitem__(self,idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

		imageB = io.imread(self.image_name_listB[idx])

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(imageB.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		#print("len of label3")
		#print(len(label_3.shape))
		#print(label_3.shape)

		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(imageB.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(imageB.shape) and 2==len(label.shape)):
			image = imageB[:,:,np.newaxis]
			label = label[:,:,np.newaxis]

		# #vertical flipping
		# # fliph = np.random.randn(1)
		# flipv = np.random.randn(1)
		#
		# if flipv>0:
		# 	image = image[::-1,:,:]
		# 	label = label[::-1,:,:]
		# #vertical flip

		sample = {'imageB':imageB, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample

class SalCDDataset(Dataset):
	def __init__(self,img_name_listA,img_name_listB,lbl_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_listA = img_name_listA
		self.image_name_listB = img_name_listB
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_listA)

	def __getitem__(self,idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

		imageA = io.imread(self.image_name_listA[idx])
		imageB = io.imread(self.image_name_listB[idx])

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(imageA.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		#print("len of label3")
		#print(len(label_3.shape))
		#print(label_3.shape)

		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(imageA.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(imageA.shape) and 2==len(label.shape)):
			imageA = imageA[:,:,np.newaxis]
			imageB = imageB[:,:,np.newaxis]
			label = label[:,:,np.newaxis]

		# #vertical flipping
		# # fliph = np.random.randn(1)
		# flipv = np.random.randn(1)
		#
		# if flipv>0:
		# 	image = image[::-1,:,:]
		# 	label = label[::-1,:,:]
		# #vertical flip

		sample = {'imageA':imageA, 'imageB':imageB, 'label':label}

		if self.transform:
			sample = self.transform(sample)
		# print(np.unique(sample['label'].unsqueeze(0)))
		# exit(-1)

		return sample