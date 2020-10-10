#!/usr/bin/env python
# coding: utf-8

#importing required libraries
import os
#import pydicom
import numpy as np
import nibabel as nib
import glob
from PIL import Image
#from imread import imread, imsave
#import mahotas
import nilearn
from nilearn import image
from nilearn.image import resample_img
import SimpleITK as sitk
import scipy
from scipy import interpolate
import torch
import matplotlib.pyplot as plt
from nilearn.plotting import show

#read required files and check for their shape
path = glob.glob('selected/*')
file_names = []
file_data = []
file_info = []
file_header = []
data_shapes = []
for images in path:
    file_names.append(images)
    data_load = nib.load(images)
    file_data.append(data_load)
    info = data_load.get_data()
    file_info.append(info)
    header = data_load.header
    file_header.append(header)

#get voxel data and pixel data
vox_data = []
data_shapes = []
for headers in file_header:
    a = headers.get_zooms()
    vox_data.append(a)
for shapes in file_data:
    pix_shape = shapes.shape
    data_shapes.append(pix_shape) 
print(vox_data[1])
print(len(data_shapes),len(vox_data))

#processing for isotropic voxels
target_resolution = []
for (x_p,y_p,z_p),(x_v,y_v,z_v) in zip(data_shapes,vox_data):
    required_pixel_resolution = ((x_p*x_v)/z_v,(y_p*y_v)/z_v,(z_p*z_v)/z_v)
    required_pixel_resolution = (int(required_pixel_resolution[0]),int(required_pixel_resolution[1]),int(required_pixel_resolution[2]))
    print(required_pixel_resolution)
    target_resolution.append(required_pixel_resolution) 

# making data to isotropic voxels
required_resolution = []
required_new_resolution_data = []
#target_affine = np.eye(4)
for resolution,file,y in zip(target_resolution,file_names,file_data):
    new_resolution = resample_img(file, target_affine=y.affine, target_shape=resolution)
    new_resolution_data = new_resolution.get_data()
    new_resolution_data = np.asarray(new_resolution_data)
    required_new_resolution_data.append(new_resolution_data)
    required_resolution.append(new_resolution)
print(required_resolution[1].shape)


#plotting original data and resampled data
from nilearn import plotting

plotting.plot_stat_map(file_data[5],
                       title="original resolution")
plotting.plot_stat_map(required_resolution[5],cut_coords=(0, 14, -7),
                       title="Resampled resolution")
plotting.show()

# checking shapes of resampled data
required_new_resolution_array_data= np.asarray(required_new_resolution_data)
print(required_new_resolution_array_data.shape)
print(required_new_resolution_array_data[1].shape)

#downsampling resampled data by removing alternative slices
gts = []
x = np.arange(1,131,2)
r,_,_,_ = required_new_resolution_array_data.shape
for i in range(r):
    test = required_new_resolution_array_data[i,:,:,:]
    lbs = []
    for i in x:
        slices = test[:,:,i]
        lbs.append(slices)    
    lbs = np.asarray(lbs)
    lbs = np.moveaxis(lbs,0,-1)
    gts.append(lbs)
print(gts[1].shape)
print(type(gts[1]))
print(required_resolution[1].shape)

#converting 3d to 5d
print(gts[1].shape)
required_arr_4d = []
required_arr_5d = []
for i in gts:
    #swapped = np.moveaxis(i, 2, 0)  
    arr4d = np.expand_dims(i, 0)
    arr5d = np.expand_dims(arr4d,0)
    #arr5d = np.moveaxis(arr5d,4,2)
    required_arr_4d.append(arr4d)
    required_arr_5d.append(arr5d)
print(required_arr_4d[1].shape)
print(required_arr_5d[1].shape)


#getting torch tensors from numpy array
tor_5d_arr = []
for x in required_arr_5d:
    tor_data = torch.from_numpy(x)
    tor_5d_arr.append(tor_data)
print(tor_5d_arr[1].shape)  

#Interpolation of downsampled data for adding noise
needed = tor_5d_arr[1]
tor_inter = torch.nn.functional.interpolate(needed, size=(192,192,130), scale_factor=None, mode='trilinear', align_corners=None)
print(tor_inter.shape)

#Getting numpy array from pytorch tensors 
pt2np = tor_inter.numpy()
pt2np.shape

#Verifying noise from interpolated image
print(pt2np.mean())
print(required_new_resolution_array_data[1].mean())

#Squeezing 5d to 3d for plotting
sqez1 = np.squeeze(pt2np, axis=0)
sqez2 = np.squeeze(sqez1,axis = 0)
plt.imshow(sqez2[:,:,39], cmap='gray')

#checking for the noise in the image slice with isotropic file
fig = plt.figure(figsize = (18, 8))
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(sqez2[:,:,39], cmap= "gray")
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(required_new_resolution_array_data[1][:,:,39],cmap= "gray"

req_swap = np.moveaxis(required_new_resolution_array_data[1],2,0)


#3d volume verification of isotropic volume and interpolated volume
swapped = np.moveaxis(sqez2,2,0)
print(swapped.shape)
plt_sqeez = sitk.GetImageFromArray(swapped)
req_plt_sqeez = sitk.GetImageFromArray(req_swap)
sitk.Show(plt_sqeez)
sitk.Show(req_plt_sqeez)
           

import torch
import torch.nn as nn

def create_conv_bn_relu(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )

def create_double_conv(in_channels, mid_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        create_conv_bn_relu(in_channels, mid_channels, kernel_size, padding),
        create_conv_bn_relu(mid_channels, out_channels, kernel_size, padding),
    )

def upsample(x1, x2):

    # here you can also use ConvTranspose2d. There are some drawbacks to that.
    # ConvTranspose2d (aka. Deconvolution) creats checkerboard pattern. You can
    # fix that with more convolution layers after ConvTranspose2d
    # more about it -> https://distill.pub/2016/deconv-checkerboard/
    # In my experience basic bilinear upsample is faster and it dosen't require
    # more conv layers after upsample

    return torch.nn.functional.interpolate(
        x1,
        size=(x2.size()[2], x2.size()[3], x2.size()[4]),
        scale_factor=None,
        mode='trilinear',
        align_corners=True
    )

class UNet(nn.Module):
    def __init__(self, n_channels, output_channels):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.output_channels = output_channels

        self.double_conv_down_0 = create_double_conv(1, 10, 10)
        self.double_conv_down_1 = create_double_conv(10, 10, 10)
        self.double_conv_down_2 = create_double_conv(10, 10, 10)
        self.double_conv_down_3 = create_double_conv(10, 10, 10)
        self.double_conv_down_4 = create_double_conv(10, 10, 10)

        self.maxpool_1 = nn.MaxPool3d((1, 2, 2))
        self.maxpool_2 = nn.MaxPool3d((1, 2, 2))
        self.maxpool_3 = nn.MaxPool3d((1, 2, 2))
        self.maxpool_4 = nn.MaxPool3d((1, 2, 2))

        self.double_conv_up_1 = create_double_conv(20, 10, 10)
        self.double_conv_up_2 = create_double_conv(20, 10, 10)
        self.double_conv_up_3 = create_double_conv(20, 10, 10)
        self.double_conv_up_4 = create_double_conv(20, 10, 10)

        self.output_conv = nn.Conv3d(10, 1, kernel_size=1)

    def forward(self, x):
        # x is in BCDHW format

        x1 = self.double_conv_down_0(x)
        x = self.maxpool_1(x1)
        x2 = self.double_conv_down_1(x)
        x = self.maxpool_2(x2)
        x3 = self.double_conv_down_2(x)
        x = self.maxpool_3(x3)
        x4 = self.double_conv_down_3(x)
        #x = self.maxpool_4(x4)
        #x5 = self.double_conv_down_4(x)

        x = upsample(x4, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.double_conv_up_1(x)

        x = upsample(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = self.double_conv_up_2(x)

        x = upsample(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = self.double_conv_up_3(x)

        x = upsample(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = self.double_conv_up_4(x)

        output = self.output_conv(x)

        return output


if __name__ == '__main__':

    import numpy as np

    #a = pt2np

    model = UNet(10, 10)
    output = model(tor_inter.float())

    print (output.size())

import matplotlib.pyplot as plt

pt2np1 = output.detach().numpy()
sqez1 = np.squeeze(pt2np1, axis=0)
sqez2 = np.squeeze(sqez1,axis = 0)
plt.imshow(sqez2[:,:,39], cmap='gray')


