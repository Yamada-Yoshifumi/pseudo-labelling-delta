import numpy as np
import tifffile
from cellpose import models, io, transforms, core, utils
from omnipose.utils import normalize99
import os
import tensorflow as tf
import cv2
import glob
from joblib import Parallel, delayed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import time, os, sys
from cellpose import models, core, utils
from cellpose import io, transforms
from cellpose import models
from omnipose.utils import normalize99
from pathlib import Path
import os
from skimage.util import img_as_ubyte
from cellpose import io
use_GPU = core.use_gpu()
from delta.data import saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.utilities import cfg
from delta.model import unet_rois, unet_seg
from delta.data import trainGenerator_seg
use_GPU = core.use_gpu()
print('>>> GPU activated? %d'%use_GPU)
import logging
logger = logging.getLogger(__name__)
from scene_generation import scene_generation

# for plotting 
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

dimension = (220, 64)

from scipy.spatial import ConvexHull
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.
    # get the convex hull for the points
    hull_points = np.array([points[vertex] for vertex in ConvexHull(points).vertices])

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

def cell_statistics(image_array):
    
    average_length_images_array = []
    average_width_images_array = []

    for image in image_array:
        cell_length_image_array = []
        cell_width_image_array = []
        cell_pixels_image = []
        cell_index = 1
        while True:
            one_cell = np.where(np.array(image) == cell_index)
            
            if len(one_cell) == 2:
                if len(one_cell[0]) == 0:
                    break
                for pixel_x, pixel_y in zip(one_cell[0], one_cell[1]):
                    cell_pixels_image.append([int(pixel_x), int(pixel_y)])
                bounding_box = minimum_bounding_rectangle(cell_pixels_image)
                scale_1 = np.sqrt((bounding_box[0][0]-bounding_box[1][0])**2 + (bounding_box[0][1]-bounding_box[1][1])**2)
                scale_2 = np.sqrt((bounding_box[1][0]-bounding_box[2][0])**2 + (bounding_box[1][1]-bounding_box[2][1])**2)
                cell_length_image_array.append(max(scale_1, scale_2))
                cell_width_image_array.append(min(scale_1, scale_2))
                cell_index += 1
            else:
                break
        if len(cell_length_image_array) == 0:
            average_length_images_array.append(0.)
            average_width_images_array.append(0.)
        else:
            average_length_images_array.append(sum(cell_length_image_array)/len(cell_length_image_array))
            average_width_images_array.append(sum(cell_width_image_array)/len(cell_width_image_array))
    
    return [sum(average_length_images_array)/len(average_length_images_array), sum(average_width_images_array)/len(average_width_images_array)]


from PIL import Image
import tifffile

training_set = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_omni/convolutions"
conv_files = io.get_image_files(training_set)
#mask_files = io.get_image_files(training_set + "/masks")

number_of_trenches = 200
tile_number_per_image = 40
time_frame_each_ite = 20
time_frame_number = 1597
time_frame = 0

# define parameters
mask_threshold = -1 
verbose = 0 # turn on if you want to see more output 
use_gpu = use_GPU #defined above
transparency = True # transparency in flow output
rescale=None # give this a number if you need to upscale or downscale your images
omni = True # we can turn off Omnipose mask reconstruction, not advised 
flow_threshold = 0 # default is .4, but only needed if there are spurious masks to clean up; slows down output
resample = True #whether or not to run dynamics on rescaled grid or original grid 
cluster=True # use DBSCAN clustering



training_set_in_use = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_omni"
training_data_dir = training_set_in_use + "/training_data"
training_data_tiled_dir = training_set_in_use + "/training_data_tiled"
temp_data_dir = training_set_in_use + "/temp_training_data"

model_name = "cellpose_residual_on_style_on_concatenation_off_omni_nclasses_4_mixed_data_small_set_omnipose_2022_11_17_13_15_40.008098_epoch_601_loss_0.60"
model = models.CellposeModel(gpu=use_GPU, pretrained_model=f"{training_set_in_use}/models/{model_name}")

init_convs_files = conv_files[0: number_of_trenches*time_frame_each_ite]
for j,temp_conv in enumerate(init_convs_files):
    Image.fromarray(np.array(tifffile.imread(temp_conv))).save(f"{training_data_dir}/train_{str(j).zfill(5)}.tif")

files = io.get_image_files(training_data_dir)
imgs = [io.imread(f) for f in files]

for k in range(len(imgs)):
    img = transforms.move_min_dim(imgs[k]) # move the channel dimension last
    if len(img.shape)>2:
        # imgs[k] = img[:,:,1] # could pick out a specific channel
        imgs[k] = np.mean(img,axis=-1) # or just turn into grayscale 
        
    imgs[k] = normalize99(imgs[k])

chans = [0,0] #this means segment based on first channel, no second channel 
n = [0] # make a list of integers to select which images you want to segment
n = range(len(imgs)) # or just segment them all 

masks, flows, styles = model.eval([imgs[j] for i in n],channels=chans,rescale=rescale,mask_threshold=mask_threshold,transparency=transparency,
                                flow_threshold=flow_threshold,omni=omni,cluster=cluster, resample=resample,verbose=verbose)

Parallel(n_jobs=-2)(delayed(io.save_masks)(image, mask, flow, file_name, 
            png=False,
            tif=True, # whether to use PNG or TIF format
            suffix='', # suffix to add to files if needed 
            save_flows=False, # saves both RGB depiction as *_flows.png and the raw components as *_dP.tif
            save_outlines=False, # save outline images 
            dir_above=0, # save output in the image directory or in the directory above (at the level of the image directory)
            in_folders=True, # save output in folders (recommended)
            save_txt=False, # txt file for outlines in imageJ
            save_ncolor=False) for image, mask, flow, file_name in zip(imgs, masks, flows, files))

for i in range(0, int(time_frame_number/time_frame_each_ite)):
    # ---------------------------------------------------------------------
    #Generate Test Data
    if time_frame + number_of_trenches*time_frame_each_ite < len(conv_files):
        temp_convs_files = conv_files[time_frame:time_frame + number_of_trenches*time_frame_each_ite]
    else:
        temp_convs_files = conv_files[time_frame:-1]
    for j,temp_conv in enumerate(temp_convs_files):
        Image.fromarray(np.array(tifffile.imread(temp_conv))).save(f"{training_set_in_use}/temp_training_data/train_{str(j).zfill(5)}.tif")

    # ---------------------------------------------------------------------
    #Test
    # Load config:
    imgs = [io.imread(f) for f in temp_convs_files]
    for k in range(len(imgs)):
        img = transforms.move_min_dim(imgs[k]) # move the channel dimension last
        if len(img.shape)>2:
            # imgs[k] = img[:,:,1] # could pick out a specific channel
            imgs[k] = np.mean(img,axis=-1) # or just turn into grayscale 
            
        imgs[k] = normalize99(imgs[k])

    chans = [0,0] #this means segment based on first channel, no second channel 

    n = [0] # make a list of integers to select which images you want to segment
    n = range(len(imgs)) # or just segment them all 

    masks, flows, styles = model.eval([imgs[i] for i in n],channels=chans,rescale=rescale,mask_threshold=mask_threshold,transparency=transparency,
                                    flow_threshold=flow_threshold,omni=omni,cluster=cluster, resample=resample,verbose=verbose)
    
    files = io.get_image_files(f"{training_set_in_use}/temp_training_data/")
    Parallel(n_jobs=-2)(delayed(io.save_masks)(image, mask, flow, file_name, 
            png=False,
            tif=True, # whether to use PNG or TIF format
            suffix='', # suffix to add to files if needed 
            save_flows=False, # saves both RGB depiction as *_flows.png and the raw components as *_dP.tif
            save_outlines=False, # save outline images 
            dir_above=0, # save output in the image directory or in the directory above (at the level of the image directory)
            in_folders=True, # save output in folders (recommended)
            save_txt=False, # txt file for outlines in imageJ
            save_ncolor=False) for image, mask, flow, file_name in zip(imgs, masks, flows, files))
    
    #-------------------------------------------------------
    #Compare cell statistics
    masks_files = glob.glob(f"{training_data_dir}/masks/*.tif")
    temp_masks_files = glob.glob(f"{temp_data_dir}/masks/*tif")
    temp_convs_files = glob.glob(f"{temp_data_dir}/*tif")
    masks_files = sorted(masks_files)
    temp_masks_files = sorted(temp_masks_files)
    temp_convs_files = sorted(temp_convs_files)

    masks = np.array([list(tifffile.imread(mask_file)) for mask_file in masks_files], dtype=list)
    temp_masks = np.array([list(tifffile.imread(temp_mask_file)) for temp_mask_file in temp_masks_files], dtype=list)

    for j in range(0, time_frame_each_ite):
        Image.fromarray(np.array(tifffile.imread(temp_masks_files[j*number_of_trenches+7]))).save(f"{training_set_in_use}/results/1/masks/train_{str(time_frame+j*number_of_trenches+7).zfill(5)}.tif")
        Image.fromarray(np.array(tifffile.imread(temp_masks_files[j*number_of_trenches+14]))).save(f"{training_set_in_use}/results/2/masks/train_{str(time_frame+j*number_of_trenches+14).zfill(5)}.tif")
        Image.fromarray(np.array(tifffile.imread(temp_masks_files[j*number_of_trenches+21]))).save(f"{training_set_in_use}/results/3/masks/train_{str(time_frame+j*number_of_trenches+21).zfill(5)}.tif")
        Image.fromarray(np.array(tifffile.imread(temp_masks_files[j*number_of_trenches+28]))).save(f"{training_set_in_use}/results/4/masks/train_{str(time_frame+j*number_of_trenches+28).zfill(5)}.tif")
        Image.fromarray(np.array(tifffile.imread(temp_masks_files[j*number_of_trenches+35]))).save(f"{training_set_in_use}/results/5/masks/train_{str(time_frame+j*number_of_trenches+35).zfill(5)}.tif")

        Image.fromarray(np.array(tifffile.imread(temp_convs_files[j*number_of_trenches+7]))).save(f"{training_set_in_use}/results/1/convolutions/train_{str(time_frame+j*number_of_trenches+7).zfill(5)}.tif")
        Image.fromarray(np.array(tifffile.imread(temp_convs_files[j*number_of_trenches+14]))).save(f"{training_set_in_use}/results/2/convolutions/train_{str(time_frame+j*number_of_trenches+14).zfill(5)}.tif")
        Image.fromarray(np.array(tifffile.imread(temp_convs_files[j*number_of_trenches+21]))).save(f"{training_set_in_use}/results/3/convolutions/train_{str(time_frame+j*number_of_trenches+21).zfill(5)}.tif")
        Image.fromarray(np.array(tifffile.imread(temp_convs_files[j*number_of_trenches+28]))).save(f"{training_set_in_use}/results/4/convolutions/train_{str(time_frame+j*number_of_trenches+28).zfill(5)}.tif")
        Image.fromarray(np.array(tifffile.imread(temp_convs_files[j*number_of_trenches+35]))).save(f"{training_set_in_use}/results/5/convolutions/train_{str(time_frame+j*number_of_trenches+35).zfill(5)}.tif")
    
    masks_stat = cell_statistics(masks)
    temp_masks_stat = cell_statistics(temp_masks)

    if abs(temp_masks_stat[0] - masks_stat[0])/masks_stat[0] > 0.05 or abs(temp_masks_stat[1] - masks_stat[1])/masks_stat[1] > 0.05:

        for k,temp_conv in enumerate(temp_convs_files):
            Image.fromarray(np.array(tifffile.imread(temp_conv))).save(f"{training_data_dir}/train_{str(k).zfill(5)}.tif")
        for k,temp_mask in enumerate(temp_masks_files):
            Image.fromarray(np.array(tifffile.imread(temp_mask))).save(f"{training_data_dir}/masks/train_{str(k).zfill(5)}_cp_masks.tif")

        from random import randint

        masks_dir = training_data_dir + "/masks"
        convs_dir = training_data_dir
        masks = glob.glob(f"{masks_dir}/*.tif")
        convs = glob.glob(f"{convs_dir}/*.tif")
        masks = sorted(masks)
        convs = sorted(convs)

        print(len(masks))
        tile_length = 40
        training_samples = 200

        for j in range(0, training_samples):
            value = randint(0, len(masks) - 1)

            mask = tifffile.imread(masks[value])
            conv = tifffile.imread(convs[value])
            if mask.shape != dimension:
                mask = np.array(cv2.resize(mask, (dimension[1], dimension[0]), 0, 0, interpolation = cv2.INTER_NEAREST))
            if conv.shape != dimension:
                conv = np.array(cv2.resize(conv, (dimension[1], dimension[0]), 0, 0, interpolation = cv2.INTER_NEAREST))

            mask_tile = np.array(mask)
            conv_tile = np.array(conv)
            for j in range(0, tile_length - 1):

                value = randint(0, len(masks) - 1)

                mask = tifffile.imread(masks[value])
                conv = tifffile.imread(convs[value])
                if mask.shape != dimension:
                    mask = np.array(cv2.resize(mask, (dimension[1], dimension[0]), 0, 0, interpolation = cv2.INTER_NEAREST))
                if conv.shape != dimension:
                    conv = np.array(cv2.resize(conv, (dimension[1], dimension[0]), 0, 0, interpolation = cv2.INTER_NEAREST))
                
                mask_tile = np.concatenate([mask_tile, np.array(mask)], axis = 1)
                conv_tile = np.concatenate([conv_tile, np.array(conv)], axis = 1)
                
            conv_tile = conv_tile/conv_tile.max()
            conv_tile = img_as_ubyte(conv_tile)
            Image.fromarray(mask_tile.astype(np.ubyte)).save(f"{training_data_tiled_dir}/train_{str(j).zfill(5)}_masks.png")
            Image.fromarray(conv_tile).save(f"{training_data_tiled_dir}/train_{str(j).zfill(5)}.png")

        # ------------------------------------------------------
        # Train
        # Load config:
        output = io.load_train_test_data(training_data_tiled_dir, None, '', "_masks", False, False, True)
        images, labels, image_names, test_images, test_labels, image_names_test = output
        
        cpmodel_path = model.train(images, labels, train_files=image_names,
                                           test_data=test_images, test_labels=test_labels, test_files=image_names_test,
                                           model_name = model_name,
                                           learning_rate=1e-4, channels=4,
                                           save_path=training_set_in_use, save_every=False,
                                           save_each=False,
                                           save_same = True,
                                           rescale=rescale,n_epochs=50,
                                           batch_size=10, 
                                           min_train_masks=1,
                                           SGD=(not False),
                                           tyx=None)
        model.pretrained_model = cpmodel_path
        logger.info('model trained and saved to %s'%cpmodel_path)

    time_frame += number_of_trenches*time_frame_each_ite
    model = models.CellposeModel(gpu=use_GPU, pretrained_model=f"{training_set_in_use}/models/{model_name}")