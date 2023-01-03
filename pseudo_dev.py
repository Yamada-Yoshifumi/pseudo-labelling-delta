import numpy as np
import tifffile
from cellpose import models, io, transforms, core, utils
from omnipose.utils import normalize99
import os
import tensorflow as tf
import cv2
import glob
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from scene_generation import scene_generation
from delta.data import saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.utilities import cfg
from delta.model import unet_rois, unet_seg
from delta.data import trainGenerator_seg
use_GPU = core.use_gpu()
print('>>> GPU activated? %d'%use_GPU)

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
training_set_in_use = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta"
training_set = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_delta"

conv_files = io.get_image_files(training_set + "/convolutions")
#mask_files = io.get_image_files(training_set + "/masks")

number_of_trenches = 200
time_frame_each_ite = 20
time_frame_number = 1597

time_frame = 0
training_set = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_delta"
conv_files = io.get_image_files(training_set + "/convolutions")
training_set_in_use = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta"
masks_dir = training_set_in_use + "/masks/"
convs_dir = training_set_in_use + "/convolutions/"
temp_masks_dir = training_set_in_use + "/temp_convolutions/segmentation/"
temp_convs_dir = training_set_in_use + "/temp_convolutions/"

for i in range(0, int(time_frame_number/time_frame_each_ite)):
    # ---------------------------------------------------------------------
    #Generate Test Data
    if time_frame + number_of_trenches*time_frame_each_ite < len(conv_files):
        temp_convs_files = conv_files[time_frame:time_frame + number_of_trenches*time_frame_each_ite]
    else:
        temp_convs_files = conv_files[time_frame:-1]
    for j,temp_conv in enumerate(temp_convs_files):
        Image.fromarray(np.array(tifffile.imread(temp_conv))).save(f"{training_set_in_use}/temp_convolutions/train_{str(j).zfill(5)}.tif")

    # ---------------------------------------------------------------------
    #Test
    # Load config:
    cfg.load_config(json_file="/home/ameyasu/.delta/config_mothermachine_mixture.json", presets="mothermachine")

    # Input image sequence (change to whatever images sequence you want to evaluate):
    inputs_folder = cfg.eval_movie
    savefile = cfg.model_file_rois

    # # For mother machine instead:
    # cfg.load_config(presets='mothermachine')

    # # Images sequence (change to whatever images sequence you want to evaluate):
    # inputs_folder = os.path.join(cfg.eval_movie,'cropped_rois')

    # Outputs folder:
    outputs_folder = os.path.join(inputs_folder, "segmentation")
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    # List files in inputs folder:
    unprocessed = sorted(
        glob.glob(inputs_folder + "/*.tif") + glob.glob(inputs_folder + "/*.png")
    )

    # Load up model:
    model = unet_seg(input_size=cfg.target_size_rois + (1,), pretrained_weights=savefile)
    #model.load_weights(savefile)

    # Process
    while unprocessed:
        # Pop out filenames
        ps = min(1500, len(unprocessed))  # 4096 at a time
        to_process = unprocessed[0:ps]
        del unprocessed[0:ps]

        # Input data generator:
        predGene = predictGenerator_seg(
            inputs_folder,
            files_list=to_process,
            target_size=cfg.target_size_rois,
            crop=cfg.crop_windows,
        )
        # Predictions:
        results = model.predict(predGene, verbose=1)
        # Post process results (binarize + light morphology-based cleaning):
        results = np.squeeze(results)
        #results = np.argmax(results, axis=3)[:,:,:]
        results = postprocess(results, min_size = 20, crop=cfg.crop_windows)
        for i, result in enumerate(results):
            ret, results[i] = cv2.connectedComponents(np.uint8(result))

        # Save to disk:
        saveResult_seg(outputs_folder, results, files_list=to_process)

    #-------------------------------------------------------
    #Compare cell statistics

    masks_files = io.get_image_files(masks_dir)
    temp_masks_files = io.get_image_files(temp_masks_dir)
    temp_convs_files = io.get_image_files(temp_convs_dir)

    print("Time Frame Number: " + str(time_frame))

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
    print("Length Change: " + str((temp_masks_stat[0] - masks_stat[0])/masks_stat[0]))
    print("Width Change: " + str((temp_masks_stat[1] - masks_stat[1])/masks_stat[1]))
    if abs(temp_masks_stat[0] - masks_stat[0])/masks_stat[0] > 0.05 or abs(temp_masks_stat[1] - masks_stat[1])/masks_stat[1] > 0.05:
        
        length_variation = (temp_masks_stat[0] - masks_stat[0])/masks_stat[0]
        width_variation = (temp_masks_stat[1] - masks_stat[1])/masks_stat[1]
        
        for f in glob.glob(masks_dir + "*.tif"):
            os.remove(f)

        for f in glob.glob(convs_dir + "*.tif"):
            os.remove(f)

        if abs(length_variation) > 0.1 or abs(width_variation) > 0.1:
            valid_pseudo_proportion = 1. - (abs(length_variation) - 0.1)/(abs(length_variation) - 0.05)
            valid_time_frames = int(time_frame_each_ite*valid_pseudo_proportion)
            for k,temp_conv in enumerate(temp_convs_files[0:(valid_time_frames*number_of_trenches)]):
                Image.fromarray(np.array(tifffile.imread(temp_conv))).save(f"{training_set_in_use}/convolutions/train_{str(k).zfill(5)}.tif")
            for k,temp_mask in enumerate(temp_masks_files[0:(valid_time_frames*number_of_trenches)]):
                Image.fromarray(np.array(tifffile.imread(temp_mask))).save(f"{training_set_in_use}/masks/train_{str(k).zfill(5)}.tif")
            length = masks_stat[0]*(1 + np.sign(length_variation)*0.1)*0.065*0.6
            sample_size = (time_frame_each_ite - valid_time_frames)*number_of_trenches
            sample_size = min(sample_size, 1000)
            scene_generation(length = length, length_var = length/4, width = 0.95, width_var = 0.2, sim_length = sample_size + 100, sample_size = sample_size, initialised = True, save_dir=training_set_in_use)
            time_frame += number_of_trenches*valid_time_frames
        else:
            for k,temp_conv in enumerate(temp_convs_files):
                Image.fromarray(np.array(tifffile.imread(temp_conv))).save(f"{training_set_in_use}/convolutions/train_{str(k).zfill(5)}.tif")
            for k,temp_mask in enumerate(temp_masks_files):
                Image.fromarray(np.array(tifffile.imread(temp_mask))).save(f"{training_set_in_use}/masks/train_{str(k).zfill(5)}.tif")
            time_frame += number_of_trenches*time_frame_each_ite

        # ------------------------------------------------------
        # Train
        # Load config:
        cfg.load_config(json_file="/home/ameyasu/.delta/config_mothermachine_mixture.json", presets="mothermachine")

        # Files:
        training_set = cfg.training_set_rois
        savefile = cfg.model_file_rois

        print(cfg.training_set_rois)
        print(cfg.target_size_rois)

        # Parameters:
        batch_size = 40
        epochs = 30
        steps_per_epoch = 100
        patience = 5

        # Data generator parameters:
        data_gen_args = dict(
            rotation=0,
            rotations_90d=False,
            horizontal_flip=False,
            vertical_flip=False,
            illumination_voodoo=False,
            gaussian_noise=0.03,
            gaussian_blur=1,
        )

        # Generator init:
        myGene = trainGenerator_seg(
            batch_size,
            os.path.join(training_set, "convolutions/"),
            os.path.join(training_set, "masks/"),
            None,
            augment_params=data_gen_args,
            target_size=cfg.target_size_rois,
        )

        # Define model:
        model = unet_seg(input_size=cfg.target_size_rois + (1,), pretrained_weights=savefile)
        model.summary()
        # Callbacks:
        model_checkpoint = ModelCheckpoint(
            savefile, monitor="loss", verbose=1, save_best_only=True
        )
        early_stopping = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=patience)

        # Train:
        history = model.fit(
            myGene,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[model_checkpoint, early_stopping],
        )
    else:
        time_frame += number_of_trenches*time_frame_each_ite