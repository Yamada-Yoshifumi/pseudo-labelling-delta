import numpy as np
import tifffile
from cellpose import models, io, transforms, core, utils
from omnipose.utils import normalize99
import os
import tensorflow as tf
import glob
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from delta.data import saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.model import unet_seg
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

number_of_trenches = 20
time_frame_each_ite = 10
time_frame_number = 1597

time_frame = 0
training_set = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_delta"
conv_files = io.get_image_files(training_set + "/convolutions")
training_set_in_use = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta"
masks_dir = training_set_in_use + "/masks/"
temp_masks_dir = training_set_in_use + "/temp_convolutions/segmentation/"
temp_convs_dir = training_set_in_use + "/temp_convolutions/"

for i in range(0, int(time_frame_number/time_frame_each_ite)):
    # ---------------------------------------------------------------------
    #Generate Test Data
    temp_convs_files = conv_files[time_frame:time_frame + number_of_trenches*time_frame_each_ite]
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
        ps = min(4096, len(unprocessed))  # 4096 at a time
        to_process = unprocessed[0:ps]
        del unprocessed[0:ps]

        # Input data generator:
        predGene = predictGenerator_seg(
            inputs_folder,
            files_list=to_process,
            target_size=cfg.target_size_rois,
            crop=cfg.crop_windows,
        )

        # mother machine: Don't crop images into windows
        if not cfg.crop_windows:
            # Predictions:
            results = model.predict(predGene, verbose=1)[:, :, :, 0]

        # 2D: Cut into overlapping windows
        else:
            img = readreshape(
                os.path.join(inputs_folder, to_process[0]),
                target_size=cfg.target_size_rois,
                crop=True,
            )
            # Create array to store predictions
            results = np.zeros((len(to_process), img.shape[0], img.shape[1], 1))
            # Crop, segment, stitch and store predictions in results
            for j in range(len(to_process)):
                # Crop each frame into overlapping windows:
                windows, loc_y, loc_x = utils.create_windows(
                    next(predGene)[0, :, :], target_size=cfg.target_size_rois
                )
                # We have to play around with tensor dimensions to conform to
                # tensorflow's functions:
                windows = windows[:, :, :, np.newaxis]
                # Predictions:
                pred = model.predict(windows, verbose=1, steps=windows.shape[0])
                # Stich prediction frames back together:
                pred = utils.stitch_pic(pred[:, :, :, 0], loc_y, loc_x)
                pred = pred[np.newaxis, :, :, np.newaxis]  # Mess around with dims

                results[j] = pred

        # Post process results (binarize + light morphology-based cleaning):
        results = postprocess(results, crop=cfg.crop_windows)

        # Save to disk:
        saveResult_seg(outputs_folder, results, files_list=to_process)

    #-------------------------------------------------------
    #Compare cell statistics
    masks_files = io.get_image_files(masks_dir)
    temp_masks_files = io.get_image_files(temp_masks_dir)
    temp_convs_files = io.get_image_files(temp_convs_dir)

    masks = np.array([list(tifffile.imread(mask_file)) for mask_file in masks_files], dtype=list)
    temp_masks = np.array([list(tifffile.imread(temp_mask_file)) for temp_mask_file in temp_masks_files], dtype=list)

    for j in range(0, time_frame_each_ite):
        Image.fromarray(np.array(tifffile.imread(temp_masks_files[i*number_of_trenches+7]))).save(f"{training_set_in_use}/results/7/train_{str(time_frame+j*number_of_trenches+7).zfill(5)}.tif")
        Image.fromarray(np.array(tifffile.imread(temp_masks_files[i*number_of_trenches+14]))).save(f"{training_set_in_use}/results/14/train_{str(time_frame+j*number_of_trenches+14).zfill(5)}.tif")
    
    masks_stat = cell_statistics(masks)
    temp_masks_stat = cell_statistics(temp_masks)

    if abs(temp_masks_stat[0] - masks_stat[0])/masks_stat[0] > 0.05 or abs(temp_masks_stat[1] - masks_stat[1])/masks_stat[1] > 0.05:
        for i,temp_conv in enumerate(temp_convs_files):
            Image.fromarray(np.array(tifffile.imread(temp_conv))).save(f"{training_set_in_use}/convolutions/train_{str(i).zfill(5)}.tif")
        for i,temp_mask in enumerate(temp_masks_files):
            Image.fromarray(np.array(tifffile.imread(temp_mask))).save(f"{training_set_in_use}/masks/train_{str(i).zfill(5)}.tif")

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
        batch_size = 20
        epochs = 600
        steps_per_epoch = 100
        patience = 50

        # Data generator parameters:
        data_gen_args = dict(
            rotation=3,
            shiftX=0.1,
            shiftY=0.1,
            zoom=0.25,
            horizontal_flip=True,
            vertical_flip=True,
            rotations_90d=True,
            histogram_voodoo=True,
            illumination_voodoo=True,
            gaussian_noise=0.03,
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

    time_frame += number_of_trenches*time_frame_each_ite