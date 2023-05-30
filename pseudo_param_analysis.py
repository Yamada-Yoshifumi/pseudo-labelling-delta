from pseudo_dev import PseudoLabelling
import matplotlib.pyplot as plt
import matplotlib
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import HyperBandScheduler
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools._mask as _mask_util
import pycocotools.mask as mask_util
import pylab
from ray.air import session
import ray
from ray import tune, air
from benchmark import minimum_bounding_contour, minimum_bounding_rectangle, loss_f, write_json, get_cells, draw_contour
import delta.utilities as utils
from delta.data import saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.data import trainGenerator_seg
from delta.model import unet_seg
from delta.utilities import cfg
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from scene_generation import scene_generation
import zarr
from PIL import Image
from joblib import Parallel, delayed
import tifffile
import os
import glob
import cv2

'''
change initial model
change ground truth folder
change initial model index in pseudo dev
'''

pylab.rcParams['figure.figsize'] = (10.0, 8.0)
annType = ['segm', 'bbox', 'keypoints']
annType = annType[0]  # specify type here
apType = [1, 8]
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'

print('Running demo for *%s* results.' % (annType))
hyperband = HyperBandScheduler(metric="score", mode="max", max_t=40)

ratios_list = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
#ratios_list = [32.0, 64.0, 128.0]
prev_three_lengths = []
epochs_list = np.linspace(10, 80, 8, dtype=int).tolist()
#epochs_list = np.linspace(60, 80, 3, dtype=int).tolist()
indices_list = np.linspace(0, 7, 8, dtype=int).tolist()
#indices_list = np.linspace(5, 7, 3, dtype=int).tolist()
sizes_list = [(ratio, epoch, index) for ratio, epoch, index in zip(ratios_list, epochs_list, indices_list)]
search_space = {
    "params": tune.grid_search(sizes_list)
}

def length_mapping(length):
    length = length*0.065*1.5
    
    d_length = length - prev_three_lengths[2]
    
    if d_length > 0.05:
        length *= 1.1
    elif d_length <= - 0.05:
        length *= 0.9
    
    prev_three_lengths.pop(0)
    prev_three_lengths.append(length)
    return length
def erode(path, kernel):
    img = tifffile.imread(path)
    img = np.array(img)
    result = img.copy()
    kernel_size = (kernel.shape[0], kernel.shape[1])
    half_length = int((kernel_size[0]-1)/2)
    diameter = kernel_size[0]
    for j in range(len(img)):
        for i in range(len(img[0])):
            if j > half_length and j < len(img) - half_length and i > half_length and i < len(img[0]) - half_length:
                temp_img = []
                for m in range(half_length+1):
                    for n in range(half_length + 1 - m):
                        temp_img.append(img[j+m][i+n])
                        temp_img.append(img[j-m][i+n])
                        temp_img.append(img[j+m][i-n])
                        temp_img.append(img[j-m][i-n])
                #temp_img = img[j-half_length:j+half_length, i-half_length:i+half_length]
                if not all([pix == img[j][i] for pix in temp_img]):
                    result[j][i] = 0
    Image.fromarray(result).save(path)
    return result
def objective(config):
    pseudo_labelling_pipeline = PseudoLabelling()
    params = config
    training_set_in_use = f"/home/ameyasu/cuda_ws/src/pseudo_labelling_synth/{params[2]}/training_set_in_use_delta"
    training_set = "/home/ameyasu/cuda_ws/src/pseudo_labelling_synth/training_set_delta"
    groundtruth_set = "/home/ameyasu/cuda_ws/src/full_curve_test"
    symbac_generation_set = "/home/ameyasu/cuda_ws/src/pseudo_labelling_synth/training_set_in_use_delta/results/symbac_generations"
    conv_files = sorted(glob.glob(training_set + "/convolutions/*.tif"))

    temp_evaluation_dir = training_set_in_use + "/results/temp_evaluation"
    temp_groundtruth_dir = training_set_in_use + "/results/temp_groundtruth"

    time_frame = 0

    masks_dir = training_set_in_use + "/masks/"
    convs_dir = training_set_in_use + "/convolutions/"
    temp_masks_dir = training_set_in_use + "/temp_convolutions/segmentation/"
    temp_convs_dir = training_set_in_use + "/temp_convolutions/"

    for f in glob.glob(masks_dir + "*.tif"):
        os.remove(f)

    for f in glob.glob(convs_dir + "*.tif"):
        os.remove(f)

    for f in glob.glob(temp_masks_dir + "*.tif"):
        os.remove(f)

    for f in glob.glob(temp_convs_dir + "*.tif"):
        os.remove(f)
    #init

    for i in pseudo_labelling_pipeline.trench_indices:
        if not os.path.exists(f"{training_set_in_use}/results/{i}/masks/"):
            os.makedirs(f"{training_set_in_use}/results/{i}/masks/")


    temp_convs_files = conv_files[time_frame:time_frame + pseudo_labelling_pipeline.number_of_trenches*pseudo_labelling_pipeline.time_frame_each_ite]
    for j,temp_conv in enumerate(temp_convs_files):
        Image.fromarray(np.uint8(255*pseudo_labelling_pipeline.normalize99(np.array(tifffile.imread(temp_conv))))).save(f"{training_set_in_use}/convolutions/train_{str(j).zfill(5)}.tif")

    # ---------------------------------------------------------------------
    #Test
    # Load config:
    cfg.load_config(json_file="/home/ameyasu/.delta/config_mothermachine_mixture.json", presets="mothermachine")

    # Input image sequence (change to whatever images sequence you want to evaluate):
    inputs_folder = training_set_in_use + "/convolutions/"
    savefile = training_set_in_use + "/models/1.5_0.775.hdf5"
    outputs_folder = os.path.join(inputs_folder, "../masks")

    _ = pseudo_labelling_pipeline.predicting_delta(inputs_folder, outputs_folder, savefile, metrics_while_predicting=False)
    initialised = True
    i = 0
    while True:
        # ---------------------------------------------------------------------
        #Generate Test Data
        if time_frame + pseudo_labelling_pipeline.number_of_trenches*pseudo_labelling_pipeline.time_frame_each_ite < len(conv_files):
            temp_convs_files = conv_files[time_frame:time_frame + pseudo_labelling_pipeline.number_of_trenches*pseudo_labelling_pipeline.time_frame_each_ite]
        else:
            break
        for j,temp_conv in enumerate(temp_convs_files):
            Image.fromarray(np.uint8(255*pseudo_labelling_pipeline.normalize99(np.array(tifffile.imread(temp_conv))))).save(f"{training_set_in_use}/temp_convolutions/train_{str(j).zfill(5)}.tif")

        # ---------------------------------------------------------------------
        #Test
        # Load config:
        cfg.load_config(json_file="/home/ameyasu/.delta/config_mothermachine_mixture.json", presets="mothermachine")

        # Input image sequence (change to whatever images sequence you want to evaluate):
        inputs_folder = temp_convs_dir
        # Outputs folder:
        outputs_folder = os.path.join(inputs_folder, "segmentation")
        savefile = training_set_in_use + "/models/1.5_0.775.hdf5"
        if pseudo_labelling_pipeline.fine_tuned:
            savefile = savefile.replace('/1.5_0.775.hdf5', '/tuned_model.hdf5')
        _ = pseudo_labelling_pipeline.predicting_delta(inputs_folder, outputs_folder, savefile, metrics_while_predicting=False)
        #-------------------------------------------------------
        #Compare cell statistics

        masks_files = glob.glob(masks_dir+"*.tif")
        temp_masks_files = glob.glob(temp_masks_dir + "*.tif")
        temp_convs_files = glob.glob(temp_convs_dir + "*.tif")
        masks_files = sorted(masks_files)
        temp_masks_files = sorted(temp_masks_files)
        temp_convs_files = sorted(temp_convs_files)

        print("Time Frame Number: " + str(time_frame))
        print("Frame Number: " + str(i))

        masks = np.array([tifffile.imread(mask_file) for mask_file in masks_files])
        temp_masks = np.array([tifffile.imread(temp_mask_file) for temp_mask_file in temp_masks_files])

        if pseudo_labelling_pipeline.use_SymBac:
            for j in range(0, int(len(temp_masks_files)/pseudo_labelling_pipeline.number_of_trenches)):
                for k in range(pseudo_labelling_pipeline.number_of_trenches):
                    ret, result = cv2.connectedComponents(np.uint8(tifffile.imread(temp_masks_files[j*pseudo_labelling_pipeline.number_of_trenches+k])))
                    Image.fromarray(np.uint8(result)).save(f"{training_set_in_use}/results/{pseudo_labelling_pipeline.trench_indices[k]}/masks/train_{str(i + j).zfill(5)}.tif")
        else:
            for j in range(0, pseudo_labelling_pipeline.step_forward_each_ite):
                for k in range(pseudo_labelling_pipeline.number_of_trenches):
                    ret, result = cv2.connectedComponents(np.uint8(tifffile.imread(temp_masks_files[j*pseudo_labelling_pipeline.number_of_trenches+k])))
                    Image.fromarray(np.uint8(result)).save(f"{training_set_in_use}/results/{pseudo_labelling_pipeline.trench_indices[k]}/masks/train_{str(i + j).zfill(5)}.tif")

        masks_stat = pseudo_labelling_pipeline.cell_statistics(masks)
        temp_masks_stat = pseudo_labelling_pipeline.cell_statistics(temp_masks)

        

        if len(prev_three_lengths) == 3:

            print(prev_three_lengths)
            length = length_mapping(temp_masks_stat[0])

            print("Length Change: " + str((temp_masks_stat[0] - masks_stat[0])/(masks_stat[0]+1e-5)))
            print("Width Change: " + str((temp_masks_stat[1] - masks_stat[1])/(masks_stat[1]+1e-5)))
            length_variation = (temp_masks_stat[0] - masks_stat[0])/(masks_stat[0]+1e-5)
            valid_pseudo_proportion = 1. - (abs(length_variation) - 0.1)/(abs(length_variation) + 1e-5)
            valid_time_frames = int(pseudo_labelling_pipeline.time_frame_each_ite*valid_pseudo_proportion)
            print()
            
            if abs(temp_masks_stat[0] - masks_stat[0])/(masks_stat[0]+1e-5) > 0.1:
                    
                #width_variation = (temp_masks_stat[1] - masks_stat[1])/(masks_stat[1]+1e-5)
                epochs = params[1]
                for f in glob.glob(masks_dir + "*.tif"):
                    os.remove(f)

                for f in glob.glob(convs_dir + "*.tif"):
                    os.remove(f)

                #if abs(length_variation) > 0.1:
                            
                for k,temp_conv in enumerate(temp_convs_files[0:(valid_time_frames*pseudo_labelling_pipeline.number_of_trenches)]):
                    Image.fromarray(np.array(tifffile.imread(temp_conv))).save(f"{training_set_in_use}/convolutions/train_{str(k).zfill(5)}.tif")
                for k,temp_mask in enumerate(temp_masks_files[0:(valid_time_frames*pseudo_labelling_pipeline.number_of_trenches)]):
                    Image.fromarray(np.array(tifffile.imread(temp_mask))).save(f"{training_set_in_use}/masks/train_{str(k).zfill(5)}.tif")
                #length = masks_stat[0]*(1 + np.sign(length_variation)*0.1)*0.065
                
                if not os.path.exists(f"{symbac_generation_set}/{i}/masks/"):
                    os.makedirs(f"{symbac_generation_set}/{i}/masks/")
                with open(f"{symbac_generation_set}/{i}/length_record.txt", 'w+') as f:
                    f.write(str(length) + "\n")
                sample_size = abs(int(pseudo_labelling_pipeline.number_of_trenches*pseudo_labelling_pipeline.time_frame_each_ite*(1-valid_pseudo_proportion)*params[0]/valid_pseudo_proportion))
                sample_size = min(sample_size, 4000)
                sample_size = 100*(sample_size//100)
                lengths = np.array([ 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
                widths = [ 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925]
                min_index = (np.abs(lengths - length)).argmin()
                length = lengths[min_index]
                width = widths[min_index]
            
                #scene_generation(length = length, length_var = 0.1, width = width, width_var = 0.1, sim_length = sample_size+100, sample_size = sample_size, label_masks=True, initialised = True, save_dir=training_set_in_use+"/")
                gen_i = 0
                gen_i_end = int(sample_size // 400)
                if gen_i_end >= 1:
                    while gen_i < gen_i_end:
                        try:
                            scene_generation(length = length, length_var = 0.1, width = width, width_var = 0.1, sim_length = 450, sample_size=400, label_masks=True, initialised = True, save_dir=training_set_in_use+"/")
                            gen_i+=1
                        except:
                            print("glitched")
                else:
                    passed = False
                    while not passed:
                        try:
                            scene_generation(length = length, length_var = 0.1, width = width, width_var = 0.1, sim_length = sample_size + 60, sample_size= sample_size + 10, label_masks=True, initialised = True, save_dir=training_set_in_use+"/")
                            passed = True
                        except:
                            print("glitched")
                            passed = False
                kernel = np.ones((5,5), np.uint8)
                sorted_synth_list = sorted(glob.glob(masks_dir + "synth*.tif"))
                Parallel(n_jobs=-4)(delayed(erode)(path, kernel) for path in sorted_synth_list)
                for f in glob.glob(convs_dir + "*.tif"):
                    Image.fromarray(np.uint8(255*(pseudo_labelling_pipeline.normalize99(np.array(tifffile.imread(f)))))).save(f)
                time_frame += pseudo_labelling_pipeline.number_of_trenches*valid_time_frames
                i += valid_time_frames

                # ------------------------------------------------------
                # Train
                # Load config:
                
                pseudo_labelling_pipeline.training_delta(epochs, training_set_in_use, 1, initialised, savefile)
                for f in glob.glob(convs_dir + "synth_*.tif"):
                    os.remove(f)
                for f in glob.glob(masks_dir + "synth_*.tif"):
                    os.remove(f)
                pseudo_labelling_pipeline.fine_tuned = True
                savefile = savefile.replace('/1.5_0.775.hdf5', '/tuned_model.hdf5')
                _ = pseudo_labelling_pipeline.predicting_delta(inputs_folder, outputs_folder, savefile = savefile, metrics_while_predicting=False)
                masks_files = glob.glob(masks_dir+"*.tif")
                temp_masks_files = glob.glob(temp_masks_dir + "*.tif")
                temp_convs_files = glob.glob(temp_convs_dir + "*.tif")
                masks_files = sorted(masks_files)
                temp_masks_files = sorted(temp_masks_files)
                temp_convs_files = sorted(temp_convs_files)
                if pseudo_labelling_pipeline.use_SymBac:
                    for j in range(0, int(len(temp_masks_files)/pseudo_labelling_pipeline.number_of_trenches)):
                        for k in range(pseudo_labelling_pipeline.number_of_trenches):
                            ret, result = cv2.connectedComponents(np.uint8(tifffile.imread(temp_masks_files[j*pseudo_labelling_pipeline.number_of_trenches+k])))
                            Image.fromarray(np.uint8(result)).save(f"{training_set_in_use}/results/{pseudo_labelling_pipeline.trench_indices[k]}/masks/train_{str(i + j).zfill(5)}.tif")
                else:
                    for j in range(0, pseudo_labelling_pipeline.step_forward_each_ite):
                        for k in range(pseudo_labelling_pipeline.number_of_trenches):
                            ret, result = cv2.connectedComponents(np.uint8(tifffile.imread(temp_masks_files[j*pseudo_labelling_pipeline.number_of_trenches+k])))
                            Image.fromarray(np.uint8(result)).save(f"{training_set_in_use}/results/{pseudo_labelling_pipeline.trench_indices[k]}/masks/train_{str(i + j).zfill(5)}.tif")
            else:
                if pseudo_labelling_pipeline.use_SymBac:
                    time_frame += pseudo_labelling_pipeline.number_of_trenches*pseudo_labelling_pipeline.time_frame_each_ite
                    i+= pseudo_labelling_pipeline.time_frame_each_ite
                else:
                    time_frame += pseudo_labelling_pipeline.number_of_trenches*pseudo_labelling_pipeline.step_forward_each_ite
                    i += pseudo_labelling_pipeline.step_forward_each_ite
        else:
            length = temp_masks_stat[0]*0.065*1.5
            prev_three_lengths.append(length)
            if pseudo_labelling_pipeline.use_SymBac:
                time_frame += pseudo_labelling_pipeline.number_of_trenches*pseudo_labelling_pipeline.time_frame_each_ite
                i+= pseudo_labelling_pipeline.time_frame_each_ite
            else:
                time_frame += pseudo_labelling_pipeline.number_of_trenches*pseudo_labelling_pipeline.step_forward_each_ite
                i += pseudo_labelling_pipeline.step_forward_each_ite
        
        

if __name__ == "__main__":
    for i in range(8):
    #for i in range(3):
        objective(sizes_list[i])

'''
ray.init(address = "auto")

from ray.air import FailureConfig

failure_config = FailureConfig(
    max_failures=5,
)
tuner = tune.Tuner(
    tune.with_resources(
        objective,
        resources=tune.PlacementGroupFactory(
            [
                {"CPU": 2, "GPU": 0.5},
                {"CPU": 1},
                {"accelerator_type:A100": 0.1}
            ],
            strategy="PACK",
        ),
    ),
    tune_config=tune.TuneConfig(
#        scheduler=hyperband,
        max_concurrent_trials=9
    ),
    run_config=air.RunConfig(local_dir="./results",
                             name="symbac_param_analysis", 
                             failure_config=failure_config,),
    param_space=search_space,
)
results = tuner.fit()
'''