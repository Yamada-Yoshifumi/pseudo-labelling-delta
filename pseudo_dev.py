import numpy as np
import tifffile
import os
import tensorflow as tf
import cv2
import glob
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from scene_generation import scene_generation
from delta.data import saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.utilities import cfg
from delta.model import unet_rois, unet_seg
from delta.data import trainGenerator_seg
import json
from joblib import Parallel, delayed
from PIL import Image
import tifffile
from scipy.spatial import ConvexHull
from benchmark import get_cells, minimum_bounding_rectangle, minimum_bounding_contour, write_json

class PseudoLabelling():

    def __init__(self) -> None:
        self.trench_indices = ["515", "1823", "212", "978", "509", "2642", "693", "1910", "97", "1095", "1865"]
        self.models_list = ["1.2_0.75.hdf5", "1.5_0.775.hdf5", "2.0_0.8.hdf5", "2.5_0.825.hdf5", "3.0_0.85.hdf5", "3.5_0.875.hdf5", "4.0_0.9.hdf5", "4.5_0.925.hdf5"]
        self.init_model_index = 5
        self.use_SymBac = True
        #self.number_of_trenches = len(self.trench_indices) 
        self.number_of_trenches = len(self.trench_indices)
        self.time_frame_each_ite = 20
        self.step_forward_each_ite = 20
        self.time_frame_number = 1570
        self.continuous_tuning = True
        self.synthetic_only = True
        self.prev_length  = 0
        self.fine_tuned = False

    def normalize99(self, Y,lower=0.01,upper=99.99):
        """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile 
        Upper and lower percentile ranges configurable. 
        
        Parameters
        ----------
        Y: ndarray, float
            Component array of lenth N by L1 by L2 by ... by LN. 
        upper: float
            upper percentile above which pixels are sent to 1.0
        
        lower: float
            lower percentile below which pixels are sent to 0.0
        
        Returns
        --------------
        normalized array with a minimum of 0 and maximum of 1
        """
        X = Y.copy()
        return np.interp(X, (np.percentile(X, lower), np.percentile(X, upper)), (0, 1))

    def erode(self, path, kernel):
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

    def cell_statistics(self, image_array):
        
        average_length_images_array = []
        average_width_images_array = []
        image_number = 0
        for image in image_array:
            labeled_image = np.array(cv2.connectedComponents(np.int8(image))[1])
            #print(labeled_image)
            cell_length_image_array = []
            cell_width_image_array = []
            cell_index = 1
            #if image_number == 0:
            #    fig = plt.figure()
            while True:
                cell_pixels_image = []
                one_cell = np.where(labeled_image == cell_index)
                if len(one_cell) == 2:
                    if len(one_cell[0]) == 0:
                        break
                    for pixel_x, pixel_y in zip(one_cell[0], one_cell[1]):
                        cell_pixels_image.append([int(pixel_x), int(pixel_y)])
                    bounding_box = minimum_bounding_rectangle(cell_pixels_image)
                    #if image_number == 0:
                        #print(bounding_box)
                        #plt.plot([bounding_box[0][0], bounding_box[1][0]], [bounding_box[0][1], bounding_box[1][1]], marker = 'o', color='red')
                        #plt.plot([bounding_box[1][0], bounding_box[2][0]], [bounding_box[1][1], bounding_box[2][1]], marker = 'o', color='red')
                        #plt.plot([bounding_box[2][0], bounding_box[3][0]], [bounding_box[2][1], bounding_box[3][1]], marker = 'o', color='red')
                        #plt.plot([bounding_box[3][0], bounding_box[0][0]], [bounding_box[3][1], bounding_box[0][1]], marker = 'o', color='red')
                    scale_1 = np.sqrt((bounding_box[0][0]-bounding_box[1][0])**2 + (bounding_box[0][1]-bounding_box[1][1])**2)
                    scale_2 = np.sqrt((bounding_box[1][0]-bounding_box[2][0])**2 + (bounding_box[1][1]-bounding_box[2][1])**2)
                    cell_length_image_array.append(max(scale_1, scale_2))
                    cell_width_image_array.append(min(scale_1, scale_2))
                    cell_index += 1
                else:
                    break
            #if image_number == 0:
            #    imgplot = plt.imshow(labeled_image)
            #    imgplot.set_cmap('magma_r')
            #    plt.savefig("debug_folder/" + str(image_number)+".png")

            image_number += 1
            if not len(cell_length_image_array) == 0:
                #print("cell length: " + str(sum(cell_length_image_array)/len(cell_length_image_array)))
                average_length_images_array.append(sum(cell_length_image_array)/len(cell_length_image_array))
                average_width_images_array.append(sum(cell_width_image_array)/len(cell_width_image_array))
        #print("average cell length: " + str(sum(average_length_images_array)/len(average_length_images_array)))
        return [sum(average_length_images_array)/len(average_length_images_array), sum(average_width_images_array)/len(average_width_images_array)]



    def benchmark(self, temp_eval, ground_truth, index_temp_data, iou):

        ground_truth = [np.asarray(cv2.resize(f, (64, 256), interpolation=cv2.INTER_NEAREST)) for f in ground_truth]
        pseudo_model_cells = []
        ground_truth_cells = []
        ground_truth_cells, true_cell_sizes, cell_index = get_cells(ground_truth, index_temp_data, "groundtruth", ground_truth, False)
        pseudo_model_cells, pseudo_cell_sizes, pseudo_cell_index = get_cells(temp_eval, index_temp_data, "pseudo", ground_truth, False)

        imgIds= [index_temp_data]
        cocoGt = COCO('/home/ameyasu/cuda_ws/src/pseudo_labelling_real/groundtruth_temp_data.json')
        cocoSmall = cocoGt.loadRes('/home/ameyasu/cuda_ws/src/pseudo_labelling_real/pseudo_temp_data.json')  
        annType = ['segm','bbox','keypoints']
        annType = annType[0]      #specify type here
        cocoEval = COCOeval(cocoGt,cocoSmall,annType)
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        iou = cocoEval.stats[0]
        del cocoGt, cocoEval
        return iou

    def training_delta(self, epochs, training_set_in_use, training_set = 0, initialised = False, savefile = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta/models/pretrained_model.hdf5"):

        cfg.load_config(json_file="/home/ameyasu/.delta/config_mothermachine_mixture.json", presets="mothermachine")

        # Files:
        if training_set == 0:
            training_set = cfg.training_set_rois
        else:
            training_set = training_set_in_use

        # Parameters:
        batch_size = 20
        steps_per_epoch = 100
        patience = 1

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
        if self.synthetic_only:
            myGene = trainGenerator_seg(
                batch_size,
                os.path.join(training_set, "convolutions"),
                os.path.join(training_set, "masks"),
                None,
                augment_params=data_gen_args,
                keyword='synth',
                target_size=cfg.target_size_rois,
            )
        else:
            myGene = trainGenerator_seg(
                batch_size,
                os.path.join(training_set, "convolutions"),
                os.path.join(training_set, "masks"),
                None,
                augment_params=data_gen_args,
                target_size=cfg.target_size_rois,
            )

        # Define model:
        if initialised:
            if self.continuous_tuning and self.fine_tuned:
                savefile = savefile.replace(self.models_list[self.init_model_index], 'tuned_model.hdf5')
            model = unet_seg(input_size=cfg.target_size_rois + (1,), pretrained_weights=savefile)
            print("Pretrained weights from: " + savefile)
        else:
            model = unet_seg(input_size=cfg.target_size_rois + (1,))
        model.summary()
        # Callbacks:
        savefile = savefile.replace(self.models_list[self.init_model_index], 'tuned_model.hdf5')
        print("Save tuned model to: " + savefile)

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
        del myGene
        del model

    def predicting_delta(self, temp_evaluation_dir, temp_evaluation_output_dir, savefile, temp_groundtruth_dir="", pseudo_error_history=[], metrics_while_predicting=False, training_i = 0):
        cfg.load_config(json_file="/home/ameyasu/.delta/config_mothermachine_mixture.json", presets="mothermachine")
        inputs_folder = temp_evaluation_dir
        outputs_folder = temp_evaluation_output_dir
        if not os.path.exists(outputs_folder):
            os.makedirs(outputs_folder)

        # List files in inputs folder:
        unprocessed = sorted(
            glob.glob(inputs_folder + "/*.tif") + glob.glob(inputs_folder + "/*.png")
        )

        # Load up model:
        model = unet_seg(input_size=cfg.target_size_seg + (1,))
        model.load_weights(savefile)
        print("Predicting using :" + savefile)

        # Process
        while unprocessed:
            # Pop out filenames
            ps = min(2048, len(unprocessed))  # 4096 at a time
            to_process = unprocessed[0:ps]
            del unprocessed[0:ps]

            # Input data generator:
            predGene = predictGenerator_seg(
                inputs_folder,
                files_list=to_process,
                target_size=cfg.target_size_seg,
                crop=False,
            )
            results = model.predict(predGene, verbose=1)[:, :, :, 0]

            # Post process results (binarize + light morphology-based cleaning):
            results = postprocess(results, crop=False, min_size=20)
            for i, result in enumerate(results):
                ret, results[i] = cv2.connectedComponents(np.uint8(result))
            # Save to disk:
            saveResult_seg(outputs_folder, results, files_list=to_process)
            if metrics_while_predicting:
                temp_evaluation_files = sorted(glob.glob(temp_evaluation_dir + "/masks"))
                temp_groundtruth_files = sorted(glob.glob(temp_groundtruth_dir + "/masks"))
                temp_evaluation = [tifffile.imread(dir) for dir in temp_evaluation_files]
                temp_groundtruth = [tifffile.imread(dir) for dir in temp_groundtruth_files]
                for loop_in_temp_data in range(self.time_frame_each_ite*self.number_of_trenches):
                    iou = 0
                    iou = self.benchmark(temp_evaluation, temp_groundtruth, loop_in_temp_data, iou)
                    #iou = benchmark(temp_evaluation, temp_groundtruth, loop_in_temp_data)
                    pseudo_error_history[training_i] += iou/(self.time_frame_each_ite*self.number_of_trenches)
        return pseudo_error_history

    def length_mapping(self, length):
        if self.prev_length > 0:
            if length > self.prev_length:
                length *= 1.1
            else:
                length *= 0.9
        return length*0.065*1.3

    def main(self):

        training_set_in_use = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta"
        training_set = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_delta"
        groundtruth_set = "/home/ameyasu/cuda_ws/src/full_curve_test"
        symbac_generation_set = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta/results/symbac_generations"
        conv_files = sorted(glob.glob(training_set + "/convolutions"))

        time_frame = 0*self.number_of_trenches
        training_set = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_delta"
        conv_files = sorted(glob.glob(training_set + "/convolutions/*.tif"))

        training_set_in_use = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta"
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

        for i in self.trench_indices:
            if self.synthetic_only:
                if not os.path.exists(f"{training_set_in_use}/results/{i}_synth/masks/"):
                    os.makedirs(f"{training_set_in_use}/results/{i}_synth/masks/")
            else:
                if not os.path.exists(f"{training_set_in_use}/results/{i}/masks/"):
                    os.makedirs(f"{training_set_in_use}/results/{i}/masks/")
            if not self.use_SymBac:
                if not os.path.exists(f"{training_set_in_use}/results/{i}_pseudo/masks/"):
                    os.makedirs(f"{training_set_in_use}/results/{i}_pseudo/masks/")

        temp_convs_files = conv_files[time_frame:time_frame + self.number_of_trenches*self.time_frame_each_ite]
        for j,temp_conv in enumerate(temp_convs_files):
            Image.fromarray(np.uint8(255*self.normalize99(np.array(tifffile.imread(temp_conv))))).save(f"{training_set_in_use}/convolutions/train_{str(j).zfill(5)}.tif")

        # ---------------------------------------------------------------------
        #Test
        # Load config:
        cfg.load_config(json_file="/home/ameyasu/.delta/config_mothermachine_mixture.json", presets="mothermachine")

        # Input image sequence (change to whatever images sequence you want to evaluate):
        inputs_folder = training_set_in_use + "/convolutions/"
        savefile = training_set_in_use + "/models/" + self.models_list[self.init_model_index]
        outputs_folder = os.path.join(inputs_folder, "../masks")
        
        _ = self.predicting_delta(inputs_folder, outputs_folder, savefile, metrics_while_predicting=False)
        initialised = False
        i = 0
        jumped = False
        lengths_record = []
        length = 0
        while True:
            '''
            if i >= 500 and not jumped:
                i = 700
                time_frame = 700*self.number_of_trenches
                jumped = True
            '''
            # ---------------------------------------------------------------------
            #Generate Test Data
            if time_frame + self.number_of_trenches*self.time_frame_each_ite < len(conv_files):
                temp_convs_files = conv_files[time_frame:time_frame + self.number_of_trenches*self.time_frame_each_ite]
            else:
                break
            for j,temp_conv in enumerate(temp_convs_files):
                Image.fromarray(np.uint8(255*self.normalize99(np.array(tifffile.imread(temp_conv))))).save(f"{training_set_in_use}/temp_convolutions/train_{str(j).zfill(5)}.tif")

            # ---------------------------------------------------------------------
            #Test
            # Load config:
            cfg.load_config(json_file="/home/ameyasu/.delta/config_mothermachine_mixture.json", presets="mothermachine")

            # Input image sequence (change to whatever images sequence you want to evaluate):
            inputs_folder = cfg.eval_movie
            
            savefile = training_set_in_use + "/models/" + self.models_list[self.init_model_index]
            if self.fine_tuned:
                savefile = savefile.replace(self.models_list[self.init_model_index], 'tuned_model.hdf5')
            
            # Outputs folder:
            outputs_folder = os.path.join(inputs_folder, "segmentation")

            _ = self.predicting_delta(inputs_folder, outputs_folder, savefile, metrics_while_predicting=False)
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

            #masks = np.array([list(cv2.connectedComponents(np.uint8(tifffile.imread(mask_file)))[1]) for mask_file in masks_files], dtype=list)
            #temp_masks = np.array([list(cv2.connectedComponents(np.uint8(tifffile.imread(temp_mask_file)))[1]) for temp_mask_file in temp_masks_files], dtype=list)
            masks = np.array([tifffile.imread(mask_file) for mask_file in masks_files])
            temp_masks = np.array([tifffile.imread(temp_mask_file) for temp_mask_file in temp_masks_files])

            if self.use_SymBac:
                for j in range(0, int(len(temp_masks_files)/self.number_of_trenches)):
                    for k in range(self.number_of_trenches):
                        ret, result = cv2.connectedComponents(np.uint8(tifffile.imread(temp_masks_files[j*self.number_of_trenches+k])))
                        if self.synthetic_only:
                            Image.fromarray(np.uint8(result)).save(f"{training_set_in_use}/results/{self.trench_indices[k]}_synth/masks/train_{str(i + j).zfill(5)}.tif")
                        else:
                            Image.fromarray(np.uint8(result)).save(f"{training_set_in_use}/results/{self.trench_indices[k]}/masks/train_{str(i + j).zfill(5)}.tif")
            else:
                for j in range(0, self.step_forward_each_ite):
                    for k in range(self.number_of_trenches):
                        ret, result = cv2.connectedComponents(np.uint8(tifffile.imread(temp_masks_files[j*self.number_of_trenches+k])))
                        Image.fromarray(np.uint8(result)).save(f"{training_set_in_use}/results/{self.trench_indices[k]}_pseudo/masks/train_{str(i + j).zfill(5)}.tif")

            
            masks_stat = self.cell_statistics(masks)
            temp_masks_stat = self.cell_statistics(temp_masks)
            print("Length Change: " + str((temp_masks_stat[0] - masks_stat[0])/(masks_stat[0]+1e-4)))
            print("Width Change: " + str((temp_masks_stat[1] - masks_stat[1])/(masks_stat[1]+1e-4)))
            length_variation = (temp_masks_stat[0] - masks_stat[0])/(masks_stat[0]+1e-4)
            width_variation = (temp_masks_stat[1] - masks_stat[1])/(masks_stat[1]+1e-4)
            #the higher the length variation, the lower this variable, the less pseudo labels will be valid
            valid_pseudo_proportion = min(1 - (abs(length_variation) - 0.1)/(abs(length_variation) + 1e-4), 1 - (abs(width_variation) - 0.1)/(abs(width_variation) + 1e-4))
            valid_time_frames = int(self.time_frame_each_ite*valid_pseudo_proportion)
            #new mapping function
            width = temp_masks_stat[1]*0.065*1.2
            length = self.length_mapping(temp_masks_stat[0])

            if ((abs(length_variation) > 0.1 or abs(width_variation) > 0.1) and not (self.prev_length >= 4.0 and length >= 4.0)) or (not initialised and length < 3.0):
                if not initialised:
                    valid_pseudo_proportion = 0.5
                    valid_time_frames = int(self.time_frame_each_ite)
                initialised = True
                if self.use_SymBac:

                    for f in glob.glob(masks_dir + "*.tif"):
                        os.remove(f)

                    for f in glob.glob(convs_dir + "*.tif"):
                        os.remove(f)
                                
                    for k,temp_conv in enumerate(temp_convs_files[0:(valid_time_frames*self.number_of_trenches)]):
                        Image.fromarray(np.array(tifffile.imread(temp_conv))).save(f"{training_set_in_use}/convolutions/train_{str(k).zfill(5)}.tif")
                    for k,temp_mask in enumerate(temp_masks_files[0:(valid_time_frames*self.number_of_trenches)]):
                        Image.fromarray(np.array(tifffile.imread(temp_mask))).save(f"{training_set_in_use}/masks/train_{str(k).zfill(5)}.tif")
                    
                    #original mapping function
                    #length = temp_masks_stat[0]*0.4*0.065
                    #if length > 3:
                    #    length = (length - 3)**2 + 1.5
                    #elif length < 3:
                    #    length = 1.5
                    
                    
                    if not os.path.exists(f"{symbac_generation_set}/{i}/masks/"):
                        os.makedirs(f"{symbac_generation_set}/{i}/masks/")
                    with open(f"{symbac_generation_set}/{i}/length_record.txt", 'w+') as f:
                        f.write(str(length) + "\n")
                    
                    print("length: " + str(length))
                    print("Width: " + str(width))
                    #without the multiplier 10, sample size to pseudo labels size would correspond to the valid_pseudo_proportion
                    #with the multiplier, we get to tune the actual ratio
                    sample_size = abs(int(self.number_of_trenches*self.time_frame_each_ite*(1 - valid_pseudo_proportion)*16/(valid_pseudo_proportion)))
                    sample_size = min(sample_size, 900)
                    sample_size = 100*(sample_size//100)
                    #lengths = np.array([ 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
                    #widths = [ 0.775, 0.8, 0.825, 0.85, 0.875, 0.9 , 0.925 ]
                    #min_index = (np.abs(lengths - length)).argmin()
                    #length = lengths[min_index]
                    #min_index = (np.abs(widths - width)).argmin()
                    #width = widths[min_index]
                    '''
                    if i >= 560 and i <= 580:
                        length = 2.4
                    elif i > 580 and i <= 600:
                        length = 2.1
                    elif i > 600 and i <= 640:
                        length = 1.8
                    elif i>640 and i<= 700:
                        length = 1.5
                    '''
                    if width < 0.9:
                        width = 0.9
                    print(sample_size)
                    print("Length: " + str(length))
                    print("Width: " + str(width))
                
                    #scene_generation(length = length, length_var = 0.1, width = width, width_var = 0.1, sim_length = sample_size+100, sample_size = sample_size, label_masks=True, initialised = True, save_dir=training_set_in_use+"/")
                    gen_i = 0
                    gen_i_end = int(sample_size // 300)
                    if gen_i_end >= 1:
                        if length <= 3.0:
                            passed = False
                            while not passed:
                                gen_result = scene_generation(length = length, length_var = 0.1, width = 0.8, width_var = 0.2, sim_length = int(sample_size/5) + 100, sample_size=int(sample_size/5), label_masks=True, initialised = True, save_dir=training_set_in_use+"/")
                                if gen_result:    
                                    passed = True
                                else:
                                    print("glitched")

                        while gen_i < gen_i_end:
                            gen_result = scene_generation(length = length, length_var = 0.1, width = width, width_var = 0.2, sim_length = 350, sample_size=300, label_masks=True, initialised = True, save_dir=training_set_in_use+"/")
                            if gen_result:    
                                gen_i+=1
                            else:
                                print("glitched")
                    else:
                        passed = False
                        while not passed:
                            gen_result = scene_generation(length = length, length_var = 0.1, width = width, width_var = 0.2, sim_length = 350, sample_size=300, label_masks=True, initialised = True, save_dir=training_set_in_use+"/")
                            if gen_result:    
                                passed = True
                            else:
                                print("glitched")
                    kernel = np.ones((5,5), np.uint8)
                    temp_synth_list = sorted(glob.glob(masks_dir + "*.tif"))
                    sorted_synth_list = []
                    for synth in temp_synth_list:
                        if "synth" in synth:
                            sorted_synth_list.append(synth)
                    Parallel(n_jobs=-4)(delayed(self.erode)(path, kernel) for path in sorted_synth_list)
                    for f in glob.glob(convs_dir + "*.tif"):
                        Image.fromarray(np.uint8(255*(self.normalize99(np.array(tifffile.imread(f)))))).save(f)
                    time_frame += self.number_of_trenches*valid_time_frames
                    i += valid_time_frames
                else:
                    for replace_i in range(self.number_of_trenches*self.step_forward_each_ite, self.number_of_trenches*self.time_frame_each_ite):
                        img = tifffile.imread(masks_dir+f"train_{str(replace_i).zfill(5)}.tif")
                        Image.fromarray(img).save(masks_dir + f"train_{str(replace_i - self.number_of_trenches*self.step_forward_each_ite).zfill(5)}.tif")
                        img = tifffile.imread(convs_dir+f"train_{str(replace_i).zfill(5)}.tif")
                        Image.fromarray(img).save(convs_dir + f"train_{str(replace_i - self.number_of_trenches*self.step_forward_each_ite).zfill(5)}.tif")
                    length_variation = (temp_masks_stat[0] - masks_stat[0])/(masks_stat[0]+1e-4)
                    valid_pseudo_proportion = 1. - (abs(length_variation) - 0.05)/abs(length_variation)
                    #epochs = int((1 - valid_pseudo_proportion)*120) + 1
                    for k,temp_conv in enumerate(temp_convs_files[0: self.number_of_trenches*self.step_forward_each_ite]):
                        Image.fromarray(np.array(tifffile.imread(temp_conv))).save(f"{training_set_in_use}/convolutions/train_{str(k + self.number_of_trenches*(self.time_frame_each_ite - self.step_forward_each_ite)).zfill(5)}.tif")
                    for k,temp_mask in enumerate(temp_masks_files[0: self.number_of_trenches*self.step_forward_each_ite]):
                        Image.fromarray(np.array(tifffile.imread(temp_mask))).save(f"{training_set_in_use}/masks/train_{str(k + self.number_of_trenches*(self.time_frame_each_ite - self.step_forward_each_ite)).zfill(5)}.tif")
                    time_frame += self.number_of_trenches*self.step_forward_each_ite
                    i += self.step_forward_each_ite
                savefile = training_set_in_use + "/models/" + self.models_list[self.init_model_index]
                if self.fine_tuned and self.continuous_tuning:
                    savefile = savefile.replace(self.models_list[self.init_model_index], 'tuned_model.hdf5')
                epochs = 50
                '''
                if abs(temp_masks_stat[0] - masks_stat[0])/(masks_stat[0]+1e-4) > 0.2:
                    pseudo_error_history = np.zeros(epochs)
                    for f in glob.glob(temp_evaluation_dir + "/masks/*.tif"):
                        os.remove(f)
                    for f in glob.glob(temp_evaluation_dir + "/*.tif"):
                        os.remove(f)
                    for f in glob.glob(temp_groundtruth_dir + "/masks/*.tif"):
                        os.remove(f)
                    for j in range(0, self.time_frame_each_ite):
                        for n,k in enumerate(self.trench_indices):
                            Image.fromarray(np.array(tifffile.imread(groundtruth_set + f"/{k}/groundtruth/{str(i + j).zfill(5)}.tif"))).save(f"{temp_evaluation_dir}/{str(time_frame+j*self.number_of_trenches+n).zfill(5)}.tif")
                    for j in range(0, self.time_frame_each_ite):
                        for n,k in enumerate(self.trench_indices):
                            Image.fromarray(np.array(tifffile.imread(groundtruth_set + f"/{k}/groundtruth/masks/{str(i + j).zfill(5)}_cp_masks_prediction.tif"))).save(f"{temp_groundtruth_dir}/masks/{str(time_frame+j*self.number_of_trenches+n).zfill(5)}.tif")

                    for training_i in range(epochs):
                        self.training_delta(1, training_set_in_use, 0, initialised, savefile)
                        pseudo_error_history = self.predicting_delta(temp_evaluation_dir, temp_evaluation_dir+"/masks", temp_groundtruth_dir = temp_groundtruth_dir, pseudo_error_history = pseudo_error_history, metrics_while_predicting = True, training_i = training_i)
                        
                    f1 = plt.figure()
                    plt.plot(np.array(range(1,len(pseudo_error_history)+1)), pseudo_error_history)
                    plt.xlabel("Epoch")
                    plt.ylabel("Segmentation Precision")
                    plt.savefig(f'temp_APvsEpoch_{time_frame}.png')
                    for f in glob.glob(convs_dir + "synth_*.tif"):
                        os.remove(f)
                    for f in glob.glob(masks_dir + "synth_*.tif"):
                        os.remove(f)
                    fine_tuned = True
                '''

                # ------------------------------------------------------
                # Train

                if self.use_SymBac and not self.synthetic_only:
                    self.synthetic_only = True
                    self.training_delta(epochs, training_set_in_use, 0, initialised, savefile)
                    if not self.continuous_tuning:
                        savefile = savefile.replace(self.models_list[self.init_model_index], 'tuned_model.hdf5')
                    _ = self.predicting_delta(inputs_folder, outputs_folder, savefile, metrics_while_predicting=False)
                    temp_masks_files = glob.glob(temp_masks_dir + "*.tif")
                    temp_convs_files = glob.glob(temp_convs_dir + "*.tif")
                    temp_masks_files = sorted(temp_masks_files)
                    temp_convs_files = sorted(temp_convs_files)
                    for k,temp_conv in enumerate(temp_convs_files[0:(valid_time_frames*self.number_of_trenches)]):
                        Image.fromarray(np.array(tifffile.imread(temp_conv))).save(f"{training_set_in_use}/convolutions/train_{str(k).zfill(5)}.tif")
                    for k,temp_mask in enumerate(temp_masks_files[0:(valid_time_frames*self.number_of_trenches)]):
                        Image.fromarray(np.array(tifffile.imread(temp_mask))).save(f"{training_set_in_use}/masks/train_{str(k).zfill(5)}.tif")
                    self.synthetic_only = False

                self.training_delta(epochs, training_set_in_use, 0, initialised, savefile)
                '''
                for img_i,img in enumerate(real_masks):
                    Image.fromarray(img).save(real_masks_files[img_i])
                for img_i,img in enumerate(real_convs):
                    Image.fromarray(img).save(real_convs_files[img_i])
                '''
                for f in glob.glob(convs_dir + "synth_*.tif"):
                    os.remove(f)
                for f in glob.glob(masks_dir + "synth_*.tif"):
                    os.remove(f)
                self.fine_tuned = True
                
                if self.fine_tuned:
                    savefile = savefile.replace(self.models_list[self.init_model_index], 'tuned_model.hdf5')
                _ = self.predicting_delta(inputs_folder, outputs_folder, savefile = savefile, metrics_while_predicting=False)
                temp_masks_files = glob.glob(temp_masks_dir + "*.tif")
                temp_convs_files = glob.glob(temp_convs_dir + "*.tif")
                temp_masks_files = sorted(temp_masks_files)
                temp_convs_files = sorted(temp_convs_files)

                if self.use_SymBac:
                    for j in range(0, int(len(temp_masks_files)/self.number_of_trenches)):
                        for k in range(self.number_of_trenches):
                            ret, result = cv2.connectedComponents(np.uint8(tifffile.imread(temp_masks_files[j*self.number_of_trenches+k])))
                            if self.synthetic_only:
                                Image.fromarray(np.uint8(result)).save(f"{training_set_in_use}/results/{self.trench_indices[k]}_synth/masks/train_{str(i + j).zfill(5)}.tif")
                            else:
                                Image.fromarray(np.uint8(result)).save(f"{training_set_in_use}/results/{self.trench_indices[k]}/masks/train_{str(i + j).zfill(5)}.tif")
                else:
                    for j in range(0, self.step_forward_each_ite):
                        for k in range(self.number_of_trenches):
                            ret, result = cv2.connectedComponents(np.uint8(tifffile.imread(temp_masks_files[j*self.number_of_trenches+k])))
                            Image.fromarray(np.uint8(result)).save(f"{training_set_in_use}/results/{self.trench_indices[k]}_pseudo/masks/train_{str(i + j).zfill(5)}.tif")

                for k,temp_mask in enumerate(temp_masks_files[0:(valid_time_frames*self.number_of_trenches)]):
                    Image.fromarray(np.array(tifffile.imread(temp_mask))).save(f"{training_set_in_use}/masks/train_{str(k).zfill(5)}.tif")
            else:
                if self.use_SymBac:
                    time_frame += self.number_of_trenches*self.time_frame_each_ite
                    i+= self.time_frame_each_ite
                else:
                    time_frame += self.number_of_trenches*self.step_forward_each_ite
                    i += self.step_forward_each_ite

            lengths_record.append(length)
            self.prev_length = length
        
        fig = plt.figure()
        plt.plot(lengths_record)
        plt.savefig("lengths_record.png")

if __name__ == "__main__":
    pseudo_labelling_pipeline = PseudoLabelling()
    pseudo_labelling_pipeline.main()