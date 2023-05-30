from scipy.spatial import ConvexHull
import numpy as np
from scipy.signal import argrelextrema
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import gaussian_filter 
matplotlib.use('Agg')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
import numpy as np
from pathlib import Path
import os
from cellpose import io
import math
import cv2
from benchmark import get_cells, identification_NNS_Delaunay
import tifffile
annType = ['segm','bbox','keypoints']
annType = annType[0]      #specify type here
apType = [1, 8]
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.'%(annType))

'''
change ground truth folder
change prediction folder
change plot name
'''

error_history = np.zeros((8, 2100))

for param_i in range(8):

    benchmark_maskdir = "/home/ameyasu/cuda_ws/src/pseudo_labelling_synth/{}/training_set_in_use_delta/results/515_ascending/masks".format(param_i)
    benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
    pseudo_maskfiles = [io.imread(f) for f in benchmark_maskfiles]

    groundtruth_maskdir = "/home/ameyasu/cuda_ws/src/pseudo_labelling_synth/training_set_delta/masks"
    groundtruth_maskfiles_o = io.get_image_files(groundtruth_maskdir)
    #groundtruth_maskfiles_97 = []
    #for slice_i in range(140):
    #    groundtruth_maskfiles_97.append(groundtruth_maskfiles_o[8 + 11 * slice_i])
    groundtruth_maskfiles = [np.asarray(cv2.resize(io.imread(f), (64, 256), interpolation= cv2.INTER_NEAREST)) for f in groundtruth_maskfiles_o]


    if len(groundtruth_maskfiles) != len(benchmark_maskfiles):
        print("not compatible dimensions")
    
    standard_shape = groundtruth_maskfiles[0].shape
    normalise_constant = 0
    average_size = 1
    debug = False
    identification = True
    pseudo_error_history = np.zeros(2100)
    if identification:
        for i in range(0, 2100):
            
            standard_shape = groundtruth_maskfiles[0].shape

            error = 0.
            norm_g = 0.
            norm_b = 0.
            ground_truth_cells = []
            pseudo_model_cells = []
            average_cell_size = 0
            
            
            ground_truth_cells, true_cell_sizes, cell_index = get_cells(groundtruth_maskfiles, i, "groundtruth", groundtruth_maskfiles, debug)
            pseudo_model_cells, pseudo_cell_sizes, pseudo_cell_index = get_cells(pseudo_maskfiles, i, "pseudo", groundtruth_maskfiles, debug)
            
            pseudo_error_history = identification_NNS_Delaunay(ground_truth_cells, pseudo_model_cells, groundtruth_maskfiles, pseudo_maskfiles, true_cell_sizes, pseudo_cell_sizes, pseudo_error_history, i, cell_index)
                
        error_history[param_i] = gaussian_filter(pseudo_error_history, sigma=20)



    else:
        for i in range(0, 2100):
            print(i)
            ground_truth_cells, true_cell_sizes, cell_index = get_cells(groundtruth_maskfiles, i, "groundtruth", groundtruth_maskfiles, debug)
            pseudo_model_cells, pseudo_cell_sizes, pseudo_cell_index = get_cells(pseudo_maskfiles, i, "pseudo", groundtruth_maskfiles, debug)
            #delta_large_model_cells, delta_cell_sizes, delta_cell_index = get_cells(delta_large_maskfiles, i, "delta", groundtruth_maskfiles)
            
            if debug:
                groundtruth_json = f"runtime_comparison/json/groundtruth_data_{str(i)}.json"
                pseudo_json = f"runtime_comparison/json/pseudo_data_{str(i)}.json"
            else:
                groundtruth_json = f'/home/ameyasu/cuda_ws/src/pseudo_labelling_real/groundtruth_temp_data.json'
                pseudo_json = f'/home/ameyasu/cuda_ws/src/pseudo_labelling_real/pseudo_temp_data.json'
            
            imgIds= [i]
            cocoGt = COCO(groundtruth_json)
            cocoPseudo = cocoGt.loadRes(pseudo_json)  
            cocoEval = COCOeval(cocoGt,cocoPseudo,annType)
            cocoEval.params.imgIds  = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            #small_error_history[i] += normalise_constant*cocoEval.stats[apType]/average_size
            error_history[param_i][i] = (cocoEval.stats[apType[0]]+cocoEval.stats[apType[1]])/2

f1 = plt.figure()
for i, _list in enumerate(error_history):
    error_history[i] = gaussian_filter(_list, sigma=20)
    plt.plot(np.array(range(1,len(error_history[i])+1)), error_history[i], label=f'pseudo, param set {i}')


plt.legend()
plt.xlabel("Time Frame")
plt.ylabel("Sum of Deviations from True Cell Centres")
#plt.ylabel("Precision and Recall")
plt.savefig('average_pseudo_labelling_with_symbac_pos_only_0514_ascending_size.png')