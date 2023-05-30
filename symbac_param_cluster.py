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
from pseudo_dev import erode, training_delta, predicting_delta, normalize99
import delta.utilities as utils
from delta.data import saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.data import trainGenerator_seg
from delta.model import unet_seg
from delta.utilities import cfg
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
# from scene_generation import scene_generation
import zarr
from PIL import Image
from joblib import Parallel, delayed
import tifffile
import os
import glob
import cv2
from benchmark import trench_indices
file_path = os.path.realpath(__file__)
file_path = file_path.replace('/symbac_param_cluster.py', '')
var = zarr.open("/media/ameyasu/Local Disk/SymBac/short_trench_hyperstack.zarr", mode='r')
analysis_dir = file_path + '/symbac_param_analysis_small'  # '~/rds/hpc-work/cuda/src'
# '~/rds/hpc-work/cuda/src/groundtruth'
groundtruth_dir = file_path + '/symbac_param_analysis_small/groundtruth'
lengths_list = np.linspace(2, 5.5, 8)
if not os.path.exists(groundtruth_dir):
    os.makedirs(groundtruth_dir+"/convolutions")
    os.makedirs(groundtruth_dir+"/masks")


def save_fig(i, j_index, j, var, length, trench_list_length):
    image_1 = np.array(var[j, i, :, :])
    sample = Image.fromarray(image_1)
    sample.save(analysis_dir + f"/{str(length)}/testing/convolutions/" +
                str(i*trench_list_length + j_index).zfill(5) + ".tif")
    sample.save(groundtruth_dir + "/convolutions/" +
                str(i*trench_list_length + j_index).zfill(5) + ".tif")


for length in lengths_list:
    if not os.path.exists(analysis_dir+f'/{str(length)}/training'):
        os.mkdir(analysis_dir + f"/{str(length)}")
        os.mkdir(analysis_dir + f"/{str(length)}/training")
    if not os.path.exists(analysis_dir+f'/{str(length)}/testing'):
        os.mkdir(analysis_dir + f"/{str(length)}/testing")
        os.mkdir(analysis_dir + f"/{str(length)}/testing/convolutions")
        os.mkdir(analysis_dir + f"/{str(length)}/testing/masks")
    if len(os.listdir(analysis_dir+f'/{str(length)}/testing/convolutions')) == 0:
        for i in range(800, 1100):
            Parallel(n_jobs=-2)(delayed(save_fig)(i, j_index, j, var, length,
                                                  len(trench_indices)) for j_index, j in enumerate(trench_indices))
            # for j_index, j in enumerate(trench_indices):
            #    save_fig(i, j_index, j, var, length, len(trench_indices))


pylab.rcParams['figure.figsize'] = (10.0, 8.0)
annType = ['segm', 'bbox', 'keypoints']
annType = annType[0]  # specify type here
apType = 0
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
print('Running demo for *%s* results.' % (annType))
hyperband = HyperBandScheduler(metric="score", mode="max", max_t=1000)
error_dict = {}
for length in lengths_list:
    error_dict[str(length)] = []

search_space = {
    "length": tune.grid_search(lengths_list)
}
from cellpose import models, core, utils, io, transforms
def omnipose_label(trench_index, model_name):
    use_GPU = core.use_gpu()
    print('>>> GPU activated? %d'%use_GPU)
    model_name = "/home/ameyasu/cuda_ws/src/benchmarking/cellpose_residual_on_style_on_concatenation_off_omni_omnipose_training_data_2022_10_07_00_50_12.177206_epoch_3999"
    model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_name)
    basedir = groundtruth_dir + "/convolutions"
    files = io.get_image_files(basedir)


    imgs = [io.imread(f) for f in files]

    # print some info about the images.
    nimg = len(imgs)
    print('number of images:',nimg)

    #fig = plt.figure(figsize=[40]*2) # initialize figure

    for k in range(len(imgs)):
        img = transforms.move_min_dim(imgs[k]) # move the channel dimension last
    if len(img.shape)>2:
        # imgs[k] = img[:,:,1] # could pick out a specific channel
        imgs[k] = np.mean(img,axis=-1) # or just turn into grayscale 
        imgs[k] = normalize99(imgs[k])
    
    chans = [0,0] #this means segment based on first channel, no second channel 

    n = [0] # make a list of integers to select which images you want to segment
    n = range(nimg) # or just segment them all 

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

    masks, flows, styles = model.eval([imgs[i] for i in n],channels=chans,rescale=rescale,mask_threshold=mask_threshold,transparency=transparency,
                                    flow_threshold=flow_threshold,omni=omni,cluster=cluster, resample=resample,verbose=verbose)
                
    io.save_masks(imgs, masks, flows, files, 
              png=False,
              tif=True, # whether to use PNG or TIF format
              suffix='prediction', # suffix to add to files if needed 
              save_flows=False, # saves both RGB depiction as *_flows.png and the raw components as *_dP.tif
              save_outlines=False, # save outline images 
              dir_above=0, # save output in the image directory or in the directory above (at the level of the image directory)
              in_folders=True, # save output in folders (recommended)
              save_txt=False, # txt file for outlines in imageJ
              save_ncolor=False) # save ncolor version of masks for visualization and editing 


if len(os.listdir(groundtruth_dir + "/convolutions/masks")) == 0:
    omnipose_label(trench_index = None, model_name = None)

def post_processing(path):
    kernel = np.ones((5, 5), np.uint8)
    result = tifffile.imread(path)
    result = erode(result, kernel)
    Image.fromarray(result).save(path)


'''
for length in lengths_list:
    width = 1.0
    if len(os.listdir(analysis_dir + f"/{str(length)}/training")) == 0:
        i = 0
        i_end = 4
        #scene_generation(length = 6, length_var = 0.1, width = 1.0, width_var = 0.1, sim_length = 4100, sample_size=4000, label_masks=True, initialised = True, save_dir='/tmp/test/')

        while i < i_end:
            try:
                scene_generation(length = length, length_var = 0.1, width = width, width_var = 0.1, sim_length = 1100, sample_size=1000, label_masks=True, initialised = True, save_dir= analysis_dir + f"/{str(length)}/training/")
                i+=1
            except:
                print("glitched")
        Parallel(n_jobs=-2)(delayed(post_processing)(path) for path in glob.glob(f"{analysis_dir}/{str(length)}/training/masks" + "/synth*.tif"))

for length in lengths_list:
    Image.fromarray(np.array(tifffile.imread(analysis_dir+"/synth_11111.tif"))).save(f"{analysis_dir}/{str(length)}/training/convolutions/synth_{str(11111).zfill(5)}.tif")
    Image.fromarray(np.array(tifffile.imread(analysis_dir+"/blank_trench_mask.tif"))).save(f"{analysis_dir}/{str(length)}/training/masks/synth_{str(11111).zfill(5)}.tif")
'''


def symbac_delta_training(length, initialised):
    training_delta(epochs=20, training_set_in_use=analysis_dir +
                   f"/{str(length)}/training/", training_set=1, initialised=initialised, savefile=analysis_dir + f"/{length}/model/pretrained_model.hdf5")
    initialised = True
    predicting_delta(analysis_dir+f"/{length}/testing/convolutions", analysis_dir +
                     f"/{length}/testing/masks", analysis_dir + f"/{length}/model/pretrained_model.hdf5")
    benchmark_maskdir = analysis_dir+f"/{length}/testing/masks"
    benchmark_maskfiles = sorted(glob.glob(benchmark_maskdir + "/*.tif"))
    prediction_maskfiles = [io.imread(f) for f in benchmark_maskfiles]

    groundtruth_maskdir = groundtruth_dir + "/convolutions/masks"
    groundtruth_maskfiles_o = sorted(glob.glob(groundtruth_maskdir + "/*.tif"))
    groundtruth_maskfiles = [np.asarray(cv2.resize(io.imread(
        f), (64, 256), interpolation=cv2.INTER_NEAREST)) for f in groundtruth_maskfiles_o]

    error_array = []

    for i in range(len(prediction_maskfiles)):
        groundtruth_cells, true_cell_sizes, cell_index = get_cells(
            groundtruth_maskfiles, i, "groundtruth", groundtruth_maskfiles, False)
        prediction_cells, prediction_cell_sizes, prediction_cell_index = get_cells(
            prediction_maskfiles, i, "prediction", groundtruth_maskfiles, False)
        groundtruth_json = file_path + '/groundtruth_temp_data.json'
        prediction_json = file_path + '/prediction_temp_data.json'
        cocoGt = COCO(groundtruth_json)
        cocoLarge = cocoGt.loadRes(prediction_json)
        cocoEval = COCOeval(cocoGt, cocoLarge, annType)
        cocoEval.params.imgIds = [i]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        # large_error_history[i] += normalise_constant*cocoEval.stats[apType]/average_size
        error_array.append(cocoEval.stats[apType])
        del cocoGt, cocoEval

    score = np.mean(error_array)
    return score, initialised

# @ray.remote(num_gpus=0.2)


def objective(config):
    import tensorflow as tf
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    if ray.get_gpu_ids():
        try:
            gpu = tf.config.PhysicalDevice(name=f'/physical_device:GPU:{str(ray.get_gpu_ids()[0])}', device_type='GPU')
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=2560)])
            # logical_gpus = tf.config.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(
            #    logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    length = config["length"]
    initialised = False
    timestep = 0
    for i in range(50):
        timestep += 100
        score, initialised = symbac_delta_training(length, initialised)
        checkpoint = Checkpoint.from_dict({"timestep": timestep})
        session.report({"score": score}, checkpoint=checkpoint)
        error_dict[str(length)].append(score)
    return {"score": score}


ray.init(num_cpus=10, num_gpus=1)
#ray.init(address = "10.43.74.122:6379")

tuner = tune.Tuner(
    tune.with_resources(
        objective,
        resources=tune.PlacementGroupFactory(
            [
                {"CPU": 3, "GPU": 0.5},
                {"CPU": 1},
                {"CPU": 1},
            ],
            strategy="PACK",
        ),
    ),
    tune_config=tune.TuneConfig(
        scheduler=hyperband,
    ),
    run_config=air.RunConfig(local_dir="./results",
                             name="symbac_param_analysis"),
    param_space=search_space,
)
results = tuner.fit()
print(results.get_best_result(metric="score", mode="max").config)

matplotlib.use("TkAgg")
matplotlib.interactive(True)

ax = None
for result in results:
    label = f"length={result.config['length']}"
    if ax is None:
        ax = result.metrics_dataframe.plot(
            "training_iteration", "score", label=label)
    else:
        result.metrics_dataframe.plot(
            "training_iteration", "score", ax=ax, label=label)
ax.set_ylabel("Precision")
ax.set_xlabel("Iteration")
ax.figure.savefig('symbac_analysis.png')
