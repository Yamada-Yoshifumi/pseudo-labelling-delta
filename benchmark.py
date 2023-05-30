from scipy.spatial import ConvexHull
import numpy as np
from scipy.signal import argrelextrema
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import gaussian_filter 
import matplotlib.tri as mtri
matplotlib.use('Agg')
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage import rotate
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

    for k in rval:
        temp = k[1]
        k[1] = k[0]
        k[0] = temp

    return rval

def minimum_bounding_contour(points, shape):
    mask_copy = np.zeros(shape) 
    for point in points:
        mask_copy[point[0]][point[1]] = 255
    ret, thresh = cv2.threshold(mask_copy, 127, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    contours, heir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.squeeze(np.asarray(contours))
    return contours

def loss_f(ground_truth_cells, small_model_cells, groundtruth_maskfiles, small_maskfiles, cell_sizes, small_cell_sizes, small_error_history, i, cell_index):
    ground_truth_cells_copy = ground_truth_cells.copy()
    small_model_cells_copy = small_model_cells.copy()
    cell_sizes_copy = cell_sizes.copy()
    small_cell_sizes_copy = small_cell_sizes.copy()
    t_p = 0
    f_n = len(ground_truth_cells)
    f_p = len(small_model_cells)
    j = 0
    if cell_index == 0:
        small_error_history[i] += 0 if f_n == 0 else 1
        return small_error_history

    while len(ground_truth_cells_copy) != 0:
        
        if len(small_model_cells) == 0:
            ground_truth_cells_copy.pop(0)
            cell_sizes_copy.pop(0)
            continue

        if j >= (len(small_model_cells_copy)):
            #small_error_history[i] += 1./(cell_index)
            #small_error_history[i] += 1.
            ground_truth_cells_copy.pop(0)
            cell_sizes_copy.pop(0)
            j = 0
            small_model_cells_copy = small_model_cells.copy()
            small_cell_sizes_copy = small_cell_sizes.copy()
            continue

        distance = np.sqrt((ground_truth_cells_copy[0][0] - small_model_cells_copy[j][0])**2 + (ground_truth_cells_copy[0][1] - small_model_cells_copy[j][1])**2)
        if  distance < np.sqrt(cell_sizes_copy[0][1]**2 + cell_sizes_copy[0][0]**2)/4 and distance < np.sqrt(small_cell_sizes_copy[j][1]**2 + small_cell_sizes_copy[j][0]**2)/4:
            #t_p += (distance/np.sqrt(cell_sizes_copy[0][1]**2 + cell_sizes_copy[0][0]**2))
            t_p += 1 
            ground_truth_cells_copy.pop(0)
            cell_sizes_copy.pop(0)
            small_model_cells.pop(j)
            small_model_cells_copy = small_model_cells.copy()
            small_cell_sizes.pop(j)
            small_cell_sizes_copy = small_cell_sizes.copy()
            j = 0
        else:
            j += 1
    f_n -= t_p
    f_p -= t_p
    f_n = 0 if f_n < 0 else f_n
    f_p = 0 if f_p < 0 else f_n
    if (f_n + t_p == 0 or f_p + t_p == 0):
        return small_error_history

    with open("record_tp_fp.txt", "w") as f:
        f.write("tp" + ", " + str(t_p) + "\n")
        f.write("fn" + ", " + str(f_n) + "\n")
        f.write("false positive contribution: " + str((f_n + f_p)/(f_n + t_p + f_p)))
        f.flush()
    small_error_history[i] += min(1, (f_n + f_p)/(f_n + t_p + f_p))
    return small_error_history

from scipy.spatial import Delaunay
def identification_NNS_Delaunay(ground_truth_cells, small_model_cells, groundtruth_maskfiles, small_maskfiles, cell_sizes, small_cell_sizes, small_error_history, i, cell_index):
    ground_truth_cells_copy = ground_truth_cells.copy()
    small_model_cells_copy = small_model_cells.copy()
    t_p = 0
    f_n = len(ground_truth_cells_copy)
    f_p = len(small_model_cells_copy)

    while (f_p < 3):
        arr = np.random.random_sample((1,2))*5
        small_model_cells_copy = np.append(small_model_cells_copy, arr, 0)
        f_p = len(small_model_cells_copy)
    print(small_model_cells_copy)
    tri = Delaunay(small_model_cells_copy)
    '''
    if i == 700:
        temp_plot = plt.figure()
        plt.triplot(np.asarray(small_model_cells_copy)[:,1], np.asarray(small_model_cells_copy)[:,0], list(tri.simplices))
        plt.plot(np.asarray(small_model_cells_copy)[:,1], np.asarray(small_model_cells_copy)[:,0], 'o')
        plt.ylim(max(np.asarray(small_model_cells_copy)[:,0] + 25), min(np.asarray(small_model_cells_copy)[:,0] - 25))
        #plt.rcParams["figure.figsize"] = [3.50, 7.50]
        #plt.rcParams["figure.autolayout"] = True

        plt.savefig("delaunay_demo.png")
    '''
    if cell_index == 0:
        small_error_history[i] += 0 if f_n == 0 else 1
        return small_error_history
    print("Image: " + str(i))
    j = 0
    matched_nodes = []
    for gt_cell in ground_truth_cells_copy:
        prev_node = 0
        current_node = 0
        found = False
        print("Cell: " + str(j))
        j+=1
        while not found:
            prev_node = current_node
            current_dist = np.linalg.norm(np.array(small_model_cells_copy[current_node]) - np.array(gt_cell))
            small_model_indices_delaunay = tri.simplices
            for simplex in small_model_indices_delaunay:
                if current_node in simplex:
                    for neighbour in simplex:
                        if np.linalg.norm(np.array(small_model_cells_copy[neighbour]) - np.array(gt_cell)) < current_dist:
                            found = False
                            current_node = neighbour
                            current_dist = np.linalg.norm(np.array(small_model_cells_copy[neighbour]) - np.array(gt_cell))
            if current_node == prev_node:
                found = True
                if current_node not in matched_nodes:
                    matched_nodes.append(current_node)
                    small_error_history[i] += current_dist
                else:
                    small_error_history[i] += np.linalg.norm([np.average(cell_sizes[:][0]), np.average(cell_sizes[:][1])])
    try:        
        f_p = len(small_model_cells) - len(matched_nodes)
        small_error_history[i] += f_p*np.linalg.norm([np.average(cell_sizes[:][0]), np.average(cell_sizes[:][1])])
    except:
        return small_error_history
    return small_error_history

import json
import cv2
import pycocotools.mask as mask_util
import pycocotools._mask as _mask_util
import os

def RLE(image_array, cell_index):
    rle_decoded = []
    current_i = 0
    currently_on_cell = False
    def count_cell_pixel(image_array, current_i, cell_index):
        current_count = 0
        while True:    
            current_i += 1
            if current_i== len(image_array):
                return current_count, current_i
            if image_array[current_i] == cell_index:
                current_count += 1
            else:
                break
        return current_count, current_i

    def count_background_pixel(image_array, current_i, cell_index):
        current_count = 0
        while True:
            current_i += 1
            if current_i == len(image_array):
                return current_count, current_i
            if image_array[current_i] != cell_index:
                current_count += 1
            else:
                break
        return current_count, current_i

    while (current_i < len(image_array)):
        if currently_on_cell:
            current_count, current_i = count_cell_pixel(image_array, current_i, cell_index)
        else:
            current_count, current_i = count_background_pixel(image_array, current_i, cell_index)
        currently_on_cell = not currently_on_cell
        rle_decoded.append(current_count)
    
    return rle_decoded

def binarize_cell_mask(image_array, cell_index):
    image_array_copy = image_array.copy()
    for i in range(len(image_array)):
        for j in range(len(image_array[0])):
            if image_array[i][j] != cell_index:
                image_array_copy[i][j] = 0
            else:
                image_array_copy[i][j] = 1
    return image_array_copy

def write_json(json_data, description, cell_index, large_maskfiles, i, large_model_cells, cell_sizes):
    one_cell = np.where(np.array(large_maskfiles[i]) == cell_index)
    if len(one_cell) == 2:
        if len(one_cell[0]) == 0:
            return large_model_cells, cell_sizes, json_data
        max_x = np.max(one_cell[0])
        max_y = np.max(one_cell[1])
        min_x = np.min(one_cell[0])
        min_y = np.min(one_cell[1])
        bounding_rect = (int(min_x), int(min_y), int(max_x), int(max_y))
        cell_pixels_image = []
        for pixel_x, pixel_y in zip(one_cell[0], one_cell[1]):
            cell_pixels_image.append([int(pixel_x), int(pixel_y)])
        x = np.average(one_cell[0])
        y = np.average(one_cell[1])
        large_model_cells[cell_index - 1] = [x,y]
        try:
            bounding_box = minimum_bounding_rectangle(cell_pixels_image)
            segmentation = minimum_bounding_contour(cell_pixels_image, np.array(large_maskfiles[0]).shape)
            scale_1 = np.sqrt((bounding_box[0][0]-bounding_box[1][0])**2 + (bounding_box[0][1]-bounding_box[1][1])**2)
            scale_2 = np.sqrt((bounding_box[1][0]-bounding_box[2][0])**2 + (bounding_box[1][1]-bounding_box[2][1])**2)
            if scale_1 > scale_2:
                width = scale_2
                length = scale_1
                if abs(bounding_box[0][0]-bounding_box[1][0]) < 1e-4:
                    angle = np.pi/2
                else:
                    angle = np.arctan((bounding_box[0][1]-bounding_box[1][1])/(bounding_box[0][0]-bounding_box[1][0])) + np.pi
                while angle - np.pi > 0:
                    angle -= np.pi
            else:
                width = scale_1
                length = scale_2
                if abs(bounding_box[2][0]-bounding_box[1][0]) < 1e-4:
                    angle = np.pi/2
                else:
                    angle = np.arctan((bounding_box[2][1]-bounding_box[1][1])/(bounding_box[2][0]-bounding_box[1][0])) + np.pi
                while angle - np.pi > 0:
                    angle -= np.pi
            cell_sizes[cell_index - 1] = [length, width, angle]
            segmentation = np.append(segmentation, segmentation[0])
            if description == "groundtruth":
                json_data["annotations"].append(
                    {
                        "id": cell_index,
                        "category_id": 22,
                        "iscrowd": 0,
                        #"segmentation": {
                        #    "size": [len(large_maskfiles[i]), len(large_maskfiles[i][0])],
                        #    "counts": mask_util.encode(np.asfortranarray(binarize_cell_mask(np.array(large_maskfiles[i]), cell_index)))#[[int(integer_v) for integer_v in bounding_box.ravel()]]#
                        #},
                        #"segmentation": {
                        #    "size": [len(large_maskfiles[i]), len(large_maskfiles[i][0])],
                        #    "counts": RLE(np.array(large_maskfiles[i]).ravel(), cell_index)#[[int(integer_v) for integer_v in bounding_box.ravel()]]#
                        #},
                        "segmentation": [[int(integer_v) for integer_v in segmentation.ravel()]],
                        "image_id": i,
                        "area": int(len(np.where(np.array(large_maskfiles[i]) == cell_index)[0])),
                        "bbox": [int(min_y), int(min_x), int(max_y - min_y), int(max_x - min_x)],
                    }
                )
            else:
                json_data.append({
                    "category_id": 22,
                    #"segmentation": [RLE(np.array(large_maskfiles[i]).ravel(), cell_index)],
                    #"segmentation": {
                    #    "size": [len(large_maskfiles[i]), len(large_maskfiles[i][0])],
                    #    "counts": mask_util.encode(np.asfortranarray(binarize_cell_mask(np.array(large_maskfiles[i]), cell_index)))#[[int(integer_v) for integer_v in bounding_box.ravel()]]#
                    #},
                    #"segmentation": {
                    #    "size": [len(large_maskfiles[i]), len(large_maskfiles[i][0])],
                    #    "counts": RLE(np.array(large_maskfiles[i]).ravel(), cell_index)#[[int(integer_v) for integer_v in bounding_box.ravel()]]#
                    #},#[[int(integer_v) for integer_v in bounding_box.ravel()]],#RLE(np.array(large_maskfiles[i]).ravel(), cell_index),
                    "segmentation": [[int(integer_v) for integer_v in segmentation.ravel()]],
                    "image_id": i,
                    "bbox": [int(min_y), int(min_x), int(max_y - min_y), int(max_x - min_x)],
                    "score": 1.0
                })
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            bounding_box = np.array([[bounding_rect[0], bounding_rect[1]],[bounding_rect[2], bounding_rect[3]], [bounding_rect[0]+ 1, bounding_rect[1] + 1], [bounding_rect[0] + 1, bounding_rect[1] + 1]])
            cell_sizes[cell_index - 1] = [0, 0, 0]
        return large_model_cells, cell_sizes, json_data
    else:
        return large_model_cells, cell_sizes, json_data

def get_cells( large_maskfiles, i, description, gt_maskfiles,debug):
    cell_count = int(np.max(np.array(large_maskfiles[i]).ravel()))
    cell_sizes = [(0,0,0) for _ in range(cell_count)]
    large_model_cells = [[0,0] for _ in range(cell_count)]
    cell_index = 1
    if description == "groundtruth":
        json_data = {
            "info": {
                "description": description,
                "url": "http://cocodataset.org",
                "version": "1.0",
                "year": 2017,
                "contributor": "COCO Consortium",
                "date_created": "2017/09/01"
            },
            "licenses": [
                {
                    "url": "http://creativecommons.org/licenses/by/2.0/",
                    "id": 4,
                    "name": "Attribution License"
                }
            ],
            "images": [
            {
                "id": i,
                "license": 4,
                "coco_url": "http://images.cocodataset.org/val2017/000000242287.jpg",
                "flickr_url": "http://farm3.staticflickr.com/2626/4072194513_edb6acfb2b_z.jpg",
                "width": np.size(np.array(large_maskfiles[i]), 1),
                "height": np.size(np.array(large_maskfiles[i]), 0),
                "file_name": f"{i}.jpg",
                "date_captured": "2013-11-15 02:41:42"
            }],
            "annotations": [

            ],
            "categories": [
                {
                    "supercategory": "animal",
                    "id": 22,
                    "name": "elephant"
                }
            ],
        }
    else:
        json_data = []
    for cell_index in range(1, cell_count + 1):
        large_model_cells, cell_sizes, json_data = write_json(json_data, description, cell_index, large_maskfiles, i, large_model_cells, cell_sizes)

    if len(json_data) == 0:
        json_data.append({
            "category_id": 1,
            "segmentation": [[0,0,0,0,0,0,0,0]],
            "image_id": i,
            "bbox": [0,0,0,0],
            "score": 0.0
        })
    file_path = os.path.realpath(__file__)
    file_path = file_path.replace('/benchmark.py', '')
    if not debug:
        #with open(f'/home/ameyasu/cuda_ws/src/pseudo_labelling_real/{description}_temp_data.json', 'w', encoding='utf-8') as f:
        with open(file_path + f'/{description}_temp_data.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

    else:
        #with open(f'runtime_comparison/json/{description}_data_{i}.json', 'w', encoding='utf-8') as f:
        with open(file_path + f'/{description}_data_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

    return large_model_cells, cell_sizes, cell_count

def draw_contour(groundtruth_maskfiles, i, small_json, groundtruth_json, model_name):
    mask = groundtruth_maskfiles[i]
    fig = plt.figure()
    with open(small_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for k in range(len(data)):
            vertices = data[k]["segmentation"][0]
            for j in range(0, len(vertices)-3, 2):
                plt.plot([vertices[j], vertices[j+2]], [vertices[j+1], vertices[j+3]], color='red')
    with open(groundtruth_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for k in range(len(data["annotations"])):
            vertices = data["annotations"][k]["segmentation"][0]
            for j in range(0, len(vertices)-3, 2):
                plt.plot([vertices[j], vertices[j+2]], [vertices[j+1], vertices[j+3]], color='blue')
    precision = str(round((cocoEval.stats[apType[0]]+cocoEval.stats[apType[1]])/2,2))
    plt.text(0, 0, "Precision: " + precision, fontsize=12)

    imgplot = plt.imshow(mask)
    imgplot.set_cmap('magma_r')
    plt.savefig(f"runtime_comparison/{model_name}/{str(i).zfill(5)}_AP_{precision}.png")
import numpy as np
from pathlib import Path
import os
import tifffile
import math
#trench_indices = [515, 1823, 212, 978, 1390, 509, 2642, 693, 1910, 1459, 97, 1095, 1865, 1102, 2512, 2080, 2509, 2006, 59, 1415, 133, 1877, 1768, 1926]
trench_indices = [515, 1823, 212, 978, 509, 2642, 693, 1910, 97, 1865]
#trench_indices = [515, 509, 978, 1823, 212]
#trench_indices = [693, 1910, 1865, 97, 2642]
#trench_indices = [515]
if __name__ == "__main__":
    from cellpose import io

    benchmark_methods = ["pos_only", "a_p", "derivative_mc"]
    method = benchmark_methods[1]
    debug = False
    if (method == "pos_only"):

        average = len(trench_indices)
        mixed_error_history = np.zeros(1570)
        small_error_history = np.zeros(1570)
        large_error_history = np.zeros(1570)
        pseudo_error_history = np.zeros(1570)
        for iteration in trench_indices:
            #benchmark_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/large/masks".format(iteration)
            benchmark_maskdir = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta/results/{}_pseudo/masks".format(iteration)
            benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
            large_maskfiles = [io.imread(f) for f in benchmark_maskfiles]

            #benchmark_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/small/masks".format(iteration)
            benchmark_maskdir = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta/results/{}_synth/masks".format(iteration)
            benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
            small_maskfiles = [io.imread(f) for f in benchmark_maskfiles]
            

            benchmark_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/mixed/masks".format(iteration)
            benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
            mixed_maskfiles = [io.imread(f) for f in benchmark_maskfiles]

            benchmark_maskdir = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta/results/{}/masks".format(iteration)
            benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
            pseudo_maskfiles = [io.imread(f) for f in benchmark_maskfiles]
            #for k in range(800):
            #    pseudo_maskfiles.insert(0, io.imread(benchmark_maskfiles[0]))

            groundtruth_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/groundtruth/masks/*.tif".format(iteration)
            groundtruth_maskfiles_o = sorted(glob.glob(groundtruth_maskdir))
            groundtruth_maskfiles = [np.asarray(cv2.resize(tifffile.imread(f), (64, 256), interpolation=cv2.INTER_NEAREST)) for f in groundtruth_maskfiles_o]
            

            if len(groundtruth_maskfiles) != len(benchmark_maskfiles):
                print("not compatible arrays")
                print(len(groundtruth_maskfiles))
                print(len(benchmark_maskfiles))
            '''
            if groundtruth_maskfiles[0].shape != mixed_maskfiles[0].shape:
                print("not compatible dimensions")
                print(groundtruth_maskfiles[0].shape)
                print(mixed_maskfiles[0].shape)
                break
            '''
            standard_shape = groundtruth_maskfiles[0].shape


            for i in range(1570):
            #for i in np.linspace(800, 1569, 770):
                
                i = int(i)

                error = 0.
                norm_g = 0.
                norm_b = 0.
                ground_truth_cells = []
                large_model_cells = []
                small_model_cells = []
                mixed_model_cells = []
                pseudo_model_cells = []
                delta_large_model_cells = []
                average_cell_size = 0
                
                
                small_model_cells, small_cell_sizes, small_cell_index = get_cells(small_maskfiles, i, "small", groundtruth_maskfiles, debug)
                large_model_cells, large_cell_sizes, large_cell_index = get_cells(large_maskfiles, i, "large", groundtruth_maskfiles, debug)
                mixed_model_cells, mixed_cell_sizes, mixed_cell_index = get_cells(mixed_maskfiles,  i, "mixed", groundtruth_maskfiles, debug)
                pseudo_model_cells, pseudo_cell_sizes, pseudo_cell_index = get_cells(pseudo_maskfiles,  i, "pseudo", groundtruth_maskfiles, debug)
                ground_truth_cells, true_cell_sizes, cell_index = get_cells(groundtruth_maskfiles, i, "groundtruth", groundtruth_maskfiles,  debug)
                
                mixed_error_history = identification_NNS_Delaunay(ground_truth_cells, mixed_model_cells, groundtruth_maskfiles, mixed_maskfiles, true_cell_sizes, mixed_cell_sizes, mixed_error_history, i, cell_index)
                small_error_history = identification_NNS_Delaunay(ground_truth_cells, small_model_cells, groundtruth_maskfiles, small_maskfiles, true_cell_sizes, small_cell_sizes, small_error_history, i, cell_index)
                large_error_history = identification_NNS_Delaunay(ground_truth_cells, large_model_cells, groundtruth_maskfiles, large_maskfiles, true_cell_sizes, large_cell_sizes, large_error_history, i, cell_index)
                pseudo_error_history = identification_NNS_Delaunay(ground_truth_cells, pseudo_model_cells, groundtruth_maskfiles, pseudo_maskfiles, true_cell_sizes, pseudo_cell_sizes, pseudo_error_history, i, cell_index)
                #delta_large_error_history_f = loss_f(ground_truth_cells, delta_large_model_cells, groundtruth_maskfiles, delta_large_maskfiles, true_cell_sizes, delta_cell_sizes, delta_large_error_history_f, i, cell_index)

                #pseudo_error_history_b = loss_f(ground_truth_cells, pseudo_model_cells, groundtruth_maskfiles, pseudo_maskfiles, true_cell_sizes, pseudo_cell_sizes, pseudo_error_history_b, i, pseudo_cell_index)
                #delta_large_error_history_b = loss_f(ground_truth_cells, delta_large_model_cells, groundtruth_maskfiles, delta_large_maskfiles, true_cell_sizes, delta_cell_sizes, delta_large_error_history_b, i, delta_cell_index)
            with open("record.txt", "w") as f:
                f.write(str(iteration))

        large_error_history /= average
        small_error_history /= average
        mixed_error_history /= average
        pseudo_error_history /= average
            
        smoothed_large_error_history = gaussian_filter(large_error_history, sigma=20)
        smoothed_mixed_error_history = gaussian_filter(mixed_error_history, sigma=20)
        smoothed_small_error_history = gaussian_filter(small_error_history, sigma=20)
        smoothed_pseudo_error_history = gaussian_filter(pseudo_error_history, sigma=20)

        f1 = plt.figure()
        plt.rcParams.update({'font.size': 15})
        #plt.plot(np.linspace(100,len(smoothed_large_error_history[:1100]), len(smoothed_small_error_history[100:1100])), smoothed_large_error_history[100:1100], label='large')
        #plt.plot(np.linspace(100,len(smoothed_large_error_history[:1100]), len(smoothed_small_error_history[100:1100])), smoothed_small_error_history[100:1100], label='small')
        #plt.plot(np.linspace(100,len(smoothed_large_error_history[:1100]), len(smoothed_small_error_history[100:1100])), smoothed_mixed_error_history[100:1100], label='mixed')
        #plt.plot(np.linspace(100,len(smoothed_large_error_history[:1100]), len(smoothed_small_error_history[100:1100])), smoothed_pseudo_error_history[100:1100], label='pseudo labeling')
        #plt.plot(np.array(range(1,len(smoothed_delta_large_error_history)+1)), smoothed_delta_large_error_history, label='initial model for reference')

        plt.plot(np.linspace(100,len(smoothed_mixed_error_history), len(smoothed_mixed_error_history[100:1500])), smoothed_mixed_error_history[100:1500], label='mixture model')
        plt.plot(np.linspace(100,len(smoothed_large_error_history), len(smoothed_large_error_history[100:1500])), smoothed_large_error_history[100:1500], label='Pseudo-label only')
        plt.plot(np.linspace(100,len(smoothed_small_error_history), len(smoothed_large_error_history[100:1500])), smoothed_small_error_history[100:1500], label='synthetic only')
        plt.plot(np.linspace(100,len(smoothed_mixed_error_history), len(smoothed_pseudo_error_history[100:1500])), smoothed_pseudo_error_history[100:1500], label='two-step fine-tuning')
        #plt.plot(np.array(range(1,len(mixed_error_history)+1)), mixed_error_history, label='mixed')
        #plt.plot(np.array(range(1,len(pseudo_error_history)+1)), pseudo_error_history, label='pseudo labelling')
        #plt.plot(np.array(range(1,len(delta_large_error_history)+1)), delta_large_error_history, label='initial model for reference')

        plt.legend()
        plt.xlabel("Time Frame")
        plt.ylabel("Sum of Deviations from True Cell Center")
        plt.savefig('0530_pseudo_comparison_pos_only_unfamiliar.png')
        #plt.savefig('average_pseudo_labelling_with_symbac_0516_pos_only.png')


    elif(method=="a_p"):
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

        annType = ['segm','bbox','keypoints']
        annType = annType[0]      #specify type here
        apType = [1, 8]
        prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
        print('Running demo for *%s* results.'%(annType))

        average = len(trench_indices)
        large_error_history = np.zeros(1570)
        mixed_error_history = np.zeros(1570)
        small_error_history = np.zeros(1570)
        pseudo_error_history = np.zeros(1570)
        pseudo_normal_error_history = np.zeros(1570)
        '''
        if average == 1:
            for f in glob.glob("runtime_comparison/large/*.png"):
                os.remove(f)
            for f in glob.glob("runtime_comparison/small/*.png"):
                os.remove(f)
            for f in glob.glob("runtime_comparison/mixed/*.png"):
                os.remove(f)
            for f in glob.glob("runtime_comparison/pseudo/*.png"):
                os.remove(f)
        '''
        for iteration in trench_indices:
            with open("record_trench.txt", "w") as f:
                f.write("trench: " + str(iteration))
            
            #benchmark_maskdir =  "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta/results/{}_pseudo/masks".format(iteration)
            benchmark_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/large/masks".format(iteration)
            benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
            large_maskfiles = [io.imread(f) for f in benchmark_maskfiles]

            benchmark_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/small/masks".format(iteration)
            #benchmark_maskdir =  "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta/results/{}_synth/masks".format(iteration)
            benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
            small_maskfiles = [io.imread(f) for f in benchmark_maskfiles]
            
            #benchmark_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/mixed/masks".format(iteration)
            benchmark_maskdir =  "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta/results/{}/masks".format(iteration)
            benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
            pseudo_normal_maskfiles = [io.imread(f) for f in benchmark_maskfiles]

            benchmark_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/mixed/masks".format(iteration)
            benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
            mixed_maskfiles = [io.imread(f) for f in benchmark_maskfiles]

            benchmark_maskdir = "/home/ameyasu/cuda_ws/src/pseudo_labelling_real/training_set_in_use_delta/results/{}_synth/masks".format(iteration)
            benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
            pseudo_maskfiles = [io.imread(f) for f in benchmark_maskfiles]
            
            groundtruth_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/groundtruth/masks".format(iteration)
            groundtruth_maskfiles_o = io.get_image_files(groundtruth_maskdir)
            groundtruth_maskfiles = [np.asarray(cv2.resize(io.imread(f), (64, 256), interpolation= cv2.INTER_NEAREST)) for f in groundtruth_maskfiles_o]
            '''
            for k in range(200):  
                mixed_maskfiles.pop(500)              
                groundtruth_maskfiles.pop(500)
                pseudo_normal_maskfiles.pop(500)
            for k in range(100):
                large_maskfiles.pop(500)
            for k in range(50):
                small_maskfiles.pop(500)
            '''
            if len(groundtruth_maskfiles) != len(benchmark_maskfiles):
                print("not compatible dimensions")
            
            standard_shape = groundtruth_maskfiles[0].shape
            normalise_constant = 0
            average_size = 1

            for i in range(1570):
            #for i in np.linspace(800, 1569, 770):
                print(i)
                
                ground_truth_cells, true_cell_sizes, cell_index = get_cells(groundtruth_maskfiles, int(i), "groundtruth", groundtruth_maskfiles, debug)
                small_model_cells, small_cell_sizes, small_cell_index = get_cells(small_maskfiles, int(i), "small", groundtruth_maskfiles, debug)
                large_model_cells, large_cell_sizes, large_cell_index = get_cells(large_maskfiles, int(i), "large", groundtruth_maskfiles, debug)
                mixed_model_cells, mixed_cell_sizes, mixed_cell_index = get_cells(mixed_maskfiles, int(i), "mixed", groundtruth_maskfiles,debug)
                pseudo_model_cells, pseudo_cell_sizes, pseudo_cell_index = get_cells(pseudo_maskfiles, int(i), "pseudo", groundtruth_maskfiles, debug)
                #pseudo_normal_model_cells, pseudo_normal_cell_sizes, pseudo_normal_cell_index = get_cells(pseudo_normal_maskfiles, int(i), "pseudo_normal", groundtruth_maskfiles, debug)
                
                if debug:
                    groundtruth_json = f"runtime_comparison/json/groundtruth_data_{str(i)}.json"
                    small_json = f"runtime_comparison/json/small_data_{str(i)}.json"
                    large_json = f"runtime_comparison/json/large_data_{str(i)}.json"
                    mixed_json = f"runtime_comparison/json/mixed_data_{str(i)}.json"
                    pseudo_json = f"runtime_comparison/json/pseudo_data_{str(i)}.json"
                else:
                    groundtruth_json = f'/home/ameyasu/cuda_ws/src/pseudo_labelling_real/groundtruth_temp_data.json'
                    small_json = f'/home/ameyasu/cuda_ws/src/pseudo_labelling_real/small_temp_data.json'
                    large_json = f'/home/ameyasu/cuda_ws/src/pseudo_labelling_real/large_temp_data.json'
                    mixed_json = f'/home/ameyasu/cuda_ws/src/pseudo_labelling_real/mixed_temp_data.json'
                    pseudo_json = f'/home/ameyasu/cuda_ws/src/pseudo_labelling_real/pseudo_temp_data.json'
                    pseudo_normal_json = f'/home/ameyasu/cuda_ws/src/pseudo_labelling_real/pseudo_normal_temp_data.json'
                '''
                if len(true_cell_sizes)>1 and len(true_cell_sizes[0]) == 3:
                    try:
                        average_length = np.average(np.squeeze(true_cell_sizes[:][0]))
                        average_width = np.average(np.squeeze(true_cell_sizes[:][1]))
                        average_size = average_length*average_width
                    except:
                        average_length = true_cell_sizes[0][0]
                        average_width = true_cell_sizes[0][1]
                        average_size = average_length*average_width
                    if (i < 20):
                        normalise_constant += average_size/20
                        continue
                '''
                imgIds= [int(i)]
                
                cocoGt = COCO(groundtruth_json)
                cocoSmall = cocoGt.loadRes(small_json)  
                cocoEval = COCOeval(cocoGt,cocoSmall,annType)
                cocoEval.params.imgIds  = imgIds
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                #small_error_history[i] += normalise_constant*cocoEval.stats[apType]/average_size
                small_error_history[int(i)] += (cocoEval.stats[apType[0]]+cocoEval.stats[apType[1]])/2

                #if (average == 1):
                #    index = trench_indices[0]
                #    draw_contour(groundtruth_maskfiles, i, small_json, groundtruth_json, "small")
                
                cocoGt = COCO(groundtruth_json)
                cocoLarge = cocoGt.loadRes(large_json)
                cocoEval = COCOeval(cocoGt,cocoLarge,annType)
                cocoEval.params.imgIds  = imgIds
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                #large_error_history[i] += normalise_constant*cocoEval.stats[apType]/average_size
                large_error_history[int(i)] += (cocoEval.stats[apType[0]]+cocoEval.stats[apType[1]])/2
                
                #if (average == 1):
                #    index = trench_indices[0]
                #    draw_contour(groundtruth_maskfiles, i, large_json, groundtruth_json, "large")
                

                
                cocoGt = COCO(groundtruth_json)
                cocoMixed = cocoGt.loadRes(mixed_json)
                cocoEval = COCOeval(cocoGt,cocoMixed,annType)
                cocoEval.params.imgIds  = imgIds
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                #mixed_error_history[i] += normalise_constant*cocoEval.stats[apType]/average_size
                mixed_error_history[int(i)] += (cocoEval.stats[apType[0]]+cocoEval.stats[apType[1]])/2
                
                #if (average == 1):
                #    index = trench_indices[0]
                #    draw_contour(groundtruth_maskfiles, i, mixed_json, groundtruth_json, "mixed")
  
                cocoGt = COCO(groundtruth_json)
                cocoPseudo = cocoGt.loadRes(pseudo_json)
                cocoEval = COCOeval(cocoGt,cocoPseudo,annType)
                cocoEval.params.imgIds  = imgIds
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                #pseudo_error_history[i] += normalise_constant*cocoEval.stats[apType]/average_size
                pseudo_error_history[int(i)] += (cocoEval.stats[apType[0]]+cocoEval.stats[apType[1]])/2
                '''
                #if (average == 1):
                #    index = trench_indices[0]
                #    draw_contour(groundtruth_maskfiles, i, pseudo_json, groundtruth_json, "pseudo")
                
                cocoGt = COCO(groundtruth_json)
                cocoPseudoNormal = cocoGt.loadRes(pseudo_normal_json)
                cocoEval = COCOeval(cocoGt,cocoPseudoNormal,annType)
                cocoEval.params.imgIds  = imgIds
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                #pseudo_error_history[i] += normalise_constant*cocoEval.stats[apType]/average_size
                pseudo_normal_error_history[int(i)] += (cocoEval.stats[apType[0]]+cocoEval.stats[apType[1]])/2
                '''

        large_error_history /= average
        small_error_history /= average
        mixed_error_history /= average
        pseudo_error_history /= average
        #pseudo_normal_error_history /= average
        '''
        smooth_window = np.ones(100)*0.01

        smoothed_large_error_history = np.zeros(len(large_error_history)-len(smooth_window))
        for i in range(len(large_error_history)-len(smooth_window)):
            temp_list = [smooth_window[j]*large_error_history[j+i] for j in range(len(smooth_window))]
            smoothed_large_error_history[i] = np.sum(temp_list)

        smoothed_small_error_history = np.zeros(len(small_error_history)-len(smooth_window))
        for i in range(len(small_error_history)-len(smooth_window)):
            temp_list = [smooth_window[j]*small_error_history[j+i] for j in range(len(smooth_window))]
            smoothed_small_error_history[i] = np.sum(temp_list)

        smoothed_mixed_error_history = np.zeros(len(mixed_error_history)-len(smooth_window))
        for i in range(len(mixed_error_history)-len(smooth_window)):
            temp_list = [smooth_window[j]*mixed_error_history[j+i] for j in range(len(smooth_window))]
            smoothed_mixed_error_history[i] = np.sum(temp_list)

        smoothed_pseudo_error_history = np.zeros(len(pseudo_error_history)-len(smooth_window))
        for i in range(len(pseudo_error_history)-len(smooth_window)):
            temp_list = [smooth_window[j]*pseudo_error_history[j+i] for j in range(len(smooth_window))]
            smoothed_pseudo_error_history[i] = np.sum(temp_list)
        '''
        smoothed_large_error_history = gaussian_filter(large_error_history, sigma=20)
        smoothed_mixed_error_history = gaussian_filter(mixed_error_history, sigma=20)
        smoothed_small_error_history = gaussian_filter(small_error_history, sigma=20)
        smoothed_pseudo_error_history = gaussian_filter(pseudo_error_history, sigma=20)
        #smoothed_pseudo_normal_error_history = gaussian_filter(pseudo_normal_error_history, sigma=20)
        '''
        smoothed_large_error_history = large_error_history
        smoothed_mixed_error_history = mixed_error_history
        smoothed_small_error_history = small_error_history
        smoothed_pseudo_error_history = pseudo_error_history
        '''
        f1 = plt.figure()
        plt.rcParams.update({'font.size': 22})

        #plt.plot(np.linspace(100,len(smoothed_mixed_error_history[:1500]), len(smoothed_mixed_error_history[100:1500])), smoothed_mixed_error_history[100:1500], label='mixture model')
        #plt.plot(np.linspace(100,len(smoothed_large_error_history[:1500]), len(smoothed_large_error_history[100:1500])), smoothed_large_error_history[100:1500], label='Pseudo-label only')
        #plt.plot(np.linspace(100,len(smoothed_small_error_history[:1500]), len(smoothed_large_error_history[100:1500])), smoothed_small_error_history[100:1500], label='synthetic only')
        
        plt.plot(np.linspace(100,len(smoothed_large_error_history[:1500]), len(smoothed_large_error_history[100:1500])), smoothed_large_error_history[100:1500], label='large')
        plt.plot(np.linspace(100,len(smoothed_small_error_history[:1500]), len(smoothed_large_error_history[100:1500])), smoothed_small_error_history[100:1500], label='small')
        plt.plot(np.linspace(100,len(smoothed_mixed_error_history[:1500]), len(smoothed_mixed_error_history[100:1500])), smoothed_mixed_error_history[100:1500], label='mixed')
        plt.plot(np.linspace(100,len(smoothed_mixed_error_history[:1500]), len(smoothed_pseudo_error_history[100:1500])), smoothed_pseudo_error_history[100:1500], label='synthetic-only fine-tuning')
        #plt.plot(np.linspace(100,len(smoothed_mixed_error_history[:1500]), len(smoothed_pseudo_error_history[100:1500])), smoothed_pseudo_error_history[100:1500], label='continuous fine-tuning')

        plt.legend()
        plt.xlabel("Time Frame")
        plt.ylabel("Precision and Recall")
        plt.savefig('0530_small_large_mixed_pseudo_cell_identification_overall_synth_only.png')
        #plt.savefig('0530_pseudo_comparison_identification_overall.png')

    else:
        large_error_history = np.zeros(1570)
        mixed_error_history = np.zeros(1570)
        small_error_history = np.zeros(1570)
        for iteration in trench_indices:
            with open("record_trench.txt", "w") as f:
                f.write("trench: " + str(iteration))
            benchmark_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/large/masks".format(iteration)
            benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
            large_maskfiles = [io.imread(f) for f in benchmark_maskfiles]

            benchmark_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/small/masks".format(iteration)
            benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
            small_maskfiles = [io.imread(f) for f in benchmark_maskfiles]

            benchmark_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/mixed/masks".format(iteration)
            benchmark_maskfiles = io.get_image_files(benchmark_maskdir)
            mixed_maskfiles = [io.imread(f) for f in benchmark_maskfiles]

            groundtruth_maskdir = "/home/ameyasu/cuda_ws/src/full_curve_test/{}/groundtruth/masks".format(iteration)
            groundtruth_maskfiles_o = io.get_image_files(groundtruth_maskdir)
            groundtruth_maskfiles = [np.asarray(cv2.resize(io.imread(f), (64, 256), interpolation= cv2.INTER_NEAREST)) for f in groundtruth_maskfiles_o]
            

            current_frame = 0
            large_m_c_lengths = np.zeros(1570)
            small_m_c_lengths = np.zeros(1570)
            mixed_m_c_lengths = np.zeros(1570)
            for i in range(0, 1570):
                error = 0.
                norm_g = 0.
                norm_b = 0.
                average_cell_size = 0
                print(i)
                large_errors_count = 0
                small_errors_count = 0
                mixed_errors_count = 0

                small_model_cells, small_cell_sizes, small_cell_index = get_cells(small_maskfiles, i, "small", groundtruth_maskfiles, debug)
                large_model_cells, large_cell_sizes, large_cell_index = get_cells(large_maskfiles, i, "large", groundtruth_maskfiles, debug)
                mixed_model_cells, mixed_cell_sizes, mixed_cell_index = get_cells(mixed_maskfiles, i, "mixed", groundtruth_maskfiles, debug)
                large_m_c_lengths[i] = large_cell_sizes[0][0]
                small_m_c_lengths[i] = small_cell_sizes[0][0]
                mixed_m_c_lengths[i] = mixed_cell_sizes[0][0]
            
            window_length = 51
            half_length = int((window_length-1)/2)
            large_gradients = large_m_c_lengths
            small_gradients = small_m_c_lengths
            mixed_gradients = mixed_m_c_lengths
            for k in range(half_length, 1570-half_length):
                large_error_history[k] += 1 if (large_gradients[k]>(np.mean(large_gradients[k-half_length:k+half_length]) + 3*np.sqrt(np.var(large_gradients[k-half_length:k+half_length]))) or large_gradients[k]<(np.mean(large_gradients[k-half_length:k+half_length]) - 3*np.sqrt(np.var(large_gradients[k-half_length:k+half_length])))) else 0
                small_error_history[k] += 1 if (small_gradients[k]>(np.mean(small_gradients[k-half_length:k+half_length]) + 3*np.sqrt(np.var(small_gradients[k-half_length:k+half_length]))) or small_gradients[k]<(np.mean(small_gradients[k-half_length:k+half_length]) - 3*np.sqrt(np.var(small_gradients[k-half_length:k+half_length])))) else 0
                mixed_error_history[k] += 1 if (mixed_gradients[k]>(np.mean(mixed_gradients[k-half_length:k+half_length]) + 3*np.sqrt(np.var(mixed_gradients[k-half_length:k+half_length]))) or mixed_gradients[k]<(np.mean(mixed_gradients[k-half_length:k+half_length]) - 3*np.sqrt(np.var(mixed_gradients[k-half_length:k+half_length])))) else 0
            #for k in argrelextrema(large_gradients, np.greater, order=half_length)[0]:
            #    large_error_history[k] += 1
            #for k in argrelextrema(small_gradients, np.greater, order=half_length)[0]:
            #    small_error_history[k] += 1
            #for k in argrelextrema(mixed_gradients, np.greater, order=half_length)[0]:
            #    mixed_error_history[k] += 1

        large_error_history = large_error_history/len(trench_indices)
        small_error_history = small_error_history/len(trench_indices)
        mixed_error_history = mixed_error_history/len(trench_indices)
    
        smooth_window = np.ones(100)*0.01

        smoothed_large_error_history = np.zeros(len(large_error_history)-len(smooth_window))
        for i in range(len(large_error_history)-len(smooth_window)):
            temp_list = [smooth_window[j]*large_error_history[j+i] for j in range(len(smooth_window))]
            smoothed_large_error_history[i] = np.sum(temp_list)

        smoothed_small_error_history = np.zeros(len(small_error_history)-len(smooth_window))
        for i in range(len(small_error_history)-len(smooth_window)):
            temp_list = [smooth_window[j]*small_error_history[j+i] for j in range(len(smooth_window))]
            smoothed_small_error_history[i] = np.sum(temp_list)

        smoothed_mixed_error_history = np.zeros(len(mixed_error_history)-len(smooth_window))
        for i in range(len(mixed_error_history)-len(smooth_window)):
            temp_list = [smooth_window[j]*mixed_error_history[j+i] for j in range(len(smooth_window))]
            smoothed_mixed_error_history[i] = np.sum(temp_list)

        f1 = plt.figure()
        plt.plot(np.array(range(1,len(smoothed_large_error_history)+1)), smoothed_large_error_history, label='large')
        plt.plot(np.array(range(1,len(smoothed_small_error_history)+1)), smoothed_small_error_history, label='small')
        plt.plot(np.array(range(1,len(smoothed_mixed_error_history)+1)), smoothed_mixed_error_history, label='mixed')

        plt.legend()
        plt.xlabel("Time Frame")
        plt.ylabel("Mother cell error")
        plt.savefig('average_pseudo_labelling_with_symbac_m_c_only.png')