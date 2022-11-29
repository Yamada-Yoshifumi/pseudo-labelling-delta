import numpy as np
from pathlib import Path
import os
from cellpose import io
import math
import matplotlib.pyplot as plt

average = 5
large_error_history = np.zeros(1597)
mixed_error_history = np.zeros(1597)
small_error_history = np.zeros(1597)

for iteration in range(1, 1 + average):
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
    groundtruth_maskfiles = io.get_image_files(groundtruth_maskdir)
    groundtruth_maskfiles = [io.imread(f) for f in groundtruth_maskfiles]


    if len(groundtruth_maskfiles) != len(benchmark_maskfiles):
        print("not compatible dimensions")

    for i in range(0, len(groundtruth_maskfiles)):
    #for i in range(500, 550):
        error = 0.
        norm_g = 0.
        norm_b = 0.
        ground_truth_cells = []
        large_model_cells = []
        small_model_cells = []
        mixed_model_cells = []
        cell_index = 1
        average_cell_size = 0
        print(i)
        while True:
            one_cell = np.where(np.array(large_maskfiles[i]) == cell_index)
            
            if len(one_cell) == 2:
                if len(one_cell[0]) == 0:
                    cell_index = 1
                    break
                x = np.average(one_cell[0])
                y = np.average(one_cell[1])
                large_model_cells.append([x,y])
                cell_index += 1
            else:
                cell_index = 1
                break

        while True:
            one_cell = np.where(np.array(mixed_maskfiles[i]) == cell_index)
            
            if len(one_cell) == 2:
                if len(one_cell[0]) == 0:
                    cell_index = 1
                    break
                x = np.average(one_cell[0])
                y = np.average(one_cell[1])
                mixed_model_cells.append([x,y])
                cell_index += 1
            else:
                cell_index = 1
                break

        while True:
            one_cell = np.where(np.array(small_maskfiles[i]) == cell_index)
            
            if len(one_cell) == 2:
                if len(one_cell[0]) == 0:
                    cell_index = 1
                    break
                x = np.average(one_cell[0])
                y = np.average(one_cell[1])
                small_model_cells.append([x,y])
                cell_index += 1
            else:
                cell_index = 1
                break
        while True:
            one_cell = np.where(np.array(groundtruth_maskfiles[i]) == cell_index)
            
            if len(one_cell) == 2:
                if len(one_cell[0]) == 0:
                    break
                x = np.average(one_cell[0])
                y = np.average(one_cell[1])
                ground_truth_cells.append([x,y])
                cell_index += 1
            else:
                break
        for row in groundtruth_maskfiles[i]:
            for pixel in row:
                if pixel != 0:
                    average_cell_size += 1
        average_cell_size /= cell_index
        ground_truth_cells_copy = ground_truth_cells.copy()
        small_model_cells_copy = small_model_cells.copy()

        j = 0

        while len(ground_truth_cells_copy) != 0:
            
            if len(small_model_cells) == 0:
                ground_truth_cells_copy.pop(0)
                #small_error_history[i] += 1./(cell_index)
                small_error_history[i] += 1.
                continue

            if j >= (len(small_model_cells_copy) - 1):
                    #small_error_history[i] += 1./(cell_index)
                    small_error_history[i] += 1.
                    ground_truth_cells_copy.pop(0)
                    j = 0
                    small_model_cells_copy = small_model_cells.copy()
                    break
            if (ground_truth_cells_copy[0][0] - small_model_cells_copy[j][0])**2 + (ground_truth_cells_copy[0][1] - small_model_cells_copy[j][1])**2 <= average_cell_size/4:
                cell_pixel = groundtruth_maskfiles[i][int(ground_truth_cells_copy[0][0])][int(ground_truth_cells_copy[0][1])]
                matched_p = 0
                unmatched_p = 0
                gt_list = np.where(groundtruth_maskfiles[i] == cell_pixel)
                gt_list = [(gt_list[0][i], gt_list[1][i]) for i in range(len(gt_list[0]))]
                benchmark_list = np.where(small_maskfiles[i] == cell_pixel)
                benchmark_list = [(benchmark_list[0][i], benchmark_list[1][i]) for i in range(len(benchmark_list[0]))]
                matched_p = len(set(gt_list).intersection(set(benchmark_list)))
                unmatched_p = len(gt_list) + len(benchmark_list) - 2*matched_p
                if (matched_p+unmatched_p) == 0:
                    #small_error_history[i] += 1./(cell_index)
                    small_error_history[i] += 1.
                else:
                    #small_error_history[i] += (1./(cell_index))*(unmatched_p/(unmatched_p + matched_p))
                    small_error_history[i] += unmatched_p/(unmatched_p + matched_p)
                ground_truth_cells_copy.pop(0)
                small_model_cells.pop(j)
                small_model_cells_copy = small_model_cells.copy()
                j = 0
            else:
                small_model_cells_copy.pop(j)
                j += 1

        j = 0
        large_model_cells_copy = large_model_cells.copy()
        ground_truth_cells_copy = ground_truth_cells.copy()
        while len(ground_truth_cells_copy) != 0:
            
            if len(large_model_cells) == 0:
                ground_truth_cells_copy.pop(0)
                #large_error_history[i] += 1./(cell_index)
                large_error_history[i] += 1.
                continue

            if j >= (len(large_model_cells_copy) - 1):
                    #large_error_history[i] += 1./(cell_index)
                    large_error_history[i] += 1.
                    ground_truth_cells_copy.pop(0)
                    j = 0
                    large_model_cells_copy = large_model_cells.copy()
                    break
            
            if (ground_truth_cells_copy[0][0] - large_model_cells_copy[j][0])**2 + (ground_truth_cells_copy[0][1] - large_model_cells_copy[j][1])**2 <= average_cell_size/4:
                cell_pixel = groundtruth_maskfiles[i][int(ground_truth_cells_copy[0][0])][int(ground_truth_cells_copy[0][1])]
                matched_p = 0
                unmatched_p = 0
                #print(cell_pixel)
                gt_list = np.where(groundtruth_maskfiles[i] == cell_pixel)
                gt_list = [(gt_list[0][i], gt_list[1][i]) for i in range(len(gt_list[0]))]
                benchmark_list = np.where(large_maskfiles[i] == cell_pixel)
                benchmark_list = [(benchmark_list[0][i], benchmark_list[1][i]) for i in range(len(benchmark_list[0]))]
                matched_p = len(set(gt_list).intersection(set(benchmark_list)))
                unmatched_p = len(gt_list) + len(benchmark_list) - 2*matched_p
                if (matched_p+unmatched_p) == 0:
                    #large_error_history[i] += 1./(cell_index)
                    large_error_history[i] += 1.
                else:
                    #large_error_history[i] += (1./(cell_index))*(unmatched_p/(unmatched_p + matched_p))
                    large_error_history[i] += unmatched_p/(unmatched_p + matched_p)
                ground_truth_cells_copy.pop(0)
                large_model_cells.pop(j)
                large_model_cells_copy = large_model_cells.copy()
                j = 0
            else:
                large_model_cells_copy.pop(j)
                j += 1
        j = 0
        #print(cell_errors)

        mixed_model_cells_copy = mixed_model_cells.copy()
        ground_truth_cells_copy = ground_truth_cells.copy()
        while len(ground_truth_cells_copy) != 0:
            
            if len(mixed_model_cells) == 0:
                ground_truth_cells_copy.pop(0)
                #large_error_history[i] += 1./(cell_index)
                mixed_error_history[i] += 1.
                continue

            if j >= (len(mixed_model_cells_copy) - 1):
                    #large_error_history[i] += 1./(cell_index)
                    mixed_error_history[i] += 1.
                    ground_truth_cells_copy.pop(0)
                    j = 0
                    mixed_model_cells_copy = mixed_model_cells.copy()
                    break
            
            if (ground_truth_cells_copy[0][0] - mixed_model_cells_copy[j][0])**2 + (ground_truth_cells_copy[0][1] - mixed_model_cells_copy[j][1])**2 <= average_cell_size/4:
                cell_pixel = groundtruth_maskfiles[i][int(ground_truth_cells_copy[0][0])][int(ground_truth_cells_copy[0][1])]
                matched_p = 0
                unmatched_p = 0
                #print(cell_pixel)
                gt_list = np.where(groundtruth_maskfiles[i] == cell_pixel)
                gt_list = [(gt_list[0][i], gt_list[1][i]) for i in range(len(gt_list[0]))]
                benchmark_list = np.where(mixed_maskfiles[i] == cell_pixel)
                benchmark_list = [(benchmark_list[0][i], benchmark_list[1][i]) for i in range(len(benchmark_list[0]))]
                matched_p = len(set(gt_list).intersection(set(benchmark_list)))
                unmatched_p = len(gt_list) + len(benchmark_list) - 2*matched_p
                if (matched_p+unmatched_p) == 0:
                    #large_error_history[i] += 1./(cell_index)
                    mixed_error_history[i] += 1.
                else:
                    #large_error_history[i] += (1./(cell_index))*(unmatched_p/(unmatched_p + matched_p))
                    mixed_error_history[i] += unmatched_p/(unmatched_p + matched_p)
                ground_truth_cells_copy.pop(0)
                mixed_model_cells.pop(j)
                mixed_model_cells_copy = mixed_model_cells.copy()
                j = 0
            else:
                mixed_model_cells_copy.pop(j)
                j += 1
        j = 0
        #print(cell_errors)

large_error_history /= average
small_error_history /= average
mixed_error_history /= average
    
smooth_window = np.ones(50)*0.02
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

plt.plot(np.array(range(1,len(smoothed_large_error_history)+1)), smoothed_large_error_history, label='large')
plt.plot(np.array(range(1,len(smoothed_small_error_history)+1)), smoothed_small_error_history, label='small')
plt.plot(np.array(range(1,len(smoothed_mixed_error_history)+1)), smoothed_mixed_error_history, label='mixed')

plt.legend()
plt.xlabel("Time Frame")
plt.ylabel("Number of Segmentation Error")
plt.show()