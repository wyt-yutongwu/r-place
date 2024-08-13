import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
from utils import *
from disjoint_set import *
import math
import random
import time
import pickle
import numpy as np
import cv2
import csv
import sys
from os.path import exists
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from scipy.sparse import csr_matrix 
import networkx as nx
import re
from place_graphcut_user_emb.fGreedy_set_cover import *
import struct
import multiprocessing

def agg_cluster(user_emb, connectivity_mtx, dist_threshold):

    agglo = AgglomerativeClustering(distance_threshold=dist_threshold, linkage="ward", connectivity=connectivity_mtx, n_clusters=None)
    agglo_fit = agglo.fit(user_emb)
    return agglo_fit.labels_

def segment_2d(in_image, in_users, in_userembs, sigma, k, min_size, uo_threshold, ue_threshold, name, alpha):
    
    start_time = time.time()
    height, width, band = in_image.shape
    # print("Height:  " + str(height))
    # print("Width:   " + str(width))
    
    ###### convert color space, RGB to LAB  #####
    in_image_lab = cv2.cvtColor(in_image,cv2.COLOR_RGB2LAB)
    in_image_lab = in_image_lab.astype('int')                       #uin8 causes overflow
    smooth_red_band = gaussian_filter(in_image_lab[:, :, 0], sigma)
    smooth_green_band = gaussian_filter(in_image_lab[:, :, 1], sigma)
    smooth_blue_band = gaussian_filter(in_image_lab[:, :, 2], sigma)

    ######  build graph                    #####
    edges_size = width * height * 4
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    num = 0
    row = []
    col = []
    empty = set()
    # considers "plus" edges and diagonal edges as well. 
    # adding 2nd hop neighbors would sacrifice spatial connectivity. 
    # note how the MST edges that cross internal boundaries have highest weight.
    #num_vertices = 0
    for y in range(height):
        for x in range(width):

            if in_image[y,x,0] == 1:
                empty.add(int(y * width + x))
                continue
            if x < width - 1 and in_image[y,x+1,0] != 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int(y * width + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y)
                row.append(int(edges[num, 0]))
                col.append(int(edges[num, 1]))
                num += 1
            if y < height - 1 and in_image[y+1,x,0] != 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + x)
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x, y + 1)
                row.append(int(edges[num, 0]))
                col.append(int(edges[num, 1]))
                num += 1

            if (x < width - 1) and (y < height - 2) and in_image[y+1, x+1, 0] != 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y + 1)
                row.append(int(edges[num, 0]))
                col.append(int(edges[num, 1]))
                num += 1

            if (x < width - 1) and (y > 0) and in_image[y-1, x+1,0] != 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y - 1) * width + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y - 1)
                row.append(int(edges[num, 0]))
                col.append(int(edges[num, 1]))
                num += 1

    ##### Segment                       #####
    u, threshold = segment_graph(width*height, num, edges, in_users.reshape(-1) if in_users is not None else None, \
                        in_userembs.reshape(-1,120) if in_userembs is not None else None, k)

    num_cc = u.num_sets()
    print("Number of regions: {}".format(num_cc))

    clustered_user_embs = u.get_comp_user()
    data = np.ones(len(row))
    connectivity_mtx = csr_matrix((data, (np.array(row), np.array(col))),  
                          shape = (width * height, width * height))
    G=nx.from_scipy_sparse_array(connectivity_mtx, parallel_edges=False, create_using=None, edge_attribute='weight')
    components = nx.connected_components(G)
    pred_labels = np.zeros(width * height, dtype='ushort')
    comp_counter = 1
    for com in components:

        nodes = np.array(list(com))
        if len(nodes) == 1:
            if nodes[0] not in empty:
                pred_labels[nodes[0]] = comp_counter
                comp_counter += 1
            continue

        conn_mtx = nx.adjacency_matrix(G, nodelist=nodes)
        user_emb = clustered_user_embs[nodes]
        labels = agg_cluster(user_emb, conn_mtx, ue_threshold)
        max_val = np.max(labels)
        labels += comp_counter
        comp_counter += max_val
        pred_labels[nodes] = labels
    pred_labels = pred_labels.reshape((width, height))
        
    elapsed_time = time.time() - start_time
    print(
        "Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
            int(elapsed_time % 60)) + " seconds")
    
    return u, threshold, edges, num, pred_labels, elapsed_time


def seg(time):
    ### For running large number of files
    sigma = 0.5   #gaussian blur strength. 
    k = 30       #higher values favor bigger size components
    min_size = 50     #minimum size of clusters
    uo_threshold = 1 #threshold for simple user overlap merging
    ue_threshold = 30
    alpha = 0.7
    rgb_path = "../data/canvas/canvas_"+time+".npy"
    with open(rgb_path,'rb') as f:
        in_image = np.load(f)
    height, width, band = in_image.shape

    ##### get user information                  #####
    users_path = "../data/user/user_"+time+".npy"
    with open(users_path,'rb') as f:
        in_users = np.load(f, allow_pickle=True)


    #node2vec user embeddings
    users_emb_path   = "../node2vec_emb.pkl"
    with open(users_emb_path,'rb') as f:
        users_emb = pickle.load(f)
    users_ind_path   = "../user_index.pkl"
    with open(users_ind_path,'rb') as f:
        users_ind = pickle.load(f)
    
    in_userembs = np.zeros((1001,1001,120))
    present, absent = 0,0
    for i in range(1001):
        for j in range(1001):
            user = in_users[i,j]
            if(user in users_ind):
                in_userembs[i,j] = users_emb[users_ind[user]]
                present += 1
            else:
                absent += 1

    print("Num of users present: {}, users absent: {}".format(present, absent))
    u, threshold, edges, num, predLabels, elapsed_time = segment_2d(in_image, in_users, in_userembs, sigma, k, min_size, uo_threshold, ue_threshold, "", alpha)
    np.save("/scratch/yw180/place/data/labels/label_"+str(time)+".npy", predLabels)


def mp_worker(inputs): #This function should run the segmentation
    seg(str(inputs))

def proj_worker(update_label_file, num):
    print('length of update lst', len(settings.curr_up_lst))
    updated_labels = set()
    for update in settings.curr_up_lst:
        if update not in settings.ul_idx_lst[num]:
            continue
        start_line = settings.ul_idx_lst[num][update][0]
        end_line = settings.ul_idx_lst[num][update][1]
        with open(update_label_file, "rb") as f:
            length = len(struct.pack("I", 0))
            f.seek(start_line*length)
            i = start_line
            while i < end_line:
                up = struct.unpack("I", f.read(length))[0]
                snap = struct.unpack("I", f.read(length))[0]
                lab = struct.unpack("I", f.read(length))[0]
                i+=3
                marker = (snap, lab)
                settings.ls_lst[num][marker] += -1
                updated_labels.add(marker)
    if num == 9 and (1490999399, 50) in updated_labels:
        print("updated", settings.ls_lst[num][(1490999399, 50)])
    return num, updated_labels