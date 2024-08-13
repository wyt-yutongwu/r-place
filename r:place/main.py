from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from utils import *
from disjoint_set import *
import time
import pickle
import numpy as np
import cv2
from os.path import exists
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from scipy.sparse import csr_matrix 
import networkx as nx
from place_graphcut_user_emb.fGreedy_set_cover import *
from parallel_run import *
from merge_drawings import *

if __name__ == "__main__":

    start_time = time.time()
    users_emb_path = "../node2vec_emb.pkl"
    users_ind_path   = "../user_index.pkl"
    update_user_path = "/scratch/yw180/place/data/update_user_dict.pkl"
    sorted_update_drawing = "/scratch/yw180/place/data/final_drawing/sorted_update_drawing.csv"
    sorted_tile_placements_idx =  "/scratch/yw180/place/data/sorted_tile_placements_idx.csv"
    final_update_canvas_file = "/home/yw180/place/data/updates/update_1491238734000.npy"
    label_folder = "/scratch/yw180/place/data/all_labels/labels/"
    update_folder = "/home/yw180/place/data/updates/"
    overlap_folder = "/scratch/yw180/place/data/final_drawing/overlap/"
    canvas_folder = "/scratch/yw180/place/data/canvas/"
    line_emb_path = "/home/yw180/place/data/merged_8/line_user_emb.pkl"
    bound_dict_path = "/home/yw180/place/data/merged_8/new_bound.pkl"

    out = "/scratch/yw180/place/data/user_emb_merge.csv"

    threshold = 0.85
    bound_threshold = 0.4
    print('threshold:', threshold)
    print('bound_threshold', bound_threshold)
    merge_user_emb(threshold, out, bound_threshold, bound_dict_path, line_emb_path)
    elapsed_time = time.time() - start_time
    print(
        "time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
            int(elapsed_time % 60)) + " seconds")




