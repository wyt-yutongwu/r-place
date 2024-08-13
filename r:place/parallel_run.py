import multiprocessing
import re
from place_graphcut_user_emb.fGreedy_set_cover import *
import os
from drawing import *
from merge_drawings import *

def p_run(label_dir, update_dir, label_update_dir, threshold_folder, threshold):
    tasks = list()
    label_lst = [[] for x in range(10)]
    count = 0
    for filename in os.listdir(label_dir):
        result = re.search('label_(.*).npy', filename)
        snap = int(result.group(1)) / 1000
        last_digit = int(str(int(snap))[-1])
        label_lst[last_digit].append(snap)
        count += 1
    print("count:", count)
    for i in range(0,10):
        #(label_dir, label_lst, update_dir, label_update_file, num)
        tasks.append((label_dir, label_lst[i], update_dir, label_update_dir+str(i), i, threshold, threshold_folder))
    num_processes = 10
    p = multiprocessing.Pool(num_processes).starmap(init_fGreedy, tasks) #worker function should be defined in a different file
    return p



