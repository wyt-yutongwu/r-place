from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
from utils import *
from disjoint_set import *
import numpy as np
import csv
import os
import heapq
import struct
from worker import *

### Storing not in memory, parallel
### num: the last digit that the current process handles
def init_fGreedy(label_dir, label_lst, update_dir, label_update_file, num, threshold, threshold_folder):
    threshold_file = threshold_folder + str(num)
    label_size = dict()
    label_up_idx = dict()
    if len(label_lst) == 0:
         return label_size, label_up_idx, num, dict()
    curr_loc = 0
    curr_idx = 0
    for filename in label_lst:
        
        label = np.load(label_dir+"label_"+str(int(filename))+"000.npy", allow_pickle=True)
        #label_flat = label.flatten()
        # result = re.search('label_(.*).npy', filename)
        t = int(filename)
        update = np.load(update_dir+"update_"+str(t)+"000.npy")
        curr_label_up_dict = dict()
        largest_lab = 0
        for i in range(0, 1001):
            for j in range(0, 1001):
                curr_label = label[j,i]
                if curr_label == 0:
                    continue
                if curr_label > largest_lab:
                    largest_lab = curr_label
                curr_update = update[j,i]
                marker = (t, curr_label)
                if marker not in curr_label_up_dict:
                    curr_label_up_dict[marker] = [curr_update]
                else:
                    curr_label_up_dict[marker].append(curr_update)
                    
            
        label_up_idx[t] = np.array(np.zeros((largest_lab + 1)), dtype = np.uint32)
        for marker in curr_label_up_dict:
            curr_label = marker[1]
            size_of_marker = len(curr_label_up_dict[marker])
            if size_of_marker > threshold:
                label_size[marker] = size_of_marker
            else:
                #(snap, lab, size, idx)
                with open(threshold_file, 'ab') as f:
                    f.write(struct.pack("I", t))
                    f.write(struct.pack("I", curr_label))
                    f.write(struct.pack("I", size_of_marker)) 
            label_up_idx[t][curr_label] = curr_idx
            with open(label_update_file, 'ab') as f:
                # f.write(struct.pack("I", marker[0]))
                # f.write(struct.pack("I", marker[1]))
                for item in curr_label_up_dict[marker]:
                    f.write(struct.pack("I", item))   
            curr_idx += 2
            with open(label_update_file+"_idx", 'ab') as f:
                f.write(struct.pack("Q", curr_loc))
                curr_loc += size_of_marker
                f.write(struct.pack("Q", curr_loc))
            # ### Debug starts here
            # if debug:
            #     print("marker", marker)
            #     print("in debugging")
            #     print("curr_idx:", curr_idx)
            #     print("size of marker:", size_of_marker)
            #     np.save("/scratch/yw180/place/data/debug.npy", curr_label_up_dict)
            # ### Debug ends here
        # print(update_label_idx)
            
    return label_size, label_up_idx, num

def create_heap(label_size_lst):
    heap = list()
    for l in label_size_lst:
        for key in l:
            tup = (-1 * l[key], key)
            heap.append(tup)
    return heap


### label_up_idx_lst, label_size_lst: should be ordered based on last digit
### 
def fGreedy(count, updates, label_update_dir, label_up_idx_lst, out, heap, threshold, threshold_folder):
    # create heap
    num_proj = 0
    heapq.heapify(heap)
    pre_exit = False
    # create list to keep track of whether update is used
    while count < 16559408:
        if len(heap) == 0:
            pre_exit = True
            break
        # Get current largest and check whether they are in the same size in label_size
        max_tup = heapq.heappop(heap)
        max_size = -1 * max_tup[0]
        if max_size <= 0:
            pre_exit = True
            break
        max_lab = max_tup[1]
        last_digit = int(str(max_lab[0])[-1])
        t = max_lab[0]
        l = max_lab[1]
        true_size = 0
        idx_lines_start = label_up_idx_lst[last_digit][t][l]
        lines_start = 0
        lines_end = 0
        with open(label_update_dir+str(last_digit)+"_idx", "rb") as f:
            length = len(struct.pack("Q", 0))
            f.seek(idx_lines_start*length)
            lines_start = struct.unpack("Q", f.read(length))[0]
            lines_end = struct.unpack("Q", f.read(length))[0]

        # Change label size for all labels associated with the update
        with open(label_update_dir+str(last_digit), "rb") as f:
            length = len(struct.pack("I", 0))
            f.seek(lines_start*length)
            for i in range(lines_end - lines_start):
                up = struct.unpack("I", f.read(length))[0]
                if not updates[up]:
                    true_size += 1
        if true_size == 0: 
            continue
        elif true_size == max_size:
            num_proj += 1
            with open(label_update_dir+str(last_digit), "rb") as f:
                line_out = [max_lab[0], max_lab[1]]
                with open(out, 'a') as file_out:
                    writer = csv.writer(file_out, delimiter = ",")
                    length = len(struct.pack("I", 0))
                    f.seek(lines_start*length)
                    for i in range(lines_end - lines_start):
                        up = struct.unpack("I", f.read(length))[0]
                        if not updates[up]:
                            count += 1
                            line_out.append(up)
                            updates[up] = 1
                    writer.writerow(line_out)
        elif true_size < threshold:
            with open(threshold_folder+str(last_digit), 'ab') as f:
                f.write(struct.pack("I", t))
                f.write(struct.pack("I", l))
                f.write(struct.pack("I", true_size)) 
        else:
            tup = (-1 * true_size, max_lab)
            heapq.heappush(heap, tup)
            if heap[0][1] == max_lab:
                num_proj += 1
                heapq.heappop(heap)
                with open(label_update_dir+str(last_digit), "rb") as f:
                    line_out = [max_lab[0], max_lab[1]]
                    with open(out, 'a') as file_out:
                        writer = csv.writer(file_out, delimiter = ",")
                        length = len(struct.pack("I", 0))
                        f.seek(lines_start*length)
                        for i in range(lines_end - lines_start):
                            up = struct.unpack("I", f.read(length))[0]
                            if not updates[up]:
                                count += 1
                                line_out.append(up)
                                updates[up] = 1
                        writer.writerow(line_out)
    print("num of project written:", num_proj)
    return count, updates, pre_exit

def init_from_threshold_file(threshold_folder, threshold):
    heap = list()
    for i in range(0, 10):
        threshold_file = threshold_folder+str(i)
        file_size = os.path.getsize(threshold_file)
        with open(threshold_file, "rb") as f:
            counter = 0
            length = len(struct.pack("I", 0))
            while counter < (file_size / length):
                t = struct.unpack("I", f.read(length))[0]
                lab = struct.unpack("I", f.read(length))[0]
                size = struct.unpack("I", f.read(length))[0]
                if size > threshold:
                    tup = (-1 * size, (t, lab))
                    heap.append(tup)
                counter += 3
    return heap

def rebuild_final_canvas_from_seg(update_file, seg_output):
    update_pos = dict()
    update_canv = np.load(update_file)
    # Create inverse indexed <update, position> dictionary
    for x in range(0, 1001):
        for y in range(0, 1001):
            if update_canv[y,x] != 0:
                up = int(update_canv[y,x])
                update_pos[up] = (y,x)
    # Find the partitions where final updates belong
    canv = np.array(np.zeros((1001, 1001)))
    counter = 1
    counter_used = False
    with open(seg_output, 'r')as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            updates = line[2:]
            for u in updates:
                u = int(u)
                if u in update_pos:
                    counter_used = True
                    pos = update_pos[u]
                    canv[pos[0],pos[1]] = counter
            if counter_used:
                counter_used = False
                counter += 1
    return canv

def rebuild_final_canvas_from_seg_merged(update_file, seg_output):
    update_pos = dict()
    update_canv = np.load(update_file)
    # Create inverse indexed <update, position> dictionary
    for x in range(0, 1001):
        for y in range(0, 1001):
            if update_canv[y,x] != 0:
                up = int(update_canv[y,x])
                update_pos[up] = (y,x)
    # Find the partitions where final updates belong
    canv = np.array(np.zeros((1001, 1001)))
    counter = 1
    counter_used = False
    with open(seg_output, 'r')as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            for u in line:
                u = int(u)
                if u in update_pos:
                    counter_used = True
                    pos = update_pos[u]
                    canv[pos[0],pos[1]] = counter
            if counter_used:
                counter_used = False
                counter += 1
    return canv

def plot_n_largest_drawing(n, seg_output, update_folder, canvas_folder,output_folder):
    counter = 0
    snapshot_lst = np.zeros(n)
    lab_lst = np.zeros(n)
    up_lst = list()
    with open(seg_output, mode ='r')as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            if counter >= n:
                break
            snap = int(line[0])
            snapshot_lst[counter] = snap
            lab = int(line[1])
            lab_lst[counter] = lab
            up = [int(x) for x in line[2:]]
            up_lst.append(up)
            counter += 1

    for k in range(0, n):
        up = set(up_lst[k])
        canvas = np.load(canvas_folder+"canvas_"+str(int(snapshot_lst[k]))+"000.npy")
        update = np.load(update_folder+"update_"+str(int(snapshot_lst[k]))+"000.npy")
        out = np.array(255*np.ones((1001, 1001, 3)))
        for i in range(0, 1001):
            for j in range(0, 1001):
                if update[j,i] in up:
                    out[j,i] = canvas[j,i]
        out = out.astype('uint8')
        plt.figure(frameon=False, figsize = (100,100))
        fig, a = plt.subplots()

        a.axis('off')

        plt.imshow(out)
        plt.show()
        plt.savefig(output_folder+str(k)+".png", dpi=400)

        plt.clf()
