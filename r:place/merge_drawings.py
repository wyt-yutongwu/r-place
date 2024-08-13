import csv
import pickle
import os
import numpy as np

        
def make_set_cover(file):
    set_cover = list()
    set_update = dict()
    update_set_cover = dict()
    with open(file,"r") as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            snap = int(line[0])
            lab = int(line[1])
            set_cover.append((snap,lab))
            set_update[(snap,lab)] = set()
            for u in line[2:]:
                update_set_cover[int(u)] = (snap,lab)

    return set_cover, set_update, update_set_cover

def reformat_result():
    set_cover_line = dict()
    line_count = 0
    with open("/home/yw180/place/data/merged_85/merged_85.csv", "r") as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            count = 0
            snapshot = 0
            label = 0
            for item in line:
                if count % 2 == 0:
                    count += 1
                    snapshot = int(item)
                else:
                    count += 1
                    label = int(item)
                    set_cover_line[(snapshot, label)] = line_count
            line_count += 1
    set_cover, set_update, update_set = make_set_cover("/scratch/yw180/place/data/seg_analysis_out.csv")
    update_lab = dict()
    lab_update = dict()
    for i in update_set:
        s = update_set[i]
        l = set_cover_line[s]
        update_lab[i] = l
        if l not in lab_update:
            lab_update[l] = set()
        lab_update[l].add(i)

    with open("/home/yw180/place/data/merged_85/merged_update_lab.csv", "a") as file_out:
        writer = csv.writer(file_out, delimiter = ",")
        print("writing first")
        for i in update_lab:
            line = [i, update_lab[i]]
            writer.writerow(line)
    

    with open("/home/yw180/place/data/merged_85/merged_lab_update.csv", "a") as file_out:
        writer = csv.writer(file_out, delimiter = ",")
        print('writing second')
        for i in lab_update:
            line = lab_update[i]
            writer.writerow(line)


def reformat_2nd_merge_result():
    lab_line = dict()
    update_lab = dict()
    with open("/home/yw180/place/data/merged_8/merged_update_lab.csv", "r") as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            update = int(line[0])
            lab = int(line[1])
            update_lab[update] = lab

    line_count = 0
    with open("/home/yw180/place/data/merged_8/merge_user_emb/merge_user_emb.csv", "r") as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            for item in line:
                item = int(item)
                lab_line[item] = line_count
            line_count += 1
    print('line count', line_count)
    update_new_lab = dict()
    new_lab_update = dict()

    for update in update_lab:
        new_lab = lab_line[update_lab[update]]
        update_new_lab[update] = new_lab
        if new_lab not in new_lab_update:
            new_lab_update[new_lab] = set()
        new_lab_update[new_lab].add(update)

    with open("/home/yw180/place/data/merged_8/merge_user_emb/merged_update_lab.csv", "w") as file_out:
        writer = csv.writer(file_out, delimiter = ",")
        for update in update_new_lab:
            line = [update, update_new_lab[update]]
            writer.writerow(line)
    
    with open("/home/yw180/place/data/merged_8/merge_user_emb/merged_lab_update.csv", "w") as file_out:
        writer = csv.writer(file_out, delimiter = ",")
        for lab in new_lab_update:
            line = new_lab_update[lab]
            writer.writerow(line)



def merge_across_time_main(threshold, out, bound_threshold):
    
    set_cover, set_update, update_set = make_set_cover("/scratch/yw180/place/data/seg_analysis_out.csv")
    set_cover_s = set(set_cover)
    set_size_path = "/scratch/yw180/place/data/set_size.pkl"
    with open(set_size_path, 'rb') as f:
        set_size = pickle.load(f)
    bound_dict_path = "/scratch/yw180/place/data/set_bound.pkl"
    with open(bound_dict_path, 'rb') as f:
        bound_dict = pickle.load(f)
    for i in range(0, len(set_cover)):
        s = set_cover[i]
        if s not in set_cover_s:
            continue
        snap = s[0]
        lab = s[1]
        merge_with_iou_size(snap, lab, bound_dict, threshold, bound_threshold, set_size, out, set_cover_s)


def create_label_update():
    label_folder = "/scratch/yw180/place/data/all_labels/labels/"
    update_folder = "/home/yw180/place/data/updates/"
    start_snap = 1490918688
    end_snap = 1491238734
    for i in range(start_snap, end_snap + 1):
        label_file = label_folder + "label_"+str(i)+"000.npy"
        update_file = update_folder + "update_"+str(i)+"000.npy"
        if not os.path.exists(label_file): 
            print("cont")
            continue
        label_canv = np.load(label_file)
        update_canv = np.load(update_file)
        result_dict = dict()
        for y in range(0, len(label_canv)):
            for x in range(0, len(label_canv[0])):
                lab = label_canv[y,x]
                up = update_canv[y,x]
                if up == 0:
                    continue
                if lab not in result_dict:
                    result_dict[lab] = list()
                result_dict[lab].append(up)
        with open("/scratch/yw180/place/data/lu/lu_"+str(i)+"000.csv", "w") as out:
            writer = csv.writer(out, delimiter = ",")
            for lab in result_dict:
                result_dict[lab].insert(0, lab)
                writer.writerow(result_dict[lab])


def make_bounding(snapshot):
    label_dict = dict()
    label_canv = np.load("/scratch/yw180/place/data/all_labels/labels/label_"+str(snapshot)+"000.npy")
    for x in range(0, 1001):
        for y in range(0, 1001):
            curr_lab = label_canv[y,x]
            if curr_lab not in label_dict:
                # min x, max x, min y, max y
                label_dict[curr_lab] = [x,x,y,y]
            else:
                if x < label_dict[curr_lab][0]:
                    label_dict[curr_lab][0] = x
                elif x > label_dict[curr_lab][1]:
                    label_dict[curr_lab][1] = x
                if y < label_dict[curr_lab][2]:
                    label_dict[curr_lab][2] = y
                elif y > label_dict[curr_lab][3]:
                    label_dict[curr_lab][3] = y
    with open("/scratch/yw180/place/data/bounds/b_"+str(snapshot)+"000.pkl", "wb") as f:
        pickle.dump(label_dict, f)

    
def get_iou(tup1, tup2):
    if tup1[0] == tup1[1]:
        tup1[1] += 1
    if tup1[2] == tup1[3]:
        tup1[3] += 1
    if tup2[0] == tup2[1]:
        tup2[1] += 1
    if tup2[2] == tup2[3]:
        tup2[3] += 1
    left = max(tup1[0], tup2[0])
    right = min(tup1[1], tup2[1])
    top = max(tup1[2], tup2[2])
    bottom = min(tup1[3], tup2[3])
    if left > right or bottom < top:
        return 0.0 
    intersect_area = (right - left) * (bottom - top)
    box1_area = (tup1[1] - tup1[0]) * (tup1[3] - tup1[2])
    box2_area = (tup2[1] - tup2[0]) * (tup2[3] - tup2[2])
    union_area = box1_area + box2_area - intersect_area
    iou = intersect_area / union_area
    return iou

def make_set_size():
    lu_dir = "/scratch/yw180/place/data/lu/"
    set_size_dict = dict()
    set_cover = dict()
    with open("/scratch/yw180/place/data/seg_analysis_out.csv", "r") as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            snap = int(line[0])
            lab = int(line[1])
            if snap not in set_cover:
                set_cover[snap] = set()
            set_cover[snap].add(lab)
    for snapshot in set_cover:
        lu_file = lu_dir+"lu_"+str(snapshot) + "000.csv"
        with open(lu_file, "r") as f:
            reader = csv.reader(f, delimiter=',')
            for line in reader:
                lab = int(line[0])
                if lab in set_cover[snapshot]:
                    set_size_dict[(snapshot, lab)] = len(line[1:])
    with open("/scratch/yw180/place/data/set_size.pkl", "wb") as f:
        pickle.dump(set_size_dict, f)
  
def make_set_bound():
    set_cover = dict()
    bound_dir = "/scratch/yw180/place/data/bounds/"
    bound_dict = dict()
    with open("/scratch/yw180/place/data/seg_analysis_out.csv", "r") as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            snap = int(line[0])
            lab = int(line[1])
            if snap not in set_cover:
                set_cover[snap] = set()
            set_cover[snap].add(lab)
    for snapshot in set_cover:
        bound_file = bound_dir + "b_" + str(snapshot) + "000.pkl"
        with open(bound_file, 'rb') as f:
            curr_bound = pickle.load(f)
            for lab in curr_bound:
                if lab in set_cover[snapshot]:
                    bound_dict[(snapshot, lab)] = curr_bound[lab]
    with open("/scratch/yw180/place/data/set_bound.pkl", "wb") as f:
        pickle.dump(bound_dict, f)    


def merge_with_iou_size(snapshot, label, bound_dict, threshold, bound_threshold, set_size, out, set_cover):
    merge_list = list()
    merge_set = set()
    merge_list.append(snapshot)
    merge_list.append(label)
    size = set_size[(snapshot, label)]
    tup = bound_dict[(snapshot, label)]
    for s in set_cover:
        curr_snap = s[0]
        curr_lab = s[1]
        if curr_snap == snapshot and curr_lab == label:
            continue
        curr_tup = bound_dict[(curr_snap, curr_lab)]
        iou_val = get_iou(curr_tup, tup)
        if iou_val > bound_threshold:
            curr_size = set_size[(curr_snap, curr_lab)]
            size_val = min(curr_size, size) / max(curr_size, size)
            if size_val > threshold:
                merge_list.append(curr_snap)
                merge_list.append(curr_lab)
                merge_set.add((curr_snap, curr_lab))
    for pair in merge_set:
        set_cover.remove(pair)
    with open(out, "a") as file_out:
        writer = csv.writer(file_out, delimiter = ",")
        print("writing...")
        writer.writerow(merge_list)

def make_new_bound_dict():
    bound_dict_path = "/scratch/yw180/place/data/set_bound.pkl"
    with open(bound_dict_path, 'rb') as f:
        bound_dict = pickle.load(f) 
    new_bound_dict = dict()
    with open("/home/yw180/place/data/merged_8/merged_80.csv", "r") as f:
        reader = csv.reader(f, delimiter=',')
        line_count = 0
        for line in reader:
            left = 1001
            right = 0
            top = 1001
            bottom = 0
            snapshot = 0
            label = 0
            count = 0
            for item in line:
                item = int(item)
                if count % 2 == 0:
                    snapshot = item
                else:
                    label = item
                    tup = bound_dict[(snapshot, label)]
                    if tup[0] < left:
                        left = tup[0]
                    if tup[1] > right:
                        right = tup[1]
                    if tup[2] < top:
                        top = tup[2]
                    if tup[3] > bottom:
                        bottom = tup[3]
                count += 1
            new_bound_dict[line_count] = (left,right,top,bottom)
            line_count += 1
    with open("/home/yw180/place/data/merged_8/new_bound.pkl", "wb") as f:
        pickle.dump(new_bound_dict, f) 

# Creates user embedding for each cover
def make_user_emb(sorted_tile_placement_path, users_emb_path,users_ind_path, reformated_lab_update_path, out):
    update_user = dict()
    with open(sorted_tile_placement_path, "r") as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for line in reader:
            u = int(line[0])
            user = line[2]
            update_user[u] = user
    
    line_emb = dict()

    #node2vec user embeddings
    with open(users_emb_path,'rb') as f:
        users_emb = pickle.load(f)
    with open(users_ind_path,'rb') as f:
        users_ind = pickle.load(f)
    
    line_count = 0
    with open(reformated_lab_update_path, "r") as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            emb = np.zeros(120)
            for item in line:
                user = update_user[int(item)]
                if user not in users_ind:
                    continue
                emb += users_emb[users_ind[user]]
            emb = emb / len(line)
            line_emb[line_count] = emb
            line_count += 1
    with open(out, "wb") as f:
        pickle.dump(line_emb, f)   

# Reformats output of merging
def make_lab_update(reformated_lab_update_path):
    lab_update = dict()
    line_count = 0
    with open(reformated_lab_update_path, "r") as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            lab_update[line_count] = [int(x) for x in line]
            line_count += 1
    return lab_update

#### Merge set covers based on user embeddings
def merge_user_emb(threshold, out, overlap_threshold, bound_dict_path, line_emb_path):
    with open(bound_dict_path, 'rb') as f:
        bound_dict = pickle.load(f) 
    with open(line_emb_path, 'rb') as f:
        line_emb = pickle.load(f) 
    total_line = 39879
    for i in range(0, total_line + 1):
        if i not in line_emb:
            continue
        res = compare_emb(i, threshold, line_emb, bound_dict, overlap_threshold)
        for l in res:
            del line_emb[l]
        with open(out, "a") as file_out:
            writer = csv.writer(file_out, delimiter = ",")
            writer.writerow(res)
            
#### Calculates consine similarity for a selected set cover against 
#### all other drawings.
#### Returns a list of all drawings to be merged.
def compare_emb(line, threshold, line_emb,bound_dict, overlap_threshold):
    tup1 = list(bound_dict[line])
    # print(tup1)
    merge_out = [line]
    emb = line_emb[line]
    for l in line_emb:
        if l == line: continue
        tup2 = list(bound_dict[l])
        # print(tup2)
        if get_iou(tup1, tup2) <= overlap_threshold:
            continue
        curr_emb = line_emb[l]
        cos_sim = np.dot(emb,curr_emb)/(np.linalg.norm(emb)*np.linalg.norm(curr_emb))
        if cos_sim > threshold:
            merge_out.append(l)
    return merge_out

       
