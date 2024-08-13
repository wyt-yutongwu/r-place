import os
import sys
import numpy as np
import math
import random
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score, adjusted_rand_score
from sklearn.metrics import adjusted_rand_score
np.seterr(over='ignore')
from disjoint_set import *

# edge weight function that needs code to rgb mapping
code_to_rgb = {0: [255,255,255], 1: [228,228,228], 2: [136,136,136],\
        3: [34,34,34], 4: [255,167,209], 5: [229,0,0],\
            6: [229,149,0], 7: [160,106,66], 8: [229,217,0],\
        9: [148,224,68], 10: [2,190,1], 11: [0,229,240],\
               12: [0,131,199], 13: [0,0,234], 14: [224,74,255],\
           15: [130,0,128], 16: [220,220,220]}

code_to_lab = {0: [255, 128, 128], 1: [231, 128, 128], 2: [145, 128, 128],\
              3: [34, 128, 128], 4: [199, 166, 120], 5: [122, 202, 190],\
               6: [174, 149, 201], 7: [127, 145, 159], 8: [217, 114, 213],\
                9: [209, 81, 194], 10: [171, 59, 194], 11: [212, 87, 110],\
                 12: [133, 123, 84], 13: [75, 202, 27], 14: [153, 208, 66],\
                  15: [77, 187, 92], 16: [224, 128, 128]}

def normalize_channel(channel):
    mean, std  = channel.mean(), channel.std()
    return (channel - mean)/std, mean, std

def unnormalize_channel(channel, mean, std):
    return std*channel+mean

def get_threshold(size, c):
    return c / size

# randomly creates RGB
def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(0, 255)
    rgb[1] = random.randint(0, 255)
    rgb[2] = random.randint(0, 255)
    return rgb

# edge weight function between pixels
def diff(red_band, green_band, blue_band, x1, y1, x2, y2):
    result = math.sqrt(
        (red_band[y1, x1] - red_band[y2, x2])**2 + (green_band[y1, x1] - green_band[y2, x2])**2 + (
            blue_band[y1, x1] - blue_band[y2, x2])**2)
    return result

############## for segmenting 3d graph             ###############
def diff_plus(canvas, x1, y1, x2, y2):
    result = math.sqrt(
        (canvas[y1, x1, 0] - canvas[y2, x2, 0])**2 + (canvas[y1, x1, 1] - canvas[y2, x2, 1])**2 + (
            canvas[y1, x1, 2] - canvas[y2, x2, 2])**2)
    return result

# edge weight function between pixels
def diff_userembs(userembs, x1, y1, x2, y2):
    result = np.linalg.norm(userembs[x1,y1,:]-userembs[x2,y2,:])
    return result

def diff_user_rgb(alpha, userembs, red_band, green_band, blue_band, x1, y1, x2, y2):
   result1 = diff(red_band, green_band, blue_band, x1, y1, x2, y2)
   result2 = diff_userembs(userembs, x1, y1, x2, y2)
   result = alpha * result1 + (1-alpha) * result2
   #print("result1:", result1, "result2:", result2, "result:", result)
   return result
    # result1= diff(red_band, green_band, blue_band, x1, y1, x2, y2) 
    # result2 = diff_userembs(userembs, x1, y1, x2, y2)
    #result = np.min(result1, result2)
    # return result1 if result1 < result2 else result2

def entropy_helper(clustering):
    labels = np.unique(clustering)
    N = len(clustering)
    probs = np.zeros(len(labels))
    for ind, label in enumerate(labels):
        probs[ind] = float((clustering == label).sum())/N
    return entropy(probs)

def VOI(pred,gt):
    return entropy_helper(pred) + entropy_helper(gt) - 2*mutual_info_score(pred,gt)

# takes long time to compute
def segmentation_covering(pred,gt):
    
    size = len(pred)
    pred_labels = np.unique(pred)
    gt_labels = np.unique(gt)
    print("Computing segmentation covering.")
    print("Total number of ground truth regions is {}.".format(len(gt_labels)))

    result = 0

    for ind, gt_label in enumerate(gt_labels):
        # print("Computing segmentation cover for ind {}.".format(ind))
        sys.stdout.flush()
        gt_region = (gt == gt_label)
        best_iou = 0
        for pred_label in pred_labels:
            pred_region = (pred == pred_label)
            iou = float(np.logical_and(gt_region,pred_region).sum())/float(np.logical_or(gt_region,pred_region).sum())
            if(iou > best_iou):
                best_iou = iou
        result += gt_region.sum()*best_iou

    return result/size

def compute_metrics(height, width, predLabels, labelCanvas):

    ri = adjusted_rand_score(predLabels.reshape(-1),labelCanvas[:height,:width].reshape(-1)),
    print("Rand index is: {:.4f}".format(ri[0]))
    sys.stdout.flush()
    # ari =  adjusted_rand_score(predLabels.reshape(-1),labelCanvas[:height,:width].reshape(-1))
    # print("Adjusted rand index is: {:.4f}".format(ari))
    voi = VOI(predLabels.reshape(-1),labelCanvas[:height,:width].reshape(-1))
    print("Variation of information is: {:.4f}".format(voi))        #uses natural log
    sys.stdout.flush()
    # sc = segmentation_covering(predLabels.reshape(-1),labelCanvas[:height,:width].reshape(-1))
    # print("Segmentation covering is: {:.4f}".format(sc))
    # sys.stdout.flush()
    return voi

def compute_metrics_with_mask(height, width, predLabels, labelCanvas, mask):
    flat_pred_lab = predLabels.reshape(-1)
    flat_lab_canv = labelCanvas.reshape(-1)
    flat_mask = mask.reshape(-1)
    m_predLables = flat_pred_lab[flat_mask != 0]
    m_labelCanvas = flat_lab_canv[flat_mask != 0]
    ri = adjusted_rand_score(m_predLables,m_labelCanvas),
    print("Adjusted rand index is: {:.4f}".format(ri[0]))
    sys.stdout.flush()
    # ari =  adjusted_rand_score(predLabels.reshape(-1),labelCanvas[:height,:width].reshape(-1))
    # print("Adjusted rand index is: {:.4f}".format(ari))
    voi = VOI(m_predLables,m_labelCanvas)
    print("Variation of information is: {:.4f}".format(voi))        #uses natural log
    sys.stdout.flush()
    # sc = segmentation_covering(predLabels.reshape(-1),labelCanvas[:height,:width].reshape(-1))
    # print("Segmentation covering is: {:.4f}".format(sc))
    # sys.stdout.flush()
    return ri[0], voi

def segment_graph(num_vertices, num_edges, edges, users=None, userembs=None, c=200):
    # sort edges by weight (3rd column)
    edges[0:num_edges, :] = edges[edges[0:num_edges, 2].argsort(kind='stable')] #don't reorder duplicates
    # make a disjoint-set forest
    u = disjoint_set(num_vertices, users, userembs)
    # init thresholds
    threshold = np.zeros(shape=num_vertices, dtype=float)
    for i in range(num_vertices):
        threshold[i] = get_threshold(1, c)                             

    # # for each edge, in non-decreasing weight order...
    for i in range(num_edges):
        pedge = edges[i, :]

        # components connected by this edge
        a = u.find(pedge[0])
        b = u.find(pedge[1])
        if a != b:
           # np.linalg.norm(u.get_useremb(a)-u.get_useremb(b))
            #print("pedge[2]", pedge[2], "threshold[a]", threshold[a], "threshold[b]", threshold[b])
            if (pedge[2] <= threshold[a]) and (pedge[2] <= threshold[b]):
                u.join(a, b)
                a = u.find(a)
                threshold[a] = pedge[2] + get_threshold(u.size(a), c)

    return u, threshold

def count_total_segs(folder):
    sum = 0
    for filename in os.listdir(folder):
        lab = np.load(folder+filename)
        sum += np.max(lab)
    return sum