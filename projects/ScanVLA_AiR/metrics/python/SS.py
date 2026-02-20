import os

import torch
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

def scanpath2clusters(meanshift, scanpath):
    string = []
    xs = list(scanpath[:,1])
    ys = list(scanpath[:,0])

    for i in range(len(xs)):
        symbol = meanshift.predict([[xs[i], ys[i]]])[0]
        string.append(symbol)
    return string

def improved_rate(meanshift, scanpaths):
    Nc = len(meanshift.cluster_centers_)
    Nb, Nw = 0, 0
    for scanpath in scanpaths:
        string = scanpath2clusters(meanshift, scanpath)
        for i in range(len(string)-1):
            if string[i]==string[i+1]:
                Nw += 1
            else:
                Nb += 1
    return (Nb-Nw)/Nc

def zero_one_similarity(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0
def nw_matching(pred_string, gt_string, gap=0.0):
    # NW string matching with zero_one_similarity
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]
            b = gt_string[j - 1]
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)
            delete = F[i - 1, j] + gap
            insert = F[i, j - 1] + gap
            F[i, j] = np.max([match, delete, insert])
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))

def SS_Score(pred_scanpath, gt_scanpaths):
    scanpaths = gt_scanpaths
    xs, ys = [], []
    for scanpath in scanpaths:
        xs += list(scanpath[:, 1])
        ys += list(scanpath[:, 0])

    gt_gaze = np.concatenate((np.vstack(xs), np.vstack(ys)), axis=1)
    bandwidth = estimate_bandwidth(gt_gaze)
    rates = []
    factors = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    for factor in factors:
        bd = bandwidth * factor
        if bd <= 0.0:
            bd = None

        ms = MeanShift(bandwidth=bd)
        ms.fit(gt_gaze)
        rate = improved_rate(ms, scanpaths)
        rates.append(rate)
    rates = np.vstack(rates)

    best_bd = factors[np.argmax(rates)] * bandwidth
    if best_bd <= 0.0:
        best_bd = None
    best_ms = MeanShift(bandwidth=best_bd)
    best_ms.fit(gt_gaze)

    # save best_ms for evaluation
    gt_strings = []
    for gt_scanpath in scanpaths:
        gt_string = scanpath2clusters(best_ms, gt_scanpath)
        gt_strings.append(gt_string)

    scores=[]
    pre = scanpath2clusters(best_ms, pred_scanpath)
    for gt in gt_strings:
        score = nw_matching(pre,gt)
        scores.append(score)
    ans = np.array(scores).mean()
    return ans 

    # scores=[]
    # for pre_scanpath in pred_scanpath:
    #     pre=scanpath2clusters(best_ms,pre_scanpath)
    #     for gt in gt_strings:
    #         score = nw_matching(pre,gt)
    #         scores.append(score)
    # ans = np.array(scores).mean()
    # return ans 