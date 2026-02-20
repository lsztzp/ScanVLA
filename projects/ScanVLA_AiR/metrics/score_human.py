import os
import numpy as np
from scipy import io
from tqdm import tqdm
from AiR.metrics.matlab.matlab_score_gen import human_matlab_scores
from AiR.metrics.utils import gtspath, score_all_gts


def score_human(metrics=('scanmatch', 'tde', 'mutimatch')):
    print(f'comptering--AiR--dataset in {metrics}')
    gtspathdir = os.listdir(gtspath)

    scores_all_data = {"all": {}, "right": {}, "wrong": {}}
    count_data = {"all": 0, "right": 0, "wrong": 0}
    for qid_gt_name in tqdm(gtspathdir):
        gt_path = os.path.join(gtspath, qid_gt_name)
        # print(f'index:{index} --- name: {gt_name}')

        gt_fixations = io.loadmat(gt_path)

        img_size = gt_fixations["img_size"][0]
        h, w = img_size[0], img_size[1]

        for performance in ("all", "right", "wrong", ):
            gt_fixations_performance = gt_fixations[f'fixations_{performance}']

            # matlab 读取 格式问题
            if len(gt_fixations_performance.shape) == 2 and len(gt_fixations_performance) > 0:
                gt_fixations_performance = gt_fixations_performance[0]
            # reshape gtFixations
            for n in range(len(gt_fixations_performance)):
                gt_fixation = gt_fixations_performance[n].astype(np.float)
                gt_fixation[:, 0] /= h
                gt_fixation[:, 1] /= w
                # gt_fixation = np.clip(gt_fixation, 0, 0.98)
                gt_fixation[:, 0] *= 600
                gt_fixation[:, 1] *= 800
                gt_fixations_performance[n] = gt_fixation

            scores_qid_performance = {}
            count_human = 0
            for n in range(len(gt_fixations_performance)):
                gt_fixation = gt_fixations_performance[n].astype(np.float)
                other = np.delete(gt_fixations_performance, n, axis=0)
                scores_gts = score_all_gts(gt_fixation, other, metrics)

                # add this human score
                for metric, score in scores_gts.items():
                    if metric in scores_qid_performance:
                        scores_qid_performance[metric] += score
                    else:
                        scores_qid_performance[metric] = score
                count_human += 1

            # 数据集中 某些问题-图片对中错误的扫视路径为0条，不应计算其分数和增加统计数
            if not scores_qid_performance:
                continue
            # process all human scores as this data score
            for metric, score in scores_qid_performance.items():
                scores_qid_performance[metric] /= count_human
            # add this data score
            for metric, score in scores_qid_performance.items():
                if metric in scores_all_data[performance]:
                    scores_all_data[performance][metric] += score
                else:
                    scores_all_data[performance][metric] = score
            count_data[performance] += 1

    # prcess all data scores
    for performance, value in scores_all_data.items():
        for metric, score in value.items():
            scores_all_data[performance][metric] /= count_data[performance]

    with open("AiR/metrics/huaman-score.txt", "w") as f:
        scores_all_data_matlab = human_matlab_scores()
        f.write("----------------------human----score--------------------\n")
        for performance in ('right', 'wrong'):
            f.write(f"------------------{performance}----------------------\n")
            for metric, score in scores_all_data_matlab[performance].items():
                f.write(f"matlab----{performance}------{metric}:   {score}\n")

            for metric, score in scores_all_data[performance].items():
                if metric in ("scanmatch", "tde_h", "tde_m"):
                    continue
                f.write(f"python----{performance}------{metric}:   {score}\n")

    return scores_all_data

print(score_human())


