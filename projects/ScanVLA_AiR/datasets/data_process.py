import os
from pathlib import Path
from os.path import join
import json
import numpy as np
from scipy import io

import pickle
# default_datasets_dir = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/AiR/")
default_datasets_dir = Path("/data/lyt/01-Datasets/01-ScanPath-Datasets/AiR/")
save_dir_path = "/data/lyt/01-Datasets/01-ScanPath-Datasets/AiR/air_processed/"

AiR_stimuli_dir = default_datasets_dir / "stimuli"
AiR_fixations_dir = default_datasets_dir / "fixations"
AiR_attention_bbox_dir = default_datasets_dir / "attention_reasoning"

scanpath_len = []
for phase in ("train", "validation", "test"):
    AiR_fixations_file = join(AiR_fixations_dir,
                                   "AiR_fixations_" + phase + ".json")

    with open(AiR_fixations_file) as json_file:
        fixations = json.load(json_file)

    qid_to_sub = {} # 一个问题对应多个 扫视路径，每条存储在对应 index 中
    qid_to_img = {}
    for index, fixation in enumerate(fixations):
        qid_to_sub.setdefault(fixation['question_id'], []).append(index)
        qid_to_img[fixation['question_id']] = fixation['image_id']
    qids = list(qid_to_sub.keys())

    performancesCount = {"True": 0, "False": 0}
    non_scanpath_count = {"True": 0, "False": 0}
    for idx in range(len(qids)):
        question_id = qids[idx]
        img_name = qid_to_img[question_id]
        # print(f"img_name: {img_name}, question_id: {question_id}-----------------")
        fixs_all = []
        fixs_right = []
        fixs_wrong = []
        question = ""
        for ids in qid_to_sub[question_id]:
            # 获取当前参与者数据
            fixation = fixations[ids]
            question = fixation["question"]

            origin_size_y, origin_size_x = fixation["height"], fixation["width"]
            pos_x = np.array(fixation["X"]).astype(np.float32)
            pos_y = np.array(fixation["Y"]).astype(np.float32)
            gt_fixation = np.vstack((pos_y, pos_x)).T
            scanpath_len.append(len(gt_fixation))
            performance = fixation["subject_answer"] == fixation["answer"] and fixation["subject_answer"] != "faild"
            performancesCount[str(performance)] += 1
            # gt_fixation = np.expand_dims(gt_fixation, axis=0)

            fixs_all.append(gt_fixation)
            if performance:
                fixs_right.append(gt_fixation)
            else:
                fixs_wrong.append(gt_fixation)
        if not fixs_right:
            non_scanpath_count['True'] += 1

        if not fixs_wrong:
            non_scanpath_count['False'] += 1

        # fixs_all = np.array(fixs_all)
        # fixs_right = np.array(fixs_right)
        # fixs_wrong = np.array(fixs_wrong)

        fixations_all = np.empty((len(fixs_all)), dtype=np.object)
        for i in range(len(fixs_all)):
            fixations_all[i] = fixs_all[i]

        fixations_right = np.empty((len(fixs_right)), dtype=np.object)
        for i in range(len(fixs_right)):
            fixations_right[i] = fixs_right[i]

        fixations_wrong = np.empty((len(fixs_wrong)), dtype=np.object)
        for i in range(len(fixs_wrong)):
            fixations_wrong[i] = fixs_wrong[i]

        # io.savemat(os.path.join(save_dir_path, str(question_id) + ".mat"), {
        #     "img_name": img_name,
        #     "question": question,
        #     "question_id": question_id,
        #     "img_size": [origin_size_y, origin_size_x],
        #     "fixations_all": fixations_all,
        #     "fixations_right": fixations_right,
        #     "fixations_wrong": fixations_wrong,
        # })

        data = {
            "img_name": img_name,
            "question": question,
            "question_id": question_id,
            "img_size": [origin_size_y, origin_size_x],
            "fixations_all": fixations_all,
            "fixations_right": fixations_right,
            "fixations_wrong": fixations_wrong,
        }

        # 1. 创建目录 + 写入Pickle
        os.makedirs(save_dir_path, exist_ok=True)
        pkl_path = os.path.join(save_dir_path, f"{question_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(phase, "--------------non_scanpath_count-------------", non_scanpath_count)
    print(phase, "----------------performancesCount-----------", performancesCount)

print(f"aveg-len---{np.array(scanpath_len).mean()}")
