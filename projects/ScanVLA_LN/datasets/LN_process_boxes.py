'''
some code is from cvpr2024 pixel align large language model
'''

r"""Build Localized Narrative dataset tfrecord from jsonl.

python scenic/projects/pixel_llm/tools/build_ln_tfrecord.py \
--output_dir ~/Datasets/LN \
--ln_anno_path ~/Datasets/LN/annotations \
--coco_path ~/Datasets/coco
"""

import argparse
import collections
import json
import os
from typing import NamedTuple, List

import numpy as np
import torch
import tqdm
from scipy import interpolate


class TimedPoint(NamedTuple):
    x: float
    y: float
    t: float


class TimedUtterance(NamedTuple):
    utterance: str
    start_time: float
    end_time: float


class LocalizedNarrative(NamedTuple):
    """Represents a Localized Narrative annotation.

    Visit https://google.github.io/localized-narratives/index.html?file-formats=1
    for the documentation of each field.
    """
    dataset_id: str
    image_id: str
    annotator_id: int
    caption: str
    timed_caption: List[TimedUtterance]
    traces: List[List[TimedPoint]]
    voice_recording: str

    def __repr__(self):
        truncated_caption = self.caption[:60] + '...' if len(
            self.caption) > 63 else self.caption
        truncated_timed_caption = self.timed_caption[0].__str__()
        truncated_traces = self.traces[0][0].__str__()
        return (f'{{\n'
                f' dataset_id: {self.dataset_id},\n'
                f' image_id: {self.image_id},\n'
                f' annotator_id: {self.annotator_id},\n'
                f' caption: {truncated_caption},\n'
                f' timed_caption: [{truncated_timed_caption}, ...],\n'
                f' traces: [[{truncated_traces}, ...], ...],\n'
                f' voice_recording: {self.voice_recording}\n'
                f'}}')


def annotations_in_file(filename: str):
    """Yields all `LocalizedNarrative` dic in a given file.

    Args:
      filename: File to load the Localized Narratives from.

    Yields:
      LN dic.
    """
    with open(filename, 'r') as file_handler:
        for line in file_handler:
            yield LocalizedNarrative(**json.loads(line))


def trace2coord(traces, timed_caption):
    """Computing the average location with intergral."""

    t_arr = np.array([trace['t'] for trace in traces])
    x_arr = np.array([trace['x'] for trace in traces])
    y_arr = np.array([trace['y'] for trace in traces])

    assert np.all(t_arr[:-1] < t_arr[1:])  # 1

    # get the indices that would sort t_arr
    sort_indices = np.argsort(t_arr)

    # sort t_arr, x_arr, y_arr using the indices
    t_arr = t_arr[sort_indices]
    x_arr = x_arr[sort_indices]
    y_arr = y_arr[sort_indices]

    # Clip x and y values to be within [0, 1], as they represent normalized coordinates. 人在观察屏幕，坐标不可能超出范围
    x_arr = np.clip(x_arr, 0.0001, 0.9999)
    y_arr = np.clip(y_arr, 0.0001, 0.9999)

    num_points = args.num_samples_per_trace
    for dic in timed_caption:
        start_time = dic['start_time']
        end_time = dic['end_time']

        x_interpolator = interpolate.interp1d(
            t_arr, x_arr, fill_value='extrapolate'
        )
        y_interpolator = interpolate.interp1d(
            t_arr, y_arr, fill_value='extrapolate'
        )

        t_values = np.linspace(start_time, end_time, num=num_points)
        x_values = x_interpolator(t_values)
        y_values = y_interpolator(t_values)

        # Clip interpolated x and y values to be within [0, 1], 人在观察屏幕时，坐标不可能超出范围
        x_values = np.clip(x_values, 0.0, 1.0)
        y_values = np.clip(y_values, 0.0, 1.0)

        if t_values[-1] - t_values[0] < 1e-5:
            integral_x = np.mean(x_values)
            integral_y = np.mean(y_values)
        else:
            # calculate integral (average) x and y values
            integral_x = np.trapz(x_values, t_values) / (t_values[-1] - t_values[0])
            integral_y = np.trapz(y_values, t_values) / (t_values[-1] - t_values[0])

        dic['integral_x'] = integral_x
        dic['integral_y'] = integral_y
        dic['min_x'] = np.min(x_values)
        dic['min_y'] = np.min(y_values)
        dic['max_x'] = np.max(x_values)
        dic['max_y'] = np.max(y_values)
        dic['sampled_x'] = x_values.tolist()
        dic['sampled_y'] = y_values.tolist()

    return timed_caption


def is_valid_img_infos(image_id, img_infos):
    is_valid = False
    for img_info in img_infos:
        traces = []
        for trace in img_info.traces:
            traces.extend(trace)
        if len(traces) <= 1:
            print(f'no traces {len(traces)} for {image_id}', '=' * 10)
            continue
        is_valid = True
    return is_valid


# def process_record(image_path, image_id, img_infos):
#     """Creates a sequence example from a list of dict."""
#     # captions = [img_info.caption for img_info in img_infos]

#     timed_captions = []
#     for img_info in img_infos:
#         timed_caption = img_info.timed_caption
#         traces = []
#         for trace in img_info.traces:
#             traces.extend(trace)
#         if len(traces) <= 1:
#             print(f'no traces {len(traces)} for {image_id}', '=' * 10)
#             continue
#         timed_caption = trace2coord(traces, timed_caption)
#         center_x = np.array([[dic['integral_x']] for dic in timed_caption])
#         center_y = np.array([[dic['integral_y']] for dic in timed_caption])

#         timed_captions.append({
#             'center_x': center_x,
#             'center_y': center_y,
#             'utterance': [dic['utterance'] for dic in timed_caption],
#             'string': img_info.caption
#         })

#     assert timed_captions, 'no timed captions'

    # dataset_id = img_infos[0].dataset_id
    # feature = {
    #     'image_path': image_path,
    #     'image_id': image_id,
    #     'dataset_id': dataset_id,
    #     'center_x': [torch.tensor(dic['center_x'], dtype=torch.float32) for dic in timed_captions],
    #     'center_y': [torch.tensor(dic['center_y'], dtype=torch.float32) for dic in timed_captions],
    #     'caption_string': [dic['string'] for dic in timed_captions],
    #     'caption_utterance': [dic['utterance'] for dic in timed_captions],
    # }

    # return feature

def process_record(image_path, image_id, img_infos):
    """Creates a sequence example from a list of dict."""
    # captions = [img_info.caption for img_info in img_infos]

    timed_captions = []
    for img_info in img_infos:
        timed_caption = img_info.timed_caption
        traces = []
        for trace in img_info.traces:
            traces.extend(trace)
        if len(traces) <= 1:
            print(f'no traces {len(traces)} for {image_id}', '=' * 10)
            continue
        timed_caption = trace2coord(traces, timed_caption)

        # [N, ]
        fixation_x = np.array([dic['integral_x'] for dic in timed_caption])
        fixation_y = np.array([dic['integral_y'] for dic in timed_caption])
        
        # [N, 4]
        bbox_np = np.array([
            [dic['min_x'], dic['min_y'], dic['max_x'], dic['max_y']]
            for dic in timed_caption
        ])

        # # [N, 2]
        # center_np = np.array(
        #     [[dic['integral_x'], dic['integral_y']] for dic in timed_caption]
        # )
        # # [N, num_points, 2]
        # point_np = np.array([
        #     np.stack([dic['sampled_x'], dic['sampled_y']], axis=-1)
        #     for dic in timed_caption
        # ])

        timed_captions.append({
            'fixation_x': fixation_x,
            'fixation_y': fixation_y,
            'bbox_np': bbox_np,
            'utterance': [dic['utterance'] for dic in timed_caption],
            'caption': img_info.caption
        })

    assert timed_captions, 'no timed captions'

    dataset_id = img_infos[0].dataset_id
    feature = {
        'image_path': image_path,
        'image_id': image_id,
        'dataset_id': dataset_id,
        'fixation_x': [dic['fixation_x'] for dic in timed_captions],
        'fixation_y': [dic['fixation_y'] for dic in timed_captions],
        'bbox_np': [dic['bbox_np'] for dic in timed_captions],
        'caption': [dic['caption'] for dic in timed_captions],
        'utterance': [dic['utterance'] for dic in timed_captions],
    }

    return feature


def main(args):
    annotation_files = {
        'coco_train': [
            os.path.join(
                args.ln_anno_path,
                f'coco_train_localized_narratives-{i:05d}-of-00004.jsonl',
            )
            for i in range(4)
        ],
        'coco_val': [
            os.path.join(
                args.ln_anno_path, 'coco_val_localized_narratives.jsonl'
            )
        ],
    }

    image_dirs = {
        'coco_val': os.path.join(args.coco_path, 'val2017/'),
        'coco_train': os.path.join(args.coco_path, 'train2017/'),
    }

    for dataset_name, image_dir in image_dirs.items():
        id2anno = collections.defaultdict(list)
        loco_ds = annotation_files[dataset_name]

        for annotation_file in tqdm.tqdm(
                loco_ds, desc=f'dataset: {dataset_name}', position=0
        ):
            annotations = annotations_in_file(annotation_file)
            for annotation in tqdm.tqdm(
                    annotations, desc=f'process file: {annotation_file}', position=1
            ):
                image_id = annotation.image_id
                id2anno[image_id].append(annotation)

        num_images = len(id2anno)
        output_path_dir = os.path.join(
            args.output_dir,
            dataset_name,
            args.folder_name
        )
        os.makedirs(output_path_dir, exist_ok=True)

        pbar = tqdm.tqdm(
            id2anno.items(),
            desc=f'writing to {output_path_dir}',
            total=len(id2anno),
        )

        num_exampels = 0
        id_pos_projection = {}
        id_pos_projection_path = os.path.dirname(output_path_dir) + '/' + dataset_name + '.json'

        for image_id, anno_list in pbar:

            if not is_valid_img_infos(image_id, anno_list):
                continue
            image_path = os.path.join(image_dir, f'{int(image_id):012d}.jpg')
            # record = process_record(image_path, image_id, anno_list)
            record = process_record(image_path, image_id, anno_list)

            for i in range(len(record['caption'])):
                # 每条轨迹/caption都需要单独存储
                dic_each = {'image_path': record['image_path'], 
                            'image_id': record['image_id'],
                            'fixation_x': record['fixation_x'][i],
                            'fixation_y': record['fixation_y'][i], 
                            'bbox_np': record['bbox_np'][i], 
                            'caption': record['caption'][i],
                            'utterance': record['utterance'][i]}
                output_path = output_path_dir + '/' + image_id + '_' + str(i) + '.pck'
                torch.save(dic_each, output_path)
                save_to_projection = {'image_id': image_id, 'image_path': image_path, 'record_path': output_path}
                id_pos_projection[num_exampels] = save_to_projection
                num_exampels += 1

                if num_exampels % 1000 == 0:
                    print(f'Wrote {dataset_name} with {num_exampels} examples')

        # percentiles = [20,70,90,97]   #根据refgaze数据集，简单设定时间阈值
        # thresholds = np.percentile(time_record, percentiles)

        with open(id_pos_projection_path, 'w') as f:
            json.dump(id_pos_projection, f)

        # print(f"thresholds is {thresholds}")

        print(f'Wrote {dataset_name} with total {num_exampels} examples')
        print('Everything Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LN dataset')
    parser.add_argument('--output_dir', type=str, default='/data/lyt/01-Datasets/01-ScanPath-Datasets/LN/',
                        help='output_dir')
    parser.add_argument('--ln_anno_path', type=str,
                        default='/data/lyt/01-Datasets/01-ScanPath-Datasets/LN/annotations/', help='ln_anno_path')
    parser.add_argument('--coco_path', type=str, default='/data/lyt/01-Datasets/01-ScanPath-Datasets/coco/',
                        help='coco_path')
    parser.add_argument('--folder_name', type=str, default='pytorch_record', help='folder_name')
    parser.add_argument('--num_samples_per_trace', type=int, default=16, help='num_samples_per_trace')
    args = parser.parse_args()

    time_record = []
    main(args)