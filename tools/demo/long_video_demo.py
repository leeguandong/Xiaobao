# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import random
from collections import deque
from operator import itemgetter

import os
import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]

path = r"E:\comprehensive_library\mmaction2_add"


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 predict different labels in a long video demo')
    parser.add_argument('--config',
                        default=os.path.join(path, r"\configs\recognition\tsn\tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py"),
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        default=os.path.join(path, r"weights\tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth"),
                        help='checkpoint file/url')
    parser.add_argument('--video_path',
                        default=os.path.join(path, r"data\test\sample-mp4-file.mp4"),
                        help='video file/url')
    parser.add_argument('--label',
                        default=os.path.join(path, r"tools\data\kinetics\label_map_k400.txt"),
                        help='label file')
    parser.add_argument('--out_file',
                        default=os.path.join(path, r"data\test\sample-mp4.mp4"),
                        help='output result file in video/json')
    parser.add_argument(
        '--input-step',
        type=int,
        default=3,  # 在视频中的每 N 帧中选取一帧作为输入，默认为 1
        help='input step for sampling frames')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.2,  # 动作识别的分数阈值，只有分数大于阈值的动作类型会被显示，默认为 0.01
        help='recognition score threshold')
    parser.add_argument(
        '--stride',
        type=float,
        default=0,
        help=('the prediction stride equals to stride * sample_length '
              '(sample_length indicates the size of temporal window from '
              'which you sample frames, which equals to '
              'clip_len x frame_interval), if set as 0, the '
              'prediction stride is 1'))
    # 默认情况下，脚本为每帧给出单独预测，较为耗时。可以设定 STRIDE 参数进行加速，
    # 此时脚本将会为每 STRIDE x sample_length 帧做一次预测（sample_length 指模型采帧时的时间窗大小，
    # 等于 clip_len x frame_interval）。例如，若 sample_length 为 64 帧且 STRIDE 设定为 0.5，模型将每 32 帧做一次预测。
    # 若 STRIDE 设为 0，模型将为每帧做一次预测。STRIDE 的理想取值为 (0, 1] 间，若大于 1，脚本亦可正常执行。STRIDE 默认值为 0。
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--label-color',
        nargs='+',
        type=int,
        default=(255, 255, 255),
        help='font color (B, G, R) of the labels in output video')
    parser.add_argument(
        '--msg-color',
        nargs='+',
        type=int,
        default=(128, 128, 128),
        help='font color (B, G, R) of the messages in output video')
    args = parser.parse_args()
    return args


def show_results_video(result_queue,
                       text_info,
                       thr,
                       msg,
                       frame,
                       video_writer,
                       label_color=(255, 255, 255),
                       msg_color=(128, 128, 128)):
    if len(result_queue) != 0:
        text_info = {}
        results = result_queue.popleft()
        for i, result in enumerate(results):
            selected_label, score = result
            if score < thr:
                break
            location = (0, 40 + i * 20)
            text = selected_label + ': ' + str(round(score, 2))
            text_info[location] = text
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                        label_color, THICKNESS, LINETYPE)
    elif len(text_info):
        for location, text in text_info.items():
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                        label_color, THICKNESS, LINETYPE)
    else:
        cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, msg_color,
                    THICKNESS, LINETYPE)
    video_writer.write(frame)
    return text_info


def get_results_json(result_queue, text_info, thr, msg, ind, out_json):
    if len(result_queue) != 0:
        text_info = {}
        results = result_queue.popleft()
        for i, result in enumerate(results):
            selected_label, score = result
            if score < thr:
                break
            text_info[i + 1] = selected_label + ': ' + str(round(score, 2))
        out_json[ind] = text_info
    elif len(text_info):
        out_json[ind] = text_info
    else:
        out_json[ind] = msg
    return text_info, out_json


def show_results(model, data, label, args):
    frame_queue = deque(maxlen=args.sample_length)
    result_queue = deque(maxlen=1)

    cap = cv2.VideoCapture(args.video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    msg = 'Preparing action recognition ...'
    text_info = {}
    out_json = {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (frame_width, frame_height)

    ind = 0
    video_writer = None if args.out_file.endswith('.json') \
        else cv2.VideoWriter(args.out_file, fourcc, fps, frame_size)
    prog_bar = mmcv.ProgressBar(num_frames)
    backup_frames = []

    while ind < num_frames:
        ind += 1
        prog_bar.update()
        ret, frame = cap.read()
        if frame is None:
            # drop it when encounting None
            continue
        backup_frames.append(np.array(frame)[:, :, ::-1])
        if ind == args.sample_length:
            # provide a quick show at the beginning
            frame_queue.extend(backup_frames)
            backup_frames = []
        elif ((len(backup_frames) == args.input_step
               and ind > args.sample_length) or ind == num_frames):
            # pick a frame from the backup
            # when the backup is full or reach the last frame
            chosen_frame = random.choice(backup_frames)
            backup_frames = []
            frame_queue.append(chosen_frame)

        ret, scores = inference(model, data, args, frame_queue)

        if ret:
            num_selected_labels = min(len(label), 5)
            scores_tuples = tuple(zip(label, scores))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]
            result_queue.append(results)

        if args.out_file.endswith('.json'):
            text_info, out_json = get_results_json(result_queue, text_info,
                                                   args.threshold, msg, ind,
                                                   out_json)
        else:
            text_info = show_results_video(result_queue, text_info,
                                           args.threshold, msg, frame,
                                           video_writer, args.label_color,
                                           args.msg_color)

    cap.release()
    cv2.destroyAllWindows()
    if args.out_file.endswith('.json'):
        with open(args.out_file, 'w') as js:
            json.dump(out_json, js)


def inference(model, data, args, frame_queue):
    if len(frame_queue) != args.sample_length:
        # Do no inference when there is no enough frames
        return False, None

    cur_windows = list(np.array(frame_queue))
    if data['img_shape'] is None:
        data['img_shape'] = frame_queue[0].shape[:2]

    cur_data = data.copy()
    cur_data['imgs'] = cur_windows
    cur_data = args.test_pipeline(cur_data)
    cur_data = collate([cur_data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        cur_data = scatter(cur_data, [args.device])[0]
    with torch.no_grad():
        scores = model(return_loss=False, **cur_data)[0]

    if args.stride > 0:
        pred_stride = int(args.sample_length * args.stride)
        for _ in range(pred_stride):
            frame_queue.popleft()

    # for case ``args.stride=0``
    # deque will automatically popleft one element

    return True, scores


def main():
    args = parse_args()

    args.device = torch.device(args.device)

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    data = dict(img_shape=None, modality='RGB', label=-1)
    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.data.test.pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0
    args.sample_length = sample_length
    args.test_pipeline = test_pipeline

    show_results(model, data, label, args)


if __name__ == '__main__':
    main()
