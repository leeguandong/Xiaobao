'''
@Time    : 2022/11/21 19:08
@Author  : leeguandon@gmail.com
'''
import sys

sys.path.append('/home/ivms/local_disk/mmaction2_add')

import os
import json
import argparse
from pathlib import Path
from concurrent import futures
from action import ActionDetection


path = "E:/comprehensive_library/mmaction2_add"


def parse_args():
    parser = argparse.ArgumentParser(description="football")
    parser.add_argument("--dataset_dir",
                        default=os.path.join(path, r"\data\test"))
    parser.add_argument("--config",
                        default=os.path.join(path, r'\apps\FootballAction\configs\configs.yaml'))

    args = parser.parse_args()
    return args


def get_frames_pcm(video_path, frames, pcm_filename, fps=5):
    cmd = 'ffmpeg -hwaccel cuvid -c:v h264_cuvid -v 0 -i %s -r %d -q 0  -c:v h264_nvenc %s/%s.jpg' % (video_path, fps,
                                                                                                      frames, '%08d')
    os.system(cmd)

    cmd = 'ffmpeg -hwaccel cuvid -c:v h264_cuvid -y -i %s -acodec pcm_s16le -f s16le -ac 1 -ar 16000 -c:v h264_nvenc %s -v 0' % (
        video_path, pcm_filename)
    os.system(cmd)


if __name__ == '__main__':
    args = parse_args()
    dataset_dir = args.dataset_dir
    cfg_file = args.config

    model_predict = ActionDetection(cfg_file)
    model_predict.load_model()

    temp_file = './temp'
    frames = os.path.join(temp_file, "frames")
    Path(temp_file).mkdir(parents=True, exist_ok=True)
    Path(frames).mkdir(parents=True, exist_ok=True)

    results = []
    for video_file in Path(dataset_dir).glob("football_test.mp4"):
        pcm_filename = os.path.join(temp_file, str(video_file.stem) + ".pcm")
        # with futures.ProcessPoolExecutor(max_workers=10) as executer:
        #     fs = executer.submit(get_frames_pcm, video_file, frames, pcm_filename)
        get_frames_pcm(video_file, frames, pcm_filename)

        bmn_results, action_results = model_predict.infer(frames, pcm_filename)
        results.append({
            'video_name': video_file,
            'bmn_results': bmn_results,
            'action_results': action_results
        })

        # 将视频片段裁剪出来
        clip_txt = os.path.join(temp_file, "clip.txt")
        with open(clip_txt, "w", encoding="utf-8") as f:
            for index, bmn in enumerate(bmn_results):
                cmd = f"ffmpeg -y -i {str(video_file)} -c copy -ss {bmn['start']} " \
                      f"-to {bmn['end']} {temp_file}/{str(index)}.mp4 "
                os.system(cmd)
                f.write(f"file {str(index)}.mp4 \n")
        cmd = f"ffmpeg -y -f concat -i {clip_txt} -c copy {temp_file}/{str(video_file.stem)}_clip.mp4"
        os.system(cmd)

    with open('results.json', 'w', encoding='utf-8') as f:
        data = json.dumps(results, indent=4, ensure_ascii=False)
        f.write(data)
