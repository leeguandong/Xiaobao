'''
@Time    : 2022/11/24 15:31
@Author  : leeguandon@gmail.com
'''
import os
import json
import argparse
from pathlib import Path


def crop_json(video_file, bmn_results, action_results, temp_file):
    video_file = Path(video_file)
    bmn_dir = os.path.join(temp_file, "bmn")
    Path(bmn_dir).mkdir(parents=True, exist_ok=True)
    action_dir = os.path.join(temp_file, 'action')
    Path(action_dir).mkdir(parents=True, exist_ok=True)

    clip_bmn_txt = os.path.join(temp_file, "clip_bmn.txt")
    with open(clip_bmn_txt, "w", encoding="utf-8") as f:
        for index, bmn in enumerate(bmn_results):
            cmd = f"ffmpeg -y -i {str(video_file)} -c copy -ss {bmn['start']} " \
                  f"-to {bmn['end']} {bmn_dir}/{str(index)}.mp4 "
            os.system(cmd)
            f.write(f"file {str(index)}.mp4 \n")
    cmd = f"ffmpeg -y -f concat -i {clip_bmn_txt} -c copy {temp_file}/{str(video_file.stem)}_clip_bmn.mp4"
    os.system(cmd)

    clip_action_txt = os.path.join(temp_file, "clip_action.txt")
    with open(clip_action_txt, "w", encoding="utf-8") as f:
        for index, action in enumerate(action_results):
            cmd = f"ffmpeg -y -i {str(video_file)} -c copy -ss {action['start_time']} -to {action['end_time']} " \
                  f"{action_dir}/{str(index)}_{action['label_name']}_{action['classify_score']}.mp4"
            os.system(cmd)
            f.write(f"file {str(index)}_{action['label_name']}_{action['classify_score']}.mp4 \n")
    cmd = f"ffmpeg -y -f concat -i {clip_action_txt} -c copy {temp_file}/{str(video_file.stem)}_clip_action.mp4"
    os.system(cmd)


def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('--video_dir',
                        default=r'E:\comprehensive_library\Xiaobao\data\test\football_test.mp4',
                        type=str, help='source video directory')
    parser.add_argument("--json", default=r"E:\comprehensive_library\Xiaobao\apps\football\clip\results.json")
    parser.add_argument("--save_dir", default=r"E:\comprehensive_library\Xiaobao\data\cache")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    json_file = args.json
    video_file = args.video_dir
    save_dir = args.save_dir

    with open(json_file, 'r', encoding='utf-8') as f:
        json_file = json.load(f)

    bmn_results = json_file[0]['bmn_results']
    action_results = json_file[0]['action_results']
    crop_json(video_file, bmn_results, action_results, save_dir)
