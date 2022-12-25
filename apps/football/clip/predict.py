import os
import sys
import json
from pathlib import Path

sys.path.append('action_detect')
from action import ActionDetection

if __name__ == '__main__':
    # dataset_dir = "/workspace/PaddleVideo/applications/FootballAction/datasets/EuroCup2016"
    dataset_dir = r"E:\comprehensive_library\Xiaobao\data\test"

    model_predict = ActionDetection(cfg_file="./configs/configs.yaml")
    model_predict.load_model()

    video_url = os.path.join(dataset_dir, 'url.list')
    with open(video_url, 'r') as f:
        lines = f.readlines()
    lines = [os.path.join(dataset_dir, k.strip()) for k in lines]

    results = []
    for line in lines:
        video_name = line

        imgs_path = os.path.join(str(Path(video_name).parent), "frames/football_test")
        pcm_path = os.path.join(str(Path(video_name).parent), "pcm/football_test.pcm")

        bmn_results, action_results = model_predict.infer(imgs_path, pcm_path)
        results.append({
            'video_name': line,
            'bmn_results': bmn_results,
            'action_results': action_results
        })

    with open('results.json', 'w', encoding='utf-8') as f:
        data = json.dumps(results, indent=4, ensure_ascii=False)
        f.write(data)
