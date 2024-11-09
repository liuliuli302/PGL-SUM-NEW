# 以上述代码为参考，写一个计算单个summary分数的函数
import json
from os import listdir
import h5py
import numpy as np
import pprint
from evaluation_metrics import evaluate_summary
from generate_summary import generate_single_summary


def compute_single_summary_score(
    scores: np.ndarray,
    shot_bound: np.ndarray,
    n_frames: np.ndarray,
    positions: np.ndarray,
    user_summary: np.ndarray,
    eval_method: str,
):
    """
    Compute the f_score of a single summary
    Args:
        scores(np.ndarray): importance scores of the frames
        shot_bound(np.ndarray): shot boundaries
        n_frames(np.ndarray): number of frames
        positions(np.ndarray): positions of the frames
        user_summary(np.ndarray): user summary
        eval_method(str): evaluation method to be used
    Returns:
        f_score: float, f_score of the summary
    """
    summary = generate_single_summary(shot_bound, scores, n_frames, positions)
    f_score = evaluate_summary(summary, user_summary, eval_method)

    return f_score


def get_video_name_dict(hdf: h5py.File):
    """
    Get a dictionary of video names from the h5py file
    Args:
        hdf(h5py.File): h5py file
    Returns:
        video_name_dict(dict): dictionary of video names
    """
    keys = list(hdf.keys())
    video_name_dict = {}
    for video_name in keys:
        video_name_dict[video_name] = hdf.get(video_name).get("video_name")[()]
    # 将dict中value转换为str,并将其中的空格替换为下划线，然后将key和value反转
    video_name_dict = {
        value.decode("utf-8").replace(" ", "_"): key
        for key, value in video_name_dict.items()
    }
    return video_name_dict


def get_gt(hdf: h5py.File, video_name: str):
    """
    Get the ground truth summary of a video
    Args:
        hdf(h5py.File): h5py file
        video_name(str): name of the video
    Returns:
        user_summary(np.ndarray): ground truth summary
    """
    video_name_dict = get_video_name_dict(hdf)
    user_summary = np.array(hdf.get(video_name + "/user_summary"))
    shot_bound = np.array(hdf.get(video_name + "/change_points"))
    n_frames = np.array(hdf.get(video_name + "/n_frames"))
    positions = np.array(hdf.get(video_name + "/picks"))
    return user_summary, shot_bound, n_frames, positions, video_name_dict


def read_and_process_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    max_index = max(map(int, data.keys()))
    result_array = [data.get(str(i), None) for i in range(max_index + 1)]

    for i in range(1, len(result_array)):
        if result_array[i] is None:
            result_array[i] = result_array[i - 1]

    return result_array


def process_scores(scores: np.ndarray, sample_interval: int, nframes: int):
    score_processed = np.zeros(nframes, dtype=np.float32)
    for i in range(nframes // 16):
        pos_left, pos_right = i * sample_interval, (i + 1) * sample_interval
        if i * sample_interval >= nframes:
            score_processed[pos_left:-1] = scores[i]
        else:
            score_processed[pos_left:pos_right] = scores[i]
    
    return score_processed


# 示例用法
if __name__ == "__main__":
    result_path = (
        "../PGL-SUM-Modify/Summaries/PGL-SUM/exp1/SumMe/results/split0/SumMe_155.json"
    )
    dataset_path = (
        "../PGL-SUM-Modify/data/datasets/SumMe/eccv16_dataset_summe_google_pool5.h5"
    )
    dataset = "SumMe"
    eval_method = "max"
    with open(result_path) as f:  # read the json file ...
        data = json.loads(f.read())
        keys = list(data.keys())
        hdf = h5py.File(dataset_path, "r")
        for video_name in keys:  # for each video inside that json file ...
            scores = np.asarray(
                data[video_name]
            )  # read the importance scores from frames
            user_summary, shot_bound, n_frames, positions, video_name_dict = get_gt(
                hdf=hdf, video_name=video_name
            )
            score_processed = process_scores(
                scores=scores, sample_interval=15, nframes=n_frames
            )
            f_score = compute_single_summary_score(
                score_processed,
                shot_bound,
                n_frames,
                positions,
                user_summary,
                eval_method,
            )
            print(f"{video_name}:{f_score}")
