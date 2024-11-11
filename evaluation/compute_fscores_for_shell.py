# 以上述代码为参考，写一个计算单个summary分数的函数
import argparse
import json
from os import listdir
from pathlib import Path
import h5py
import numpy as np
import pprint
from evaluation_metrics import evaluate_summary
from generate_summary import generate_single_summary
import scipy.io as scio


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
    user_summary = np.array(hdf.get(video_name + "/user_summary"))
    shot_bound = np.array(hdf.get(video_name + "/change_points"))
    n_frames = np.array(hdf.get(video_name + "/n_frames"))
    positions = np.array(hdf.get(video_name + "/picks"))
    return user_summary, shot_bound, n_frames, positions


def read_and_process_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    max_index = max(map(int, data.keys()))
    result_array = [data.get(str(i), None) for i in range(max_index + 1)]

    for i in range(1, len(result_array)):
        if result_array[i] is None:
            result_array[i] = result_array[i - 1]

    return result_array


def process_scores(scores: np.ndarray, frame_interval: int, nframes: int):
    score_processed = np.zeros(nframes, dtype=np.float32)
    for i in range(nframes // 16):
        pos_left, pos_right = i * frame_interval, (i + 1) * frame_interval
        if i * frame_interval >= nframes:
            score_processed[pos_left:-1] = scores[i]
        else:
            score_processed[pos_left:pos_right] = scores[i]

    return score_processed


def get_tvsum_video_name_dict_from_matfile(mat_file_path: str):
    # load mat file from mat_file_path
    mat_file = h5py.File(mat_file_path, "r")
    tvsum50_video_name = mat_file['tvsum50']["video"][:]
    video_name_dict = {}
    for i, ref_array in enumerate(tvsum50_video_name):
        for ref in ref_array:
            name = mat_file[ref]
            str_values = ''.join(chr(val[0]) for val in name)
            video_name_dict[str_values] = f"video_{i+1}"
    # pprint.pprint(video_name_dict)
    return video_name_dict


def parse_args():
    parser = argparse.ArgumentParser()

    # Required argument
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--eval_method", type=str, required=True)

    # Optional arguments with defaults
    parser.add_argument("--frame_interval", type=int, default=16)

    return parser.parse_args()


def main(
    root_path: str,
    dataset_name: str,
    frame_interval: int,
    eval_method: str
):
    # Get paths needed.
    h5py_file_name = "eccv16_dataset_" + dataset_name.lower() + "_google_pool5.h5"

    dataset_dir = Path(root_path, dataset_name)
    scores_dir = Path(dataset_dir, "scores")
    splits_dir = Path(dataset_dir, "splits")
    output_dir = Path(dataset_dir, "output")
    h5py_file_path = Path(dataset_dir, h5py_file_name)
    output_file_path = Path(output_dir, "result.json")

    hdf = h5py.File(h5py_file_path, "r")

    # Create paths if dont exits
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get the video names
    file_names = listdir(scores_dir)

    # Get the video name dictionary
    if dataset_name == "TVSum":
        video_name_dict = get_tvsum_video_name_dict_from_matfile(
            mat_file_path=Path(dataset_dir, "ydata-tvsum50.mat")
        )
    else:
        video_name_dict = get_video_name_dict(hdf)

    # Initialize the results dictionary
    results = {}

    # Iterate over the videos
    for file_name in file_names:

        video_name = file_name.split(".")[0]
        video_index_name = video_name_dict[video_name]
        # Get the ground truth summary
        user_summary, shot_bound, n_frames, positions = get_gt(
            hdf=hdf, video_name=video_index_name
        )

        # Read the importance scores
        json_file_path = Path(scores_dir, video_name + ".json")
        scores = read_and_process_json(json_file_path)

        scores = process_scores(
            scores=scores, frame_interval=frame_interval, nframes=n_frames
        )

        # Compute the f_score
        f_score = compute_single_summary_score(
            scores=scores,
            shot_bound=shot_bound,
            n_frames=n_frames,
            positions=positions,
            user_summary=user_summary,
            eval_method=eval_method,
        )

        # Store the result
        results[video_name] = f_score

        # Print the result
        print(f"{video_name}: {f_score}")

    # Calculate the average f_score and save it
    avg_f_score = np.mean(list(results.values()))
    results["average_f_score"] = avg_f_score

    # Save the results
    with open(output_file_path, "w") as file:
        json.dump(results, file, indent=4)


# 示例用法
if __name__ == "__main__":
    args = parse_args()
    main(
        root_path=args.root_path,
        dataset_name=args.dataset_name,
        frame_interval=args.frame_interval,
        eval_method=args.eval_method
    )
