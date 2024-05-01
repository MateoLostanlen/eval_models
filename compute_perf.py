import glob
import os
from utils import xywh2xyxy, box_iou
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Evaluate predictions
def evaluate_predictions(pred_folder, gt_folder, conf_th=0.1, cat=None):
    nb_fp, nb_tp, nb_fn = 0, 0, 0

    # Create a list of all image filenames (without the file extension)
    gt_filenames = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(gt_folder, "*.txt"))
    ]
    pred_filenames = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(pred_folder, "*.txt"))
    ]

    # Evaluate each image based on the presence of ground truth and prediction files
    all_filenames = set(gt_filenames + pred_filenames)
    if cat is not None:
        all_filenames = [f for f in all_filenames if cat == f.split("_")[0].lower()]
    for filename in all_filenames:
        gt_file = os.path.join(gt_folder, f"{filename}.txt")
        pred_file = os.path.join(pred_folder, f"{filename}.txt")

        gt_boxes = []
        # Read ground truth boxes if file exists
        if os.path.isfile(gt_file) and os.path.getsize(gt_file) > 0:
            with open(gt_file, "r") as f:
                gt_boxes = [
                    xywh2xyxy(np.array(line.strip().split(" ")[1:5]).astype(float))
                    for line in f.readlines()
                ]

        gt_matches = np.zeros(len(gt_boxes), dtype=bool)

        # Read prediction boxes if file exists
        if os.path.isfile(pred_file) and os.path.getsize(pred_file) > 0:
            with open(pred_file, "r") as f:
                pred_boxes = [line.strip().split(" ") for line in f.readlines()]

            for pred_box in pred_boxes:
                try:
                    _, x, y, w, h, conf = map(float, pred_box)
                except:
                    print(pred_file)
                if conf < conf_th:
                    continue
                pred_box = xywh2xyxy(np.array([x, y, w, h]))

                if gt_boxes:
                    matches = [box_iou(pred_box, gt_box) > 0.1 for gt_box in gt_boxes]
                    if any(matches):
                        nb_tp += 1
                        gt_matches = gt_matches | matches
                    else:
                        nb_fp += 1
                else:
                    nb_fp += 1

        if gt_boxes:
            nb_fn += len(gt_boxes) - np.sum(gt_matches)

    precision = nb_tp / (nb_tp + nb_fp) if (nb_tp + nb_fp) > 0 else 0
    recall = nb_tp / (nb_tp + nb_fn) if (nb_tp + nb_fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {"precision": precision, "recall": recall, "f1_score": f1_score}


def find_best_conf_threshold(pred_folder, gt_folder, conf_thres_range, cat=None):
    best_conf_thres = 0
    best_f1_score = 0
    best_precision = 0
    best_recall = 0

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_thres, cat)
        if results["f1_score"] > best_f1_score:
            best_conf_thres = conf_thres
            best_f1_score = results["f1_score"]
            best_precision = results["precision"]
            best_recall = results["recall"]

    return best_conf_thres, best_f1_score, best_precision, best_recall


def evaluate_multiple_pred_folders(pred_folders, gt_folder, conf_thres_range, cat=None):
    # Initialize a DataFrame to store the results
    results_df = pd.DataFrame(
        columns=[
            "Prediction Folder",
            "Best Threshold",
            "Best F1 Score",
            "Precision",
            "Recall",
        ]
    )

    for pred_folder in pred_folders:
        best_conf_thres, best_f1_score, best_precision, best_recall = (
            find_best_conf_threshold(pred_folder, gt_folder, conf_thres_range, cat)
        )

        # Use loc to append data to the DataFrame to avoid potential issues
        results_df.loc[len(results_df.index)] = [
            pred_folder.split("/")[2],
            best_conf_thres,
            best_f1_score,
            best_precision,
            best_recall,
        ]

    return results_df


def find_best_conf_threshold_and_plot(
    pred_folder, gt_folder, conf_thres_range, plot=True
):
    f1_scores, precisions, recalls = [], [], []

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_thres)
        f1_scores.append(results["f1_score"])
        precisions.append(results["precision"])
        recalls.append(results["recall"])

    # Find the best confidence threshold
    best_idx = np.argmax(f1_scores)
    best_conf_thres = conf_thres_range[best_idx]
    best_f1_score = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    if plot:

        # Plotting the metrics
        plt.figure(figsize=(10, 6))
        plt.plot(
            conf_thres_range, f1_scores, label="F1 Score", color="blue", marker="o"
        )
        plt.plot(
            conf_thres_range,
            precisions,
            label="Precision",
            color="green",
            linestyle="--",
        )
        plt.plot(conf_thres_range, recalls, label="Recall", color="red", linestyle="-.")

        # Highlight the best configuration
        plt.scatter(
            best_conf_thres, best_f1_score, color="blue", s=100, edgecolor="k", zorder=5
        )
        plt.scatter(
            best_conf_thres,
            best_precision,
            color="green",
            s=100,
            edgecolor="k",
            zorder=5,
        )
        plt.scatter(
            best_conf_thres, best_recall, color="red", s=100, edgecolor="k", zorder=5
        )

        plt.text(
            best_conf_thres,
            best_f1_score,
            f" Best F1: {best_f1_score:.2f}\n Precision: {best_precision:.2f}\n Recall: {best_recall:.2f}",
            fontsize=9,
            verticalalignment="bottom",
        )

        plt.title("F1 Score, Precision, and Recall vs. Confidence Threshold")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    return best_conf_thres, best_f1_score, best_precision, best_recall
