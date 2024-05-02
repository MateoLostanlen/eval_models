import glob
import fiftyone as fo
import os
import shutil
import numpy as np
import random
from datetime import datetime
from utils import xywh2xyxy, box_iou


def read_file(file):
    if os.path.isfile(file) and os.path.getsize(file) > 0:
        with open(file, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            return np.zeros((0, 5))
        boxes = np.array([l.split(" ")[1:] for l in lines]).astype("float")
        boxes[:, :4] = [xywh2xyxy(box) for box in boxes[:, :4]]
        return boxes
    else:
        return np.zeros((0, 5))


conf_th = 0.11


ds_name = "DS-71c1fd51-v2"
IMG_FOLDER = f"Data/{ds_name}/images/val"
gt_folder = f"Data/{ds_name}/labels/val"

imgs = glob.glob(f"{IMG_FOLDER}/*")
imgs.sort()

folder_name = "DS-71c1fd51-v2_yolov8s_exp_10"
pred_folder = f"Data/test_preds/{folder_name}/labels"
print(folder_name)


error = []

tn_imgs, tp_imgs, fp_imgs, fn_imgs = [], [], [], []
for file in imgs:
    filename = os.path.basename(file).split(".")[0]
    gt_file = os.path.join(gt_folder, f"{filename}.txt")
    pred_file = os.path.join(pred_folder, f"{filename}.txt")

    gt_boxes = read_file(gt_file)
    pred_boxes = read_file(pred_file)
    pred_boxes = pred_boxes[pred_boxes[:, -1] > conf_th]

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        tn_imgs.append(file)

    elif len(pred_boxes) == 0 and len(gt_boxes) > 0:
        fn_imgs.append(file)

    elif len(pred_boxes) > 0 and len(gt_boxes) == 0:
        fp_imgs.append(file)

    else:
        if box_iou(pred_boxes[:, :4], gt_boxes).max() > 0:
            tp_imgs.append(file)

        else:
            fn_imgs.append(file)
            fp_imgs.append(file)


# Remove existing
ds_list = fo.list_datasets()
for ds in ds_list:
    dataset = fo.load_dataset(ds)
    dataset.delete()


label_name = "smoke"

for imgs, ds_name in [
    (tn_imgs, "tn"),
    (tp_imgs, "tp"),
    (fp_imgs, "fp"),
    (fn_imgs, "fn"),
]:
    samples = []
    for img_file in imgs:
        _, file = os.path.split(img_file)
        sample = fo.Sample(filepath=img_file)
        filename = os.path.basename(file).split(".")[0]
        gt_file = os.path.join(gt_folder, f"{filename}.txt")
        pred_file = os.path.join(pred_folder, f"{filename}.txt")

        if os.path.isfile(gt_file):
            with open(gt_file) as f:
                lines = f.readlines()

            detections = []

            for line in lines:
                if len(line) > 0:
                    bounding_box = [float(li) for li in line.split(" ")[1:5]]
                    bounding_box[0] -= bounding_box[2] / 2
                    bounding_box[1] -= bounding_box[3] / 2

                    detections.append(
                        fo.Detection(label="smoke", bounding_box=bounding_box)
                    )

            # Store detections in a field name of your choice
            sample["gt"] = fo.Detections(detections=detections)

        if os.path.isfile(pred_file):
            with open(pred_file) as f:
                lines = f.readlines()

            detections = []

            for line in lines:
                if len(line) > 0:
                    score = float(line.split(" ")[-1])
                    if score > conf_th:
                        bounding_box = [float(li) for li in line.split(" ")[1:5]]
                        bounding_box[0] -= bounding_box[2] / 2
                        bounding_box[1] -= bounding_box[3] / 2

                        detections.append(
                            fo.Detection(
                                label="smoke",
                                bounding_box=bounding_box,
                                confidence=score,
                            )
                        )

            # Store detections in a field name of your choice
            sample[folder_name] = fo.Detections(detections=detections)

        samples.append(sample)

    dataset = fo.Dataset(ds_name)
    dataset.add_samples(samples)
    dataset.persistent = True

if __name__ == "__main__":
    # Ensures that the App processes are safely launched on Windows
    session = fo.launch_app(dataset)
    session.wait()


imgs = glob.glob("/home/mateo/Desktop/pyronear_ds_03_2024_sub4/images/val/*.jpg")
imgs.sort()

gt_folder = "/home/mateo/Desktop/pyronear_ds_03_2024_sub4/labels/val"

pred_folders = glob.glob(f"test_preds/**/labels")
pred_folders.sort()
pred_folders = [f for f in pred_folders if "pyronear_ds_03_2024_sub3" in f]
len(pred_folders)

ds_list = fo.list_datasets()
for ds in ds_list:
    dataset = fo.load_dataset(ds)
    dataset.delete()

samples = []

label_name = "smoke"
ds_name = "new_dl"

for img_file in imgs:
    if "pyro" in os.path.basename(img_file):

        _, file = os.path.split(img_file)
        sample = fo.Sample(filepath=img_file)

        gt_file = img_file.replace("images", "labels").replace(".jpg", ".txt")

        if os.path.isfile(gt_file):

            with open(gt_file) as f:
                lines = f.readlines()

            # Convert detections to FiftyOne format
            detections = []

            for line in lines:
                # Bounding box coordinates should be relative values
                # in [0, 1] in the following format:
                # [top-left-x, top-left-y, width, height]
                if len(line) > 0:
                    score = float(line.split(" ")[-1])

                    bounding_box = [float(li) for li in line.split(" ")[1:5]]
                    bounding_box[0] -= bounding_box[2] / 2
                    bounding_box[1] -= bounding_box[3] / 2

                    detections.append(
                        fo.Detection(label="smoke", bounding_box=bounding_box)
                    )

            # Store detections in a field name of your choice
            sample["gt"] = fo.Detections(detections=detections)

        for label_folder in pred_folders:

            label_name = os.path.normpath(label_folder.replace(".", "")).split(os.sep)[
                -2
            ]
            label_name = label_name.replace("-", "")
            label_file = os.path.join(label_folder, file.replace(".jpg", ".txt"))

            if os.path.isfile(label_file):

                with open(label_file) as f:
                    lines = f.readlines()

                # Convert detections to FiftyOne format
                detections = []

                for line in lines:
                    # Bounding box coordinates should be relative values
                    # in [0, 1] in the following format:
                    # [top-left-x, top-left-y, width, height]
                    if len(line) > 0:
                        score = float(line.split(" ")[-1])
                        if score > 0.15:

                            bounding_box = [float(li) for li in line.split(" ")[1:5]]
                            bounding_box[0] -= bounding_box[2] / 2
                            bounding_box[1] -= bounding_box[3] / 2

                            detections.append(
                                fo.Detection(
                                    label="smoke",
                                    bounding_box=bounding_box,
                                    confidence=score,
                                )
                            )

                # Store detections in a field name of your choice
                sample[label_name] = fo.Detections(detections=detections)

        samples.append(sample)

dataset = fo.Dataset("sub3")
dataset.add_samples(samples)
dataset.persistent = True


if __name__ == "__main__":
    # Ensures that the App processes are safely launched on Windows
    session = fo.launch_app(dataset)
    session.wait()
