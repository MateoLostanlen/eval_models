import subprocess
import glob

source = "Data/DS-71c1fd51-v2/images/val"

weights = glob.glob("Data/cp/*")

print(len(weights))

for weight in weights:

    cmd = f"yolo predict model={weight} iou=0.01 conf=0.01 source={source} save=False save_txt save_conf project=Data/test_preds name={source.split('/')[-3]}_{weight.split('/')[2].split('.')[0]}"
    print(f"* Command:\n{cmd}")
    subprocess.call(cmd, shell=True)
