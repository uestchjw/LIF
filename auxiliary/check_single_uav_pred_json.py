import os
import json
import random
import matplotlib.pyplot as plt


single_uav_prediction_path = "/mnt/storage/hjw/code/uav3d/perception/bevfusion/h95_2324.json"
with open(single_uav_prediction_path, "r") as f:
    single_uav_preds = json.load(f) # dict
    # len(single_uav_predictions.keys()): 17000 = 14000(train samples) + 3000(val samples)
    # Each key represents a sample_token

# Randomly select a sample for visualization
random_sample_token = random.choice(list(single_uav_preds.keys()))
random_single_uav_pred = single_uav_preds[random_sample_token] # list, len=5


sample_annotation_path = "/mnt/storage/hjw/code/uav3d/collaborative_perception/bevfusion/data/uav3d/v1.0/v1.0-trainval/sample_annotation.json"
with open(sample_annotation_path, "r") as f:
    sample_annotation_all = json.load(f)
gt_boxes = [box for box in sample_annotation_all if box["sample_token"] == random_sample_token]
gt_x = [box["translation"][0] for box in gt_boxes]
gt_y = [-box["translation"][1] for box in gt_boxes] # Note that -

plt.figure()
plt.scatter(gt_x, gt_y, c="g", label = "gt")


for i in range(len(random_single_uav_pred)):
    selected_uav_pred = random_single_uav_pred[i]
    if selected_uav_pred == []:
        continue
    uav_x = [box[0] for box in selected_uav_pred]
    uav_y = [-box[1] for box in selected_uav_pred] # Note that -
    plt.scatter(uav_x, uav_y, s=4, label = f"uav_{i}")

plt.xlabel("Forward of UAV")
plt.ylabel("Left of UAV")
plt.legend()
plt.savefig("Visualization_of_single_uav_pred.png")
