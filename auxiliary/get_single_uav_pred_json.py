#! 2025.2.2: H95
def get_single_uav_pred_json():
    """
        How to get the single uav predictioins?
        Step1: (Done) create new .pkl files
        Step2: (Done) replace the original .pkl file with new .pkl files
        Step3: (Done) use single-agent model (/perception/detr3d) for inference
        Step4: (Doing) merge the generated small files into a single large .json file
    """

    inference_output_path = "/mnt/storage/hjw/code/uav3d/collaborative_perception/bevfusion/data/uav3d/v1.0/h95"
    single_uav_pred_save_path = "/mnt/storage/hjw/code/uav3d/visualization/step4/single_uav_preds.json"
    filenum = len(os.listdir(inference_output_path))

    sample_data_path = '/mnt/storage/hjw/code/uav3d/collaborative_perception/bevfusion/data/uav3d/v1.0/v1.0-trainval/sample_data.json'
    with open(sample_data_path, 'r') as f:
        sample_data = json.load(f)

    calibrated_sensor_path = "/mnt/storage/hjw/code/uav3d/collaborative_perception/bevfusion/data/uav3d/v1.0/v1.0-trainval/calibrated_sensor.json"
    with open(calibrated_sensor_path, 'r') as f:
        calibrated_sensor_all = json.load(f)

    single_pred = {}
    num = 0
    for record in os.listdir(inference_output_path):
        # record: a7aeb2c3f13443b18c9dd01d77c2461e_id_4.yaml
        num += 1
        print(f"Doing {num}/{filenum}")
        sample_token = record.split('_')[0]
        uav_id = record[-6]

        full_path = os.path.join(inference_output_path, record)
        with open(full_path, 'r') as f:
            data = yaml.safe_load(f)
        boxes = np.array(data["boxes_3d"])[:, 0:7]   #! [300, 9] -> [300, 7]
        scores = np.array(data["scores_3d"])
        scores_reshape = scores[:, np.newaxis]
        boxes = np.hstack((boxes, scores_reshape)) #! [300, 8]
        boxes = boxes[scores>0.3] # detection threshold = 0.3

        #! ego2global
        selected_camera = [i for i in sample_data if i['sample_token']==sample_token and f"BOTTOM_id_{uav_id}" in i['filename']]
        assert len(selected_camera) == 1
        calibrated_sensor_token = selected_camera[0]["calibrated_sensor_token"]
        calibrated_sensor = [i for i in calibrated_sensor_all if i["token"] == calibrated_sensor_token]
        assert len(calibrated_sensor) == 1
        ego2global_translation = calibrated_sensor[0]["translation"] #! list, [-218.84188842773438, 123.0099868774414, 65.52840103149414]

        #! each uav's coordinate to uav_id_0's coordinate
        if uav_id == "1":
            ego2global_translation[0] -= 20
        elif uav_id == "2":
            ego2global_translation[1] -= 20
        elif uav_id == "3":
            ego2global_translation[1] += 20
        elif uav_id == "4":
            ego2global_translation[0] += 20
        else:
            pass

        #! from uav_id_0's coordinate to world coordiante
        for i in range(boxes.shape[0]):
            boxes[i, 0] += ego2global_translation[0]
            boxes[i, 1] += ego2global_translation[1]

        if sample_token not in single_pred.keys():
            single_pred[sample_token] = []
            single_pred[sample_token].append(boxes.tolist())
        else:
            single_pred[sample_token].append(boxes.tolist())
    
    with open(single_uav_pred_save_path, "w") as f:
        json.dump(single_pred, f)


if __name__ == "__main__":
  get_single_uav_pred_json()
