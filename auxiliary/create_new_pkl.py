# 根据已有的.pkl文件, 创建同一sample下以其他UAV作为ego的.pkl文件: 5倍于之前的数量
# todo: 阉割版, 不知道怎么算角度, 先设为0, 只得到box的中心位置
def create_new_pkl():
    val_pkl_path = r"G:\Datasets\uav3d\v1.0\v1.0\uav3d_infos_train.pkl"
    # val_pkl_path = r"G:\Datasets\uav3d\v1.0\v1.0\uav3d_infos_val.pkl"
    with open(val_pkl_path, 'rb') as f:
        val_pkl = pickle.load(f)
    print(f"Val part origin number: {len(val_pkl['infos'])}") # Val part origin number: 3000
    # print(val_pkl.keys()) # ['infos', 'metadata']
    
    new_val_pkl = {}
    new_val_pkl["metadata"] = val_pkl["metadata"]
    new_val_pkl["infos"] = []
    
    sample_annotation_path = r"G:\Datasets\uav3d\v1.0\v1.0\v1.0-trainval\sample_annotation.json"
    with open(sample_annotation_path, 'r') as f:
        sample_annotation_all = json.load(f)
    
    for i in range(len(val_pkl["infos"])):
        print(f"Doing {i+1}/{len(val_pkl['infos'])}")
        sample = val_pkl["infos"][i]
        sample_annotation = [mm for mm in sample_annotation_all if mm["sample_token"] == sample["token"]]
        #! sample.keys(): ['lidar_path', 'token', 'sweeps', 'cams', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag', 'location']
        # print(sample["lidar_path"]) # 空
        # print(sample["token"]) # dbc9b41e6b01466da29d7db5cd8ef9e4
        # print(sample["sweeps"]) # []
        # print(sample["cams"].keys()) # ['CAMERA_FRONT_id_0', 'CAMERA_BACK_id_0', 'CAMERA_LEFT_id_0', 'CAMERA_RIGHT_id_0', 'CAMERA_BOTTOM_id_0', 'CAMERA_FRONT_id_1', 'CAMERA_BACK_id_1', 'CAMERA_LEFT_id_1', 'CAMERA_RIGHT_id_1', 'CAMERA_BOTTOM_id_1', 'CAMERA_FRONT_id_2', 'CAMERA_BACK_id_2', 'CAMERA_LEFT_id_2', 'CAMERA_RIGHT_id_2', 'CAMERA_BOTTOM_id_2', 'CAMERA_FRONT_id_3', 'CAMERA_BACK_id_3', 'CAMERA_LEFT_id_3', 'CAMERA_RIGHT_id_3', 'CAMERA_BOTTOM_id_3', 'CAMERA_FRONT_id_4', 'CAMERA_BACK_id_4', 'CAMERA_LEFT_id_4', 'CAMERA_RIGHT_id_4', 'CAMERA_BOTTOM_id_4']
        # print(sample["lidar2ego_translation"]) # [0.0, 0.0, 0.0]
        # print(sample["lidar2ego_rotation"]) # [1.0, 0.0, 0.0, 0.0]
        # print(sample["ego2global_translation"]) # [-68.47067993164063, -159.22780212402344, 0]
        # # translation和 对应id的bottom的translation相同: calibrated_sensor.json中
        # print(sample["ego2global_rotation"]) # [1.0, 0.0, 0.0, 0.0]
        # print(sample["timestamp"]) # 1701498620885503488
        # # 重头戏: "gt_boxes"
        # print(sample["gt_boxes"].shape) # numpy.ndarray, [N, 7]
        # print(sample["gt_boxes"])
        
        # print(sample["gt_names"])      # numpy.ndarray: (N,) ['car' 'car' ... 'car']
        # print(sample["gt_velocity"])   # numpy.ndarray, [N, 2], 每个都是[nan nan]
        # print(sample["num_lidar_pts"]) # numpy.ndarray: 全是-1   [-1 -1 -1 ... -1]
        # print(sample["num_radar_pts"]) # numpy.ndarray: 全是-1
        # print(sample["valid_flag"])    # numpy.ndarray: 全是True
        # print(sample["location"])      # Atlanta

        for j in range(5):
            new_sample = {}
            new_sample["lidar_path"] = sample["lidar_path"]
            new_sample["token"] = sample["token"]
            new_sample["sweeps"] = sample["sweeps"]
            
            # cams要调换顺序, 按照id_j在第一个, 其余顺序不变
            new_sample["cams"] = {}
            for key in sample["cams"].keys():
                if key.split("_")[-1] == f"{j}":
                    new_sample["cams"][key] = sample["cams"][key]
            for key in sample["cams"].keys():
                if key.split("_")[-1] != f"{j}":
                    new_sample["cams"][key] = sample["cams"][key]

            new_sample["lidar2ego_translation"] = sample["lidar2ego_translation"]
            new_sample["lidar2ego_rotation"] = sample["lidar2ego_rotation"]

            new_sample["ego2global_translation"] = new_sample["cams"][f"CAMERA_BOTTOM_id_{j}"]["ego2global_translation"]
            new_sample["ego2global_rotation"] = sample["ego2global_rotation"]
            new_sample["timestamp"] = sample["timestamp"]

            boxes = []
            for kk in range(len(sample_annotation)):
                # box = sample_annotation[kk]["translation"] + sample_annotation[kk]["size"]
                #! 阉割版: yaw = 0
                box = sample_annotation[kk]["translation"] + sample_annotation[kk]["size"] + [0.0]
                box[0] = box[0] - new_sample["ego2global_translation"][0]
                box[1] = box[1] - new_sample["ego2global_translation"][1]
                # 判断在[-102.4, +102.4]范围内
                if box[0] > -102.4 and box[0] < 102.4 and box[1] > -102.4 and box[1] < 102.4:
                    boxes.append(box)
            boxes = np.array(boxes).reshape(-1, 7)
            new_sample["gt_boxes"] = boxes
            num_boxes = boxes.shape[0]
            new_sample["gt_names"] = np.array(['car']*num_boxes)
            new_sample["gt_velocity"] = np.full((num_boxes, 2), np.nan)
            new_sample["num_lidar_pts"] = np.full(num_boxes, -1)
            new_sample["num_radar_pts"] = np.full(num_boxes, -1)
            new_sample["valid_flag"] = np.full(num_boxes, True)
            new_sample["location"] = sample["location"]

            new_val_pkl["infos"].append(new_sample)

    # save .pkl
    with open(r"G:\Datasets\uav3d\v1.0\v1.0\uav3d_infos_train_5ego.pkl", 'wb') as f:
        pickle.dump(new_val_pkl, f)
    # with open(r"G:\Datasets\uav3d\v1.0\v1.0\uav3d_infos_val_5ego.pkl", 'wb') as f:
    #     pickle.dump(new_val_pkl, f)

if __name__ == "__main__":
    create_new_pkl()
