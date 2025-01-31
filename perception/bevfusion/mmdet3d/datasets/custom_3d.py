import tempfile
from os import path as osp

import mmcv
from mmcv.parallel import DataContainer #! H95
import numpy as np
import torch #! H95
from ..core.bbox import LiDARInstance3DBoxes #! H95
from copy import deepcopy #! H95
from torch.utils.data import Dataset

from mmdet.datasets import DATASETS

from ..core.bbox import get_box_type
from .pipelines import Compose
from .utils import extract_result_dict

import time

@DATASETS.register_module()
class Custom3DDataset(Dataset):
    """Customized 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.

    Args:
        dataset_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(
        self,
        dataset_root,
        ann_file,
        pipeline=None,
        classes=None,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        self.epoch = -1
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self, "pipeline"):
            for transform in self.pipeline.transforms:
                if hasattr(transform, "set_epoch"):
                    transform.set_epoch(epoch)
        
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        return mmcv.load(ann_file)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - lidar_path (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info["point_cloud"]["lidar_idx"]
        lidar_path = osp.join(self.dataset_root, info["pts_path"])

        input_dict = dict(
            lidar_path=lidar_path, sample_idx=sample_idx, file_name=lidar_path
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos
            if self.filter_empty_gt and ~(annos["gt_labels_3d"] != -1).any():
                return None
        return input_dict

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # print(input_dict["token"])
        #! input_dict.keys(): ['token', 'sample_idx', 'lidar_path', 'sweeps', 'timestamp', 'location', 'ego2global', 'lidar2ego', 'image_paths', 'lidar2camera', 'lidar2image', 'camera2ego', 'camera_intrinsics', 'camera2lidar', 'ann_info']
        #!     input_dict["image_paths"]: len = 5

        #! H95: 添加单体感知结果后
        #! inpupt_dict.keys(): ['token', 'sample_idx', 'lidar_path', 'sweeps', 'timestamp', 'location', 'single_pred', 'ego2global', 'lidar2ego', 'image_paths', 'lidar2camera', 'lidar2image', 'camera2ego', 'camera_intrinsics', 'camera2lidar', 'ann_info']
        if input_dict is None:
            return None
        #! 
        # if input_dict["token"] == "933662bda2c945489372ae5a4a7abd57":
            # print(input_dict["single_pred"])
            #! e.g.: input_dict["single_pred"] 和 标准box能对应的上
            # [9.70540618, 79.022331, -0.762298, 4.9053378, 2.05179667, 1.534105, -1.5734708, 0.97925251]
            # print(input_dict["ann_info"]["gt_bboxes_3d"])
            #! input_dict["anno_info"]["gt_bboxes_3d"]: LiDARInstance3DBoxes
            #! 和nuscenes_dataset.py返回的结果一致
            # tensor([[-7.4315e+01,  7.8894e+01, -5.5465e-03,  4.1928e+00,  1.8162e+00,  1.4738e+00,  -1.5712e+00,  0.0000e+00,  0.0000e+00],                                                                                          
            #         [ 9.8626e+00,  7.8933e+01,  7.3985e-02,  4.6110e+00,  2.2417e+00,  1.6673e+00,  -1.5707e+00,  0.0000e+00,  0.0000e+00],                                                                                          
            #         [-7.6892e+01,  8.2391e+01, -2.3888e-02,  4.9742e+00,  2.0384e+00,  1.5543e+00,  -1.5711e+00,  0.0000e+00,  0.0000e+00],                                                                                          
            #         [ 9.6960e+00,  6.8412e+01, -5.8792e-03,  4.7175e+00,  1.8948e+00,  1.3009e+00,  -1.5711e+00,  0.0000e+00,  0.0000e+00],                                                                                          
            #         [-7.7576e+01, -3.2912e+01, -1.1668e-02,  4.1812e+00,  1.9941e+00,  1.3853e+00,  -1.5749e+00,  0.0000e+00,  0.0000e+00]])) 
            # pass
            # raise Exception

        #! 既然顺序有关系, 那我就全部推倒重来: 把single_pred放到gt_bboxes_3d里面, 一起处理再拿出来
        single_pred_origin = input_dict["single_pred"] # list, len一定等于5
        gt_boxes_origin = input_dict["ann_info"]["gt_bboxes_3d"].tensor
        # gt_box的数量: 分离的时候要用
        gt_len = gt_boxes_origin.shape[0]
        # assert gt_len > 0, f"{input_dict['token']}: gt_len <= 0"
        if gt_len <= 0:
            print(f"{input_dict['token']}: gt_len <= 0")
            raise Exception


        z_mean = gt_boxes_origin[:, 2].mean()

        # len_list = [] # 记录每个uav的长度
        pred_list = []
        conf_list = []
        for i in range(len(single_pred_origin)):
            box_list = torch.tensor(single_pred_origin[i]) # [N, 8], 最后一维是condfidence
            #! 20250125, 01:25: 加判断条件
            if box_list.shape[0] <= 0:
                continue
            # if box_list.shape[0] <= 0:
            #     print("Stop here")
            #     raise Exception
            confidence = box_list[:, -1].reshape(-1, 1) # [N, 1]
            conf_padding = torch.full((confidence.shape[0], 1), i) # [N, 1]的指示uav tensor
            confidence = torch.cat([confidence, conf_padding], dim=-1) # [N, 2]

            padding = torch.zeros(box_list.shape[0], 2)
            pred_uav = torch.cat([box_list[:, :-1], padding], dim=-1) # [N,7] + [N,2] = [N,9]
            pred_list.append(pred_uav)
            # len_list.append(box_list.shape[0]) # 记录该uav的record数量
            conf_list.append(confidence)
        
        #! 这里需要判断一下, 如果5个UAV都是空, 那么直接cat会报错导致卡死
        # print(f"{input_dict['token']}: len(pred_list) = {len(pred_list)}")
        if len(pred_list) == 0:
            # 进行主流的pipeline
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            example["single_pred"] = DataContainer(-1)
            if self.filter_empty_gt and (example is None or ~(example["gt_labels_3d"]._data != -1).any()):
                return None
            return example
        
        single_pred = torch.cat(pred_list, dim=0) # [5N, 9]
        single_pred[:, 2] = z_mean

        # 加一个分割record, 用来区分gt_box和single_pred_box
        gt_boxes_origin = torch.cat([gt_boxes_origin, torch.zeros((1, 9))], dim=0) # [N+1, 9]
        # print(f"gt_boxes_origin")
        gt_boxes = torch.cat([gt_boxes_origin, single_pred], dim=0) # [N + 5N, 9]

        #! 变换前的box数量
        box_num_origin = gt_boxes.shape[0]

        # 替换
        input_dict["ann_info"]["gt_bboxes_3d"].tensor = gt_boxes
        input_dict["ann_info"]["gt_labels_3d"] = np.array([0] * gt_boxes.shape[0])
        input_dict["ann_info"]["gt_names"] = np.array(["car"] * gt_boxes.shape[0])

        # 进行主流的pipeline
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        all_boxes = example["gt_bboxes_3d"].data.tensor # [N + 5N, 9]
        divider_mask = torch.all(all_boxes[:, :6]==0, dim=1)
        divider_indice = torch.where(divider_mask)[0] # tensor([5])
        assert len(divider_indice) == 1
        box_num_after = all_boxes.shape[0]

        #! 分割record居然很随机: yaw不知道咋算的
        # [ 0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -3.1060,  0.0000,  -0.0000]
        # [ 0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0142,  0.0000,  -0.0000]
        # [ 0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.1404,  0.0000,  -0.0000]
        single_pred_after = all_boxes[divider_indice[0]+1:, 0:7] # [5N, 9] -> [5N, 7], 舍掉最后2个为0的value

        # 加入conf和指示uav的tensor
        ttmp = torch.cat(conf_list, dim=0) # [5N, 2]
        #! Noting!
        # assert single_pred_after.shape[0] > 0, f"{input_dict['token']}: single_pred_after.shape[0]"

        if ttmp.shape[0] != single_pred_after.shape[0]:
            single_pred_after = torch.cat([single_pred_after, ttmp[0:single_pred_after.shape[0], :]], dim=-1) # [5N, 9]
        else:
            single_pred_after = torch.cat([single_pred_after, ttmp], dim=-1) # [5N, 9]
        example["single_pred"] = single_pred_after

        # 替换回来
        #! example.keys(): ['img', 'gt_bboxes_3d', 'gt_labels_3d', 'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix', 'metas']
        example["gt_bboxes_3d"].data.tensor = all_boxes[0:divider_indice[0], :]
        example["gt_labels_3d"] = DataContainer(torch.tensor([0] * divider_indice[0]))
        # print(example["gt_labels_3d"])
        # DataContainer(tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        
        # if input_dict["token"] == "933662bda2c945489372ae5a4a7abd57":
        #     print(example["gt_bboxes_3d"].data.tensor)
        #     print(f"single_pred_after={single_pred_after}")
        #     raise Exception

        if self.filter_empty_gt and (
            example is None or ~(example["gt_labels_3d"]._data != -1).any()
        ):
            return None

        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)

                #! 既然顺序有关系, 那我就全部推倒重来: 把single_pred放到gt_bboxes_3d里面, 一起处理再拿出来
        single_pred_origin = input_dict["single_pred"] # list, len一定等于5
        gt_boxes_origin = input_dict["ann_info"]["gt_bboxes_3d"].tensor
        # gt_box的数量: 分离的时候要用
        gt_len = gt_boxes_origin.shape[0]
        # assert gt_len > 0, f"{input_dict['token']}: gt_len <= 0"
        if gt_len <= 0:
            print(f"{input_dict['token']}: gt_len <= 0")
            raise Exception


        z_mean = gt_boxes_origin[:, 2].mean()

        # len_list = [] # 记录每个uav的长度
        pred_list = []
        conf_list = []
        for i in range(len(single_pred_origin)):
            box_list = torch.tensor(single_pred_origin[i]) # [N, 8], 最后一维是condfidence
            #! 20250125, 01:25: 加判断条件
            if box_list.shape[0] <= 0:
                continue
            # if box_list.shape[0] <= 0:
            #     print("Stop here")
            #     raise Exception
            confidence = box_list[:, -1].reshape(-1, 1) # [N, 1]
            conf_padding = torch.full((confidence.shape[0], 1), i) # [N, 1]的指示uav tensor
            confidence = torch.cat([confidence, conf_padding], dim=-1) # [N, 2]

            padding = torch.zeros(box_list.shape[0], 2)
            pred_uav = torch.cat([box_list[:, :-1], padding], dim=-1) # [N,7] + [N,2] = [N,9]
            pred_list.append(pred_uav)
            # len_list.append(box_list.shape[0]) # 记录该uav的record数量
            conf_list.append(confidence)
        
        #! 这里需要判断一下, 如果5个UAV都是空, 那么直接cat会报错导致卡死
        # print(f"{input_dict['token']}: len(pred_list) = {len(pred_list)}")
        if len(pred_list) == 0:
            # 进行主流的pipeline
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            example["single_pred"] = DataContainer(-1)
            if self.filter_empty_gt and (example is None or ~(example["gt_labels_3d"]._data != -1).any()):
                return None
            return example
        
        single_pred = torch.cat(pred_list, dim=0) # [5N, 9]
        single_pred[:, 2] = z_mean

        # 加一个分割record, 用来区分gt_box和single_pred_box
        gt_boxes_origin = torch.cat([gt_boxes_origin, torch.zeros((1, 9))], dim=0) # [N+1, 9]
        # print(f"gt_boxes_origin")
        gt_boxes = torch.cat([gt_boxes_origin, single_pred], dim=0) # [N + 5N, 9]

        #! 变换前的box数量
        box_num_origin = gt_boxes.shape[0]

        # 替换
        input_dict["ann_info"]["gt_bboxes_3d"].tensor = gt_boxes
        input_dict["ann_info"]["gt_labels_3d"] = np.array([0] * gt_boxes.shape[0])
        input_dict["ann_info"]["gt_names"] = np.array(["car"] * gt_boxes.shape[0])

        # 进行主流的pipeline
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        all_boxes = example["gt_bboxes_3d"].data.tensor # [N + 5N, 9]
        divider_mask = torch.all(all_boxes[:, :6]==0, dim=1)
        divider_indice = torch.where(divider_mask)[0] # tensor([5])
        assert len(divider_indice) == 1
        box_num_after = all_boxes.shape[0]

        #! 分割record居然很随机: yaw不知道咋算的
        # [ 0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -3.1060,  0.0000,  -0.0000]
        # [ 0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0142,  0.0000,  -0.0000]
        # [ 0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.1404,  0.0000,  -0.0000]
        single_pred_after = all_boxes[divider_indice[0]+1:, 0:7] # [5N, 9] -> [5N, 7], 舍掉最后2个为0的value

        # 加入conf和指示uav的tensor
        ttmp = torch.cat(conf_list, dim=0) # [5N, 2]
        #! Noting!
        # assert single_pred_after.shape[0] > 0, f"{input_dict['token']}: single_pred_after.shape[0]"

        if ttmp.shape[0] != single_pred_after.shape[0]:
            single_pred_after = torch.cat([single_pred_after, ttmp[0:single_pred_after.shape[0], :]], dim=-1) # [5N, 9]
        else:
            single_pred_after = torch.cat([single_pred_after, ttmp], dim=-1) # [5N, 9]
        example["single_pred"] = single_pred_after

        # 替换回来
        #! example.keys(): ['img', 'gt_bboxes_3d', 'gt_labels_3d', 'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix', 'metas']
        example["gt_bboxes_3d"].data.tensor = all_boxes[0:divider_indice[0], :]
        example["gt_labels_3d"] = DataContainer(torch.tensor([0] * divider_indice[0]))



        #! 23, 24: 参考上面, 对single_pred也进行坐标变换  
        # single_pred_origin = h95_dict["single_pred"]
        # first_dim_values = torch.arange(single_pred_origin.shape[0]).view(-1,1,1).expand(-1, single_pred_origin.shape[1], 1) # [5, N, 1]
        # h95_tmp = torch.cat([single_pred_origin, first_dim_values], dim = -1) # (5, N, 9)
        # h95_tmp = h95_tmp.reshape(-1, 9) # [5, N, 9] -> [5N, 9]
        # h95_dict["ann_info"]["gt_bboxes_3d"] = LiDARInstance3DBoxes(h95_tmp, box_dim=h95_tmp.shape[-1], origin=(0.5, 0.5, 0))
        # h95_dict["ann_info"]["gt_labels_3d"] = np.array([0] * h95_tmp.shape[0])
        # h95_dict["ann_info"]["gt_names"] = np.array(["car"] * h95_tmp.shape[0])
        # h95_example = self.pipeline(h95_dict)

        # example["single_pred"] = DataContainer(h95_example["gt_bboxes_3d"].data.tensor)



        # example["single_pred"] = DataContainer(input_dict["single_pred"])
        # example["single_pred"] = DataContainer(torch.tensor(input_dict["single_pred"], requires_grad=False))
        #! example: dict, keys() = ['img', 'gt_bboxes_3d', 'gt_labels_3d', 'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix', 'metas', 'single_pred']
        #!     example["single_pred"]: <class 'mmcv.parallel.data_container.DataContainer'>
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        return class_names

    def format_results(self, outputs, pklfile_prefix=None, submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results, \
                tmp_dir is the temporal directory created for saving json \
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, "results")
            out = f"{pklfile_prefix}.pkl"
        mmcv.dump(outputs, out)
        return outputs, tmp_dir

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, "data loading pipeline is not provided"
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)

            if data is None:
                idx = self._rand_another(idx)
                continue

            data['points'] = 0
            data['gt_masks_bev'] = 0

            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
