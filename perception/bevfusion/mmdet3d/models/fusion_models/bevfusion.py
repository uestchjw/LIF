from typing import Any, Dict
import json
import yaml
import pickle
import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
import time #! H95

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        #! H95: box指示tensor，加到对应位置上去
        # self.box_tensor = torch.rand(256).cuda()
        # self.box_tensor = nn.Parameter(torch.rand(256, 1, 1).cuda()) #! 07, 08
        # self.box_MLP = torch.nn.Linear(7,32) #! 11, 12
        # self.box_tensor = nn.Parameter(torch.rand(256).cuda()) #! 15, 16
        
        #! H95: 19, 20
        # self.single_pred_path = "./data/uav3d/v1.0/h95_sample_sum/single_pred.json"
        # with open(self.single_pred_path) as f:
        #     self.single_pred_all = json.load(f)
        # self.sample_data_path = "./data/uav3d/v1.0/v1.0-trainval/sample_data.json"
        # with open(self.sample_data_path, 'r') as f:
        #     self.sample_data_all = json.load(f)
        # self.calibrated_sensor_path = "./data/uav3d/v1.0/v1.0-trainval/calibrated_sensor.json"
        # with open(self.calibrated_sensor_path, 'r') as f:
        #     self.calibrated_sensor_all = json.load(f)

        
        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W) #! H95: [5, 3, 448, 800]
        #! H95: <class 'mmdet.models.backbones.resnet.ResNet'>
        x = self.encoders["camera"]["backbone"](x) #! H95: tuple 

        #! H95: <class 'mmdet3d.models.necks.second.SECONDFPN'>
        x = self.encoders["camera"]["neck"](x) #! H95: list
        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W) #! H95: [1, 5, 512, 56, 100]
        #! <class 'mmdet3d.models.vtransforms.lss.LSSTransform'>
        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x #! H95: [1, 256, 256, 256]

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        single_pred,  #! H95
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        points=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            # print(f"There are {len(gt_bboxes_3d[0])} gt_bboxes_3d")
            # x = [ box[0] for box in gt_bboxes_3d[0] ]
            # y = [ box[1] for box in gt_bboxes_3d[0] ]
            # print(f"max x: {max(x)}, min x: {min(x)}")
            # print(f"max y: {max(y)}, min y: {min(y)}")
            # box的范围不会超过: ±102.4，和论文一致
            # 102.4/256 = 0.4，看起来分辨率是0.4m
            #! points:            tensor([0.], device='cuda:0', dtype=torch.float16)
            #! camera2ego:        torch.Size([1, 5, 4, 4])
            #! camera_intrinsics: torch.Size([1, 5, 4, 4])
            #! img_aug_matrix:    torch.Size([1, 5, 4, 4])
            #! gt_masks_bev:      tensor([0], device='cuda:0')
            #! gt_bboxes_3d[0]:   <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>
            #! gt_labels_3d[0]:   tensor([0, 0, 0, 0, 0], device='cuda:0')
            outputs = self.forward_single(
                img,
                single_pred,  #! H95
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        single_pred,  #! H95
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        # want = {'filename': metas[0]['filename'],
        #         'sample_token': metas[0]['token'],
        #         'gt_bboxes_3d': gt_bboxes_3d[0].tensor.tolist()}
        # id = metas[0]['filename'][0].split("id_")[1].split("/")[0]
        # yaml_name = metas[0]['token'] + '_' + "id_" + id + ".yaml"
        # with open("/mnt/storage/hjw/code/uav3d/perception/bevfusion/" + yaml_name, 'w') as f:
        #     yaml.dump(want, f)
        # raise Exception
        #! print(img.shape) # train/val: [1, 5, 3, 448, 800]
        #! lidar2ego: [1,4,4]的单位矩阵
        #! camera_intrinsics: [1, 5, 4, 4]
        #! camera2lidar: [1, 5, 4, 4]
        #! metas: list, len=1
        #!   metas[0]: dict, ['filename', 'timestamp', 'ori_shape', 'img_shape', 'lidar2image', 'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'token', 'lidar_path']
        #!     metas[0]['filename']: 
        #!       ['./data/uav3d/v1.0/sweeps/town06_row3/CAMERA_FRONT_id_0/1701307796749329408.png', 
        #!        './data/uav3d/v1.0/sweeps/town06_row3/CAMERA_BACK_id_0/1701307796749329408.png', 
        #!        './data/uav3d/v1.0/sweeps/town06_row3/CAMERA_LEFT_id_0/1701307796749329408.png', 
        #!        './data/uav3d/v1.0/sweeps/town06_row3/CAMERA_RIGHT_id_0/1701307796749329408.png', 
        #!        './data/uav3d/v1.0/sweeps/town06_row3/CAMERA_BOTTOM_id_0/1701307796749329408.png']
        #!     metas[0]['token']: e.g.:933662bda2c945489372ae5a4a7abd57, 是sample.json中的一个token

        #! gt_labels_3d: list, len=1, tensor([0, 0, 0, 0, 0], device='cuda:0')


        #! 卡死的时候: 
        #! gt_labels_3d = [tensor([0, 0, 0, 0, 0], device='cuda:0')]
        #! single_pred[0].shape: torch.Size([24, 9])
        # breakpoint()
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
                #! when train: feature.shape = [1, 256, 256, 256]
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)
        #! when train: len(features)=1
        if not self.training:
            # avoid OOM
            features = features[::-1]

        #! when train: self.fuser = None
        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]
        #! H95: x = [1, 256, 256, 256]



        #! 5个UAV的single_pred全部为空时:
        if (single_pred[0].numel()==1):
            print(f"Fuck 5 UAV.")
            add_feature = torch.zeros(1, 25, 256, 256, dtype=x.dtype, device=x.device, requires_grad=False)
            y = torch.cat([x, add_feature], dim = 1) # [1, 281, 256, 256]
        else:
            add_feature_list = []
            for mm in range(5):
                mask_0 = single_pred[0][:, -1] == mm
                boxes_pred = single_pred[0][mask_0, :-1]  
                #! 20250125, 01:32 :增加判断条件, 该UAV的single_pred是否为空
                if boxes_pred.shape[0] == 0: 
                    extra_features = torch.zeros(1, 5, 256, 256, dtype=x.dtype, device=x.device, requires_grad=False)
                    add_feature_list.append(extra_features)
                    continue
                # boxes_pred: [N, 8], 最后一维是score
                mask = (boxes_pred[:, -1] > 0.7) & (boxes_pred[:, -1] < 1)
                boxes_limited = boxes_pred[mask]  # [N, 8]

                # print(boxes_limited)
                # print("HHH")
                # print(gt_bboxes_3d)
                # raise Exception
                #! 检查: 这里的值居然有超过102的
                xx = boxes_limited[:, 0]
                yy = boxes_limited[:, 1]
                xx = [ int(jj/0.8)+128 for jj in xx]
                yy = [ int(jjj/0.8)+128 for jjj in yy]
                xx = [ 0 if kk < 0 else 255 if kk > 255 else kk for kk in xx]
                yy = [ 0 if kkk < 0 else 255 if kkk > 255 else kkk for kkk in yy]

                values = boxes_limited[:, 3:] # 不要xyz
                extra_features = torch.zeros(1, 5, 256, 256, dtype=x.dtype, device=x.device, requires_grad=False)
                for pp in range(len(xx)):
                    extra_features[0, :, xx[pp], yy[pp]] = values[pp, :]
                add_feature_list.append(extra_features)
            add_feature = torch.cat(add_feature_list, dim = 1)
            y = torch.cat([x, add_feature], dim = 1) # [1, 281, 256, 256]


        #! when train: x = torch.Size([1, 256, 256, 256])
        batch_size = y.shape[0]

        # self.decoder["backbone"]: <class 'mmdet3d.models.backbones.resnet.GeneralizedResNet'>
        y = self.decoder["backbone"](y) #! len(x) = 3
        #! when train: [1, 128, 128, 128], [1, 256, 64, 64], [1, 512, 64, 64]
        # self.decoder["neck"]: <class 'mmdet3d.models.necks.lss.LSSFPN'>
        y = self.decoder["neck"](y) #! when train: torch.Size([1, 256, 128, 128])

        # print(gt_bboxes_3d[0].tensor.shape, gt_labels_3d[0].shape)
        # tmp_w = torch.min(gt_bboxes_3d[0].tensor[:, 3])
        # tmp_h = torch.min(gt_bboxes_3d[0].tensor[:, 4])
        # tmp_l = torch.min(gt_bboxes_3d[0].tensor[:, 5])
        # print(f"{metas[0]['token']}: w={tmp_w}, h={tmp_h}, l={tmp_l}")
        # torch.Size([5, 9]) torch.Size([5])
        if self.training:
            outputs = {}
            # self.heads: class 'torch.nn.modules.container.ModuleDict'>
            #   self.heads.keys(): dict_keys(['object'])
            for type, head in self.heads.items():
                #! when train: only "object"
                if type == "object":
                    # head: <class 'mmdet3d.models.heads.bbox.centerpoint.CenterHead'>
                    #   head.in_channels: 256
                    #   head.class_names: [['car']]
                    pred_dict = head(y, metas) #! tuple,len=1, tuple[0] = list, lne=1
                    #! pred_dict[0][0] = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
                        # reg torch.Size([1, 2, 128, 128])
                        # height torch.Size([1, 1, 128, 128])
                        # dim torch.Size([1, 3, 128, 128])
                        # rot torch.Size([1, 2, 128, 128])
                        # vel torch.Size([1, 2, 128, 128])
                        # heatmap torch.Size([1, 1, 128, 128])
                    #! gt_bboxes_3d: list, len=1
                    #!     gt_bboxes_3d.tensor: [N, 9] xyz, whl, yaw, 0, 0
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(y, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                #! head: <class 'mmdet3d.models.heads.bbox.centerpoint.CenterHead'>
                if type == "object":
                    pred_dict = head(y, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    #! bboxes: list, len=1
                    #!     bboxes[0]: list, len=3
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        #! boxes: <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>
                        #!     boxes.tensor: [N, 9], e.g.: [ 4.3591e+01, -1.9563e+00, -5.7049e-03,  4.7707e+00,  2.0063e+00,  1.4743e+00, -1.6090e+00,  4.5300e-05,  2.8610e-05]
                        #! scores: [N]
                        #! labels: [N]
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(y)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            #! H95: 保存outputs
            # sample_token = metas[0]["token"]
            # uav_id = metas[0]['filename'][0].split('id_')[1].split('/')[0]
            # assert batch_size == 1
            # H95_output = {}
            # H95_output["boxes_3d"] =  boxes.to("cpu").tensor.numpy().tolist()
            # H95_output["scores_3d"] = scores.cpu().numpy().tolist()
            # H95_output["labels_3d"] = labels.cpu().numpy().tolist()
            # # breakpoint()
            # with open("/mnt/storage/hjw/code/uav3d/perception/bevfusion/data/uav3d/v1.0/h95/" + sample_token + '_id_' + uav_id + '.yaml', 'w') as f:
            #     yaml.dump(H95_output, f)

            return outputs
