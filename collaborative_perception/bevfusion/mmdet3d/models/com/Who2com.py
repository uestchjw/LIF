'''
Author: Pan Shi <shipan76@mail.ustc.edu.cn>
Date: 2023-12-13 03:34:08
LastEditTime: 2024-03-19 14:52:00
LastEditors: Pan Shi
Description: 
FilePath: /shipan/bevfusion/mmdet3d/models/com/Who2com.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from .Backbone import (
    LidarEncoder,
    Conv2DBatchNormRelu,
    Sparsemax,
)


class Who2com(nn.Module):
    """When2com

    https://github.com/GT-RIPL/MultiAgentPerception

    """

    def __init__(
        self,
        config,
        n_classes=21,
        feat_channel=512,
        feat_squeezer=-1,
        attention="additive",
        has_query=True,
        sparse=False,
        layer=3,
        warp_flag=0,
        image_size=512,
        shared_img_encoder="unified",
        key_size=1024,
        query_size=32,
        compress_level=0,
        only_v2i=False,
    ):
        super().__init__()
        self.agent_num = config['agent_num']
        self.sparse = sparse
        self.key_size = key_size
        self.query_size = query_size
        self.shared_img_encoder = shared_img_encoder
        self.has_query = has_query
        self.warp_flag = warp_flag
        self.in_channels = config['in_channels']
        self.key_net = KmGenerator(
            # out_size=self.key_size, input_feat_sz=image_size / 32
            out_size=self.key_size, input_feat_sz=image_size / 2
        )
        self.attention_net = MIMOGeneralDotProductAttention(
            self.query_size, self.key_size, self.warp_flag
        )
        # # Message generator
        self.query_key_net = PolicyNet4(in_channels=self.in_channels)
        if self.has_query:
            self.query_net = KmGenerator(
                # out_size=self.query_size, input_feat_sz=image_size / 32
                out_size=self.query_size, input_feat_sz=image_size / 2
            )

        # List the parameters of each modules
        self.attention_paras = list(self.attention_net.parameters())
        # if self.shared_img_encoder == "unified":
        #     self.img_net_paras = list(self.u_encoder.parameters()) + list(
        #         self.decoder.parameters()
        #     )

        self.policy_net_paras = (
            list(self.query_key_net.parameters())
            + list(self.key_net.parameters())
            + self.attention_paras
        )
        if self.has_query:
            self.policy_net_paras = self.policy_net_paras + list(
                self.query_net.parameters()
            )

        # self.all_paras = self.img_net_paras + self.policy_net_paras
        self.all_paras = self.policy_net_paras

        # FIXME: MotionSateHead undefined
        # if self.motion_state:
        #     self.motion_cls = MotionStateHead(config)

    def argmax_select(self, warp_flag, val_mat, prob_action):
        # v(batch, query_num, channel, size, size)
        cls_num = prob_action.shape[1]

        coef_argmax = F.one_hot(prob_action.max(dim=1)[1], num_classes=cls_num).type(
            torch.FloatTensor
        )
        
        coef_argmax = coef_argmax.to(val_mat.device)
        
        coef_argmax = coef_argmax.transpose(1, 2)
        attn_shape = coef_argmax.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        coef_argmax_exp = coef_argmax.view(bats, key_num, query_num, 1, 1, 1)

        if warp_flag == 1:
            v_exp = val_mat
        else:
            v_exp = torch.unsqueeze(val_mat, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)

        output = coef_argmax_exp * v_exp  # (batch,4,channel,size,size)
        feat_argmax = output.sum(1)  # (batch,1,channel,size,size)

        # compute connect
        count_coef = copy.deepcopy(coef_argmax)
        ind = np.diag_indices(self.agent_num)
        count_coef[:, ind[0], ind[1]] = 0
        num_connect = torch.nonzero(count_coef).shape[0] / (
            self.agent_num * count_coef.shape[0]
        )

        return feat_argmax, coef_argmax, num_connect

    def activated_select(self, warp_flag, val_mat, prob_action, thres=0.10):

        coef_act = torch.mul(prob_action, (prob_action > thres).float())
        attn_shape = coef_act.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        coef_act_exp = coef_act.view(bats, key_num, query_num, 1, 1, 1)

        if warp_flag == 1:
            v_exp = val_mat
        else:
            v_exp = torch.unsqueeze(val_mat, 2)
            v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)
        # import pdb
        # pdb.set_trace()
        output = coef_act_exp * v_exp  # (batch,4,channel,size,size)
        # import pdb
        # pdb.set_trace()
        feat_act = output.sum(1)  # (batch,1,channel,size,size)
        # feat_act = feat_act.sum(1)  # (batch,1,channel,size,size)
        # compute connect
        count_coef = coef_act.clone()
        ind = np.diag_indices(self.agent_num)
        count_coef[:, ind[0], ind[1]] = 0
        num_connect = torch.nonzero(count_coef).shape[0] / (
            self.agent_num * count_coef.shape[0]
        )
        return feat_act, coef_act, num_connect


    def agents_to_batch(self, feats):
            """Concatenate the features of all agents back into a bacth.

            Args:
                feats (tensor): features

            Returns:
                The concatenated feature matrix of all agents.
            """
            feat_list = []
            for i in range(self.agent_num):
                feat_list.append(feats[:, i, :, :, :])
            feat_mat = torch.cat(tuple(feat_list), 0)

            # feat_mat = torch.flip(feat_mat, (2,))

            return feat_mat

    def forward(
        self,
        bevs,
        training=True,
        MO_flag=True,
        inference="activated",
        batch_size=1,
    ): 
        
        B, N, C, H, W = bevs.size()
        batch_size = B
        device = bevs.device
        
        # local_com_mat_update = x.clone()
        val_mat = torch.zeros(
                    B, N, N, C, H, W
                ).to(device)
        for b in range(B):
            for i in range(N):
                tg_agent = bevs[b, i]
                for j in range(N):
                    if j == i:
                        val_mat[b, i, j] = tg_agent
                    else:
                        val_mat[b, i, j] = bevs[b,j]
        # self.agent_num = agent_num
        # pass feature maps through key and query generator
        
        query_key_maps = self.query_key_net(bevs)
        keys = self.key_net(query_key_maps)

        if self.has_query:
            querys = self.query_net(query_key_maps)
        # get key and query
        key = {}
        query = {}
        key_list = []
        query_list = []

        for i in range(self.agent_num):
            key[i] = torch.unsqueeze(keys[batch_size * i : batch_size * (i + 1)], 1)
            key_list.append(key[i])
            if self.has_query:
                query[i] = torch.unsqueeze(
                    querys[batch_size * i : batch_size * (i + 1)], 1
                )
            else:
                query[i] = torch.ones(batch_size, 1, self.query_size).to("cuda")
            query_list.append(query[i])
        # import pdb
        # pdb.set_trace()
        key_mat = torch.cat(tuple(key_list), 1)
        query_mat = torch.cat(tuple(query_list), 1)
        if MO_flag:
            query_mat = query_mat
        else:
            query_mat = torch.unsqueeze(query_mat[:, 0, :], 1)

        feat_fuse, prob_action = self.attention_net(
            query_mat, key_mat, val_mat, sparse=self.sparse
        )
        
        # not related to how we combine the feature (prefer to use the agnets' own frames: to reduce the bandwidth)
        small_bis = torch.eye(prob_action.shape[1]) * 0.001
        small_bis = small_bis.reshape((1, prob_action.shape[1], prob_action.shape[2]))
        small_bis = small_bis.repeat(prob_action.shape[0], 1, 1).to(device)
        prob_action = prob_action + small_bis
        
        self.warp_flag = 1
        # inference = 'activated'
        inference = 'argmax_test'
        
        # training = True
        
        if training:
            # action = torch.argmax(prob_action, dim=1)
            # num_connect = self.agent_num - 1
            pass
        else:
            if inference == "activated":
                print("activated")
                feat_act, connect_mat, num_connect = self.activated_select(
                    self.warp_flag, val_mat, prob_action
                )
                # feat_act_mat = self.agents_to_batch(
                    # feat_act
                # )  # (batchsize*agent_num, channel, size, size)
                feat_act_mat = feat_act
                feat_fuse = feat_act_mat.detach()
            elif inference == "argmax_test":
                print("argmax_test")
                feat_argmax, connect_mat, num_connect = self.argmax_select(
                    self.warp_flag, val_mat, prob_action
                )
                feat_argmax_mat = feat_argmax
                # feat_argmax_mat = self.agents_to_batch(
                    # feat_argmax
                # )  # (batchsize*agent_num, channel, size, size)
                feat_fuse = feat_argmax_mat.detach()
        


        # feat_fuse = feat_fuse[:,0,:].squeeze(dim=1)
        feat_fuse = feat_fuse.view(B, N, C, H, W)
        return feat_fuse
        # return torch.mean(feat_fuse, dim=1)


class PolicyNet4(nn.Module):
    def __init__(self, in_channels=13, input_feat_sz=32):
        super(PolicyNet4, self).__init__()
        # feat_map_sz = input_feat_sz // 4
        # self.n_feat = int(256 * feat_map_sz * feat_map_sz)
        # self.lidar_encoder = LidarEncoder(height_feat_size=in_channels)

        # Encoder
        # down 1
        self.conv1 = Conv2DBatchNormRelu(80, 64, k_size=3, stride=1, padding=1)
        self.conv2 = Conv2DBatchNormRelu(64, 64, k_size=3, stride=1, padding=1)
        self.conv3 = Conv2DBatchNormRelu(64, 64, k_size=3, stride=2, padding=1)

        # down 2
        self.conv4 = Conv2DBatchNormRelu(64, 64, k_size=3, stride=1, padding=1)
        self.conv5 = Conv2DBatchNormRelu(64, 64, k_size=3, stride=2, padding=1)
        
        # down 3
        self.conv6 = Conv2DBatchNormRelu(64, 64, k_size=3, stride=1, padding=1)
        self.conv7 = Conv2DBatchNormRelu(64, 64, k_size=3, stride=2, padding=1)
        
        # down 4
        self.conv8 = Conv2DBatchNormRelu(64, 64, k_size=3, stride=1, padding=1)
        self.conv9 = Conv2DBatchNormRelu(64, 64, k_size=3, stride=2, padding=1)
        
        # down 5
        self.conv10 = Conv2DBatchNormRelu(64, 64, k_size=3, stride=1, padding=1)
        self.conv11 = Conv2DBatchNormRelu(64, 64, k_size=3, stride=2, padding=1)
        
        # down 6
        self.conv12 = Conv2DBatchNormRelu(64, 64, k_size=3, stride=1, padding=1)
        self.conv13 = Conv2DBatchNormRelu(64, 64, k_size=3, stride=2, padding=1)
        
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.avg_pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.avg_pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        # self.avg_pool2 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        
        
        

    def forward(self, features_map):
        # _, _, _, _, outputs1 = self.lidar_encoder(features_map)
        B, N, C, H, W = features_map.size()
        features_map = features_map.view(B * N, C, H, W)
        outputs = self.conv1(features_map)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        
        # outputs = self.avg_pool1(outputs)
        # outputs = self.avg_pool2(outputs)
        
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        
        
        
        outputs = self.conv6(outputs)
        outputs = self.conv7(outputs)
        
        # outputs = self.avg_pool3(outputs)
        # outputs = self.avg_pool4(outputs)
        
        outputs = self.conv8(outputs)
        outputs = self.conv9(outputs)
        
        #outputs = self.avg_pool1(outputs)
        
        outputs = self.conv10(outputs)
        outputs = self.conv11(outputs)
        
        # outputs = self.conv12(outputs)
        # outputs = self.conv13(outputs)
        
        return outputs


# hand-shake
class MIMOGeneralDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, query_size, key_size, warp_flag, attn_dropout=0.1):
        super().__init__()
        self.sparsemax = Sparsemax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(query_size, key_size)
        self.warp_flag = warp_flag
        print("Msg size: ", query_size, "  Key size: ", key_size)

    def forward(self, qu, k, v, sparse=True):
        # qu (batch,5,32)
        # k (batch,5,1024)
        # v (batch,5,channel,size,size)
        query = self.linear(qu)  # (batch,5,key_size)

        # normalization
        # query_norm = query.norm(p=2,dim=2).unsqueeze(2).expand_as(query)
        # query = query.div(query_norm + 1e-9)

        # k_norm = k.norm(p=2,dim=2).unsqueeze(2).expand_as(k)
        # k = k.div(k_norm + 1e-9)
        # generate the
        attn_orig = torch.bmm(
            k, query.transpose(2, 1)
        )  # (batch,5,5)  column: differnt keys and the same query

        # scaling [not sure]
        # scaling = torch.sqrt(torch.tensor(k.shape[2],dtype=torch.float32)).cuda()
        # attn_orig = attn_orig/ scaling # (batch,5,5)  column: differnt keys and the same query

        attn_orig_softmax = self.softmax(attn_orig)  # (batch,5,5)

        attn_shape = attn_orig_softmax.shape
        bats, key_num, query_num = attn_shape[0], attn_shape[1], attn_shape[2]
        attn_orig_softmax_exp = attn_orig_softmax.view(
            bats, key_num, query_num, 1, 1, 1
        )

        # if self.warp_flag == 1:
        #     v_exp = v
        # else:
        #     v_exp = torch.unsqueeze(v, 2)
        #     v_exp = v_exp.expand(-1, -1, query_num, -1, -1, -1)
        output = attn_orig_softmax_exp * v  # (batch,5,channel,size,size)
        output_sum = output.sum(1)  # (batch,1,channel,size,size)

        return output_sum, attn_orig_softmax


class KmGenerator(nn.Module):
    def __init__(self, out_size=128, input_feat_sz=32.0):
        super(KmGenerator, self).__init__()
        # feat_map_sz = input_feat_sz // 4
        feat_map_sz_h = int(256 // 32)
        feat_map_sz_w = int(256 // 32)
        self.n_feat = int(64 * feat_map_sz_h * feat_map_sz_w)
        self.fc = nn.Sequential(
            nn.Linear(self.n_feat, 256),  #
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),  #
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size),
        )  #

    def forward(self, features_map):
        outputs = self.fc(features_map.view(-1, self.n_feat))
        return outputs
    
def agents_to_batch(self, feats):
    """Concatenate the features of all agents back into a bacth.

    Args:
        feats (tensor): features

    Returns:
        The concatenated feature matrix of all agents.
    """
    feat_list = []
    for i in range(self.agent_num):
        feat_list.append(feats[:, i, :, :, :])
    feat_mat = torch.cat(tuple(feat_list), 0)

    feat_mat = torch.flip(feat_mat, (2,))

    return feat_mat
