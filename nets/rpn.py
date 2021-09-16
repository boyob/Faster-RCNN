import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils import loc2bbox


class ProposalCreator():
    def __init__(self, mode, nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=600,
                 n_test_pre_nms=3000,
                 n_test_post_nms=300,
                 min_size=16):
        self.mode = mode
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        # loc.shape=[22500, 4], score.shape=[22500].

        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        anchor = torch.from_numpy(anchor)
        if loc.is_cuda:
            anchor = anchor.cuda()
        # -----------------------------------#
        #   将RPN网络预测结果转化成建议框
        # -----------------------------------#
        roi = loc2bbox(anchor, loc)  # shape=[22500, 4].

        # -----------------------------------#
        #   防止建议框超出图像边缘
        # -----------------------------------#
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])

        # -----------------------------------#
        #   建议框的宽高的最小值不可以小于16
        # -----------------------------------#
        min_size = self.min_size * scale  # 16 = 16 * 1.
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        roi = roi[keep, :]  # shape=[22500,4] -> shape=[22430,4]. 22430是不固定的。
        score = score[keep]  # shape=[22500] -> shape=[22430]. 22430是不固定的。

        # -----------------------------------#
        #   根据得分进行排序，取出建议框
        # -----------------------------------#
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:  # 12000
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]  # 12000.

        # -----------------------------------#
        #   对建议框进行非极大抑制
        # -----------------------------------#
        keep = nms(roi, score, self.nms_thresh)
        keep = keep[:n_post_nms]  # 获取得分最高的前600个建议框。
        roi = roi[keep]
        return roi


class RegionProposalNetwork(nn.Module):
    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            mode="training",
    ):
        super(RegionProposalNetwork, self).__init__()
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(mode)
        # -----------------------------------------#
        #   生成基础先验框，shape为[9, 4]
        # -----------------------------------------#
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        n_anchor = self.anchor_base.shape[0]  # self.anchor_base.shape(9, 4).

        # -----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        # -----------------------------------------#
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # -----------------------------------------#
        #   分类预测先验框内部是否包含物体
        # -----------------------------------------#
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # -----------------------------------------#
        #   回归预测对先验框进行调整
        # -----------------------------------------#
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        # --------------------------------------#
        #   对FPN的网络部分进行权值初始化
        # --------------------------------------#
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape  # n = batch_size = 1.
        # -----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        # -----------------------------------------#
        x = F.relu(self.conv1(x))  # shape = [1, 1024, 50, 50] -> shape = [1, 512, 50, 50]。
        # -----------------------------------------#
        #   回归预测对先验框进行调整
        # -----------------------------------------#
        rpn_locs = self.loc(x)  # shape=[1, 36, 50, 50].
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # shape=[1, 22500, 4].
        # -----------------------------------------#
        #   分类预测先验框内部是否包含物体
        # -----------------------------------------#
        rpn_scores = self.score(x)  # shape=[1, 18, 50, 50].
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)  # shape=[1, 22500, 2].

        # --------------------------------------------------------------------------------------#
        #   进行softmax概率计算，每个先验框只有两个判别结果
        #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        # --------------------------------------------------------------------------------------#
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)  # shape=[1, 22500, 2].
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()  # shape=[1, 22500].
        rpn_fg_scores = rpn_fg_scores.view(n, -1)  # shape=[1, 22500].

        # ------------------------------------------------------------------------------------------------#
        #   生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        # ------------------------------------------------------------------------------------------------#
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)  # shape=(22500, 4).

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)  # shape=[600, 4].
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = torch.cat(rois, dim=0)
        roi_indices = torch.cat(roi_indices, dim=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
