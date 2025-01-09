import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import KendallRankCorrCoef


"""
Code is modified from paper: "Relational Knowledge Distillation"
Github: https://github.com/lenscloth/RKD/blob/master/metric/loss.py
"""


class RKD(nn.Module):
    """
    Relational Knowledge Distillation
    https://arxiv.org/pdf/1904.05068.pdf
    """

    def __init__(self, w_dist, w_angle):
        super(RKD, self).__init__()

        self.w_dist = w_dist
        self.w_angle = w_angle

    def forward(self, feat_s, feat_t):
        loss = self.w_dist * self.rkd_dist(
            feat_s, feat_t
        ) + self.w_angle * self.rkd_angle(feat_s, feat_t)

        return loss

    def rkd_dist(self, feat_s, feat_t):
        feat_t_dist = self.pdist(feat_t, squared=False)
        mean_feat_t_dist = feat_t_dist[feat_t_dist > 0].mean()
        feat_t_dist = feat_t_dist / mean_feat_t_dist

        feat_s_dist = self.pdist(feat_s, squared=False)
        mean_feat_s_dist = feat_s_dist[feat_s_dist > 0].mean()
        feat_s_dist = feat_s_dist / mean_feat_s_dist

        loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)

        return loss

    def rkd_angle(self, feat_s, feat_t):
        # N x C --> N x N x C
        feat_t_vd = feat_t.unsqueeze(0) - feat_t.unsqueeze(1)
        norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
        feat_t_angle = torch.bmm(norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(
            -1
        )

        feat_s_vd = feat_s.unsqueeze(0) - feat_s.unsqueeze(1)
        norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
        feat_s_angle = torch.bmm(norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(
            -1
        )

        loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)

        return loss

    def pdist(self, feat, squared=False, eps=1e-12):
        feat_square = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (
            feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod
        ).clamp(min=eps)

        if not squared:
            feat_dist = feat_dist.sqrt()

        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0

        return feat_dist


class RKDLoss(nn.Module):
    def __init__(
        self,
        w_dist=1.0,
        w_angle=1.0,
        is_bidirectional=False,
        is_filtering=False,
        filtering_config={"type": "krc", "krc_threshold": 0},
    ):
        super(RKDLoss, self).__init__()
        self.rkd = RKD(w_dist=w_dist, w_angle=w_angle)
        self.is_bidirectional = is_bidirectional
        self.is_filtering = is_filtering
        self.filtering_config = filtering_config
        if filtering_config["type"] == "krc":
            self.krc_metric = KendallRankCorrCoef(variant="b")

    def kendall_rank_correlation(self, logits_teacher, logits_student):
        batch_size = logits_teacher.size(0)
        krc_scores = torch.zeros(batch_size, device=logits_teacher.device)

        for i in range(batch_size):
            krc_scores[i] = self.krc_metric(logits_teacher[i], logits_student[i])

        return krc_scores

    def filtering_krc(self, student_logits, teacher_logits, threshold):
        krc = self.kendall_rank_correlation(teacher_logits, student_logits)
        mask = krc > threshold
        return mask, mask

    def filtering_correct(self, student_logits, teacher_logits, labels):
        teacher_preds = teacher_logits.argmax(dim=1)
        student_preds = student_logits.argmax(dim=1)

        teacher_correct = teacher_preds == labels
        student_correct = student_preds == labels

        return teacher_correct, student_correct

    def regular_forward(self, feat_s, feat_t, labels):
        hard_loss = F.cross_entropy(feat_s, labels)
        if self.is_filtering:
            if self.filtering_config["type"] == "correct":
                mask, _ = self.filtering_correct(feat_s, feat_t, labels)
            elif self.filtering_config["type"] == "krc":
                mask, _ = self.filtering_krc(
                    feat_s, feat_t, self.filtering_config["krc_threshold"]
                )
            feat_s = feat_s[mask]
            feat_t = feat_t[mask]
            if mask.sum() == 0:
                # If no features are left after filtering, return only the hard loss
                return hard_loss

        soft_loss = self.rkd(feat_s, feat_t.detach())
        kd_loss = hard_loss + soft_loss
        return kd_loss

    def bidirectional_forward(self, feat_s, feat_t, labels):
        hard_loss = F.cross_entropy(feat_s, labels)

        soft_loss = 0
        teacher_soft_loss = 0

        if self.is_filtering:
            if self.filtering_config["type"] == "correct":
                teacher_mask, student_mask = self.filtering_correct(
                    feat_s, feat_t, labels
                )
            elif self.filtering_config["type"] == "krc":
                teacher_mask, student_mask = self.filtering_krc(
                    feat_s, feat_t, self.filtering_config["krc_threshold"]
                )

            # Filter based on teacher correctness for student learning from teacher
            if teacher_mask.sum() > 0:
                feat_s_filtered = feat_s[teacher_mask]
                feat_t_filtered = feat_t[teacher_mask]

                # Compute soft loss for student learning from teacher
                soft_loss = self.rkd(feat_s_filtered, feat_t_filtered.detach())

            # Filter based on student correctness for teacher learning from student
            if student_mask.sum() > 0:
                feat_s_filtered = feat_s[student_mask]
                feat_t_filtered = feat_t[student_mask]

                # Compute soft loss for teacher learning from student
                teacher_soft_loss = self.rkd(feat_t_filtered, feat_s_filtered.detach())
        else:
            soft_loss = self.rkd(feat_s, feat_t.detach())
            teacher_soft_loss = self.rkd(feat_t, feat_s.detach())

        kd_loss = hard_loss + soft_loss + teacher_soft_loss
        return kd_loss

    def forward(self, feat_s, feat_t, labels):
        if self.is_bidirectional:
            return self.bidirectional_forward(feat_s, feat_t, labels)
        else:
            return self.regular_forward(feat_s, feat_t, labels)
