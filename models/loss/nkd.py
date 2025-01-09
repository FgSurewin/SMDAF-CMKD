import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import KendallRankCorrCoef



"""
Code is modified from paper: "From Knowledge Distillation to Self-Knowledge Distillation: A Unified Approach with Normalized Loss and Customized Soft Labels"
GitHub: https://github.com/yzd-v/cls_KD/blob/1.0/mmcls/models/dis_losses/nkd.py
"""

class NKD(nn.Module):
    """PyTorch version of NKD"""

    def __init__(
        self,
        temp=1.0,
        gamma=1.5,
    ):
        super(NKD, self).__init__()

        self.temp = temp
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logit_s, logit_t, gt_label):

        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        # N*class
        N, c = logit_s.shape
        s_i = self.log_softmax(logit_s)
        t_i = F.softmax(logit_t, dim=1)
        # N*1
        s_t = torch.gather(s_i, 1, label)
        t_t = torch.gather(t_i, 1, label).detach()

        loss_t = -(t_t * s_t).mean()

        mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
        logit_s = logit_s[mask].reshape(N, -1)
        logit_t = logit_t[mask].reshape(N, -1)

        # N*class
        S_i = self.log_softmax(logit_s / self.temp)
        T_i = F.softmax(logit_t / self.temp, dim=1)

        loss_non = (T_i * S_i).sum(dim=1).mean()
        loss_non = -self.gamma * (self.temp**2) * loss_non

        return loss_t + loss_non


class NKDLoss(nn.Module):
    def __init__(
        self,
        temp=1.0,
        gamma=1.5,
        is_bidirectional=False,
        is_filtering=False,
        filtering_config={"type": "krc", "krc_threshold": 0},
    ):
        super(NKDLoss, self).__init__()
        self.nkd = NKD(temp=temp, gamma=gamma)
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

    def regular_forward(self, logit_s, logit_t, gt_label):
        hard_loss = F.cross_entropy(logit_s, gt_label)
        if self.is_filtering:
            if self.filtering_config["type"] == "correct":
                mask, _ = self.filtering_correct(logit_s, logit_t, gt_label)
            elif self.filtering_config["type"] == "krc":
                mask, _ = self.filtering_krc(
                    logit_s, logit_t, self.filtering_config["krc_threshold"]
                )
            logit_s = logit_s[mask]
            logit_t = logit_t[mask]
            gt_label = gt_label[mask]
            if mask.sum() == 0:
                # If no logits are left after filtering, return only the hard loss
                return hard_loss

        soft_loss = self.nkd(logit_s, logit_t.detach(), gt_label)
        kd_loss = hard_loss + soft_loss
        return kd_loss

    def bidirectional_forward(self, logit_s, logit_t, gt_label):
        hard_loss = F.cross_entropy(logit_s, gt_label)

        soft_loss = 0
        teacher_soft_loss = 0

        if self.is_filtering:
            if self.filtering_config["type"] == "correct":
                teacher_mask, student_mask = self.filtering_correct(
                    logit_s, logit_t, gt_label
                )
            elif self.filtering_config["type"] == "krc":
                teacher_mask, student_mask = self.filtering_krc(
                    logit_s, logit_t, self.filtering_config["krc_threshold"]
                )

            # Filter based on teacher correctness for student learning from teacher
            if teacher_mask.sum() > 0:
                logit_s_filtered = logit_s[teacher_mask]
                logit_t_filtered = logit_t[teacher_mask]
                gt_label_filtered = gt_label[teacher_mask]

                # Compute soft loss for student learning from teacher
                soft_loss = self.nkd(
                    logit_s_filtered, logit_t_filtered.detach(), gt_label_filtered
                )

            # Filter based on student correctness for teacher learning from student
            if student_mask.sum() > 0:
                logit_s_filtered = logit_s[student_mask]
                logit_t_filtered = logit_t[student_mask]
                gt_label_filtered = gt_label[student_mask]

                # Compute soft loss for teacher learning from student
                teacher_soft_loss = self.nkd(
                    logit_t_filtered, logit_s_filtered.detach(), gt_label_filtered
                )
        else:
            soft_loss = self.nkd(logit_s, logit_t.detach(), gt_label)
            teacher_soft_loss = self.nkd(logit_t, logit_s.detach(), gt_label)

        kd_loss = hard_loss + soft_loss + teacher_soft_loss
        return kd_loss

    def forward(self, logit_s, logit_t, gt_label):
        if self.is_bidirectional:
            return self.bidirectional_forward(logit_s, logit_t, gt_label)
        else:
            return self.regular_forward(logit_s, logit_t, gt_label)
