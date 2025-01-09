import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss.dist import DISTLoss
from models.loss.c2kd import C2KDLoss
from models.loss.shake import SHAKE
from models.loss.kl import KL
from models.loss.rkd import RKDLoss
from models.loss.nkd import NKDLoss


class KDLoss(nn.Module):
    def __init__(self, kd_type="kl", **kwargs):
        super(KDLoss, self).__init__()
        self.kd_type = kd_type
        self.kwargs = kwargs
        if self.kd_type == "kl":
            self.loss_fn = KL(**kwargs)
        elif self.kd_type == "dist":
            self.loss_fn = DISTLoss(**kwargs)
        elif self.kd_type == "rkd":
            self.loss_fn = RKDLoss(**kwargs)
        elif self.kd_type == "nkd":
            self.loss_fn = NKDLoss(**kwargs)
        elif self.kd_type == "c2kd":
            self.loss_fn = C2KDLoss(**kwargs)
        elif self.kd_type == "shake":
            self.loss_fn = SHAKE(**kwargs)
        else:
            raise ValueError(f"Unsupported KD type: {self.kd_type}")

    def kl_loss(self, logits_dict):
        student_logits = logits_dict["student_logits"]
        teacher_logits = logits_dict["teacher_logits"]
        labels = logits_dict["labels"]
        return self.loss_fn(student_logits, teacher_logits, labels)

    def dist_loss(self, logits_dict):
        student_logits = logits_dict["student_logits"]
        teacher_logits = logits_dict["teacher_logits"]
        labels = logits_dict["labels"]
        return self.loss_fn(student_logits, teacher_logits, labels)

    def rkd_loss(self, logits_dict):
        student_logits = logits_dict["student_logits"]
        teacher_logits = logits_dict["teacher_logits"]
        labels = logits_dict["labels"]
        return self.loss_fn(student_logits, teacher_logits, labels)

    def nkd_loss(self, logits_dict):
        student_logits = logits_dict["student_logits"]
        teacher_logits = logits_dict["teacher_logits"]
        labels = logits_dict["labels"]
        return self.loss_fn(student_logits, teacher_logits, labels)

    def c2kd_loss(self, logits_dict):
        student_logits = logits_dict["student_logits"]
        teacher_logits = logits_dict["teacher_logits"]
        proxy_student_logits = logits_dict["proxy_student_logits"]
        proxy_teacher_logits = logits_dict["proxy_teacher_logits"]
        labels = logits_dict["labels"]
        return self.loss_fn(
            student_logits,
            teacher_logits,
            proxy_student_logits,
            proxy_teacher_logits,
            labels,
        )

    def shake_loss(self, logits_dict):
        student_logits = logits_dict["student_logits"]
        teacher_logits = logits_dict["teacher_logits"]
        proxy_teacher_logits = logits_dict["proxy_teacher_logits"]
        labels = logits_dict["labels"]
        return self.loss_fn(
            student_logits,
            teacher_logits,
            proxy_teacher_logits,
            labels,
        )

    def forward(self, logits_dict):
        if self.kd_type == "kl":
            return self.kl_loss(logits_dict)
        elif self.kd_type == "dist":
            return self.dist_loss(logits_dict)
        elif self.kd_type == "rkd":
            return self.rkd_loss(logits_dict)
        elif self.kd_type == "nkd":
            return self.nkd_loss(logits_dict)
        elif self.kd_type == "c2kd":
            return self.c2kd_loss(logits_dict)
        elif self.kd_type == "shake":
            return self.shake_loss(logits_dict)
        else:
            raise ValueError(f"Unsupported KD type: {self.kd_type}")
