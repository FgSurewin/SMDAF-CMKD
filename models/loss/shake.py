import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss.dist import DIST

"""
Code is modified from paper "Shadow Knowledge Distillation: Bridging Offline and Online Knowledge Transfer"
GitHub: https://github.com/lliai/SHAKE
"""


class SHAKE(nn.Module):
    def __init__(
        self,
        temperature=2.0,
        gamma=1.0,
        alpha=1.0,
        beta=1.0,
        kd_loss_type="kl",
    ) -> None:
        super().__init__()
        self.temp = temperature
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.kd_loss_type = kd_loss_type
        if kd_loss_type == "kl":
            self.loss_fn = self.kl_loss_fn
        elif kd_loss_type == "dist":
            self.loss_fn = DIST(tau=temperature)

    def kl_loss_fn(self, student_logits, teacher_logits):
        return F.kl_div(
            F.log_softmax(student_logits / self.temp, dim=1),
            F.softmax(teacher_logits / self.temp, dim=1),
            reduction="batchmean",
        ) * (self.temp**2)

    def forward(self, student_logits, teacher_logits, proxy_teacher_logits, labels):
        hard_loss = F.cross_entropy(student_logits, labels)
        soft_loss = self.loss_fn(student_logits, teacher_logits.detach())
        proxy_teacher_student_soft_loss = self.loss_fn(
            proxy_teacher_logits, student_logits.detach()
        )
        student_proxy_teacher_soft_loss = self.loss_fn(
            student_logits, proxy_teacher_logits.detach()
        )
        proxy_teacher_hard_loss = F.cross_entropy(proxy_teacher_logits, labels)
        if self.kd_loss_type == "kl":
            teacher_soft_loss = F.mse_loss(
                proxy_teacher_logits, teacher_logits.detach()
            )
        elif self.kd_loss_type == "dist":
            teacher_soft_loss = self.loss_fn(
                proxy_teacher_logits, teacher_logits.detach()
            )

        return (
            self.gamma * hard_loss
            + self.alpha * soft_loss
            + self.beta
            * (
                proxy_teacher_student_soft_loss
                + proxy_teacher_hard_loss
                + student_proxy_teacher_soft_loss
                + teacher_soft_loss
            )
        )
