import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import KendallRankCorrCoef

from models.loss.dist import DIST

"""
Code is implemented based on paper: "C2KD: Bridging the Modality Gap for Cross-Modal Knowledge Distillation"
There is no official implementation available from original paper.
"""

class C2KDLoss(nn.Module):
    def __init__(
        self,
        lambda_1=1.0,
        lambda_2=1.0,
        lambda_3=1.0,
        krc_threshold=0,
        temperature=2.0,
        kd_loss_type="kl",
        filtering_config={"type": "krc", "krc_threshold": 0},
    ):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.krc_threshold = krc_threshold
        self.temperature = temperature
        self.krc_metric = KendallRankCorrCoef(variant="b")
        self.cross_entropy = nn.CrossEntropyLoss()
        self.filtering_config = filtering_config
        if kd_loss_type == "kl":
            self.loss_fn = self.kl_divergence_loss
        elif kd_loss_type == "dist":
            self.loss_fn = DIST(tau=temperature)

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def kl_divergence_loss(self, logits_student, logits_teacher):
        return F.kl_div(
            F.log_softmax(logits_student / self.temperature, dim=-1),
            F.softmax(logits_teacher / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature**2)

    def ntkd_loss(self, logits_student, logits_teacher, target):
        gt_mask = self._get_gt_mask(logits_student, target)
        # Mask out the target class logits
        pred_student_part2 = (
            logits_student - 1000.0 * gt_mask
        )  # Set class value as large negative number, so it will become 0 after softmax.
        pred_teacher_part2 = (
            logits_teacher - 1000.0 * gt_mask
        )  # Set class value as large negative number, so it will become 0 after softmax.
        nckd_loss = self.loss_fn(pred_student_part2, pred_teacher_part2)
        return nckd_loss

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

    def forward(
        self,
        logits_student,
        logits_teacher,
        logits_proxy_student,
        logits_proxy_teacher,
        labels,
    ):
        # Calculate KRC filter
        # krc = self.kendall_rank_correlation(logits_student, logits_proxy_teacher)

        # Supervision loss
        loss_ce_student = self.cross_entropy(logits_student, labels)
        loss_ce_teacher = self.cross_entropy(logits_proxy_teacher, labels)

        # Distillation losses
        loss_kd_teacher_to_proxy_teacher = self.loss_fn(
            logits_teacher, logits_proxy_teacher.detach()
        )
        loss_kd_proxy_teacher_to_teacher = self.loss_fn(
            logits_proxy_teacher, logits_teacher.detach()
        )
        loss_kd_student_to_proxy_student = self.loss_fn(
            logits_student, logits_proxy_student.detach()
        )
        loss_kd_proxy_student_to_student = self.loss_fn(
            logits_proxy_student, logits_student.detach()
        )

        # Filter out samples with KRC < w
        # valid_samples = krc > self.krc_threshold
        # filtered_logits_proxy_student = logits_proxy_student[valid_samples]
        # filtered_logits_proxy_teacher = logits_proxy_teacher[valid_samples]
        # filtered_labels = labels[valid_samples]

        # Adjusted batch size
        # batch_size = len(valid_samples)
        non_target_losses_student_to_teacher = 0
        non_target_losses_teacher_to_student = 0
        if self.filtering_config["type"] == "correct":
            teacher_mask, student_mask = self.filtering_correct(
                logits_proxy_student, logits_proxy_teacher, labels
            )
        elif self.filtering_config["type"] == "krc":
            teacher_mask, student_mask = self.filtering_krc(
                logits_proxy_student,
                logits_proxy_teacher,
                self.filtering_config["krc_threshold"],
            )

        if teacher_mask.sum() > 0:
            filtered_logits_proxy_student = logits_proxy_student[teacher_mask]
            filtered_logits_proxy_teacher = logits_proxy_teacher[teacher_mask]
            filtered_labels = labels[teacher_mask]

            non_target_losses_student_to_teacher = self.ntkd_loss(
                filtered_logits_proxy_student,
                filtered_logits_proxy_teacher.detach(),
                filtered_labels,
            )
        if student_mask.sum() > 0:
            filtered_logits_proxy_student = logits_proxy_student[student_mask]
            filtered_logits_proxy_teacher = logits_proxy_teacher[student_mask]
            filtered_labels = labels[student_mask]

            non_target_losses_teacher_to_student = self.ntkd_loss(
                filtered_logits_proxy_teacher,
                filtered_logits_proxy_student.detach(),
                filtered_labels,
            )

        # Total combined loss
        total_loss = (
            loss_ce_student
            + loss_ce_teacher
            + self.lambda_1
            * (loss_kd_teacher_to_proxy_teacher + loss_kd_proxy_teacher_to_teacher)
            + self.lambda_2
            * (loss_kd_student_to_proxy_student + loss_kd_proxy_student_to_student)
            + self.lambda_3
            * (
                non_target_losses_student_to_teacher
                + non_target_losses_teacher_to_student
            )
        )

        return total_loss
