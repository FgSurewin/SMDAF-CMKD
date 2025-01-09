import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import KendallRankCorrCoef


class KL(nn.Module):
    def __init__(
        self,
        temperature=2.0,
        alpha=1.0,
        beta=1.0,
        is_bidirectional=False,
        is_filtering=False,
        filtering_config={"type": "krc", "krc_threshold": 0},
    ) -> None:
        super().__init__()
        self.temp = temperature
        self.alpha = alpha
        self.beta = beta
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

    def regular_forward(self, student_logits, teacher_logits, labels):
        hard_loss = F.cross_entropy(student_logits, labels)

        if self.is_filtering:
            if self.filtering_config["type"] == "correct":
                mask, _ = self.filtering_correct(student_logits, teacher_logits, labels)
            elif self.filtering_config["type"] == "krc":
                mask, _ = self.filtering_krc(
                    student_logits,
                    teacher_logits,
                    self.filtering_config["krc_threshold"],
                )
            student_logits = student_logits[mask]
            teacher_logits = teacher_logits[mask]
            if mask.sum() == 0:
                # If no logits are left after filtering, return only the hard loss
                return self.alpha * hard_loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temp, dim=1),
            F.softmax(teacher_logits.detach() / self.temp, dim=1),
            reduction="batchmean",
        ) * (self.temp**2)
        return self.alpha * hard_loss + self.beta * soft_loss

    def bidirectional_forward(self, student_logits, teacher_logits, labels):
        # Compute hard loss for student
        hard_loss = F.cross_entropy(student_logits, labels)

        soft_loss = 0
        teacher_soft_loss = 0

        if self.is_filtering:
            if self.filtering_config["type"] == "correct":
                teacher_mask, student_mask = self.filtering_correct(
                    student_logits, teacher_logits, labels
                )
            elif self.filtering_config["type"] == "krc":
                teacher_mask, student_mask = self.filtering_krc(
                    student_logits,
                    teacher_logits,
                    self.filtering_config["krc_threshold"],
                )

            # Filter based on teacher correctness for student learning from teacher
            if teacher_mask.sum() > 0:
                student_logits_filtered = student_logits[teacher_mask]
                teacher_logits_filtered = teacher_logits[teacher_mask]

                # Compute soft loss for student learning from teacher
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits_filtered / self.temp, dim=1),
                    F.softmax(teacher_logits_filtered.detach() / self.temp, dim=1),
                    reduction="batchmean",
                ) * (self.temp**2)

            # Filter based on student correctness for teacher learning from student
            if student_mask.sum() > 0:
                student_logits_filtered = student_logits[student_mask]
                teacher_logits_filtered = teacher_logits[student_mask]

                # Compute soft loss for teacher learning from student
                teacher_soft_loss = F.kl_div(
                    F.log_softmax(teacher_logits_filtered / self.temp, dim=1),
                    F.softmax(student_logits_filtered.detach() / self.temp, dim=1),
                    reduction="batchmean",
                ) * (self.temp**2)
        else:
            # Compute soft loss for student learning from teacher without filtering
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / self.temp, dim=1),
                F.softmax(teacher_logits.detach() / self.temp, dim=1),
                reduction="batchmean",
            ) * (self.temp**2)

            # Compute soft loss for teacher learning from student without filtering
            teacher_soft_loss = F.kl_div(
                F.log_softmax(teacher_logits / self.temp, dim=1),
                F.softmax(student_logits.detach() / self.temp, dim=1),
                reduction="batchmean",
            ) * (self.temp**2)

        # Combine the losses
        total_loss = self.alpha * hard_loss + self.beta * (
            soft_loss + teacher_soft_loss
        )
        return total_loss

    def forward(self, student_logits, teacher_logits, labels):
        if self.is_bidirectional:
            return self.bidirectional_forward(student_logits, teacher_logits, labels)
        else:
            return self.regular_forward(student_logits, teacher_logits, labels)


class BidirectionalKL(nn.Module):
    def __init__(self, temperature=2.0, alpha=1.0, beta=1.0) -> None:
        super().__init__()
        self.temp = temperature
        self.alpha = alpha
        self.beta = beta
        self.kl_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        hard_loss = F.cross_entropy(student_logits, labels)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temp, dim=1),
            F.softmax(teacher_logits.detach() / self.temp, dim=1),
            reduction="batchmean",
        ) * (self.temp**2)
        teacher_soft_loss = F.kl_div(
            F.log_softmax(teacher_logits / self.temp, dim=1),
            F.softmax(student_logits.detach() / self.temp, dim=1),
            reduction="batchmean",
        ) * (self.temp**2)
        return self.alpha * hard_loss + self.beta * (soft_loss + teacher_soft_loss)
