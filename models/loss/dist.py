import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import KendallRankCorrCoef


"""
Code is modified from paper: "Knowledge Distillation from A Stronger Teacher"
GitHub: https://github.com/hunto/DIST_KD
"""


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(
        a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps
    )


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss


class DISTLoss(nn.Module):
    def __init__(
        self,
        beta=1.0,
        gamma=1.0,
        temperature=1.0,
        is_bidirectional=False,
        is_filtering=False,
        filtering_config={"type": "krc", "krc_threshold": 0},
    ):
        super(DISTLoss, self).__init__()
        self.dist = DIST(beta=beta, gamma=gamma, tau=temperature)
        self.beta = beta
        self.gamma = gamma
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

    def regular_forward(self, z_s, z_t, labels):
        hard_loss = F.cross_entropy(z_s, labels)
        if self.is_filtering:
            if self.filtering_config["type"] == "correct":
                mask, _ = self.filtering_correct(z_s, z_t, labels)
            elif self.filtering_config["type"] == "krc":
                mask, _ = self.filtering_krc(
                    z_s,
                    z_t,
                    self.filtering_config["krc_threshold"],
                )
            z_s = z_s[mask]
            z_t = z_t[mask]
            if mask.sum() == 0:
                # If no logits are left after filtering, return only the hard loss
                return hard_loss

        soft_loss = self.dist(z_s, z_t.detach())
        kd_loss = hard_loss + soft_loss
        return kd_loss

    def bidirectional_forward(self, student_logits, teacher_logits, labels):
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
                soft_loss = self.dist(
                    student_logits_filtered, teacher_logits_filtered.detach()
                )

            # Filter based on student correctness for teacher learning from student
            if student_mask.sum() > 0:
                student_logits_filtered = student_logits[student_mask]
                teacher_logits_filtered = teacher_logits[student_mask]

                # Compute soft loss for teacher learning from student
                teacher_soft_loss = self.dist(
                    teacher_logits_filtered, student_logits_filtered.detach()
                )
        else:
            soft_loss = self.dist(student_logits, teacher_logits.detach())
            teacher_soft_loss = self.dist(teacher_logits, student_logits.detach())

        kd_loss = hard_loss + soft_loss + teacher_soft_loss
        return kd_loss

    def forward(self, z_s, z_t, labels):
        if self.is_bidirectional:
            return self.bidirectional_forward(z_s, z_t, labels)
        else:
            return self.regular_forward(z_s, z_t, labels)
