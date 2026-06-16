import torch
import torch.nn as nn
import torch.nn.functional as F
from pino.model import Native3DTransformer

class DINOHead(nn.Module):
    """
    Simplified DINO projection head: a 2-layer MLP with L2 normalization.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return x

class DINO_3DGS(nn.Module):
    """
    Wrapper for Teacher and Student networks.
    """
    def __init__(self, config):
        super().__init__()
        embed_dim = config.model.embed_dim
        out_dim = config.dino.out_dim
        
        self.student = Native3DTransformer(embed_dim=embed_dim, depth=config.model.depth, num_heads=config.model.num_heads)
        self.teacher = Native3DTransformer(embed_dim=embed_dim, depth=config.model.depth, num_heads=config.model.num_heads)
        
        self.student_head = DINOHead(in_dim=embed_dim, out_dim=out_dim)
        self.teacher_head = DINOHead(in_dim=embed_dim, out_dim=out_dim)
        
        # Initialize teacher with student weights
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        
        # Turn off gradients for teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
            
        self.momentum = config.dino.momentum_teacher

    @torch.no_grad()
    def update_teacher(self):
        """EMA update of the teacher."""
        for param_student, param_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            param_teacher.data.mul_(self.momentum).add_((1 - self.momentum) * param_student.data)
        for param_student, param_teacher in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_teacher.data.mul_(self.momentum).add_((1 - self.momentum) * param_student.data)

    def forward_student(self, patch_features, patch_centers):
        x = self.student(patch_features, patch_centers)
        return self.student_head(x)

    @torch.no_grad()
    def forward_teacher(self, patch_features, patch_centers):
        x = self.teacher(patch_features, patch_centers)
        return self.teacher_head(x)

class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, teacher_temp: float = 0.04, student_temp: float = 0.1):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.register_buffer("center", torch.zeros(1, 1, out_dim))
        self.center_momentum = 0.9

    def forward(self, student_output, teacher_output):
        teacher_out = teacher_output.detach()
        teacher_out = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1)
        
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)
        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=(0, 1), keepdim=True)
        batch_center = batch_center / (teacher_output.shape[0] * teacher_output.shape[1])
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
