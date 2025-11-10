import torch
import torch.nn as nn
import timm

class PromptTunedViT(nn.Module):
    """
    Few-Shot Prompt-Tuned Vision Transformer (ViT)
    Author: Sanjar Raximjonov
    """

    def __init__(
        self,
        base_model: str = 'vit_base_patch16_224',
        num_classes: int = 2,
        prompt_len: int = 10,
    ):
        super().__init__()

        self.vit = timm.create_model(base_model, pretrained=True)

        for param in self.vit.parameters():
            param.requires_grad = False

        self.embed_dim = self.vit.embed_dim

        self.prompts = nn.Parameter(torch.randn(prompt_len, self.embed_dim))

        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        """
        B = x.shape[0]

        x = self.vit.patch_embed(x)  # [B, N, D]

        P = self.prompts.unsqueeze(0).expand(B, -1, -1)  # [B, L, D]
        x = torch.cat([P, x], dim=1)  # [B, L+N, D]

        for blk in self.vit.blocks:
            x = blk(x)

        x = x.mean(dim=1)

        out = self.head(x)
        return out
