import torch
import torch.nn as nn
import timm


class PromptTunedViT(nn.Module):
    def __init__(self, num_classes=2, prompt_length=5):
        super().__init__()

        self.base_model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True
        )
        embed_dim = self.base_model.embed_dim

        self.prompt = nn.Parameter(torch.randn(1, prompt_length, embed_dim))
        self.num_classes = num_classes

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.base_model.head = nn.Identity()

        cls_token = self.base_model.pos_embed[:, :1, :]
        patch_positions = self.base_model.pos_embed[:, 1:, :]

        prompt_positions = patch_positions[:, :prompt_length, :]

        new_pos_embed = torch.cat(
            [cls_token, prompt_positions, patch_positions],
            dim=1
        )

        self.base_model.pos_embed = nn.Parameter(new_pos_embed)

    def forward(self, x):
        B = x.shape[0]

        x = self.base_model.patch_embed(x)
        x = torch.cat([self.prompt.repeat(B, 1, 1), x], dim=1)

        cls = self.base_model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        pos = self.base_model.pos_embed[:, :x.size(1), :]
        x = x + pos

        x = self.base_model.pos_drop(x)
        x = self.base_model.blocks(x)
        x = self.base_model.norm(x)

        out = self.classifier(x[:, 0])
        return out
