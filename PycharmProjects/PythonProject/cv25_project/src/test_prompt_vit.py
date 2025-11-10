from src.models.prompt_vit import PromptTunedViT
import torch

model = PromptTunedViT(num_classes=2, prompt_len=10)
x = torch.randn(2, 3, 224, 224)

y = model(x)

print("✅ Model working!")
print("Input shape:", x.shape)
print("Output shape:", y.shape)
