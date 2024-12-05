import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

content_image = load_image('content.jpg', transform)
style_image = load_image('style.jpg', transform)
generated_image = content_image.clone().requires_grad_(True)

model = models.vgg19(pretrained=True).features.eval()
optimizer = optim.Adam([generated_image], lr=0.01)

for step in range(300):
    optimizer.zero_grad()
    generated_features = model(generated_image)
    content_features = model(content_image)
    style_features = model(style_image)
    # Dummy loss computation as example
    loss = torch.mean((generated_features - content_features) ** 2) + torch.mean((generated_features - style_features) ** 2)
    loss.backward()
    optimizer.step()
