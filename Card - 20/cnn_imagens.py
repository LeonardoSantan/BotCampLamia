from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc1(x)
        return x

model = SimpleCNN()


image = Image.open('./Card - 20/image.png').convert('L')
transform_real = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
input_image = transform_real(image).unsqueeze(0)
output = model(input_image)
predicted_class = output.argmax(dim=1).item()
