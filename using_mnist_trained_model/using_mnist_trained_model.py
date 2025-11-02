import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1_bn(self.conv1(x))
        x = nn.functional.relu(x)
        x = self.conv2_bn(self.conv2(x))
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 12 * 12)
        x = self.fc1_bn(self.fc1(x))
        x = nn.functional.relu(x)
        x = self.dropout(x)  
        x = self.fc2(x)
        return x

model = Net()

# Load pretrained weights
pretrained_path = '../../Srcs/mnist trained model better.pth'
model.load_state_dict(torch.load(pretrained_path))
model.eval()  # Set the model to evaluation mode

def predict_digit(image):
    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output)
        return predicted.item()

# Load the image using PIL
image_path = '../../Srcs/8-digit-icon.png'
pil_image = Image.open(image_path)

# Convert the image to grayscale
gray_image = pil_image.convert('L')

# Resize the grayscale image to 28x28 pixels
gray_image = gray_image.resize((28, 28))

# Convert grayscale PIL image to NumPy array
numpy_image = np.array(gray_image)
# Apply the same transformations as during training
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
input_tensor = transform(numpy_image)

# Reshape to match batch size of 1 (since it's a single image)
input_tensor = input_tensor.unsqueeze(0)


# Example inference
predicted_digit = predict_digit(input_tensor)
print(f'Predicted digit: {predicted_digit}')