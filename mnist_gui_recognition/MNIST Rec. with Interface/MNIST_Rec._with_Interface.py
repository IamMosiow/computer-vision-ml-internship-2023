import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# Create the main window
root = tk.Tk()
root.title("Digit Recognizer")
window_width = 305
window_height = 350
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
root.iconbitmap("../icon.ico")

img = None
image_path = None

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size= 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1_bn(self.conv1(x))
        x = nn.functional.relu(x)
        x = self.conv2_bn(self.conv2(x))
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv3_bn(self.conv3(x))
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 5 * 5)
        x = self.fc1_bn(self.fc1(x))
        x = nn.functional.relu(x)
        x = self.dropout(x)  
        x = self.fc2_bn(self.fc2(x))
        x = nn.functional.relu(x)
        x = self.dropout(x)  
        x = self.fc3(x)
        return x

model = Net()
pretrained_path = '../../Srcs/mnist trained model v3.pth'
model.load_state_dict(torch.load(pretrained_path))
model.eval()  # Set the model to evaluation mode

def predict_digit(image):
    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output)
        return predicted.item()



# Function to open an image
def open_image():
    global image_path, img_label, img
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
    img = Image.open(image_path)
    img = img.resize((300, 300))  # Resize the image for display
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

# Function to apply changes to the image
def Predict():
    global img, image_path
    if img is not None:
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
        messagebox.showinfo("Predicted digit", predicted_digit)        
    else:
        messagebox.showwarning("Warning", "Please open an image before applying changes.")

# Create buttons
open_button = tk.Button(root, text="Open Image", command=open_image)
apply_button = tk.Button(root, text="Predict", command=Predict)

# Create image label
img_label = tk.Label(root)

# Organize widgets using grid layout
open_button.grid(row=0, column=0, padx=10, pady=10)
apply_button.grid(row=0, column=1, padx=10, pady=10)
img_label.grid(row=1, columnspan=2)

# Start the main loop
root.mainloop()

