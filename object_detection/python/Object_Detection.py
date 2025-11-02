import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the COCO class names
with open('../../Srcs/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# Load the DNN model
model = cv2.dnn.readNet(model='../../Srcs/Files needed/frozen_inference_graph.pb',
                        config='../../Srcs/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')

# Create the main window
root = tk.Tk()
root.title("Object_Detection")
window_width = 1050
window_height = 800
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# Function to open an image
# Function to open an image
def open_image():
    global image_path, img_label, image
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
    image = Image.open(image_path)
    image = image.resize((1024, 768), Image.LANCZOS)  # Use Image.ANTIALIAS for high-quality downsampling
    img_tk = ImageTk.PhotoImage(image)
    img_label.config(image=img_tk)
    img_label.image = img_tk
    detect_objects()  # Call object detection when a new image is loaded

open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

# Label to display the image
img_label = tk.Label(root)
img_label.pack()

# Function to detect objects in the image
# Function to detect objects in the image
def detect_objects():
    global image
    if image is not None:
        image_width, image_height = image.size
        # Convert the PIL image to a NumPy array
        image_np = np.array(image)
        blob = cv2.dnn.blobFromImage(image=image_np, size=(1024, 768), mean=(104, 117, 123), swapRB=True)
        model.setInput(blob)
        output = model.forward()

        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > 0.4:
                class_id = int(detection[1])
                class_name = class_names[class_id-1]
                color = COLORS[class_id]
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height
                cv2.rectangle(image_np, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                cv2.putText(image_np, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Convert the NumPy array back to a PIL image
        image = Image.fromarray(image_np)
        img_tk = ImageTk.PhotoImage(image=image)
        img_label.config(image=img_tk)
        img_label.image = img_tk


root.mainloop()
