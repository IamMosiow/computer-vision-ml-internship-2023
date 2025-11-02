import cv2
import os

output_folder_base = '../data/'
folder_name = 'AbbasAli'
output_folder = os.path.join(output_folder_base, folder_name)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

face_cascade = cv2.CascadeClassifier('D:\Internship\Files needed\haarcascade\haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

count = 0
while (count <= 100):

    success, frame = camera.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imshow('Capturing Faces...', frame)
            cv2.waitKey(1) 
            face_filename = '%s/%s_%d.pgm' % (output_folder, folder_name, count)
            cv2.imwrite(face_filename, face_img)
            count += 1

cv2.destroyAllWindows()