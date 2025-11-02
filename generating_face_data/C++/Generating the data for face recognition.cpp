#include <iostream>
#include <opencv.hpp>
#include <filesystem>
#include <sstream>

#if _DEBUG
#pragma comment(lib, "opencv_world480d.lib")
#else
#pragma comment(lib, "opencv_world480.lib")
#endif

using namespace cv;
using namespace std;

int main()
{
    string output_folder_base = "../data/";
    string folder_name = "Mosiow";
    std::filesystem::path output_folder = std::filesystem::path(output_folder_base) / folder_name;
    if (!std::filesystem::exists(output_folder)) {
        std::filesystem::create_directories(output_folder);
    }

    CascadeClassifier face_cascade;
    bool load = face_cascade.load("D:/Internship/Files needed/haarcascade/haarcascade_frontalface_default.xml");
    VideoCapture camera(0);
    int count = 0;
    while (count <= 100)
    {
        Mat frame;
        bool success = camera.read(frame);
        if (success)
        {
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            vector<Rect> faces;
            Size minSize(120, 120);
            face_cascade.detectMultiScale(gray, faces, 1.3, 5, 0, minSize);
            for (const Rect& face : faces)
            {
                int x = face.x;
                int y = face.y;
                int w = face.width;
                int h = face.height;
                rectangle(frame, Point(x, y), Point(x + w, y + h), (255, 0, 0), 2);
                Mat face_img;
                Size s(200, 200);
                resize(gray(Rect(x, y, w, h)), face_img, s);
                imshow("Capturing Faces...", frame);
                waitKey(1);

                std::filesystem::path face_filename = output_folder / (folder_name + "_" + std::to_string(count) + ".pgm");
                string face_filename_str = face_filename.string();
                imwrite(face_filename_str, face_img);
                count += 1;
            }
        }
    }
    destroyAllWindows();
}

