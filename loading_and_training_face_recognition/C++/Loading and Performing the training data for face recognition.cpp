#include <iostream>
#include <opencv.hpp>
#include <filesystem>
#include <opencv2\face.hpp>

#if _DEBUG
#pragma comment(lib, "opencv_world480d.lib")
#pragma comment(lib, "opencv_face480d")
#else
#pragma comment(lib, "opencv_world480.lib")
#pragma comment(lib, "opencv_face480")
#endif

using namespace cv;
using namespace std;
using namespace cv::face;

void read_images(const std::string& path, const cv::Size& image_size,
    vector<std::string>& names,
    vector<cv::Mat>& training_images,
    vector<int>& training_labels) {
    int label = 0;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_directory()) {
            names.push_back(entry.path().filename().string());
            for (const auto& img_entry : std::filesystem::directory_iterator(entry.path())) {
                cv::Mat img = cv::imread(img_entry.path().string(), cv::IMREAD_GRAYSCALE);
                if (img.empty()) {
                    continue;
                }
                cv::resize(img, img, image_size);
                cv::imshow("training", img);
                cv::waitKey(5);
                training_images.push_back(img.clone());
                training_labels.push_back(label);
            }
            label++;
        }
    }
}


int main()
{
    string path_to_training_images = "D:/Internship/Codes/Python/2nd week - The book(OpenCV5)/Generating the data for face recognition/data";
    Size training_image_size(200, 200);
    vector<string> names;
    vector<Mat> training_images;
    vector<int> training_labels;
    read_images(path_to_training_images, training_image_size, names, training_images, training_labels);

    //Using EigenFaceRec
    //cv::Ptr<cv::face::EigenFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
 
    //UsingFisherFAceREcognizer
    //cv::Ptr<cv::face::FisherFaceRecognizer> model = cv::face::FisherFaceRecognizer::create();

    //UsingLBPHFaceRecognizer
    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();

    model->train(training_images, training_labels);

    //model->write("my_face_model.xml");

    CascadeClassifier face_cascadde;
    face_cascadde.load("../../Srcs/haarcascade_frontalface_default.xml");

    VideoCapture camera(0);
    while (waitKey(1) == -1) {
        Mat frame;
        bool success = camera.read(frame);
        if (success) {
            vector<Rect> faces;
            face_cascadde.detectMultiScale(frame, faces, 1.3, 5);
            for (const Rect& face : faces) {
                int x = face.x;
                int y = face.y;
                int w = face.width;
                int h = face.height;
                rectangle(frame, Point(x, y), Point(x + w, y + h), (255, 0, 150), 2);
                Mat gray;
                cvtColor(frame, gray, COLOR_BGR2GRAY);
                Mat roi_gray = gray(Rect(x, y, w, h));
                if (roi_gray.size == 0)
                    continue;
                resize(roi_gray, roi_gray, training_image_size);
                int label = -1;
                double confidence = 0.0;
                model->predict(roi_gray, label, confidence);
                string text = cv::format("%s, confidence=%.2f", names[label].c_str(), confidence);
                putText(frame, text, Point(x, y - 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
            }
            imshow("Face Recognition", frame);
        }
    }
}

