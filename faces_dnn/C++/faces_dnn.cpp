#include <iostream>
#include <opencv.hpp>

#if _DEBUG
#pragma comment(lib, "opencv_world480d.lib")
#else
#pragma comment(lib, "opencv_world480.lib")
#endif

int main() {
    cv::dnn::Net face_model = cv::dnn::readNetFromCaffe(
        "D:/Internship/usefull datas/faces_data/detection/deploy.prototxt",
        "D:/Internship/usefull datas/faces_data/detection/res10_300x300_ssd_iter_140000.caffemodel");
    int face_blob_height = 300;
    cv::Scalar face_average_color = cv::Scalar(104, 177, 123);
    float face_confidence_threshold = 0.995;

    cv::dnn::Net age_model = cv::dnn::readNetFromCaffe(
        "D:/Internship/usefull datas/faces_data/age_gender_classification/age_net_deploy.prototxt",
        "D:/Internship/usefull datas/faces_data/age_gender_classification/age_net.caffemodel");
    std::vector<std::string> age_labels = { "0-2", "4-6", "8-12", "15-20",
                                           "25-32", "38-43", "48-53", "60+" };

    cv::dnn::Net gender_model = cv::dnn::readNetFromCaffe(
        "D:/Internship/usefull datas/faces_data/age_gender_classification/gender_net_deploy.prototxt",
        "D:/Internship/usefull datas/faces_data/age_gender_classification/gender_net.caffemodel");
    std::vector<std::string> gender_labels = { "male", "female" };

    cv::Size age_gender_blob_size(256, 256);
    cv::Mat age_gender_average_image = cv::imread(
        "D:/Internship/usefull datas/faces_data/age_gender_classification/average_face.jpg");

    cv::VideoCapture cap(0);

    cv::Mat frame;
    while (cap.read(frame)) {

        int h = frame.rows;
        int w = frame.cols;
        float aspect_ratio = static_cast<float>(w) / h;

        // Detect faces in the frame.

        int face_blob_width = static_cast<int>(face_blob_height * aspect_ratio);
        cv::Size face_blob_size(face_blob_width, face_blob_height);

        cv::Mat face_blob = cv::dnn::blobFromImage(
            frame, 1.0, face_blob_size, face_average_color, false, false);

        face_model.setInput(face_blob);
        cv::Mat face_results = face_model.forward();

        // Iterate over the detected faces.
        for (int i = 0; i < face_results.size[2]; ++i) {
            float face_confidence = face_results.at<float>(0, 0);
            if (face_confidence > face_confidence_threshold) {

                // Get the face coordinates.
                int x0 = static_cast<int>(face_results.at<float>(0, 3) * w);
                int y0 = static_cast<int>(face_results.at<float>(0, 4) * h);
                int x1 = static_cast<int>(face_results.at<float>(0, 5) * w);
                int y1 = static_cast<int>(face_results.at<float>(0, 6) * h);

                // ... (Rest of the code goes here)

                // Draw a blue rectangle around the face.
                cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1),
                    cv::Scalar(255, 0, 0), 2);

                // ... (Continue with the rest of the drawing code)

            }
        }

        cv::imshow("Faces, age, and gender", frame);

        int k = cv::waitKey(1);
        if (k == 27) {
            break;
        }
    }

    return 0;
}