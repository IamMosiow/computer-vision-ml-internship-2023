#include <opencv.hpp>

#if _DEBUG
#pragma comment(lib, "opencv_world480d.lib")
#else
#pragma comment(lib, "opencv_world480.lib")
#endif

using namespace cv;
using namespace std;

bool compareDMatch(const cv::DMatch& a, const cv::DMatch& b) {
    return a.distance < b.distance;
}

int main()
{
    bool better = true; // change it for a differet version 

	Mat img0 = imread("../../Srcs/nasa-logo-web-rgb.png", IMREAD_GRAYSCALE);
	Mat img1 = imread("../../Srcs/vab_for_media_release.jpg", IMREAD_GRAYSCALE);
    resize(img0, img0, Size(500, 300));
    resize(img1, img1, Size(1000, 1000));

	cv::Ptr<ORB> orb = cv::ORB::create();

    std::vector<cv::KeyPoint> kp0, kp1;
    cv::Mat des0, des1;

    orb->detectAndCompute(img0, noArray(), kp0, des0);
    orb->detectAndCompute(img1, noArray(), kp1, des1);

    if (better == false)
    {
        BFMatcher bf(NORM_HAMMING, true);
        vector<DMatch> matches;
        bf.match(des0, des1, matches);
        std::sort(matches.begin(), matches.end(), [](const DMatch& a, const DMatch& b) {
            return a.distance < b.distance;
            });
        Mat img_matches;
        drawMatches(img0, kp0, img1, kp1, matches, img_matches,
            Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imshow("Matches", img_matches);
        waitKey(0);
    }
    else
    {
        cv::BFMatcher bf(NORM_HAMMING, false);
        vector<vector<DMatch>> pairs_of_matches;
        bf.knnMatch(des0, des1, pairs_of_matches, 2);
        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < pairs_of_matches.size(); i++) {
            if (pairs_of_matches[i][0].distance < 0.75 * pairs_of_matches[i][1].distance) {
                good_matches.push_back(pairs_of_matches[i][0]);
            }
        }

        std::sort(good_matches.begin(), good_matches.end(), compareDMatch);

        cv::Mat img_pairs_of_matches;
        cv::drawMatches(img0, kp0, img1, kp1, good_matches, img_pairs_of_matches,
            cv::Scalar::all(-1), cv::Scalar::all(-1),
            std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("Matches", img_pairs_of_matches);
        waitKey(0);
    }
}

