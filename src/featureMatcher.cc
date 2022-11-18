#include "featureMatcher.h"

void FeatureMatcher::matchByBruteForce(
  const cv::Mat &descriptors1, const std::vector<cv::KeyPoint> &keypoints1,
  const cv::Mat &descriptors2, const std::vector<cv::KeyPoint> &keypoints2,
  std::vector<cv::DMatch> &matches, cv::Size image_size, int method) {
  if(descriptors1.type() == CV_8U) {
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors1, descriptors2, matches);
    // std::cout << "matches size: " << matches.size() << std::endl;

    if (method == GMS) {
      std::vector<cv::DMatch> gms_matches;
      cv::xfeatures2d::matchGMS(image_size, image_size, keypoints1,
      keypoints2, matches, gms_matches);
      matches = gms_matches;
    }
    else if (method == FM_RANSAC) {
      // ransac 
      std::vector<cv::Point2f> vkpts1, vkpts2;
      for (size_t i = 0; i < matches.size(); i++)
      {
          vkpts1.push_back(keypoints1[matches[i].queryIdx].pt);
          vkpts2.push_back(keypoints2[matches[i].trainIdx].pt);
      }

      std::vector<uchar> status;
      cv::findFundamentalMat(vkpts1, vkpts2, cv::FM_RANSAC, 3, 0.99, status);

      std::vector<cv::DMatch> good_matches;
      for(int i=0; i < matches.size(); i++)
      {
          if(status[i])
              good_matches.push_back(matches[i]);
      }
      matches = good_matches;
    }
  }
}