/*
 -------------------------------------------------------------------------------
 * Frame implementation file
 *
 * Copyright (C) 2022 AutoX, Inc.
 * Author: Guyu Pan (yuxuanhuang@autox.ai)
 -------------------------------------------------------------------------------
 */

#include "frame.h"

Frame::Frame(uint64_t frame_id, const std::string& path2im, cv::Vec3d pose,
std::string type, int feature_nums): id_(frame_id), pose_(pose) {
  cv::Mat image = cv::imread(path2im, cv::IMREAD_GRAYSCALE);
  // cv::equalizeHist(image, image_);
  image_ = image.clone();
  mask_ = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
  // detect feature and compute descriptor
  cv::Ptr<cv::Feature2D> feature_detector;
  if (type == "ORB")
    feature_detector = cv::ORB::create(feature_nums);
  else if (type == "SURF")
    feature_detector = cv::xfeatures2d::SURF::create(200, 4, 3, true);
  else if (type == "SIFT")
    feature_detector = cv::SIFT::create(feature_nums);
  else if (type == "ORB2") {
    ORB_SLAM2::ORBextractor feature_detector(feature_nums, 1.2, 8, 20, 7);
    feature_detector(image_, mask_, keypoints_, descriptors_);
    cv::drawKeypoints(image_, keypoints_, image_);
    cv::cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
    cv::imshow("image_with_keypoints", image_);
    cv::waitKey(5);
    return;
  }

  feature_detector->detectAndCompute(image_, mask_, keypoints_, descriptors_);
  cv::drawKeypoints(image_, keypoints_, image_);
  cv::cvtColor(image_, image_, cv::COLOR_BGR2GRAY);
  // cv::imwrite("ORB2.png", image_);
  // cv::imshow("image_with_keypoints", image_);
}
