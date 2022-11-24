#include "visual_place_recognition/common/preprocessing.h"



cv::Mat patchNormalization(const cv::Mat &resized_image) {
  int length = 8;

  cv::Mat normalized_image = resized_image.clone();
  for (int i = 0; i < resized_image.rows / length; ++i) {
    for (int j = 0; j < resized_image.cols / length; ++j) {
      cv::Mat patch = resized_image.rowRange(length*i, length*(i+1))
        .colRange(length*j, length*(j+1)).clone();
      // std::cout << "patch image: " << patch << std::endl;
      cv::Mat mean, stddev;
      cv::meanStdDev(patch, mean, stddev);
      double mean_value, stddev_value;
      mean_value = mean.at<double>(0,0);
      stddev_value = stddev.at<double>(0,0);
      // std::cout << "mean: " << mean << " stddev: " << stddev << std::endl;

      patch.convertTo(patch, CV_32FC1);
      // substract mean
      cv::Mat temp;
      cv::subtract(patch, cv::Mat(length, length, CV_32FC1, mean_value), temp);
      // divide by std
      cv::divide(temp, cv::Mat(length, length, CV_32FC1, stddev_value), temp);
      // add 127
      cv::add(temp, cv::Mat(length, length, CV_32FC1, 127.0), temp);
      temp.convertTo(patch, CV_8UC1);
      // std::cout << "patch image2: " << patch << std::endl;

      patch.copyTo(normalized_image.rowRange(length*i, length*(i+1))
        .colRange(length*j, length*(j+1)));
    }
  }
  return normalized_image;
}

cv::Mat normalization(const cv::Mat &resized_image) {
  cv::Mat normalized_image = resized_image.clone();
  
  cv::Mat mean, stddev;
  cv::meanStdDev(resized_image, mean, stddev);
  double mean_value, stddev_value;
  mean_value = mean.at<double>(0,0);
  stddev_value = stddev.at<double>(0,0);

  normalized_image.convertTo(normalized_image, CV_32FC1);

  // substract mean
  cv::Mat temp;
  cv::subtract(normalized_image, cv::Mat(normalized_image.size(), CV_32FC1, mean_value), temp);
  // divide by std
  cv::divide(temp, cv::Mat(normalized_image.size(), CV_32FC1, stddev_value), temp);
  // add 127
  cv::add(temp, cv::Mat(normalized_image.size(), CV_32FC1, 127.0), temp);
  temp.convertTo(normalized_image, CV_8UC1);

  return normalized_image;
}