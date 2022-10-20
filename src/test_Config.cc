#include "opencv4/opencv2/opencv.hpp"

int main() {
  cv::FileStorage fs_write("config.yaml", cv::FileStorage::WRITE);
  fs_write << "feature_type" << "ORB2";
  fs_write << "feature_size" << 2000;
}