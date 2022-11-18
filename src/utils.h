#pragma once

#include <vector>

#include "opencv4/opencv2/opencv.hpp"

void stitch_images(const std::vector<cv::Mat> &images, int rows, int cols,
  cv::Mat& im_out) {
    int row = images[0].rows;
    int col = images[0].cols;
    
    im_out = cv::Mat::zeros(row * rows, col * cols, images[0].type());

    for(int i = 0; i < images.size(); ++i) {
      int r = (i / cols);
      int c = (i % cols);
      images[i].copyTo(im_out.rowRange(r*row,(r+1)*row).colRange(c*col,(c+1)*col));
    }
}