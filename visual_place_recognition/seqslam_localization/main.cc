#include <iostream>

#include "opencv4/opencv2/opencv.hpp"

#include "visual_place_recognition/common/dump.h"
#include "visual_place_recognition/common/frame.h"
#include "visual_place_recognition/common/proto/image.pb.h"
#include "visual_place_recognition/common/convert.h"


int main() {
  std::vector<Frame> vFrames_train, vFrames_test;
  dumpAutoX(FLAGS_autox_train_images_dir, vFrames_train);
  dumpAutoX(FLAGS_autox_test_images_dir, vFrames_test);

  for(auto& elem : vFrames_train) {
    Sensor::Image image_msg = cvMatToImageMsg(elem.raw_image_);
  }
}