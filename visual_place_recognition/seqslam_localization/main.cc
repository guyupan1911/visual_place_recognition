#include <iostream>

#include "visual_place_recognition/seqslam_localization/seqslam_localization.h"
#include "visual_place_recognition/common/dump.h"

int main() {
  std::vector<Frame> vFrames_train, vFrames_test;
  dumpAutoX(FLAGS_autox_train_images_dir, vFrames_train);
  dumpAutoX(FLAGS_autox_test_images_dir, vFrames_test);

  SeqSlamLocalization system;
  system.mapping(vFrames_train);

}