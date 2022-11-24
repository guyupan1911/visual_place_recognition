#include <iostream>

#include "visual_place_recognition/seqslam_localization/seqslam_localization.h"
#include "visual_place_recognition/common/dump.h"

int main() {
  SeqSlamLocalization system;

  // system.fetchMap("../map/autox_train.bin", "../map/autox_test.bin");
  system.fetchMap("../map/seasons_train.bin", "../map/seasons_test.bin");

  system.run();

}