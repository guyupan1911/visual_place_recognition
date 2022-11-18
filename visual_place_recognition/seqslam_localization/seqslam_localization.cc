#include "visual_place_recognition/seqslam_localization/seqslam_localization.h"
#include "visual_place_recognition/common/converter.h"


void SeqSlamLocalization::mapping(const std::vector<Frame> &frames) {
  std::cout << "train frame size: " << frames.size() << std::endl;
  for (const auto &elem : frames) {
    Map::Frame seriablized_frame = Converter::FrameToFrameMsg(elem);
  }
}