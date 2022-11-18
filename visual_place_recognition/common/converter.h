#include "opencv4/opencv2/opencv.hpp"

#include "visual_place_recognition/common/proto/image.pb.h"
#include "visual_place_recognition/common/proto/frame.pb.h"

#include "visual_place_recognition/common/frame.h"

class Converter{
 public:
  static cv::Mat ImageMsgToCvMat(const Sensor::Image &image_msg);

  static Sensor::Image cvMatToImageMsg(const cv::Mat &mat);

  static Map::Frame FrameToFrameMsg(const Frame &frame);
};
