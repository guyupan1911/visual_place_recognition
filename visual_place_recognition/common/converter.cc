#include "visual_place_recognition/common/converter.h"

cv::Mat Converter::ImageMsgToCvMat(const Sensor::Image &image_msg) {
  
}

Sensor::Image Converter::cvMatToImageMsg(const cv::Mat &mat) {
  Sensor::Image serializableMat;

  serializableMat.set_rows(mat.rows);
  serializableMat.set_cols(mat.cols);
  serializableMat.set_elt_type(mat.type());
  serializableMat.set_elt_size((int)mat.elemSize());

  size_t dataSize = mat.rows * mat.cols * mat.elemSize();
  serializableMat.set_mat_data(mat.data, dataSize);

  return serializableMat;
}

Map::Frame Converter::FrameToFrameMsg(const Frame &frame) {
  Map::Frame serializableFrame;

  // set id
  serializableFrame.set_id(frame.id_);
  
  // set url_to_image
  serializableFrame.set_url_to_raw_image(frame.url_to_raw_image_);

}