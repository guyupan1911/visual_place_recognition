#include "opencv4/opencv2/opencv.hpp"

#include "visual_place_recognition/common/proto/image.pb.h"

static cv::Mat ImageMsgToCvMat(const Sensor::Image &image_msg) {
  
}

static Sensor::Image cvMatToImageMsg(const cv::Mat &mat) {
  Sensor::Image serializableMat;

  serializableMat.set_rows(mat.rows);
  serializableMat.set_cols(mat.cols);
  serializableMat.set_elt_type(mat.type());
  serializableMat.set_elt_size((int)mat.elemSize());

  size_t dataSize = mat.rows * mat.cols * mat.elemSize();
  serializableMat.set_mat_data(mat.data, dataSize);

  return serializableMat;
} 