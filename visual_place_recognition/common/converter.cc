#include "visual_place_recognition/common/converter.h"

cv::Mat Converter::ImageMsgToCvMat(const Sensor::Image &image_msg) {
  cv::Mat im;
  im.create(image_msg.rows(), image_msg.cols(), image_msg.elt_type());
  size_t dataSize = image_msg.rows() * image_msg.cols() * image_msg.elt_size();
  std::copy(reinterpret_cast<uchar*>(const_cast<char*>(image_msg.mat_data().data())),
    reinterpret_cast<uchar*>(const_cast<char*>(image_msg.mat_data().data())+dataSize),
    im.data);
  return im;
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

  // set resized_image
  Sensor::Image serializableResizedImage = cvMatToImageMsg(frame.resized_image_);
  serializableFrame.mutable_resized_image()->CopyFrom(serializableResizedImage);

  // set gnss_pose
  Sensor::GnssPose serializableGnssPose;
  serializableGnssPose.set_x(frame.gnss_pose_[0]);
  serializableGnssPose.set_y(frame.gnss_pose_[1]);
  serializableGnssPose.set_z(frame.gnss_pose_[2]);
  serializableFrame.mutable_gnss_pose()->CopyFrom(serializableGnssPose);

  return serializableFrame;
}

Frame Converter::FrameMsgToFrame(const Map::Frame &frame_msg) {
  cv::Mat resized_image = ImageMsgToCvMat(frame_msg.resized_image());
  cv::Vec3d gnss_pose = {frame_msg.gnss_pose().x(), frame_msg.gnss_pose().y(),
    frame_msg.gnss_pose().z()};
  Frame temp{frame_msg.url_to_raw_image(), frame_msg.id(), resized_image,
    gnss_pose};
  return temp;
}
