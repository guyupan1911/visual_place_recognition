#include "bowMapper.h"

void BowMapper::mapping(const std::vector<MetaData>& mapping_data) {
  std::ifstream fin("voc.yml.gz");
  if (!fin) {
    std::cout << "train a vocabulary" << std::endl;
    trainVocabulary(mapping_data);
  }

  std::ifstream fin2, fin3;
  fin2.open("db.yml.gz");
  fin3.open("entry2frame.bin");

  if (!fin2 && !fin3) {
    // create a database
    createDataBase(mapping_data);
  }
}

void BowMapper::trainVocabulary(const std::vector<MetaData>& mapping_data) {
  // load config file
  cv::FileStorage fs_read("config.yaml", cv::FileStorage::READ);
  std::string feature_type;
  int feature_size;
  fs_read["feature_type"] >> feature_type;
  fs_read["feature_size"] >> feature_size;

  std::vector<cv::Mat> vDescriptors;
  for (auto& elem : mapping_data) {
    cv::Vec3d pose{elem.gnss_pose[0], elem.gnss_pose[1], elem.gnss_pose[2]};
    Frame frame(elem.image_id, elem.path_to_image, pose, feature_type,
     feature_size);
    vFrames_.push_back(frame);
    vDescriptors.push_back(frame.getDes());
  }

  // train vocabulary
  const int k = 10;
  const int L = 4;
  const DBoW3::WeightingType weight = DBoW3::TF_IDF;
  const DBoW3::ScoringType score = DBoW3::L1_NORM;

  DBoW3::Vocabulary voc(k, L, weight, score);

  std::cout << "Creating a small " << k << "^" << L << " vocabulary..."
  << std::endl;
  voc.create(vDescriptors);
  std::cout << "... done!" << std::endl;

  std::cout << std::endl << "Saving vocabulary..." << std::endl;
  std::string path_to_voc = "voc.yml.gz";
  voc.save(path_to_voc);
  std::cout << "Done" << std::endl;
}

void BowMapper::createDataBase(const std::vector<MetaData>& mapping_data) {
  // load config file
  cv::FileStorage fs_read("config.yaml", cv::FileStorage::READ);
  std::string feature_type;
  int feature_size;
  fs_read["feature_type"] >> feature_type;
  fs_read["feature_size"] >> feature_size;
  
  // load the vocabulary from disk
  std::string path_to_voc = "voc.yml.gz";
  DBoW3::Vocabulary voc(path_to_voc);

  DBoW3::Database db(voc, false, 0);  // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for (auto& elem : mapping_data) {
    cv::Vec3d pose{elem.gnss_pose[0], elem.gnss_pose[1], elem.gnss_pose[2]};
    Frame frame(elem.image_id, elem.path_to_image, pose, feature_type,
      feature_size);
    DBoW3::EntryId entryId = db.add(frame.getDes());
    databaseID_to_FrameID_.insert(std::make_pair(entryId, elem.image_id));
  }

  db.save("db.yml.gz");
  saveEntry2Frame("entry2frame.bin");
  std::cout << "... done!" << std::endl;
}

void BowMapper::saveEntry2Frame(std::string filename) {
  std::ofstream fout("entry2frame.bin");
  unsigned int size = databaseID_to_FrameID_.size();
  fout.write((char*)&size, sizeof(size));
  for (auto& elem : databaseID_to_FrameID_) {
    fout.write((char*)&elem.first, sizeof(DBoW3::EntryId));
    fout.write((char*)&elem.second, sizeof(uint64_t));
  }
  fout.close();
}