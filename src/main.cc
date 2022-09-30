#include "io.h"
#include "frame.h"
#include "DBoW3/DBoW3.h"
#include "opencv2/xfeatures2d.hpp"

using namespace DBoW3;

// map image id to image path
std::map<uint64_t, std::pair<std::string, cv::Vec3d>> train_images;
std::map<uint64_t, std::pair<std::string, cv::Vec3d>> test_images;

// map image id to frame
std::map<int64_t, Frame> train_frames, test_frames;

// map entry_id to train frame id
std::map<EntryId, int64_t> entry_id_to_frame_id;

// save descriptors in vector<Mat>
std::vector<cv::Mat> vDes_train, vDes_test;

cv::Mat mask;

bool isFileExists_ifstream(std::string name) {
  std::ifstream f(name.c_str());
  return f.good();
}

void testVocCreation() {
  if (isFileExists_ifstream("small_voc.yml.gz"))
      return;

  train_images = readImagesAndGNSS(
  "../dataset/train/recording_2020-12-22_12-04-35_images");

  // detect feature points
  int i = 0;
  for (auto& elem : train_images) {
    if (i%100 != 0)
      continue;
    Frame frame(elem.first, elem.second.first, elem.second.second, "ORB");
    train_frames.insert(std::make_pair(elem.first, frame));
    vDes_train.push_back(frame.getDes());
  }

  const int k = 10;
  const int L = 4;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;


  DBoW3::Vocabulary voc(k, L, weight, score);

  std::cout << "Creating a small " << k << "^" << L << " vocabulary..."
  << std::endl;
  voc.create(vDes_train);
  std::cout << "... done!" << std::endl;

  // cout << "Vocabulary information: " << endl
  //      << voc << endl << endl;

  // lets do something with this vocabulary
  // cout << "Matching images against themselves (0 low, 1 high): " << endl;
  // BowVector v1, v2;
  // for(size_t i = 0; i < features.size(); i++)
  // {
  //     voc.transform(features[i], v1);
  //     for(size_t j = 0; j < features.size(); j++)
  //     {
  //         voc.transform(features[j], v2);

  //         double score = voc.score(v1, v2);
  //         cout << "Image " << i << " vs Image " << j << ": " << score << endl;
  //     }
  // }

  // save the vocabulary to disk
  std::cout << std::endl << "Saving vocabulary..." << std::endl;
  voc.save("small_voc.yml.gz");
  std::cout << "Done" << std::endl;
}

void testDatabase() {
//    if(isFileExists_ifstream("small_db.yml.gz"))
//         return;

  train_images =
    readImagesAndGNSS("../dataset/train/recording_2020-12-22_12-04-35_images");

  // detect feature points
  int i = 0;
  for (auto& elem : train_images) {
    if (i % 5 != 0)
      continue;
    Frame frame(elem.first, elem.second.first, elem.second.second, "ORB");
    train_frames.insert(std::make_pair(elem.first, frame));
    vDes_train.push_back(frame.getDes());
  }

  std::cout << "Creating a small database..." << std::endl;

  // load the vocabulary from disk
  Vocabulary voc("small_voc.yml.gz");
  // Vocabulary voc("ORBvoc.txt");


  Database db(voc, false, 0);  // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for (auto& elem : train_frames) {
    EntryId eid = db.add(elem.second.getDes());
    entry_id_to_frame_id[eid] = elem.first;
  }

  std::cout << "... done!" << std::endl;

  // cout << "Database information: " << endl << db << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  std::cout << "Saving database..." << std::endl;
  db.save("small_db.yml.gz");
  std::cout << "... done!" << std::endl;

  // once saved, we can load it again
  // cout << "Retrieving database once again..." << endl;
  // Database db2("small_db.yml.gz");
  // cout << "... done! This is: " << endl << db2 << endl;

  // test_images =
  //     readImages("../dataset/test/recording_2021-02-25_13-39-06_images");

  // for(auto& elem : test_images)
  // {
  //     Frame frame(elem.first, elem.second);
  //     test_frames.insert(make_pair(elem.first,frame));
  //     vDes_test.push_back(frame.getDes());
  // }

  // QueryResults ret;
  // for(auto& elem : test_frames)
  // {   
  //     // std::cout << "mat: " << elem.second.getDes() << std::endl;
  //     db2.query(elem.second.getDes(), ret, 4);
  //     std::cout << "ret: " << ret << std::endl;
  //     // ret[0] is always the same image in this case, because we added it to the
  //     // database. ret[1] is the second best match.

  //     // cout << "Searching for Image " << i << ". " << ret << endl;
  //     // std::cout << "ret size: " << ret.size() << std::endl;
  //     // cv::Mat im_test = elem.second.getImage();
  //     // cv::Mat im_train = train_frames.find(entry_id_to_frame_id[ret[0].Id])->second.getImage();

  //     // // cv::Mat imCom(im_test.rows, 2*im_test.cols, CV_8UC3);
  //     // // im_test.copyTo(imCom.colRange(0, im_test.cols));
  //     // // im_train.copyTo(imCom.colRange(im_test.cols, 2*im_test.cols));

  //     // cv::imshow("im_test", im_test);
  //     // cv::imshow("im_train", im_train);

  //     // cv::waitKey(5);

  // }
}

void testQuery() {
  // once saved, we can load it again
  std::cout << "Retrieving database once again..." << std::endl;
  Database db2("small_db.yml.gz");
  std::cout << "... done! This is: " << std::endl;

  test_images =
    readImagesAndGNSS("../dataset/test/recording_2021-02-25_13-39-06_images");

  int i = 0;
  for (auto& elem : test_images) {
    if (i++ % 5 !=0)
        continue;
    Frame frame(elem.first, elem.second.first, elem.second.second, "ORB");
    test_frames.insert(std::make_pair(elem.first, frame));
    vDes_test.push_back(frame.getDes());
  }

  int success = 0;
  int failure = 0;
  QueryResults ret;
  for (auto& elem : test_frames) {
    // std::cout << "mat: " << elem.second.getDes() << std::endl;
    db2.query(elem.second.getDes(), ret, 1);
    // std::cout << "ret: " << ret << std::endl;
    // ret[0] is always the same image in this case, because we added it to the
    // database. ret[1] is the second best match.

    // cout << "Searching for Image " << i << ". " << ret << endl;
    // std::cout << "ret size: " << ret.size() << std::endl;
    cv::Mat im_test = elem.second.getImage();
    cv::Vec3d pose_test = elem.second.getPose();
    std::vector<cv::Mat> queryed_images;

    // GMS
    cv::Ptr<cv::Feature2D> f_detector = cv::ORB::create(10000);
    cv::BFMatcher bf_matcher(cv::NORM_HAMMING);

    std::vector<cv::DMatch> bf_matches, gms_matches;
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat des1, des2;
    f_detector->detectAndCompute(im_test, mask, kpts1, des1);

    for (int i=0; i < ret.size(); i++) {
      uint64_t frame_id = entry_id_to_frame_id[ret[i].Id];
      cv::Mat im_train = train_frames.find(frame_id)->second.getImage();
      cv::Vec3d pose_train = train_frames.find(frame_id)->second.getPose();

      // GMS
      if (0) {
        f_detector->detectAndCompute(im_train, mask, kpts2, des2);
        bf_matcher.match(des1, des2, bf_matches);

        cv::xfeatures2d::matchGMS(im_test.size(), im_train.size(),
            kpts1, kpts2, bf_matches, gms_matches);

        cv::Mat im_gms_match;
        cv::drawMatches(im_test, kpts1, im_train, kpts2, gms_matches,
            im_gms_match);
        cv::imshow("gms_match", im_gms_match);
        cv::waitKey(0);
      }


      // drawTextInfo
      std::stringstream s;
      cv::Vec3d dis = pose_train - pose_test;
      cv::Vec2d txy(dis[0], dis[1]);
      cv::Vec2d tz(dis[2]);
      s << "Score: " << ret[i].Score << " txy: " << cv::norm(txy, cv::NORM_L2)
                                      << " tz: " << cv::norm(tz, cv::NORM_L2)
                                      << std::endl;

      if (cv::norm(txy, cv::NORM_L2) < 5.0 && cv::norm(tz, cv::NORM_L2) < 1.0)
        success++;
      else
        failure++;

      int baseline = 0;
      cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN,
          1, 1, &baseline);
      cv::Mat imText = cv::Mat(im_train.rows+textSize.height+10,
          im_train.cols, im_train.type());
      im_train.copyTo(imText.rowRange(0, im_train.rows).colRange(0,
          im_train.cols));
      imText.rowRange(im_train.rows, imText.rows) =
          cv::Mat::zeros(textSize.height+10, im_train.cols, im_train.type());
      cv::putText(imText, s.str(), cv::Point(5, imText.rows-5),
          cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
      queryed_images.push_back(imText);
    }
    cv::Mat imCom =
        cv::Mat(queryed_images[0].rows, (ret.size()+1)*queryed_images[0].cols, queryed_images[0].type());
    im_test.copyTo(imCom.colRange(0, im_test.cols).rowRange(0,im_test.rows));
    for(int i=0; i<ret.size(); i++)
    {
        queryed_images[i].copyTo(imCom.colRange(im_test.cols*(i+1), im_test.cols*(i+2)));
    }

    // draw feature matches
    // cv::BFMatcher matcher(cv::NORM_HAMMING, true); 
    // vector<cv::DMatch> matches;

    // matcher.match(elem.second.getDes(), train_frames.find(entry_id_to_frame_id[ret[0].Id])->second.getDes(), matches);

    // ransac 
    // vector<cv::Point2f> vkpts1, vkpts2;
    // for (size_t i = 0; i < matches.size(); i++)
    // {
    //     vkpts1.push_back(elem.second.getKpts()[matches[i].queryIdx].pt);
    //     vkpts2.push_back(train_frames.find(entry_id_to_frame_id[ret[0].Id])->second.getKpts()[matches[i].trainIdx].pt);

    // }

    // vector<uchar> status;
    // cv::findFundamentalMat(vkpts1, vkpts2, cv::FM_RANSAC, 3, 0.99, status);

    // vector<cv::DMatch> good_matches;
    // for(int i=0; i < matches.size(); i++)
    // {
    //     if(status[i])
    //         good_matches.push_back(matches[i]);
    // }

    // cv::Mat img_match;

    // cv::drawMatches(im_test, elem.second.getKpts(), 
    //     train_frames.find(entry_id_to_frame_id[ret[0].Id])->second.getImage(),
    //     train_frames.find(entry_id_to_frame_id[ret[0].Id])->second.getKpts(),
    //     good_matches, img_match);



    static int index = 1;
    // cv::imshow("im_test", im_test);
    // cv::imshow("im_train", im_train);
    // cv::imshow("im_Compare", imCom);
    // cv::waitKey(0);
    // cv::imwrite(to_string(index)+".png", imCom);
    // cv::imshow("im_match", img_match);
    // cv::imwrite(to_string(index)+"match.png", img_match);
    index++;
  }
  std::cout << "success: " << success << " failure: " << failure << std::endl;
}

// display 4*4 train images
void draw_images() {
  train_images =
    readImages("../dataset/train/recording_2020-12-22_12-04-35_images");

  // detect feature points
  for (auto& elem : train_images) {
    Frame frame(elem.first, elem.second.first, elem.second.second, "ORB");
    train_frames.insert(std::make_pair(elem.first, frame));
    vDes_train.push_back(frame.getDes());
  }

  int step = train_frames.size() / 16;

  std::vector<Frame> collected_frames;
  int i = 1;
  for (auto& elem : train_frames) {
    if (i++ % step == 0)
      collected_frames.push_back(elem.second);
  }

  cv::Mat images = cv::Mat(4*collected_frames[0].getImage().rows,
                            4*collected_frames[0].getImage().cols,
                            collected_frames[0].getImage().type());

  for (int i=0; i < collected_frames.size(); i++) {
    int row = i / 4;
    int col = i % 4;

    collected_frames[i].getImage().copyTo
        (images.
        rowRange(row*collected_frames[0].getImage().rows, (row+1)*collected_frames[0].getImage().rows).
        colRange(col*collected_frames[0].getImage().cols, (col+1)*collected_frames[0].getImage().cols));
}

  cv::imshow("images", images);
  cv::waitKey(0);

  cv::imwrite("images.png", images);
}

void generateMask()
{
  cv::Mat mask = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
  std::cout << "rows: " << mask.rows << " cols: " << mask.cols << std::endl;
  for (int i=0; i < mask.rows; i++)  {
    for (int j=0; j < mask.cols; j++) {
      if (i < 2*mask.rows / 3)
        mask.at<char>(i, j) = 255;
      else
        mask.at<char>(i, j) = 0;
    }
  }
  cv::imwrite("mask1.png", mask);
}

int main()
{
  mask = cv::imread("mask1.png", cv::IMREAD_GRAYSCALE);

  testVocCreation();
  testDatabase();
  testQuery();

  // draw_images();

  // generateMask();
}
