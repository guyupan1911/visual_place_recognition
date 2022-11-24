#pragma once

class SeqSlamConfig {
 public:
  int ds() {
    return ds_;
  }

  int r() {
    return r_;
  }

 private:
  // preprocessing

  // 
  int r_ = 20;
  
  // get match
  int ds_ = 10;
};