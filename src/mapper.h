/*
 -------------------------------------------------------------------------------
 * Frame header file
 *
 * Copyright (C) 2022 AutoX, Inc.
 * Author: Guyu Pan (yuxuanhuang@autox.ai)
 -------------------------------------------------------------------------------
 */

#ifndef MAPPER_H_
#define MAPPER_H_

#include <vector>

class Mapper {
 public:
  virtual void mapping() = 0;
  virtual void save() = 0;
};

#endif  // MAPPER_H_