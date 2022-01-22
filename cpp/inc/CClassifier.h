//
// Created by saleh on 1/22/22.
//

#pragma once

#include <string>
#include "CTensor.h"

using namespace std;

class CClassifier {
 public:
  CClassifier(
      string &dataDir,
      int batchSize,
      bool dumpTensors);


 private:
  string m_strDataDir;
  bool m_bDumpTensors;
  int m_iBatchSize;
};

