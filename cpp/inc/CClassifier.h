//
// Created by saleh on 1/22/22.
//

#pragma once

#include <string>
#include "CTensor.h"
#include "cnpy.h"

using namespace std;

class CClassifier {
 public:
  CClassifier(string &dataDir, int batchSize, bool dumpTensors);
  int Prepare();

 protected:
  int PreloadWeights();
  int PreloadBiases();
  int PreloadTestSet();
  bool CheckValidityOfNumpyData();
  int PreloadAll();

 private:
  string m_strDataDir;
  bool m_bDumpTensors;
  int m_iBatchSize;
  vector<CTensorPtr<float>> m_vWeights;
  vector<CTensorPtr<float>> m_vBiases;
  CTensorPtr<float> m_oTestSetData;
  CTensorPtr<float> m_oTestSetLabel;
  CTensorPtr<float> m_oConfirmation;
  std::vector<cnpy::NpyArray> m_vNumpyBuff;

};

