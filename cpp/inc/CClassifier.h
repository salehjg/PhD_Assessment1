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
  CClassifier(string &dataDir, bool dumpTensors);
  int Prepare();
  void Inference();

 protected:
  int PreloadWeights();
  int PreloadBiases();
  int PreloadTestSet();
  bool CheckValidityOfNumpyData();
  int PreloadAll();
  CTensorPtr<float> LayerConv2D(CTensorPtr<float> inputTn, CTensorPtr<float> weightTn, CTensorPtr<float> biasTn, bool isValidPadding);
  CTensorPtr<float> LayerMaxPool2D(CTensorPtr<float> inputTn, const vector<unsigned> &poolSize);
  CTensorPtr<float> LayerFlatten(CTensorPtr<float> inputTn);


  void DumpTensor(CTensorPtr<float> inputTn, string nameTag);
  void DumpTensor(CTensorPtr<unsigned> inputTn, string nameTag);
  void DumpTensor(CTensorPtr<int> inputTn, string nameTag);

 private:
  string m_strDataDir;
  bool m_bDumpTensors;
  vector<CTensorPtr<float>> m_vWeights;
  vector<CTensorPtr<float>> m_vBiases;
  CTensorPtr<float> m_oTestSetData;
  CTensorPtr<float> m_oTestSetLabel;
  CTensorPtr<float> m_oConfirmation;
  std::vector<cnpy::NpyArray> m_vNumpyBuff;

  template <typename T>
  void DumpToNumpyFile(std::string npyFileName, CTensorPtr<T> inputTn, std::string npyDumpDir);
};

template<typename T>
void CClassifier::DumpToNumpyFile(std::string npyFileName, CTensorPtr<T> inputTn, std::string npyDumpDir) {
  auto shape = inputTn->GetShape();
  std::vector<unsigned long> _shape(shape.begin(), shape.end());
  cnpy::npy_save<T>(npyDumpDir+npyFileName, inputTn->Get(), _shape, "w");
}

