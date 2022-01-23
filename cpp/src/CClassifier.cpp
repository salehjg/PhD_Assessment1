//
// Created by saleh on 1/22/22.
//

#include "CClassifier.h"

/**
 * Constructor, it does not do any preparation, please use Prepare().
 * @param dataDir Path to the dump directory of the repository with the trailing `/`
 * @param batchSize Batch-size of the inference run
 * @param dumpTensors true: dump tensors, false: dont dump any tensors
 */
CClassifier::CClassifier(string &dataDir, int batchSize, bool dumpTensors) {
  m_strDataDir = dataDir + "/";
  m_iBatchSize = batchSize;
  m_bDumpTensors = dumpTensors;
}

/**
 * Pre-loads the weights.
 * @return 0 on success
 */
int CClassifier::PreloadWeights() {
  try {
    vector<string> vecPath;
    vecPath.push_back("0.conv2d_1.weight.npy");
    vecPath.push_back("1.conv2d_2.weight.npy");
    vecPath.push_back("3.conv2d_3.weight.npy");
    vecPath.push_back("4.conv2d_4.weight.npy");
    vecPath.push_back("7.dense_1.weight.npy");
    vecPath.push_back("8.dense_2.weight.npy");

    for (string &fName: vecPath) {
      std::string npyPath = m_strDataDir + "weights/numpy_npy/" + fName;
      m_vNumpyBuff.push_back(cnpy::npy_load(npyPath));
      std::vector<unsigned> __shape(m_vNumpyBuff.back().shape.begin(), m_vNumpyBuff.back().shape.end());
      m_vWeights.push_back(
          CTensorPtr<float>(
              new CTensor<float>(__shape, m_vNumpyBuff.back().data<float>())
          )
      );
    }

    return 0;
  }catch(exception &e){
    return 1;
  }
}

/**
 * Pre-loads the biases.
 * @return 0 on success
 */
int CClassifier::PreloadBiases() {
  try{
    vector<string> vecPath;
    vecPath.push_back("0.conv2d_1.bias.npy");
    vecPath.push_back("1.conv2d_2.bias.npy");
    vecPath.push_back("3.conv2d_3.bias.npy");
    vecPath.push_back("4.conv2d_4.bias.npy");
    vecPath.push_back("7.dense_1.bias.npy");
    vecPath.push_back("8.dense_2.bias.npy");

    for(string &fName: vecPath){
      std::string npyPath = m_strDataDir + "weights/numpy_npy/" + fName;
      m_vNumpyBuff.push_back(cnpy::npy_load(npyPath));
      std::vector<unsigned> __shape(m_vNumpyBuff.back().shape.begin(), m_vNumpyBuff.back().shape.end());
      m_vBiases.push_back(
          CTensorPtr<float>(
              new CTensor<float>(__shape, m_vNumpyBuff.back().data<float>())
          )
      );
    }

    return 0;
  }catch(exception &e){
    return 1;
  }
}

/**
 * Pre-loads the dataset along with the confirmation tensor.
 * @return 0 on success
 */
int CClassifier::PreloadTestSet() {
  try{
    vector<string> vecPath;
    vecPath.push_back("input_test_data.npy");
    vecPath.push_back("input_test_label.npy");
    vecPath.push_back("confirmation.npy");

    int i=0;

    for(string &fName: vecPath){
      std::string npyPath = m_strDataDir + fName;
      m_vNumpyBuff.push_back(cnpy::npy_load(npyPath));
      std::vector<unsigned> __shape(m_vNumpyBuff.back().shape.begin(), m_vNumpyBuff.back().shape.end());
      if(i==0) {
        m_oTestSetData = CTensorPtr<float>(
            new CTensor<float>(__shape, m_vNumpyBuff.back().data<float>())
        );
      }else{
        if(i==1){
          m_oTestSetLabel = CTensorPtr<float>(
              new CTensor<float>(__shape, m_vNumpyBuff.back().data<float>())
          );
        }else{
          m_oConfirmation = CTensorPtr<float>(
              new CTensor<float>(__shape, m_vNumpyBuff.back().data<float>())
          );
        }
      }
      i++;
    }

    return 0;
  }catch(exception &e){
    return 1;
  }
}

/**
 * Pre-loads the weights, the biases, the confirmation tensor, and the dataset.
 * @return 0 on success
 */
int CClassifier::PreloadAll() {
  int retVal = 0;
  retVal += PreloadWeights();
  retVal += PreloadBiases();
  retVal += PreloadTestSet();
  return retVal;
}

/**
 * Checks the content of the loaded tensor with a known pattern to make sure
 * that everything is working as they should.
 * @return true on success
 */
bool CClassifier::CheckValidityOfNumpyData() {
  auto shape = m_oConfirmation->GetShape();
  float *ptrBuff = m_oConfirmation->Get();
  if(shape.size()!=1) return false;
  if(shape[0]!=1234) return false;
  for(unsigned i=0; i<shape[1]; i++){
    if((float)i != ptrBuff[i]){
      return false;
    }
  }
  return true;
}

/**
 * Prepares the classifier (loads the necessary weights, biases, and dataset;
 *                                          along with doing some sanity checks)
 * @return returns 0 on success
 */
int CClassifier::Prepare() {
  if(PreloadAll()!=0) return 1;
  if(!CheckValidityOfNumpyData()) return 2;
  return 0;
}

