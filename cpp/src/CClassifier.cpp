//
// Created by saleh on 1/22/22.
//

#include "CClassifier.h"

CClassifier::CClassifier(string &dataDir, int batchSize, bool dumpTensors) {
  m_strDataDir = dataDir;
  m_iBatchSize = batchSize;
  m_bDumpTensors = dumpTensors;
}
