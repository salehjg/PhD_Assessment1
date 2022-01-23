//
// Created by saleh on 1/22/22.
//

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <iostream>
#include <unistd.h>
#include <csignal>
#include <GlobalHelpers.h>
#include "spdlog/spdlog.h"
#include "CClassifier.h"

CClassifier *classifier;

int main(int argc, const char* argv[]) {
  SetupModules(argc, argv);

  classifier = new CClassifier(globalArgDataPath, globalDumpTensors);

  if(classifier->Prepare()!=0){
    SPDLOG_LOGGER_ERROR(logger, "Failed to pass the preparation phase.");
  }

  classifier->Inference();

  SPDLOG_LOGGER_TRACE(logger, "The forward pass has finished.");
  delete(classifier);
  SPDLOG_LOGGER_TRACE(logger, "Closing.");
}