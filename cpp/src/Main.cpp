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

  classifier = new CClassifier(globalArgDataPath, globalBatchsize, globalDumpTensors);
  SPDLOG_LOGGER_TRACE(logger, "The forward pass has finished.");
  delete(classifier);
  SPDLOG_LOGGER_TRACE(logger, "Closing.");
}