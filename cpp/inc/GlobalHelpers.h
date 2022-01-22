//
// Created by saleh on 1/22/22.
//

#pragma once

#include <string>
#include <iostream>
#include <execinfo.h>
#include <unistd.h>
#include <csignal>
#include "spdlog/spdlog.h"
#include "CStringFormatter.h"

extern spdlog::logger *logger;
extern std::string globalArgDataPath;
extern unsigned globalBatchsize;
extern bool globalDumpTensors;

extern void SetupModules(int argc, const char* argv[]);

#define ThrowException(arg) \
    {\
    std::string _msg = CStringFormatter()<<__FILE__<<":"<<__LINE__<<": "<<arg;\
    throw std::runtime_error(_msg); \
    SPDLOG_LOGGER_ERROR(logger,_msg);\
    exit(EXIT_FAILURE);\
    }

#define ConditionCheck(condition,msgIfFalse) \
    if(!(condition)){\
    std::string _msg = CStringFormatter()<<__FILE__<<":"<<__LINE__<<": Failed "<< #condition <<": "<<msgIfFalse;\
    throw std::runtime_error(_msg); \
    SPDLOG_LOGGER_ERROR(logger,_msg);\
    exit(EXIT_FAILURE);\
    }
