//
// Created by saleh on 1/22/22.
//
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <memory>
#include "GlobalHelpers.h"
#include "argparse.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

using namespace std;

spdlog::logger *logger;
string globalArgDataPath;
unsigned globalBatchsize;
bool globalDumpTensors=false;

void Handler(int sig) {
  void *array[40];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 40);

  // print out all the frames to stderr
  cerr<<"The host program has crashed, printing call stack:\n";
  cerr<<"Error: signal "<< sig<<"\n";
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  SPDLOG_LOGGER_CRITICAL(logger,"The host program has crashed.");
  spdlog::shutdown();

  exit(SIGSEGV);
}

void HandlerInt(int sig_no)
{
  SPDLOG_LOGGER_CRITICAL(logger,"CTRL+C pressed, terminating...");
  spdlog::shutdown();
  exit(SIGINT);
}

void SetupModules(int argc, const char* argv[]){
  signal(SIGSEGV, Handler);
  signal(SIGABRT, Handler);
  signal(SIGINT, HandlerInt);

  ArgumentParser parser("Inference");

  parser.add_argument("-d", "--data", "Dump directory", true);
  parser.add_argument("-b", "--batchsize", "Batch-size", false);
  parser.add_argument(
      "-k",
      "--dumptensors",
      "Dump tensors into *.npy files in the dump directory "
      "(no value is needed for this argument)",
      false
      );
  parser.add_argument(
      "-n",
      "--nolog",
      "Disable logging (no value is needed for this argument)",
      false
      );

  try {
    parser.parse(argc, argv);
  } catch (const ArgumentParser::ArgumentNotFound& ex) {
    std::cout << ex.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  if (parser.is_help()) exit(EXIT_SUCCESS);

  {
    // HOST LOGGER
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::trace);
    console_sink->set_pattern("[%H:%M:%S.%e][%^%l%$] %v");

    auto file_sink1 = std::make_shared<spdlog::sinks::basic_file_sink_mt>("hostlog_0trace.log", true);
    file_sink1->set_level(spdlog::level::trace);
    file_sink1->set_pattern("[%H:%M:%S.%f][%^%l%$][source %s][function %!][line %#] %v");


    auto file_sink3 = std::make_shared<spdlog::sinks::basic_file_sink_mt>("hostlog_3wraning.log", true);
    file_sink3->set_level(spdlog::level::warn);
    file_sink3->set_pattern("[%H:%M:%S.%f][%^%l%$][source %s][function %!][line %#] %v");

    logger = new spdlog::logger("Inference", {console_sink, file_sink1, file_sink3});
    logger->set_level(spdlog::level::trace);

    if(parser.exists("n")) {
      logger->set_level(spdlog::level::off);
    }
  }

  if(parser.exists("b")) {
    globalBatchsize = parser.get<unsigned>("b");
  }else{
    globalBatchsize = 100;
  }
  SPDLOG_LOGGER_INFO(logger,"Batch-size: {}", globalBatchsize);

  if(parser.exists("d")) {
    globalArgDataPath = parser.get<string>("d");
    SPDLOG_LOGGER_INFO(logger,"Dump Directory: {}", globalArgDataPath);
  }

  if(parser.exists("dumptensors")) {
    globalDumpTensors = true;
    SPDLOG_LOGGER_INFO(logger,"Tensors will be dumped into separate numpy files in the data directory.");
  }
}

