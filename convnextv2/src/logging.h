#ifndef LOGGING_H
#define LOGGING_H

#include <NvInfer.h>
#include <iostream>

using namespace nvinfer1;

class Logger : public ILogger {
   public:
    Logger(Severity severity = Severity::kINFO) : reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) noexcept override {
        if (severity > reportableSeverity)
            return;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cout << "INFO: ";
                break;
            default:
                std::cout << "VERBOSE: ";
                break;
        }
        std::cout << msg << std::endl;
    }

    Severity reportableSeverity;
};

#endif
