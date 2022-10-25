// create by ausk(jinlj) 2022/10/25
#pragma once

#include "NvInferRuntimeCommon.h"
#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

using Severity = nvinfer1::ILogger::Severity;

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity < Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};
