#ifndef _TRT_COMMON_H_
#define _TRT_COMMON_H_
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>

using namespace std;

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

constexpr long double operator"" _GB(long double val) { return val * (1 << 30); }
constexpr long double operator"" _MB(long double val) { return val * (1 << 20); }
constexpr long double operator"" _KB(long double val) { return val * (1 << 10); }

// These is necessary if we want to be able to write 1_GB instead of 1.0_GB.
// Since the return type is signed, -1_GB will work as expected.
constexpr long long int operator"" _GB(long long unsigned int val) { return val * (1 << 30); }
constexpr long long int operator"" _MB(long long unsigned int val) { return val * (1 << 20); }
constexpr long long int operator"" _KB(long long unsigned int val) { return val * (1 << 10); }

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:

    Logger(): Logger(Severity::kWARNING) {}

    Logger(Severity severity): reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity{Severity::kWARNING};
};

// Locate path to file, given its filename or filepath suffix and possible dirs it might lie in
// Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path
inline std::string locateFile(const std::string& filepathSuffix, const std::vector<std::string>& directories)
{
    const int MAX_DEPTH{10};
    bool found{false};
    std::string filepath;

    for (auto& dir : directories)
    {
        filepath = dir + filepathSuffix;

        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(filepath);
            found = checkFile.is_open();
            if (found) break;
            filepath = "../" + filepath; // Try again in parent dir
        }

        if (found)
        {
            break;
        }

        filepath.clear();
    }

    if (filepath.empty()) {
        std::string directoryList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
            [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
        throw std::runtime_error("Could not find " + filepathSuffix + " in data directories:\n\t" + directoryList);
    }
    return filepath;
}

inline void readPGMFile(const std::string& fileName, uint8_t* buffer, int inH, int inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

namespace samples_common
{

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

inline bool isDebug()
{
    return (std::getenv("TENSORRT_DEBUG") ? true : false);
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj) {
            obj->destroy();
        }
    }
};

template <typename T>
inline std::shared_ptr<T> infer_object(T* obj)
{
    if (!obj) {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, InferDeleter());
}

template <class Iter>
inline std::vector<size_t> argsort(Iter begin, Iter end, bool reverse = false)
{
    std::vector<size_t> inds(end - begin);
    std::iota(inds.begin(), inds.end(), 0);
    if (reverse) {
        std::sort(inds.begin(), inds.end(), [&begin](size_t i1, size_t i2) {
            return begin[i2] < begin[i1];
        });
    }
    else
    {
        std::sort(inds.begin(), inds.end(), [&begin](size_t i1, size_t i2) {
            return begin[i1] < begin[i2];
        });
    }
    return inds;
}

inline bool readReferenceFile(const std::string& fileName, std::vector<std::string>& refVector)
{
    std::ifstream infile(fileName);
    if (!infile.is_open()) {
        cout << "ERROR: readReferenceFile: Attempting to read from a file that is not open." << endl;
        return false;
    }
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        refVector.push_back(line);
    }
    infile.close();
    return true;
}

template <typename result_vector_t>
inline std::vector<std::string> classify(const vector<string>& refVector, const result_vector_t& output, const size_t topK)
{
    auto inds = samples_common::argsort(output.cbegin(), output.cend(), true);
    std::vector<std::string> result;
    for (size_t k = 0; k < topK; ++k) {
        result.push_back(refVector[inds[k]]);
    }
    return result;
}

//...LG returns top K indices, not values.
template <typename T>
inline vector<size_t> topK(const vector<T> inp, const size_t k)
{
    vector<size_t> result;
    std::vector<size_t> inds = samples_common::argsort(inp.cbegin(), inp.cend(), true);
    result.assign(inds.begin(), inds.begin()+k);
    return result;
}

template <typename T>
inline bool readASCIIFile(const string& fileName, const size_t size, vector<T>& out)
{
    std::ifstream infile(fileName);
    if (!infile.is_open()) {
        cout << "ERROR readASCIIFile: Attempting to read from a file that is not open." << endl;
        return false;
    }
    out.clear();
    out.reserve(size);
    out.assign(std::istream_iterator<T>(infile), std::istream_iterator<T>());
    infile.close();
    return true;
}

template <typename T>
inline bool writeASCIIFile(const string& fileName, const vector<T>& in)
{
    std::ofstream outfile(fileName);
    if (!outfile.is_open()) {
        cout << "ERROR: writeASCIIFile: Attempting to write to a file that is not open." << endl;
        return false;
    }
    for (auto fn : in) {
        outfile << fn << " ";
    }
    outfile.close();
    return true;
}

inline void print_version()
{
//... This can be only done after statically linking this support into parserONNX.library
#if 0
    std::cout << "Parser built against:" << std::endl;
    std::cout << "  ONNX IR version:  " << nvonnxparser::onnx_ir_version_string(onnx::IR_VERSION) << std::endl;
#endif
    std::cout << "  TensorRT version: "
              << NV_TENSORRT_MAJOR << "."
              << NV_TENSORRT_MINOR << "."
              << NV_TENSORRT_PATCH << "."
              << NV_TENSORRT_BUILD << std::endl;
}

inline string getFileType(const string& filepath)
{
    return filepath.substr(filepath.find_last_of(".") + 1);
}

inline string toLower(const string& inp)
{
    string out = inp;
    std::transform(out.begin(), out.end(), out.begin(), ::tolower);
    return out;
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

// Struct to maintain command-line arguments.
struct Args
{
    bool runInInt8 = false;
};

// Populates the Args struct with the provided command-line parameters.
inline void parseArgs(Args& args, int argc, char* argv[])
{
    if (argc >= 1)
    {
        for (int i = 1; i < argc; ++i)
        {
            if (!strcmp(argv[i], "--int8")) args.runInInt8 = true;
        }
    }
}

template <int C, int H, int W>
struct PPM
{
    std::string magic, fileName;
    int h, w, max;
    uint8_t buffer[C * H * W];
};

struct BBox
{
    float x1, y1, x2, y2;
};

template <int C, int H, int W>
inline void writePPMFileWithBBox(const std::string& filename, PPM<C, H, W>& ppm, const BBox& bbox)
{
    std::ofstream outfile("./" + filename, std::ofstream::binary);
    assert(!outfile.fail());
    outfile << "P6" << "\n" << ppm.w << " " << ppm.h << "\n" << ppm.max << "\n";
    auto round = [](float x) -> int { return int(std::floor(x + 0.5f)); };
    const int x1 = std::min(std::max(0, round(int(bbox.x1))), W - 1);
    const int x2 = std::min(std::max(0, round(int(bbox.x2))), W - 1);
    const int y1 = std::min(std::max(0, round(int(bbox.y1))), H - 1);
    const int y2 = std::min(std::max(0, round(int(bbox.y2))), H - 1);
    for (int x = x1; x <= x2; ++x)
    {
        // bbox top border
        ppm.buffer[(y1 * ppm.w + x) * 3] = 255;
        ppm.buffer[(y1 * ppm.w + x) * 3 + 1] = 0;
        ppm.buffer[(y1 * ppm.w + x) * 3 + 2] = 0;
        // bbox bottom border
        ppm.buffer[(y2 * ppm.w + x) * 3] = 255;
        ppm.buffer[(y2 * ppm.w + x) * 3 + 1] = 0;
        ppm.buffer[(y2 * ppm.w + x) * 3 + 2] = 0;
    }
    for (int y = y1; y <= y2; ++y)
    {
        // bbox left border
        ppm.buffer[(y * ppm.w + x1) * 3] = 255;
        ppm.buffer[(y * ppm.w + x1) * 3 + 1] = 0;
        ppm.buffer[(y * ppm.w + x1) * 3 + 2] = 0;
        // bbox right border
        ppm.buffer[(y * ppm.w + x2) * 3] = 255;
        ppm.buffer[(y * ppm.w + x2) * 3 + 1] = 0;
        ppm.buffer[(y * ppm.w + x2) * 3 + 2] = 0;
    }
    outfile.write(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

} // namespace samples_common

#endif // _TRT_COMMON_H_
