#include <NvInfer.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

class Profiler final : public nvinfer1::IProfiler {
   public:
    struct Record {
        float time{0};
        int count{0};
    };
    Profiler(const char* name, const std::vector<Profiler>& srcProfilers = std::vector<Profiler>());
    void reportLayerTime(const char* layerName, float ms) noexcept override;
    friend std::ostream& operator<<(std::ostream& out, const Profiler& value);

   private:
    std::string mName;
    std::vector<std::string> mLayerNames;
    std::map<std::string, Record> mProfile;
};
