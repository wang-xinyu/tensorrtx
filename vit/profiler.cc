#include "profiler.h"
#include <NvInfer.h>
#include <algorithm>
#include <iomanip>
#include <string>

void Profiler::reportLayerTime(const char* layerName, float ms) noexcept {
    mProfile[layerName].count++;
    mProfile[layerName].time += ms;
    if (std::find(mLayerNames.begin(), mLayerNames.end(), layerName) == mLayerNames.end()) {
        mLayerNames.emplace_back(layerName);
    }
}

Profiler::Profiler(const char* name, const std::vector<Profiler>& srcProfilers) : mName(name) {
    for (const auto& srcProfiler : srcProfilers) {
        for (const auto& rec : srcProfiler.mProfile) {
            auto it = mProfile.find(rec.first);
            if (it == mProfile.end()) {
                mProfile.insert(rec);
            } else {
                it->second.time += rec.second.time;
                it->second.count += rec.second.count;
            }
        }
    }
}

std::ostream& operator<<(std::ostream& out, const Profiler& value) {
    out << "========== " << value.mName << " ==========\n";
    float totalTime = 0;
    std::string layerNameStr = "TensorRT layer name";
    int maxLayerNameLength = std::max(static_cast<int>(layerNameStr.size()), 70);
    for (const auto& elem : value.mProfile) {
        totalTime += elem.second.time;
        maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
    }

    auto old_settings = out.flags();
    auto old_precision = out.precision();
    // Output header
    {
        out << std::setfill(' ') << std::setw(maxLayerNameLength) << layerNameStr << " ";
        out << std::setw(12) << "Runtime, " << "%" << " ";
        out << std::setw(12) << "Invocations" << " ";
        out << std::setw(12) << "Runtime, ms\n";
    }
    for (size_t i = 0; i < value.mLayerNames.size(); i++) {
        const std::string layerName = value.mLayerNames[i];
        auto elem = value.mProfile.at(layerName);
        out << std::setw(maxLayerNameLength) << layerName << " ";
        out << std::setw(12) << std::fixed << std::setprecision(1) << (elem.time * 100.0F / totalTime) << "%" << " ";
        out << std::setw(12) << elem.count << " ";
        out << std::setw(12) << std::fixed << std::setprecision(2) << elem.time << "\n";
    }
    out.flags(old_settings);
    out.precision(old_precision);
    out << "========== " << value.mName << " total runtime = " << totalTime << " ms ==========\n";

    return out;
}