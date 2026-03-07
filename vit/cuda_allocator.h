#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "macros.h"

enum class OutputAllocKind : std::uint8_t { kCudaMallocAsync, kCudaMallocManaged, kCuMem };

class CudaOutputAllocator final : public nvinfer1::IOutputAllocator {
   public:
    static std::unique_ptr<CudaOutputAllocator> Create(cudaStream_t stream, int device = 0);

    explicit CudaOutputAllocator(cudaStream_t stream, OutputAllocKind kind, int device = 0);
    ~CudaOutputAllocator() override;

#if TRT_VERSION < 10000
    void* reallocateOutput(const char* tensorName, void* currentMemory, uint64_t size,
                           uint64_t alignment) TRT_NOEXCEPT override;
#else
    void* reallocateOutputAsync(const char* tensorName, void* currentMemory, uint64_t size, uint64_t alignment,
                                cudaStream_t stream) TRT_NOEXCEPT override;
#endif
    void notifyShape(const char* tensorName, nvinfer1::Dims const& dims) TRT_NOEXCEPT override;

    void* getBuffer(const std::string& tensorName) const;
    std::size_t getSize(const std::string& tensorName) const;
    OutputAllocKind kind() const;

   private:
    struct Allocation;
    Allocation allocate(std::size_t size);
    void release(const std::string& tensorName, Allocation& alloc);

    cudaStream_t stream_{};
    OutputAllocKind kind_{OutputAllocKind::kCudaMallocManaged};
    int device_{0};
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Allocation> allocations_;
};
