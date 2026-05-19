#include "cuda_allocator.h"
#include <cuda.h>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include "macros.h"
#include "utils.h"

namespace {
constexpr int kCudaVersionAsyncMin = 11020;
constexpr int kCudaVersionCuMemMin = 12000;
}  // namespace

struct CudaOutputAllocator::Allocation {
    void* ptr{nullptr};
    std::size_t size{0};
    OutputAllocKind kind{OutputAllocKind::kCudaMallocManaged};
    CUmemGenericAllocationHandle handle{};
    CUdeviceptr addr{};
    std::size_t mapped_size{0};
};

static auto getCudaRuntimeVersion() -> int {
    int version = 0;
    if (cudaRuntimeGetVersion(&version) != cudaSuccess) {
        return 0;
    }
    return version;
}

static auto getCudaDriverVersion() -> int {
    int version = 0;
    if (cudaDriverGetVersion(&version) != cudaSuccess) {
        return 0;
    }
    return version;
}

std::unique_ptr<CudaOutputAllocator> CudaOutputAllocator::Create(cudaStream_t stream, int device) {
    CHECK(cudaSetDevice(device));
    const int rt = getCudaRuntimeVersion();
    const int drv = getCudaDriverVersion();

    OutputAllocKind kind = OutputAllocKind::kCudaMallocManaged;
    if (rt >= kCudaVersionCuMemMin && drv >= kCudaVersionCuMemMin) {
        kind = OutputAllocKind::kCuMem;
    } else if (rt >= kCudaVersionAsyncMin) {
        kind = OutputAllocKind::kCudaMallocAsync;
    }
    return std::make_unique<CudaOutputAllocator>(stream, kind, device);
}

CudaOutputAllocator::CudaOutputAllocator(cudaStream_t stream, OutputAllocKind kind, int device)
    : stream_(stream), kind_(kind), device_(device) {}

CudaOutputAllocator::~CudaOutputAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& entry : allocations_) {
        release(entry.first, entry.second);
    }
}

#if TRT_VERSION < 10000
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void* CudaOutputAllocator::reallocateOutput(const char* tensorName, void* currentMemory, uint64_t size,
                                            uint64_t alignment) TRT_NOEXCEPT {
    (void)alignment;
    if (!tensorName) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto& alloc = allocations_[tensorName];
    if (alloc.ptr && size <= alloc.size) {
        return alloc.ptr;
    }
    if (alloc.ptr) {
        release(tensorName, alloc);
    } else if (currentMemory != nullptr && size == 0) {
        return currentMemory;
    }

    Allocation fresh = allocate(static_cast<std::size_t>(size));
    if (!fresh.ptr) {
        return nullptr;
    }
    alloc = fresh;
    return alloc.ptr;
}
#else
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void* CudaOutputAllocator::reallocateOutputAsync(const char* tensorName, void* currentMemory, uint64_t size,
                                                 uint64_t alignment, cudaStream_t stream) TRT_NOEXCEPT {
    (void)alignment;
    if (!tensorName) {
        return nullptr;
    }
    if (stream == nullptr) {
        stream = stream_;
    }
    stream_ = stream;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& alloc = allocations_[tensorName];
    if (alloc.ptr && size <= alloc.size) {
        return alloc.ptr;
    }
    if (alloc.ptr) {
        release(tensorName, alloc);
    } else if (currentMemory != nullptr && size == 0) {
        return currentMemory;
    }

    Allocation fresh = allocate(static_cast<std::size_t>(size));
    if (!fresh.ptr) {
        return nullptr;
    }
    alloc = fresh;
    return alloc.ptr;
}
#endif

void CudaOutputAllocator::notifyShape(const char* /*tensorName*/, nvinfer1::Dims const& /*dims*/) TRT_NOEXCEPT {}

CudaOutputAllocator::Allocation CudaOutputAllocator::allocate(std::size_t size) {
    Allocation alloc{};
    if (size == 0) {
        return alloc;
    }
    if (kind_ == OutputAllocKind::kCudaMallocAsync) {
        void* ptr = nullptr;
        if (cudaMallocAsync(&ptr, size, stream_) != cudaSuccess) {
            return alloc;
        }
        alloc.ptr = ptr;
        alloc.size = size;
        alloc.kind = OutputAllocKind::kCudaMallocAsync;
        return alloc;
    }
    if (kind_ == OutputAllocKind::kCudaMallocManaged) {
        void* ptr = nullptr;
        if (cudaMallocManaged(&ptr, size, cudaMemAttachGlobal) != cudaSuccess) {
            return alloc;
        }
        alloc.ptr = ptr;
        alloc.size = size;
        alloc.kind = OutputAllocKind::kCudaMallocManaged;
        return alloc;
    }

    if (cudaSetDevice(device_) != cudaSuccess) {
        return alloc;
    }
    if (cuInit(0) != CUDA_SUCCESS) {
        return alloc;
    }

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_;

    std::size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
        return alloc;
    }

    const std::size_t alloc_size = ((size + granularity - 1) / granularity) * granularity;
    CUmemGenericAllocationHandle handle{};
    if (cuMemCreate(&handle, alloc_size, &prop, 0) != CUDA_SUCCESS) {
        return alloc;
    }

    CUdeviceptr addr = 0;
    if (cuMemAddressReserve(&addr, alloc_size, 0, 0, 0) != CUDA_SUCCESS) {
        cuMemRelease(handle);
        return alloc;
    }

    if (cuMemMap(addr, alloc_size, 0, handle, 0) != CUDA_SUCCESS) {
        cuMemAddressFree(addr, alloc_size);
        cuMemRelease(handle);
        return alloc;
    }

    CUmemAccessDesc access_desc{};
    access_desc.location = prop.location;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(addr, alloc_size, &access_desc, 1) != CUDA_SUCCESS) {
        cuMemUnmap(addr, alloc_size);
        cuMemAddressFree(addr, alloc_size);
        cuMemRelease(handle);
        return alloc;
    }
    static_assert(sizeof(void*) == sizeof(CUdeviceptr));
    alloc.ptr = reinterpret_cast<void*>(addr);  // NOLINT(performance-no-int-to-ptr)
    alloc.size = size;
    alloc.kind = OutputAllocKind::kCuMem;
    alloc.handle = handle;
    alloc.addr = addr;
    alloc.mapped_size = alloc_size;
    return alloc;
}

void CudaOutputAllocator::release(const std::string& /*tensorName*/, Allocation& alloc) {
    if (!alloc.ptr) {
        return;
    }
    if (alloc.kind == OutputAllocKind::kCudaMallocAsync) {
        cudaFreeAsync(alloc.ptr, stream_);
    } else if (alloc.kind == OutputAllocKind::kCudaMallocManaged) {
        cudaFree(alloc.ptr);
    } else if (alloc.kind == OutputAllocKind::kCuMem) {
        cuMemUnmap(alloc.addr, alloc.mapped_size);
        cuMemRelease(alloc.handle);
        cuMemAddressFree(alloc.addr, alloc.mapped_size);
    }
    alloc = Allocation{};
}

void* CudaOutputAllocator::getBuffer(const std::string& tensorName) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(tensorName);
    if (it == allocations_.end()) {
        return nullptr;
    }
    return it->second.ptr;
}

std::size_t CudaOutputAllocator::getSize(const std::string& tensorName) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(tensorName);
    if (it == allocations_.end()) {
        return 0;
    }
    return it->second.size;
}

OutputAllocKind CudaOutputAllocator::kind() const {
    return kind_;
}
