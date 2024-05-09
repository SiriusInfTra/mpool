#include <util.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

#define CUDA_CALL(func) do { \
    auto error = func; \
    if (error != cudaSuccess) { \
        LOG(FATAL) << #func << " " << cudaGetErrorString(error); \
        exit(EXIT_FAILURE); \
    } \
    } while (0)

#define CU_CALL(func) \
    do { \
        auto err = func; \
        if (err != CUDA_SUCCESS) { \
            const char* pstr = nullptr; \
            cuGetErrorString(err, &pstr); \
            LOG(FATAL) << #func << ": " << pstr; \
            exit(EXIT_FAILURE); \
        } \
    } while (0);

using namespace mpool;

__global__ void kernel() {
    while(true) {}
}

int main1() {
    CU_CALL(cuInit(0));

    CUcontext cuContext;
    CU_CALL(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC, 0));
    void *dev_ptr;
    // CUDA_CALL(cudaMallocAsync(&dev_ptr, 32_MB));
    CU_CALL(cuMemAddressReserve(reinterpret_cast<CUdeviceptr *>(&dev_ptr), 32_MB, 32_MB, 0, 0));
    CUmemGenericAllocationHandle cu_handle;
    CUmemAllocationProp prop = {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        .location = {
        .type = CU_MEM_LOCATION_TYPE_DEVICE,
        .id = 0
        }
    };
    CU_CALL(cuMemCreate(&cu_handle, 32_MB, &prop, 0));
    CU_CALL(cuMemMap( reinterpret_cast<CUdeviceptr>(dev_ptr), 32_MB, 0, cu_handle, 0));
    CUmemAccessDesc acc_desc = {
        .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = 0},
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    };
    CU_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(dev_ptr), 32_MB, &acc_desc, 1));
    CUstream cu_stream;
    CU_CALL(cuStreamCreate(&cu_stream, CU_STREAM_DEFAULT));
    CUevent cu_event;
    CU_CALL(cuEventCreate(&cu_event, CU_EVENT_BLOCKING_SYNC));
    kernel<<<1,1,0, cu_stream>>>();
    void *dev_ptr1;
    CUDA_CALL(cudaMallocAsync(&dev_ptr1, 32_MB, cu_stream));
    CU_CALL(cuEventRecord(cu_event, cu_stream));
    // CU_CALL(cuEventSynchronize(cu_event));
    CUDA_CALL(cudaMemsetAsync(dev_ptr, 0, 32_MB, cu_stream));
    CU_CALL(cuStreamWaitEvent(cu_stream, cu_event, 0));
    CU_CALL(cuStreamSynchronize(cu_stream));
    CUDA_CALL(cudaStreamSynchronize(cu_stream));

    
    CUstream cu_stream1;
    CU_CALL(cuStreamCreate(&cu_stream1, CU_STREAM_DEFAULT));
    CU_CALL(cuStreamWaitEvent(cu_stream1, cu_event, CU_EVENT_WAIT_DEFAULT));
    CUDA_CALL(cudaMemsetAsync(dev_ptr, 0, 32_MB, cu_stream1));
    CU_CALL(cuStreamSynchronize(cu_stream1));
    LOG(INFO) << "finish";
    return 0;
}

int main() {
    CU_CALL(cuInit(0));

    CUcontext cuContext;
    CU_CALL(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC, 0));

    CUstream cu_stream;
    CU_CALL(cuStreamCreate(&cu_stream, CU_STREAM_DEFAULT));

    // void *flag = 0;
    // CU_CALL(cuMemAllocHost(&flag, sizeof(cuuint64_t)));
    CUdeviceptr flag_dev;
    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&flag_dev), sizeof(cuuint64_t)));
    CU_CALL(cuStreamWriteValue32(cu_stream, flag_dev, 0, CU_STREAM_WRITE_VALUE_DEFAULT ));
    CU_CALL(cuStreamWaitValue32(cu_stream, flag_dev, 1, CU_STREAM_WAIT_VALUE_EQ ));

    CUDA_CALL(cudaMemsetAsync(0, 0, 32_MB, cu_stream));
}