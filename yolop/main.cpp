#include "yolov5.hpp"
#include "zedcam.hpp"
#include <csignal>

static volatile bool keep_running = true;


void keyboard_handler(int sig) {
    // handle keyboard interrupt
    if (sig == SIGINT)
        keep_running = false;
}


int main(int argc, char** argv) {
    signal(SIGINT, keyboard_handler);
    cudaSetDevice(DEVICE);
    // CUcontext ctx;
    // CUdevice device;
    // cuInit(0);
    // cuDeviceGet(&device, 0);
    // cuCtxCreate(&ctx, 0, device);

    std::string wts_name = "yolop.wts";
    std::string engine_name = "yolop.engine";

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        // std::cerr << "read " << engine_name << " error!" << std::endl;
        std::cout << "Building engine..." << std::endl;
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        std::cout << "Engine has been built and saved to file." << std::endl;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // prepare data ---------------------------
    static float det_out[BATCH_SIZE * OUTPUT_SIZE];
    static int seg_out[BATCH_SIZE * IMG_H * IMG_W];
    static int lane_out[BATCH_SIZE * IMG_H * IMG_W];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 4);
    void* buffers[4];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int output_det_index = engine->getBindingIndex(OUTPUT_DET_NAME);
    const int output_seg_index = engine->getBindingIndex(OUTPUT_SEG_NAME);
    const int output_lane_index = engine->getBindingIndex(OUTPUT_LANE_NAME);
    assert(inputIndex == 0);
    assert(output_det_index == 1);
    assert(output_seg_index == 2);
    assert(output_lane_index == 3);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output_det_index], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output_seg_index], BATCH_SIZE * IMG_H * IMG_W * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers[output_lane_index], BATCH_SIZE * IMG_H * IMG_W * sizeof(int)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // create zed
    auto zed = create_camera();
    sl::Resolution image_size = zed->getCameraInformation().camera_configuration.resolution;
    sl::Mat img_zed(image_size.width, image_size.height, sl::MAT_TYPE::U8_C4, sl::MEM::GPU);
    cv::cuda::GpuMat img_ocv = slMat2cvMatGPU(img_zed);
    cv::cuda::GpuMat cvt_img(image_size.height, image_size.width, CV_8UC3);

    // store seg results
    cv::Mat tmp_seg(IMG_H, IMG_W, CV_32S, seg_out);
    // sotore lane results
    cv::Mat tmp_lane(IMG_H, IMG_W, CV_32S, lane_out);
    cv::Mat seg_res(image_size.height, image_size.width, CV_32S);
    cv::Mat lane_res(image_size.height, image_size.width, CV_32S);

    char key = ' ';
    while (keep_running and key != 'q') {
        // retrieve img
        if (zed->grab() != sl::ERROR_CODE::SUCCESS) continue;
        zed->retrieveImage(img_zed, sl::VIEW::LEFT, sl::MEM::GPU);
        cudaSetDevice(DEVICE);
        cv::cuda::cvtColor(img_ocv, cvt_img, cv::COLOR_BGRA2BGR);
        
        // preprocess ~3ms
        preprocess_img_gpu(cvt_img, (float*)buffers[inputIndex], INPUT_W, INPUT_H); // letterbox
        
        // buffers[inputIndex] = pr_img.data;
        // Run inference
        auto start = std::chrono::system_clock::now();
        // cuCtxPushCurrent(ctx);
        doInference(*context, stream, buffers, det_out, seg_out, lane_out, BATCH_SIZE);
        // cuCtxPopCurrent(&ctx);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // postprocess ~0ms
        std::vector<Yolo::Detection> batch_res;
        nms(batch_res, det_out, CONF_THRESH, NMS_THRESH);
        cv::resize(tmp_seg, seg_res, seg_res.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(tmp_lane, lane_res, lane_res.size(), 0, 0, cv::INTER_NEAREST);

        // show results
        //std::cout << res.size() << std::endl;
        visualization(cvt_img, seg_res, lane_res, batch_res, key);
    }
    // destroy windows
#ifdef SHOW_IMG
    cv::destroyAllWindows();
#endif
    // close camera
    img_zed.free();
    zed->close();
    delete zed;
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[output_det_index]));
    CUDA_CHECK(cudaFree(buffers[output_seg_index]));
    CUDA_CHECK(cudaFree(buffers[output_lane_index]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
