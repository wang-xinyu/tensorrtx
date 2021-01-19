#include <thread>
#include <vector>
#include <memory>
#include "ibnnet.h"
#include "InferenceEngine.h"

// stuff we know about the network and the input/output blobs
static const int MAX_BATCH_SIZE = 4;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;
static const int DEVICE_ID = 0;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
extern Logger gLogger;

void run_infer(std::shared_ptr<trt::IBNNet> model) {

    CHECK(cudaSetDevice(model->getDeviceID()));

    if(!model->deserializeEngine()) {
        std::cout << "DeserializeEngine Failed." << std::endl;
        return;
    }

    /* support batch input data */
    std::vector<cv::Mat> input;
    input.emplace_back( cv::Mat(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(255,255,255)) ) ;

    /* run inference */
    model->inference(input); 

    /* get output data from cudaMalloc */
    float* prob = model->getOutput();

    /* print output */
    std::cout << "\nOutput from thread_id: " << std::this_thread::get_id() << std::endl;
    if( prob != nullptr ) { 
        for (size_t batch_idx = 0; batch_idx < input.size(); ++batch_idx) {
            for (int p = 0; p < OUTPUT_SIZE; ++p) {
                std::cout<< prob[batch_idx+p] << " ";
                if ((p+1) % 10 == 0) {
                    std::cout << std::endl;
                }
            }
        }
    }
}

int main(int argc, char** argv) {

    trt::EngineConfig engineCfg { 
        INPUT_BLOB_NAME,
        OUTPUT_BLOB_NAME,
        nullptr,
        MAX_BATCH_SIZE,
        INPUT_H,
        INPUT_W,
        OUTPUT_SIZE,
        0,
        DEVICE_ID};

    if (argc == 2 && std::string(argv[1]) == "-s") {
        std::cout << "Serializling Engine" << std::endl;
        trt::IBNNet ibnnet{engineCfg, trt::IBN::A}; 
        ibnnet.serializeEngine();
        return 0;
    } else if (argc == 2 && std::string(argv[1]) == "-d") {

        /* 
         * Support multi thread inference (mthreads>1)
         * Each thread holds their own CudaEngine
         * They can run on different cuda device through trt::EngineConfig setting
        */
        int mthreads = 1; 
        std::vector<std::thread> workers;
        std::vector<std::shared_ptr<trt::IBNNet>> models;

        for(int i = 0; i < mthreads; ++i) {
            models.emplace_back( std::make_shared<trt::IBNNet>(engineCfg, trt::IBN::A) ); // For IBNB: trt::IBN::B
        }

        for(int i = 0; i < mthreads; ++i) {
            workers.emplace_back( std::thread(run_infer, models[i]) );
        }

        for(auto & worker : workers) {
            worker.join();
        } 

        return 0;
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./ibnnet -s  // serialize model to plan file" << std::endl;
        std::cerr << "./ibnnet -d  // deserialize plan file and run inference" << std::endl;
        return -1;
    }
}
