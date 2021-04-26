#include "inception_v4.h"


/**
 * Initializes Inception class params in the 
 * InceptionV4Params structure.
**/
trtx::InceptionV4Params initializeParams()
{
    trtx::InceptionV4Params params;

    params.batchSize = 1;
    params.fp16 = false;

    params.inputH = 299;
    params.inputW = 299;
    params.outputSize = 1000;

    // change weights file name here
    params.weightsFile = "../inceptionV4.wts";

    // change engine file name here
    params.trtEngineFile = "inceptionV4.engine";
    return params;
}


int main(int argc, char** argv){
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./inception -s   // serialize model to plan file" << std::endl;
        std::cerr << "./inception -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    trtx::InceptionV4Params params = initializeParams();
    trtx::InceptionV4 inceptionV4(params);

    if (std::string(argv[1]) == "-s") {
        // check if engine exists already
        std::ifstream f(params.trtEngineFile, std::ios::binary);

        // if engine does not exists build, serialize and save
        if(!f.good())
        {
            std::cout << "Building network ..." << std::endl;
            f.close();
            inceptionV4.serializeEngine();
        }

        return 1;
    } 
    else if(std::string(argv[1]) == "-d")
    {
        // deserialize
        inceptionV4.deserializeCudaEngine();
    }

    // create data
    float data[3 * params.inputH * params.inputW];
    for(int i=0; i<3*params.inputH*params.inputW; i++)
    {
        data[i] = 1.0;
    }
    
    // run inference
    float prob[params.outputSize];
    for(int i=0; i<100; i++)
    {
        auto start = std::chrono::system_clock::now();
        inceptionV4.doInference(data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // cleanup
    bool cleaned = inceptionV4.cleanUp();
    
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < params.outputSize; i++)
    {
        std::cout << prob[i] << ", ";
        if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    return 0;
}