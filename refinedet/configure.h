
#define USE_FP32  // set USE_INT8 or USE_FP16 or USE_FP32

const int num_class = 25; //num_class + 1     //Including background class

//SERIALIZE: It indicates that to generate engin by serialization, the following path needs to be set,path_wts_ and path_save_engine
//INFER: It shows that it is a reasoning mode,the following path needs to be set,path_engine
#define INFER    //SERIALIZE   INFER

const std::string path_engine = "/data_2//cmake-build-debug/refinedet_0312-now.engine";
const std::string path_wts = "/data_1/refinedet/pytorch_refinedet-master/refinedet0312.wts";
const std::string path_save_engine = "./refinedet_0312-now.engine";

//Picture folder to be detected
const char *p_dir_name = "/data_1/img/";

const float TH = 0.2;  //Confidence threshold
const int T_show = 1; //1:Show the effect      0:Test map to generate TXT
//The path to save the generated TXT when testing the map
std::string save_path_txt = "/data_1/txt/";

#define DEVICE 0  // GPU id

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 320;
static const int INPUT_W = 320;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME_arm_loc = "arm_loc";
const char* OUTPUT_BLOB_NAME_arm_conf = "arm_conf";
const char* OUTPUT_BLOB_NAME_odm_loc = "odm_loc";
const char* OUTPUT_BLOB_NAME_odm_conf = "odm_conf";

std::string label_map[] =
        {
                "background",
                "aa",
                "bb",
                "cc",
                "dd",
                "ee",
                "ff",
                "gg",
                "hh",
                "ii",
                "jj",
                "kk",
                "ll",
                "mm",
                "nn",
                "oo",
                "pp",
                "qq",
                "rr",
                "ss",
                "tt",
                "uu",
                "vv",
                "ww",
                "xx"
        };