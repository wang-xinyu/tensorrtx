
#define USE_FP32  // set USE_INT8 or USE_FP16 or USE_FP32

const int num_class = 25; //包括背景类

//SERIALIZE 表明序列化生成engin，需要设定后面的wts路径（path_wts）和保存engine路径（path_save_engine）
//INFER 表明是推理模式，需要指定engine路径（path_engine）
#define INFER    //SERIALIZE   INFER


const std::string path_engine = "/data_2//cmake-build-debug/refinedet_0312-now.engine";
const std::string path_wts = "/data_1/refinedet/pytorch_refinedet-master/refinedet0312.wts";
const std::string path_save_engine = "./refinedet_0312-now.engine";

//需要检测的图片文件夹
const char *p_dir_name = "/data_1/img/";

const float TH = 0.2;  //可信度阈值
const int T_show = 1; //1：显示看效果   0：测试map所需要的txt
//测试map时候生成的txt
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



