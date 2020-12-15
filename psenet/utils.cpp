#include "utils.h"

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::cout << "Model weight is large, it will take some time." << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }
    std::cout << "Finish load weight" << std::endl;
    return weightMap;
}

cv::RotatedRect expandBox(const cv::RotatedRect &inBox, float ratio)
{
    cv::Size size = inBox.size;
    int neww = int(size.width * ratio);
    int newh = int(size.height * ratio);
    return cv::RotatedRect(inBox.center, cv::Size(neww, newh), inBox.angle);
}

void drawRects(cv::Mat &image, cv::Mat mask, float ratio_h, float ratio_w, int stride, float expand_ratio)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarcy;
    cv::findContours(mask, contours, hierarcy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> boundRect(contours.size());
    std::vector<cv::RotatedRect> box(contours.size());
    cv::Point2f rect[4];
    for (auto i = 0; i < contours.size(); i++)
    {
        box[i] = cv::minAreaRect(cv::Mat(contours[i]));
        cv::RotatedRect expandbox = expandBox(box[i], expand_ratio);
        expandbox.points(rect);
        for (auto j = 0; j < 4; j++)
        {
            cv::line(image, cv::Point{int(rect[j].x / ratio_w * stride), int(rect[j].y / ratio_h * stride)}, cv::Point{int(rect[(j + 1) % 4].x / ratio_w * stride), int(rect[(j + 1) % 4].y / ratio_h * stride)}, cv::Scalar(0, 0, 255), 2, 8);
        }
    }
}
