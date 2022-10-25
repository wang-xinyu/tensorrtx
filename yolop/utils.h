#ifndef TRTX_YOLOV5_UTILS_H_
#define TRTX_YOLOV5_UTILS_H_

#include <dirent.h>
#include <opencv2/opencv.hpp>

#ifdef HAVE_CUDA
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>
#endif

#include <iostream>
#include "common.hpp"

#define SHOW_IMG

static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    cv::Mat tensor;
    out.convertTo(tensor, CV_32FC3, 1.f / 255.f);

    cv::subtract(tensor, cv::Scalar(0.485, 0.456, 0.406), tensor, cv::noArray(), -1);
    cv::divide(tensor, cv::Scalar(0.229, 0.224, 0.225), tensor, 1, -1);
    // std::cout << cv::format(out, cv::Formatter::FMT_NUMPY)<< std::endl;
    // assert(false);
    // cv::Mat out(input_h, input_w, CV_8UC3);
    // cv::copyMakeBorder(re, out, y, y, x, x, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
    return tensor;
}

#ifdef HAVE_CUDA
void preprocess_img_gpu(cv::cuda::GpuMat& img, float* gpu_input, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::cuda::GpuMat re(h, w, CV_8UC3);
    cv::cuda::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::cuda::GpuMat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::cuda::GpuMat tensor;
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    out.convertTo(tensor, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(tensor, cv::Scalar(0.485, 0.456, 0.406), tensor, cv::noArray(), -1);
    cv::cuda::divide(tensor, cv::Scalar(0.229, 0.224, 0.225), tensor, 1, -1);
    // cv::Mat out(input_h, input_w, CV_8UC3);
    // cv::copyMakeBorder(re, out, y, y, x, x, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));

    // to tensor
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < 3; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(tensor.size(), CV_32FC1, gpu_input + i * input_w * input_h));
    }
    cv::cuda::split(tensor, chw);
}
#endif

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

void PrintMat(cv::Mat &A)
{
  for(int i=0; i<A.rows; i++)
  {
    for(int j=0; j<A.cols; j++)
        std::cout << A.at<int>(i,j) << ' ';
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

#ifdef HAVE_CUDA
void visualization(cv::cuda::GpuMat& cvt_img, cv::Mat& seg_res, cv::Mat& lane_res, std::vector<Yolo::Detection>& res, char& key)
{
    static const std::vector<cv::Vec3b> segColor{cv::Vec3b(0, 0, 0), cv::Vec3b(0, 255, 0), cv::Vec3b(255, 0, 0)};
    static const std::vector<cv::Vec3b> laneColor{cv::Vec3b(0, 0, 0), cv::Vec3b(0, 0, 255), cv::Vec3b(0, 0, 0)};
    cv::Mat cvt_img_cpu;
    cvt_img.download(cvt_img_cpu);

    // handling seg and lane results
    for (int row = 0; row < cvt_img_cpu.rows; ++row) {
        uchar* pdata = cvt_img_cpu.data + row * cvt_img_cpu.step;
        for (int col = 0; col < cvt_img_cpu.cols; ++col) {
            int seg_idx = seg_res.at<int>(row, col);
            int lane_idx = lane_res.at<int>(row, col);
            //std::cout << "enter" << ix << std::endl;
            for (int i = 0; i < 3; ++i) {
                if (lane_idx) {
                    if (i != 2)
                        pdata[i] = pdata[i] / 2 + laneColor[lane_idx][i] / 2;
                }
                else if (seg_idx)
                    pdata[i] = pdata[i] / 2 + segColor[seg_idx][i] / 2;
            }
            pdata += 3;
        }
    }

    // handling det results
    for (size_t j = 0; j < res.size(); ++j) {
        cv::Rect r = get_rect(cvt_img_cpu, res[j].bbox);
        cv::rectangle(cvt_img_cpu, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(cvt_img_cpu, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }

#ifdef SHOW_IMG
    cv::imshow("img", cvt_img_cpu);
    key = cv::waitKey(1);
#else
    cv::imwrite("../zed_result.jpg", cvt_img_cpu);
#endif
}
#endif

#endif  // TRTX_YOLOV5_UTILS_H_