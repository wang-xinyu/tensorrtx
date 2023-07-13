#include "postprocess.h"


cv::Rect get_rect(cv::Mat &img, float bbox[4]) {
    float l, r, t, b;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);

    if (r_h > r_w) {
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (kInputH - r_w * img.rows) / 2;
        b = bbox[3] - (kInputH - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - (kInputW - r_h * img.cols) / 2;
        r = bbox[2] - (kInputW - r_h * img.cols) / 2;
        t = bbox[1];
        b = bbox[3];
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

static float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            (std::max)(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), //left
            (std::min)(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), //right
            (std::max)(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), //top
            (std::min)(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

static bool cmp(const Detection &a, const Detection &b) {
    return a.conf > b.conf;
}

void nms(std::vector<Detection> &res, float *output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;

    for (int i = 0; i < output[0]; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto &dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto &item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

void batch_nms(std::vector<std::vector<Detection>> &res_batch, float *output, int batch_size, int output_size,
               float conf_thresh, float nms_thresh) {
    res_batch.resize(batch_size);
    for (int i = 0; i < batch_size; i++) {
        nms(res_batch[i], &output[i * output_size], conf_thresh, nms_thresh);
    }
}


void process_decode_ptr_host(std::vector<Detection> &res, const float* decode_ptr_host, int bbox_element, cv::Mat& img, int count) {
    Detection det;
    for (int i = 0; i < count; i++) {
        int basic_pos = 1 + i * bbox_element;
        int keep_flag = decode_ptr_host[basic_pos + 6];
        if (keep_flag == 1) {
            det.bbox[0] = decode_ptr_host[basic_pos + 0];
            det.bbox[1] = decode_ptr_host[basic_pos + 1];
            det.bbox[2] = decode_ptr_host[basic_pos + 2];
            det.bbox[3] = decode_ptr_host[basic_pos + 3];
            det.conf = decode_ptr_host[basic_pos + 4];
            det.class_id = decode_ptr_host[basic_pos + 5];
            res.push_back(det);
        }
    }
}

void batch_process(std::vector<std::vector<Detection>> &res_batch, const float* decode_ptr_host, int batch_size, int bbox_element, const std::vector<cv::Mat>& img_batch) {
    res_batch.resize(batch_size);
    int count = static_cast<int>(*decode_ptr_host);
    count = std::min(count, kMaxNumOutputBbox);
    for (int i = 0; i < batch_size; i++) {
        auto& img = const_cast<cv::Mat&>(img_batch[i]);
        process_decode_ptr_host(res_batch[i], &decode_ptr_host[i * count], bbox_element, img, count);
    }
}


void draw_bbox(std::vector<cv::Mat> &img_batch, std::vector<std::vector<Detection>> &res_batch) {
    for (size_t i = 0; i < img_batch.size(); i++) {
        auto &res = res_batch[i];
        cv::Mat img = img_batch[i];
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(img, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string((int) res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN,
                        1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
    }
}
