#include "postprocess.h"
#include "utils.h"

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
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
    l = std::max(0.0f, l);
    t = std::max(0.0f, t);
    int width = std::max(0, std::min(int(round(r - l)), img.cols - int(round(l))));
    int height = std::max(0, std::min(int(round(b - t)), img.rows - int(round(t))));

    return cv::Rect(int(round(l)), int(round(t)), width, height);
}

cv::Rect get_rect_obb(cv::Mat& img, float bbox[4]) {
    float l, r, t, b;
    float r_w = kObbInputW / (img.cols * 1.0);
    float r_h = kObbInputH / (img.rows * 1.0);

    if (r_h > r_w) {
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (kObbInputH - r_w * img.rows) / 2;
        b = bbox[3] - (kObbInputH - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - (kObbInputW - r_h * img.cols) / 2;
        r = bbox[2] - (kObbInputW - r_h * img.cols) / 2;
        t = bbox[1];
        b = bbox[3];
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    l = std::max(0.0f, l);
    t = std::max(0.0f, t);
    int width = std::max(0, std::min(int(round(r - l)), img.cols - int(round(l))));
    int height = std::max(0, std::min(int(round(b - t)), img.rows - int(round(t))));

    return cv::Rect(int(round(l)), int(round(t)), width, height);
}

cv::Rect get_rect_adapt_landmark(cv::Mat& img, float bbox[4], float lmk[kNumberOfPoints * 3]) {
    float l, r, t, b;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] / r_w;
        r = bbox[2] / r_w;
        t = (bbox[1] - (kInputH - r_w * img.rows) / 2) / r_w;
        b = (bbox[3] - (kInputH - r_w * img.rows) / 2) / r_w;
        for (int i = 0; i < kNumberOfPoints * 3; i += 3) {
            lmk[i] /= r_w;
            lmk[i + 1] = (lmk[i + 1] - (kInputH - r_w * img.rows) / 2) / r_w;
            // lmk[i + 2]
        }
    } else {
        l = (bbox[0] - (kInputW - r_h * img.cols) / 2) / r_h;
        r = (bbox[2] - (kInputW - r_h * img.cols) / 2) / r_h;
        t = bbox[1] / r_h;
        b = bbox[3] / r_h;
        for (int i = 0; i < kNumberOfPoints * 3; i += 3) {
            lmk[i] = (lmk[i] - (kInputW - r_h * img.cols) / 2) / r_h;
            lmk[i + 1] /= r_h;
            // lmk[i + 2]
        }
    }
    l = std::max(0.0f, l);
    t = std::max(0.0f, t);
    int width = std::max(0, std::min(int(round(r - l)), img.cols - int(round(l))));
    int height = std::max(0, std::min(int(round(b - t)), img.rows - int(round(t))));

    return cv::Rect(int(round(l)), int(round(t)), width, height);
}

static float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            (std::max)(lbox[0], rbox[0]),
            (std::min)(lbox[2], rbox[2]),
            (std::max)(lbox[1], rbox[1]),
            (std::min)(lbox[3], rbox[3]),
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    float unionBoxS = (lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS;
    return interBoxS / unionBoxS;
}

static bool cmp(const Detection& a, const Detection& b) {
    if (a.conf == b.conf) {
        return a.bbox[0] < b.bbox[0];
    }
    return a.conf > b.conf;
}

void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;

    for (int i = 0; i < output[0]; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh || isnan(output[1 + det_size * i + 4]))
            continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0)
            m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
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

void batch_nms(std::vector<std::vector<Detection>>& res_batch, float* output, int batch_size, int output_size,
               float conf_thresh, float nms_thresh) {
    res_batch.resize(batch_size);
    for (int i = 0; i < batch_size; i++) {
        nms(res_batch[i], &output[i * output_size], conf_thresh, nms_thresh);
    }
}

void process_decode_ptr_host(std::vector<Detection>& res, const float* decode_ptr_host, int bbox_element, cv::Mat& img,
                             int count) {
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

void batch_process(std::vector<std::vector<Detection>>& res_batch, const float* decode_ptr_host, int batch_size,
                   int bbox_element, const std::vector<cv::Mat>& img_batch) {
    res_batch.resize(batch_size);
    int count = static_cast<int>(*decode_ptr_host);
    count = std::min(count, kMaxNumOutputBbox);
    for (int i = 0; i < batch_size; i++) {
        auto& img = const_cast<cv::Mat&>(img_batch[i]);
        process_decode_ptr_host(res_batch[i], &decode_ptr_host[i * count], bbox_element, img, count);
    }
}

void draw_bbox(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch) {
    for (size_t i = 0; i < img_batch.size(); i++) {
        auto& res = res_batch[i];
        cv::Mat img = img_batch[i];
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(img, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                        cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
    }
}

void draw_bbox_keypoints_line(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch) {
    const std::vector<std::pair<int, int>> skeleton_pairs = {
            {0, 1}, {0, 2},  {0, 5}, {0, 6},  {1, 2},   {1, 3},   {2, 4},   {5, 6},   {5, 7},  {5, 11},
            {6, 8}, {6, 12}, {7, 9}, {8, 10}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}};

    for (size_t i = 0; i < img_batch.size(); i++) {
        auto& res = res_batch[i];
        cv::Mat img = img_batch[i];
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect_adapt_landmark(img, res[j].bbox, res[j].keypoints);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                        cv::Scalar(0xFF, 0xFF, 0xFF), 2);

            for (int k = 0; k < kNumberOfPoints * 3; k += 3) {
                if (res[j].keypoints[k + 2] > 0.5) {
                    cv::circle(img, cv::Point((int)res[j].keypoints[k], (int)res[j].keypoints[k + 1]), 3,
                               cv::Scalar(0, 0x27, 0xC1), -1);
                }
            }

            for (const auto& bone : skeleton_pairs) {
                int kp1_idx = bone.first * 3;
                int kp2_idx = bone.second * 3;
                if (res[j].keypoints[kp1_idx + 2] > 0.5 && res[j].keypoints[kp2_idx + 2] > 0.5) {
                    cv::Point p1((int)res[j].keypoints[kp1_idx], (int)res[j].keypoints[kp1_idx + 1]);
                    cv::Point p2((int)res[j].keypoints[kp2_idx], (int)res[j].keypoints[kp2_idx + 1]);
                    cv::line(img, p1, p2, cv::Scalar(0, 0x27, 0xC1), 2);
                }
            }
        }
    }
}

cv::Mat scale_mask(cv::Mat mask, cv::Mat img) {
    int x, y, w, h;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    if (r_h > r_w) {
        w = kInputW;
        h = r_w * img.rows;
        x = 0;
        y = (kInputH - h) / 2;
    } else {
        w = r_h * img.cols;
        h = kInputH;
        x = (kInputW - w) / 2;
        y = 0;
    }
    cv::Rect r(x, y, w, h);
    cv::Mat res;
    cv::resize(mask(r), res, img.size());
    return res;
}

void draw_mask_bbox(cv::Mat& img, std::vector<Detection>& dets, std::vector<cv::Mat>& masks,
                    std::unordered_map<int, std::string>& labels_map) {
    static std::vector<uint32_t> colors = {0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231, 0x48F90A, 0x92CC17,
                                           0x3DDB86, 0x1A9334, 0x00D4BB, 0x2C99A8, 0x00C2FF, 0x344593, 0x6473FF,
                                           0x0018EC, 0x8438FF, 0x520085, 0xCB38FF, 0xFF95C8, 0xFF37C7};
    for (size_t i = 0; i < dets.size(); i++) {
        cv::Mat img_mask = scale_mask(masks[i], img);
        auto color = colors[(int)dets[i].class_id % colors.size()];
        auto bgr = cv::Scalar(color & 0xFF, color >> 8 & 0xFF, color >> 16 & 0xFF);

        cv::Rect r = get_rect(img, dets[i].bbox);
        for (int x = r.x; x < r.x + r.width; x++) {
            for (int y = r.y; y < r.y + r.height; y++) {
                float val = img_mask.at<float>(y, x);
                if (val <= 0.5)
                    continue;
                img.at<cv::Vec3b>(y, x)[0] = img.at<cv::Vec3b>(y, x)[0] / 2 + bgr[0] / 2;
                img.at<cv::Vec3b>(y, x)[1] = img.at<cv::Vec3b>(y, x)[1] / 2 + bgr[1] / 2;
                img.at<cv::Vec3b>(y, x)[2] = img.at<cv::Vec3b>(y, x)[2] / 2 + bgr[2] / 2;
            }
        }

        cv::rectangle(img, r, bgr, 2);

        // Get the size of the text
        cv::Size textSize =
                cv::getTextSize(labels_map[(int)dets[i].class_id] + " " + to_string_with_precision(dets[i].conf),
                                cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
        // Set the top left corner of the rectangle
        cv::Point topLeft(r.x, r.y - textSize.height);

        // Set the bottom right corner of the rectangle
        cv::Point bottomRight(r.x + textSize.width, r.y + textSize.height);

        // Set the thickness of the rectangle lines
        int lineThickness = 2;

        // Draw the rectangle on the image
        cv::rectangle(img, topLeft, bottomRight, bgr, -1);

        cv::putText(img, labels_map[(int)dets[i].class_id] + " " + to_string_with_precision(dets[i].conf),
                    cv::Point(r.x, r.y + 4), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar::all(0xFF), 2);
    }
}

void process_decode_ptr_host_obb(std::vector<Detection>& res, const float* decode_ptr_host, int bbox_element,
                                 cv::Mat& img, int count) {
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
            det.angle = decode_ptr_host[basic_pos + 7];
            res.push_back(det);
        }
    }
}

void batch_process_obb(std::vector<std::vector<Detection>>& res_batch, const float* decode_ptr_host, int batch_size,
                       int bbox_element, const std::vector<cv::Mat>& img_batch) {
    res_batch.resize(batch_size);
    int count = static_cast<int>(*decode_ptr_host);
    count = std::min(count, kMaxNumOutputBbox);
    for (int i = 0; i < batch_size; i++) {
        auto& img = const_cast<cv::Mat&>(img_batch[i]);
        process_decode_ptr_host_obb(res_batch[i], &decode_ptr_host[i * count], bbox_element, img, count);
    }
}

std::tuple<float, float, float> convariance_matrix(Detection res) {
    float w = res.bbox[2];
    float h = res.bbox[3];

    float a = w * w / 12.0;
    float b = h * h / 12.0;
    float c = res.angle;

    float cos_r = std::cos(c);
    float sin_r = std::sin(c);

    float cos_r2 = cos_r * cos_r;
    float sin_r2 = sin_r * sin_r;

    float a_val = a * cos_r2 + b * sin_r2;
    float b_val = a * sin_r2 + b * cos_r2;
    float c_val = (a - b) * cos_r * sin_r;

    return std::make_tuple(a_val, b_val, c_val);
}

static float probiou(const Detection& res1, const Detection& res2, float eps = 1e-7) {
    // Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    float a1, b1, c1, a2, b2, c2;
    std::tuple<float, float, float> matrix1 = {a1, b1, c1};
    std::tuple<float, float, float> matrix2 = {a2, b2, c2};
    matrix1 = convariance_matrix(res1);
    matrix2 = convariance_matrix(res2);
    a1 = std::get<0>(matrix1);
    b1 = std::get<1>(matrix1);
    c1 = std::get<2>(matrix1);
    a2 = std::get<0>(matrix2);
    b2 = std::get<1>(matrix2);
    c2 = std::get<2>(matrix2);

    float x1 = res1.bbox[0], y1 = res1.bbox[1];
    float x2 = res2.bbox[0], y2 = res2.bbox[1];

    float t1 = ((a1 + a2) * std::pow(y1 - y2, 2) + (b1 + b2) * std::pow(x1 - x2, 2)) /
               ((a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2) + eps);
    float t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2) + eps);
    float t3 = std::log(
            ((a1 + a2) * (b1 + b2) - std::pow(c1 + c2, 2)) /
                    (4 * std::sqrt(std::max(a1 * b1 - c1 * c1, 0.0f)) * std::sqrt(std::max(a2 * b2 - c2 * c2, 0.0f)) +
                     eps) +
            eps);

    float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
    bd = std::max(std::min(bd, 100.0f), eps);
    float hd = std::sqrt(1.0 - std::exp(-bd) + eps);

    return 1 - hd;
}

void nms_obb(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;

    for (int i = 0; i < output[0]; i++) {

        if (output[1 + det_size * i + 4] <= conf_thresh)
            continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0)
            m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (probiou(item, dets[n]) >= nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

void batch_nms_obb(std::vector<std::vector<Detection>>& res_batch, float* output, int batch_size, int output_size,
                   float conf_thresh, float nms_thresh) {
    res_batch.resize(batch_size);
    for (int i = 0; i < batch_size; i++) {
        nms_obb(res_batch[i], &output[i * output_size], conf_thresh, nms_thresh);
    }
}

static std::vector<cv::Point> get_corner(cv::Mat& img, const Detection& box) {
    float cos_value, sin_value;

    // Calculate center point and width/height
    float x1 = box.bbox[0];
    float y1 = box.bbox[1];
    float w = box.bbox[2];
    float h = box.bbox[3];
    float angle = box.angle * 180.0f / CV_PI;  // Convert radians to degrees

    // Print original angle
    std::cout << "Original angle: " << angle << std::endl;

    // Swap width and height if height is greater than or equal to width
    if (h >= w) {
        std::swap(w, h);
        angle = fmod(angle + 90.0f, 180.0f);  // Adjust angle to be within [0, 180)
    }

    // Ensure the angle is between 0 and 180 degrees
    if (angle < 0) {
        angle += 360.0f;  // Convert to positive value
    }
    if (angle > 180.0f) {
        angle -= 180.0f;  // Subtract 180 from angles greater than 180
    }

    // Print adjusted angle
    std::cout << "Adjusted angle: " << angle << std::endl;

    // Convert to normal angle value
    float normal_angle = fmod(angle, 180.0f);
    if (normal_angle < 0) {
        normal_angle += 180.0f;  // Ensure it's a positive value
    }

    // Print normal angle value
    std::cout << "Normal angle: " << normal_angle << std::endl;

    cos_value = std::cos(angle * CV_PI / 180.0f);  // Convert to radians
    sin_value = std::sin(angle * CV_PI / 180.0f);

    // Calculate each corner point
    float l = x1 - w / 2;  // Left boundary
    float r = x1 + w / 2;  // Right boundary
    float t = y1 - h / 2;  // Top boundary
    float b = y1 + h / 2;  // Bottom boundary

    // Use get_rect function to scale the coordinates
    float bbox[4] = {l, t, r, b};
    cv::Rect rect = get_rect_obb(img, bbox);

    float x_ = (rect.x + rect.x + rect.width) / 2;   // Center x
    float y_ = (rect.y + rect.y + rect.height) / 2;  // Center y
    float width = rect.width;                        // Width
    float height = rect.height;                      // Height

    // Calculate each corner point
    std::vector<cv::Point> corner_points(4);
    float vec1x = width / 2 * cos_value;
    float vec1y = width / 2 * sin_value;
    float vec2x = -height / 2 * sin_value;
    float vec2y = height / 2 * cos_value;

    corner_points[0] = cv::Point(int(round(x_ + vec1x + vec2x)), int(round(y_ + vec1y + vec2y)));  // Top-left corner
    corner_points[1] = cv::Point(int(round(x_ + vec1x - vec2x)), int(round(y_ + vec1y - vec2y)));  // Top-right corner
    corner_points[2] =
            cv::Point(int(round(x_ - vec1x - vec2x)), int(round(y_ - vec1y - vec2y)));  // Bottom-right corner
    corner_points[3] = cv::Point(int(round(x_ - vec1x + vec2x)), int(round(y_ - vec1y + vec2y)));  // Bottom-left corner

    // Check and adjust corner points to ensure the rectangle is parallel to image boundaries
    for (auto& point : corner_points) {
        point.x = std::max(0, std::min(point.x, img.cols - 1));
        point.y = std::max(0, std::min(point.y, img.rows - 1));
    }

    return corner_points;
}

void draw_bbox_obb(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch) {
    static std::vector<uint32_t> colors = {0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231, 0x48F90A, 0x92CC17,
                                           0x3DDB86, 0x1A9334, 0x00D4BB, 0x2C99A8, 0x00C2FF, 0x344593, 0x6473FF,
                                           0x0018EC, 0x8438FF, 0x520085, 0xCB38FF, 0xFF95C8, 0xFF37C7};
    for (size_t i = 0; i < img_batch.size(); i++) {
        auto& res = res_batch[i];
        auto& img = img_batch[i];
        for (auto& obj : res) {
            auto color = colors[(int)obj.class_id % colors.size()];
            auto bgr = cv::Scalar(color & 0xFF, color >> 8 & 0xFF, color >> 16 & 0xFF);
            auto corner_points = get_corner(img, obj);
            cv::polylines(img, std::vector<std::vector<cv::Point>>{corner_points}, true, bgr, 1);

            auto text = (std::to_string((int)(obj.class_id)) + ":" + to_string_with_precision(obj.conf));
            cv::Size textsize = cv::getTextSize(text, 0, 0.3, 1, nullptr);

            int width = textsize.width;
            int height = textsize.height;
            bool outside = (corner_points[0].y - height >= 3) ? true : false;
            cv::Point p1(corner_points[0].x, corner_points[0].y), p2;
            p2.x = corner_points[0].x + width;
            if (outside) {
                p2.y = corner_points[0].y - height - 3;
            } else {
                p2.y = corner_points[0].y + height + 3;
            }
            cv::rectangle(img, p1, p2, bgr, -1, cv::LINE_AA);
            cv::putText(
                    img, text,
                    cv::Point(corner_points[0].x, (outside ? corner_points[0].y - 2 : corner_points[0].y + height + 2)),
                    0, 0.3, cv::Scalar::all(255), 1, cv::LINE_AA);
        }
    }
}
