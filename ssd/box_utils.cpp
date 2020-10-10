#include<algorithm>
#include<iostream>
#include<cmath>
#include<vector>
#include<map>
#include "box_utils.h"
using namespace std;

float clamp(float x){
  return std::max(std::min(x, 1.f), 0.f);
}

std::vector<std::vector<float>> generate_ssd_priors(){
  // SSD specifications as feature map size, shrinkage, box min, box max
  float specs[6][4] = {{19, 16, 60, 105},
    {10, 32, 105, 150},
    {5, 64, 150, 195},
    {3, 100, 195, 240},
    {2, 150, 240, 285},
    {1, 300, 285, 330}};
  float aspect_ratios[2] = {2,3};
  float image_size = 300;
  float x_center, y_center, scale, h, w, size, ratio;
  std::vector<std::vector<float>> priors;

  for (size_t i = 0; i < 6; i++) {
    scale = image_size/specs[i][1];
    for (size_t j = 0; j < specs[i][0]; j++) {
      for (size_t k = 0; k < specs[i][0]; k++) {
        x_center = clamp((j + 0.5) / scale);
        y_center = clamp((k + 0.5) / scale);

        // small sized square box
        w = clamp(specs[i][2] / image_size);
        h = w;
        std::vector<float> v1 = {x_center, y_center, w, h};
        priors.push_back(v1);

        // big sized square box
        size = sqrt(specs[i][3] * specs[i][2]);
        w = clamp(size / image_size);
        h = w;
        std::vector<float> v2 = {x_center, y_center, w, h};
        priors.push_back(v2);

        // change h/w ratio of the small sized box
        w = specs[i][2]/image_size;
        h = w;
        for (float rt: aspect_ratios){
            ratio = sqrt(rt);
            std::vector<float> v3 = {x_center, y_center, clamp(w*ratio), clamp(h/ratio)};
            priors.push_back(v3);
            std::vector<float> v4 = {x_center, y_center, clamp(w/ratio), clamp(h*ratio)};
            priors.push_back(v4);
        }
      }
    }
  }

  return priors;
}

std::vector<float> convert_locations_to_boxes(std::vector<float> prior, float* location){
  float center_variance = 0.1;
  float size_variance = 0.2;

  float bx_cx, bx_cy, bx_h, bx_w;
  float bx_x1, bx_y1, bx_x2, bx_y2;

  // x_center, y_center, h, w
  bx_cx = location[0] * center_variance * prior[2] + prior[0];
  bx_cy = location[1] * center_variance * prior[3] + prior[1];
  bx_h = exp(location[2] * size_variance) * prior[2];
  bx_w = exp(location[3] * size_variance) * prior[3];

  // x1, y1, x2, y2
  bx_x1 = bx_cx - bx_h/2;
  bx_y1 = bx_cy - bx_w/2;
  bx_x2 = bx_cx + bx_h/2;
  bx_y2 = bx_cy + bx_w/2;
  std::vector<float> box = {bx_x1, bx_y1, bx_x2, bx_y2};

  return box;
}

/* Post processing script borrowed from ../yolo5/common.h model */
float iou(std::vector<float> lbox, std::vector<float> rbox) {
    float interBox[] = {
        std::max(lbox[0], rbox[0]), //left
        std::min(lbox[2], rbox[2]), //right
        std::max(lbox[1], rbox[1]), //top
        std::min(lbox[3], rbox[3]), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(const ssd::Detection& a, const ssd::Detection& b) {
    return a.conf > b.conf;
}

std::vector<ssd::Detection> nms(std::map<float, std::vector<ssd::Detection>> m, float nms_thresh = 0.5) {
  // NMS on single image of NUM_DETECTIONS detections
    std::vector<ssd::Detection> res;
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
    return res;
}

std::vector<ssd::Detection> post_process_output(float* prob, float* locations, float conf_thresh){
    // Process the detections on a single image
    std::vector<std::vector<float>> priors = generate_ssd_priors();
    std::map<float, std::vector<ssd::Detection>> m;
    float class_id;
    float* conf;

    // map from class_id : detections
    for (int i = 0; i < ssd::NUM_DETECTIONS; i++) {
        conf = std::max_element(prob + i*ssd::NUM_CLASSES, prob + (i+1)*ssd::NUM_CLASSES);
        class_id = std::distance(prob + i*ssd::NUM_CLASSES, conf);
        if (*conf <= conf_thresh) continue;

        std::vector<float> box = convert_locations_to_boxes(priors[i], locations+i*4);
        ssd::Detection det = {box, class_id+1, *conf};
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<ssd::Detection>());
        m[det.class_id].push_back(det);
    }

    return nms(m);
}
