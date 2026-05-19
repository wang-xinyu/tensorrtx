#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include "config.h"

namespace {

std::vector<cv::Point2f> getMiniBoxes(const std::vector<cv::Point>& contour, float& minSide) {
    cv::RotatedRect rect = cv::minAreaRect(contour);
    cv::Point2f raw[4];
    rect.points(raw);
    std::vector<cv::Point2f> points(raw, raw + 4);
    std::sort(points.begin(), points.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

    std::vector<cv::Point2f> box(4);
    if (points[1].y > points[0].y) {
        box[0] = points[0];
        box[3] = points[1];
    } else {
        box[0] = points[1];
        box[3] = points[0];
    }
    if (points[3].y > points[2].y) {
        box[1] = points[2];
        box[2] = points[3];
    } else {
        box[1] = points[3];
        box[2] = points[2];
    }
    minSide = std::min(rect.size.width, rect.size.height);
    return box;
}

float boxScoreFast(const float* prob, int h, int w, const std::vector<cv::Point2f>& box) {
    cv::Rect rect = cv::boundingRect(box);
    rect &= cv::Rect(0, 0, w, h);
    if (rect.empty()) {
        return 0.0f;
    }

    cv::Mat mask(rect.height, rect.width, CV_8UC1, cv::Scalar(0));
    std::vector<cv::Point> shifted;
    shifted.reserve(box.size());
    for (auto p : box) {
        shifted.emplace_back(static_cast<int>(p.x - rect.x), static_cast<int>(p.y - rect.y));
    }
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{shifted}, cv::Scalar(1));

    double sum = 0.0;
    int count = 0;
    for (int y = 0; y < rect.height; ++y) {
        const uchar* m = mask.ptr<uchar>(y);
        const float* row = prob + (rect.y + y) * w + rect.x;
        for (int x = 0; x < rect.width; ++x) {
            if (m[x]) {
                sum += row[x];
                ++count;
            }
        }
    }
    return count > 0 ? static_cast<float>(sum / count) : 0.0f;
}

std::vector<cv::Point2f> unclipBox(const std::vector<cv::Point2f>& box) {
    std::vector<cv::Point> contour;
    contour.reserve(box.size());
    for (auto p : box) {
        contour.emplace_back(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y)));
    }
    cv::RotatedRect rect = cv::minAreaRect(contour);
    float width = rect.size.width;
    float height = rect.size.height;
    float area = std::max(1.0f, static_cast<float>(cv::contourArea(contour)));
    float length = std::max(1.0f, static_cast<float>(cv::arcLength(contour, true)));
    float distance = area * kDetUnclipRatio / length;
    cv::Size2f size(width + 2.0f * distance, height + 2.0f * distance);
    cv::RotatedRect expanded(rect.center, size, rect.angle);

    cv::Point2f raw[4];
    expanded.points(raw);
    std::vector<cv::Point> expandedContour;
    for (auto p : raw) {
        expandedContour.emplace_back(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y)));
    }
    float minSide = 0.0f;
    return getMiniBoxes(expandedContour, minSide);
}

TextBox scaleBox(const std::vector<cv::Point2f>& points, float score, int outH, int outW,
                 const DetPreprocessResult& meta) {
    float widthScale = static_cast<float>(meta.src_w) / static_cast<float>(outW);
    float heightScale = static_cast<float>(meta.src_h) / static_cast<float>(outH);

    TextBox box{};
    for (int i = 0; i < 4; ++i) {
        float x = std::round(points[i].x * widthScale);
        float y = std::round(points[i].y * heightScale);
        x = std::max(0.0f, std::min(x, static_cast<float>(meta.src_w - 1)));
        y = std::max(0.0f, std::min(y, static_cast<float>(meta.src_h - 1)));
        box.points[i] = cv::Point2f(x, y);
    }
    box.score = score;
    return box;
}

}  // namespace

std::vector<TextBox> dbPostprocess(const float* prob, int outH, int outW, const DetPreprocessResult& meta) {
    cv::Mat bitmap(outH, outW, CV_8UC1);
    for (int y = 0; y < outH; ++y) {
        uchar* row = bitmap.ptr<uchar>(y);
        for (int x = 0; x < outW; ++x) {
            row[x] = prob[y * outW + x] > kDetThresh ? 255 : 0;
        }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<TextBox> boxes;
    boxes.reserve(std::min(static_cast<int>(contours.size()), kDetMaxCandidates));
    for (const auto& contour : contours) {
        if (static_cast<int>(boxes.size()) >= kDetMaxCandidates || contour.size() < 3) {
            continue;
        }

        float minSide = 0.0f;
        std::vector<cv::Point2f> points = getMiniBoxes(contour, minSide);
        if (minSide < 3.0f) {
            continue;
        }

        float score = boxScoreFast(prob, outH, outW, points);
        if (score < kDetBoxThresh) {
            continue;
        }

        std::vector<cv::Point2f> unclipped = unclipBox(points);
        std::vector<cv::Point> unclippedContour;
        for (auto p : unclipped) {
            unclippedContour.emplace_back(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y)));
        }
        float unclippedMinSide = 0.0f;
        std::vector<cv::Point2f> box = getMiniBoxes(unclippedContour, unclippedMinSide);
        if (unclippedMinSide < 5.0f) {
            continue;
        }
        boxes.push_back(scaleBox(box, score, outH, outW, meta));
    }

    std::sort(boxes.begin(), boxes.end(), [](const TextBox& a, const TextBox& b) {
        if (std::abs(a.points[0].y - b.points[0].y) > 10.0f) {
            return a.points[0].y < b.points[0].y;
        }
        return a.points[0].x < b.points[0].x;
    });
    return boxes;
}

RecResult ctcDecode(const float* prob, int timeSteps, int classCount, const std::vector<std::string>& dict) {
    std::string text;
    float confSum = 0.0f;
    int confCount = 0;
    int lastIndex = -1;
    int blankIndex = 0;

    for (int t = 0; t < timeSteps; ++t) {
        const float* row = prob + t * classCount;
        int index = static_cast<int>(std::max_element(row, row + classCount) - row);
        float score = row[index];
        int dictIndex = index - 1;
        if (index != blankIndex && index != lastIndex && dictIndex >= 0 && dictIndex < static_cast<int>(dict.size())) {
            text += dict[dictIndex];
            confSum += score;
            ++confCount;
        }
        lastIndex = index;
    }

    RecResult result;
    result.text = text;
    result.score = confCount > 0 ? confSum / static_cast<float>(confCount) : 0.0f;
    return result;
}
