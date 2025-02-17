#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "yololayer.h"
#include "segmentlayer.h"
#include "radianlayer.h"

using namespace nvinfer1;

static std::vector<cv::Vec3b> COLOR = {
    cv::Vec3b(0xFF, 0x38, 0x38),
    cv::Vec3b(0xFF, 0x9D, 0x97),
    cv::Vec3b(0xFF, 0x70, 0x1F),
    cv::Vec3b(0xFF, 0xB2, 0x1D),
    cv::Vec3b(0xCF, 0xD2, 0x31),
    cv::Vec3b(0x48, 0xF9, 0x0A),
    cv::Vec3b(0x92, 0xCC, 0x17),
    cv::Vec3b(0x3D, 0xDB, 0x86),
    cv::Vec3b(0x1A, 0x93, 0x34),
    cv::Vec3b(0x00, 0xD4, 0xBB),
    cv::Vec3b(0x2C, 0x99, 0xA8),
    cv::Vec3b(0x00, 0xC2, 0xFF),
    cv::Vec3b(0x34, 0x45, 0x93),
    cv::Vec3b(0x64, 0x73, 0xFF),
    cv::Vec3b(0x00, 0x18, 0xEC),
    cv::Vec3b(0x84, 0x38, 0xFF),
    cv::Vec3b(0x52, 0x00, 0x85),
    cv::Vec3b(0xCB, 0x38, 0xFF),
    cv::Vec3b(0xFF, 0x95, 0xC8),
    cv::Vec3b(0xFF, 0x37, 0xC7)
};

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    float l, r, t, b;
    float r_w = Yolo::INPUT_W / (img.cols * 1.f);
    float r_h = Yolo::INPUT_H / (img.rows * 1.f);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

cv::Mat get_mask(int batch, cv::Mat& img, std::vector<Seg::DetectionWithSeg> res, float* mask) 
{    
    int l, r, t, b;
    float r_w = Seg::INPUT_W / (img.cols * 1.f);
    float r_h = Seg::INPUT_H / (img.rows * 1.f);
    if (img.cols > img.rows) {
        l = 0;
        r = Seg::INPUT_W / 4;
        t = ((Seg::INPUT_H - r_w * img.rows) / 2) / 4;
        b = (Seg::INPUT_H - (Seg::INPUT_H - r_w * img.rows) / 2) / 4;
    } else {
        l = ((Seg::INPUT_W - r_h * img.cols) / 2) / 4;
        r = (Seg::INPUT_W - (Seg::INPUT_W - r_h * img.cols) / 2) / 4;
        t = 0;
        b = Seg::INPUT_H / 4;
    }
    cv::Mat ret = cv::Mat::zeros(b - t, r - l, CV_8UC3);
    for (auto result : res) {
        float box_left = (result.bbox[0] - result.bbox[2] / 2.f) / 4.f;
        float box_right = (result.bbox[0] + result.bbox[2] / 2.f) / 4.f;
        float box_top = (result.bbox[1] - result.bbox[3] / 2.f) / 4.f;
        float box_bot = (result.bbox[1] + result.bbox[3] / 2.f) / 4.f;
        for (int x = l; x < r; x++) {
            for (int y = t; y < b; y++) {
                if (x < box_left || x > box_right || y < box_top || y > box_bot) continue;
                float mask_prob = 0;
                int maskIdx = batch*160*160*Seg::MASK_NUM + x + y*160;
                for (int idx_ = 0; idx_ < Seg::MASK_NUM; idx_++) {
                    mask_prob += result.mask[idx_] * mask[maskIdx + idx_*160*160];
                }
                mask_prob = 1.0f / (1.0f + expf(-mask_prob));
                if (mask_prob > 0.5f) {
                    ret.at<cv::Vec3b>(y - t, x - l) = COLOR.at((int)result.class_id % 20);
                }
            }
        } 
    }
    cv::resize(ret, ret, cv::Size(img.cols, img.rows), 0, 0, cv::INTER_NEAREST);
    return ret;
}

cv::RotatedRect get_rotated_rect(cv::Mat& img, float bbox[4], float radian) {
    float l, r, t, b;
    float r_w = Radian::INPUT_W / (img.cols * 1.f);
    float r_h = Radian::INPUT_H / (img.rows * 1.f);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Radian::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Radian::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (Radian::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Radian::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::RotatedRect(cv::Point2f(round((l + r) / 2), round((t + b) / 2)), 
        cv::Size2f(round(r - l), round(b - t)), radian);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

template <class T>
bool cmp(const T& a, const T& b) {
    return a.conf > b.conf;
}

template <class T>
void nms(std::vector<T>& res, float *output, float conf_thresh, float nms_thresh = 0.5, unsigned max_output_count = 1000) {
    int det_size = sizeof(T) / sizeof(float);
    std::map<float, std::vector<T>> m;
    for (int i = 0; i < output[0] && i < max_output_count; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        T det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<T>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp<T>);
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

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 3;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    // silu = x * sigmoid
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

ILayer* focus(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname) {
    ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s2 = network->addSlice(input, Dims3{ 0, 1, 0 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s3 = network->addSlice(input, Dims3{ 0, 0, 1 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s4 = network->addSlice(input, Dims3{ 0, 1, 1 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
    return conv;
}

ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname) {
    auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

ILayer* bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = network->addConvolutionNd(input, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv2.weight"], emptywts);
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolutionNd(*y1, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv3.weight"], emptywts);

    ITensor* inputTensors[] = { cv3->getOutput(0), cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);

    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), lname + ".bn", 1e-4);
    auto lr = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    auto cv4 = convBlock(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, lname + ".cv4");
    return cv4;
}

ILayer* C3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv2");
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }

    ITensor* inputTensors[] = { y1, cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);

    auto cv3 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv3");
    return cv3;
}

ILayer* SPP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k1, k1 });
    pool1->setPaddingNd(DimsHW{ k1 / 2, k1 / 2 });
    pool1->setStrideNd(DimsHW{ 1, 1 });
    auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k2, k2 });
    pool2->setPaddingNd(DimsHW{ k2 / 2, k2 / 2 });
    pool2->setStrideNd(DimsHW{ 1, 1 });
    auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k3, k3 });
    pool3->setPaddingNd(DimsHW{ k3 / 2, k3 / 2 });
    pool3->setStrideNd(DimsHW{ 1, 1 });

    ITensor* inputTensors[] = { cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);

    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

ILayer* SPPF(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k, k });
    pool1->setPaddingNd(DimsHW{ k / 2, k / 2 });
    pool1->setStrideNd(DimsHW{ 1, 1 });
    auto pool2 = network->addPoolingNd(*pool1->getOutput(0), PoolingType::kMAX, DimsHW{ k, k });
    pool2->setPaddingNd(DimsHW{ k / 2, k / 2 });
    pool2->setStrideNd(DimsHW{ 1, 1 });
    auto pool3 = network->addPoolingNd(*pool2->getOutput(0), PoolingType::kMAX, DimsHW{ k, k });
    pool3->setPaddingNd(DimsHW{ k / 2, k / 2 });
    pool3->setStrideNd(DimsHW{ 1, 1 });
    ITensor* inputTensors[] = { cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);
    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname, int check_count) {
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"];
    int anchor_len = check_count * 2;
    for (int i = 0; i < wts.count / anchor_len; i++) {
        auto *p = (const float*)wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}

IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, 
    std::string lname, std::vector<IConvolutionLayer*> dets) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, lname, Yolo::CHECK_COUNT);
    PluginField plugin_fields[2];
    int netinfo[4] = {Yolo::CLASS_NUM, Yolo::INPUT_W, Yolo::INPUT_H, Yolo::MAX_OUTPUT_BBOX_COUNT};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    int scale = 8;
    std::vector<Yolo::YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        Yolo::YoloKernel kernel;
        kernel.width = Yolo::INPUT_W / scale;
        kernel.height = Yolo::INPUT_H / scale;
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        scale *= 2;
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<ITensor*> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;
}

IPluginV2Layer* addSegmentLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, 
    std::string lname, std::vector<IConvolutionLayer*> dets) {
    auto creator = getPluginRegistry()->getPluginCreator("SegmentLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, lname, Seg::CHECK_COUNT);
    PluginField plugin_fields[2];
    int netinfo[5] = {Seg::CLASS_NUM, Seg::MASK_NUM, Seg::INPUT_W, Seg::INPUT_H, Seg::MAX_OUTPUT_BBOX_COUNT};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 5;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    int scale = 8;
    std::vector<Seg::YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        Seg::YoloKernel kernel;
        kernel.width = Seg::INPUT_W / scale;
        kernel.height = Seg::INPUT_H / scale;
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        scale *= 2;
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2 *plugin_obj = creator->createPlugin("segmentlayer", &plugin_data);
    std::vector<ITensor*> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto segment = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return segment;
}

IPluginV2Layer* addRadianLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, 
    std::string lname, std::vector<IConcatenationLayer*> dets, unsigned kernel_index) {
    auto creator = getPluginRegistry()->getPluginCreator("RadianLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, lname, Radian::CHECK_COUNT);
    PluginField plugin_fields[2];
    int netinfo[4] = {Radian::CLASS_NUM, Radian::INPUT_W, Radian::INPUT_H, Radian::MAX_OUTPUT_BBOX_COUNT};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    int scale = 2;
    std::vector<Radian::YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        Radian::YoloKernel kernel;
        kernel.width = Radian::INPUT_W / scale;
        kernel.height = Radian::INPUT_H / scale;
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        scale *= 2;
    }
    plugin_fields[1].data = &kernels[kernel_index];
    plugin_fields[1].length = 1;
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2 *plugin_obj = creator->createPlugin("radianlayer", &plugin_data);
    std::vector<ITensor*> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto radian = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return radian;
}
#endif  // YOLOV5_COMMON_H_

