#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "preprocess.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.1f
#define CONF_THRESH 0.4f
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 5000 * 5000 // ensure it exceed the maximum size in the input images !

// stuff we know about the network and the input/output blobs
// static std::vector<cv::Vec3b> COLOR = {
//     cv::Vec3b(0xFF, 0x38, 0x38),
//     cv::Vec3b(0xFF, 0x9D, 0x97),
//     cv::Vec3b(0xFF, 0x70, 0x1F),
//     cv::Vec3b(0xFF, 0xB2, 0x1D),
//     cv::Vec3b(0xCF, 0xD2, 0x31),
//     cv::Vec3b(0x48, 0xF9, 0x0A),
//     cv::Vec3b(0x92, 0xCC, 0x17),
//     cv::Vec3b(0x3D, 0xDB, 0x86),
//     cv::Vec3b(0x1A, 0x93, 0x34),
//     cv::Vec3b(0x00, 0xD4, 0xBB),
//     cv::Vec3b(0x2C, 0x99, 0xA8),
//     cv::Vec3b(0x00, 0xC2, 0xFF),
//     cv::Vec3b(0x34, 0x45, 0x93),
//     cv::Vec3b(0x64, 0x73, 0xFF),
//     cv::Vec3b(0x00, 0x18, 0xEC),
//     cv::Vec3b(0x84, 0x38, 0xFF),
//     cv::Vec3b(0x52, 0x00, 0x85),
//     cv::Vec3b(0xCB, 0x38, 0xFF),
//     cv::Vec3b(0xFF, 0x95, 0xC8),
//     cv::Vec3b(0xFF, 0x37, 0xC7)
// };
static const int INPUT_H = Radian::INPUT_H;
static const int INPUT_W = Radian::INPUT_W;
static const int CLASS_NUM = Radian::CLASS_NUM;
static const int OUTPUT_SIZE = Radian::MAX_OUTPUT_BBOX_COUNT * sizeof(Radian::DetectionWithRad) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME0 = "prob0";
const char* OUTPUT_BLOB_NAME1 = "prob1";
const char* OUTPUT_BLOB_NAME2 = "prob2";
const char* OUTPUT_BLOB_NAME3 = "prob3";
static Logger gLogger;

static int get_width(int x, float gw, int divisor = 8) 
{
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) 
{
    if (x == 1) return 1;
    int r = (int)round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}

static void print_dims(Dims dim)
{
    for (int i = 0; i < dim.nbDims; i++) {
        std::cout << dim.d[i] << " ";
    }
    std::cout << std::endl;
}

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, 
    DataType dt, float& gd, float& gw, std::string& wts_name) 
{
    INetworkDefinition* network = builder->createNetworkV2(0U);
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    /* ------ yolov5 backbone------ */
    auto conv0 = convBlock(network, weightMap, *data,  get_width(64, gw), 6, 2, 1,  "model.0");
    assert(conv0);
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(512, gw), 3, 2, 1, "model.7");
    auto bottleneck_csp8 = C3(network, weightMap, *conv7->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto spp9 = SPPF(network, weightMap, *bottleneck_csp8->getOutput(0), get_width(512, gw), get_width(512, gw), 5, "model.9");
    /* ------ yolov5 head ------ */
    auto conv10 = convBlock(network, weightMap, *spp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");
    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());
    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13"); 
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");
    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());
    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(768, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    /* ------ detect ------ */
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 
        1 *(Radian::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.27.m_box.2.weight"], weightMap["model.27.m_box.2.bias"]);
    IConvolutionLayer* radian2 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 
        1 *(Radian::RAD_NUM), DimsHW{ 1, 1 }, weightMap["model.27.m_radian.2.weight"], weightMap["model.27.m_radian.2.bias"]);
    ITensor* inputResult2[] = { det2->getOutput(0), radian2->getOutput(0) };
    auto res2 = network->addConcatenation(inputResult2, 2);

    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), 
        get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");

    IConvolutionLayer* det3 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 
        1 *(Radian::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.27.m_box.3.weight"], weightMap["model.27.m_box.3.bias"]);
    IConvolutionLayer* radian3 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 
        1 *(Radian::RAD_NUM), DimsHW{ 1, 1 }, weightMap["model.27.m_radian.3.weight"], weightMap["model.27.m_radian.3.bias"]);
    ITensor* inputResult3[] = { det3->getOutput(0), radian3->getOutput(0) };
    auto res3 = network->addConcatenation(inputResult3, 2);
    
    float common_scale[] = {1., 2., 2.};
    auto upsample21 = network->addResize(*bottleneck_csp17->getOutput(0));
    assert(upsample21);
    upsample21->setResizeMode(ResizeMode::kNEAREST);
    upsample21->setScales(common_scale, 3);
    ITensor* inputTensors22[] = { upsample21->getOutput(0), bottleneck_CSP2->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), 
        get_width(640, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.23");   
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 
        1 *(Radian::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.27.m_box.1.weight"], weightMap["model.27.m_box.1.bias"]);
    IConvolutionLayer* radian1 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 
        1 *(Radian::RAD_NUM), DimsHW{ 1, 1 }, weightMap["model.27.m_radian.1.weight"], weightMap["model.27.m_radian.1.bias"]);
    ITensor* inputResult1[] = { det1->getOutput(0), radian1->getOutput(0) };
    auto res1 = network->addConcatenation(inputResult1, 2);

    auto upsample24 = network->addResize(*bottleneck_csp23->getOutput(0));
    assert(upsample24);
    upsample24->setResizeMode(ResizeMode::kNEAREST);
    upsample24->setScales(common_scale, 3);
    ITensor* inputTensors25[] = { upsample24->getOutput(0), conv0->getOutput(0) };
    auto cat25 = network->addConcatenation(inputTensors25, 2);
    auto bottleneck_csp26 = C3(network, weightMap, *cat25->getOutput(0), 
        get_width(288, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.26");
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp26->getOutput(0), 
        1 *(Radian::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.27.m_box.0.weight"], weightMap["model.27.m_box.0.bias"]);
    IConvolutionLayer* radian0 = network->addConvolutionNd(*bottleneck_csp26->getOutput(0), 
        1 *(Radian::RAD_NUM), DimsHW{ 1, 1 }, weightMap["model.27.m_radian.0.weight"], weightMap["model.27.m_radian.0.bias"]);
    ITensor* inputResult0[] = { det0->getOutput(0), radian0->getOutput(0) };
    auto res0 = network->addConcatenation(inputResult0, 2);
    
    auto yolo0 = addRadianLayer(network, weightMap, "model.27", std::vector<IConcatenationLayer *> { res0 }, 0);
    auto yolo1 = addRadianLayer(network, weightMap, "model.27", std::vector<IConcatenationLayer *> { res1 }, 1);
    auto yolo2 = addRadianLayer(network, weightMap, "model.27", std::vector<IConcatenationLayer *> { res2 }, 2);
    auto yolo3 = addRadianLayer(network, weightMap, "model.27", std::vector<IConcatenationLayer *> { res3 }, 3);
    yolo0->getOutput(0)->setName(OUTPUT_BLOB_NAME0);
    yolo1->getOutput(0)->setName(OUTPUT_BLOB_NAME1);
    yolo2->getOutput(0)->setName(OUTPUT_BLOB_NAME2);
    yolo3->getOutput(0)->setName(OUTPUT_BLOB_NAME3);
    network->markOutput(*yolo0->getOutput(0));
    network->markOutput(*yolo1->getOutput(0));
    network->markOutput(*yolo2->getOutput(0));
    network->markOutput(*yolo3->getOutput(0));
    print_dims(res0->getOutput(0)->getDimensions());
    print_dims(yolo0->getOutput(0)->getDimensions());

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, 
    float& gd, float& gw, std::string& wts_name) 
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, 
    float& gd, float& gw, std::string& img_dir) 
{
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net[0] == 'n') {
            gd = 0.33f;
            gw = 0.25f;
        } else if (net[0] == 's') {
            gd = 0.33f;
            gw = 0.50f;
        } else if (net[0] == 'm') {
            gd = 0.67f;
            gw = 0.75f;
        } else if (net[0] == 'l') {
            gd = 1.0f;
            gw = 1.0f;
        } else if (net[0] == 'x') {
            gd = 1.33f;
            gw = 1.25f;
        } else if (net[0] == 'c' && argc == 7) {
            gd = (float)atof(argv[5]);
            gw = (float)atof(argv[6]);
        } else {
            return false;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    std::string wts_name = "";
    std::string engine_name = "";
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;
    if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, img_dir)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6... or c/c6 gd gw] " 
                  << "// serialize model to plan file" << std::endl;
        std::cerr << "./yolov5 -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    if (!wts_name.empty()) {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, gd, gw, wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }
    
    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    static float prob0[BATCH_SIZE * OUTPUT_SIZE];
    static float prob1[BATCH_SIZE * OUTPUT_SIZE];
    static float prob2[BATCH_SIZE * OUTPUT_SIZE];
    static float prob3[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 5);
    float* buffers[5];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex0 = engine->getBindingIndex(OUTPUT_BLOB_NAME0);
    const int outputIndex1 = engine->getBindingIndex(OUTPUT_BLOB_NAME1);
    const int outputIndex2 = engine->getBindingIndex(OUTPUT_BLOB_NAME2);
    const int outputIndex3 = engine->getBindingIndex(OUTPUT_BLOB_NAME3);
    assert(inputIndex == 0);
    assert(outputIndex0 == 1);
    assert(outputIndex1 == 2);
    assert(outputIndex2 == 3);
    assert(outputIndex3 == 4);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex0], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex1], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex2], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex3], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    int fcount = 0;
    std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
    for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        //auto start = std::chrono::system_clock::now();
        float *buffer_idx = (float*)buffers[inputIndex];
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            imgs_buffer[b] = img;
            size_t size_image = img.cols * img.rows * 3;
            size_t size_image_dst = INPUT_H * INPUT_W * 3;
            //copy data to pinned memory
            memcpy(img_host, img.data, size_image);
            //copy data to device memory
            CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
            preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);
            buffer_idx += size_image_dst;
        }
        // Run inference
        auto start = std::chrono::system_clock::now();
        (*context).enqueue(BATCH_SIZE, (void**)buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(
            prob0, buffers[outputIndex0], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            prob1, buffers[outputIndex1], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            prob2, buffers[outputIndex2], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            prob3, buffers[outputIndex3], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " 
                    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                    << "ms" << std::endl;
        std::vector<float*> prob { prob0, prob1, prob2, prob3 };
        std::vector<float> layer_prob { 0, 0, 0, 0 };
        unsigned total_count;
        float avg_width = 0.;
        float avg_area = 0.;
        unsigned best_index = 0;
        for (int i = 0; i < prob.size(); i++) {
            double avg_prob = 0.;
            unsigned count = BATCH_SIZE * OUTPUT_SIZE;
            total_count += count;
            for (int j = 1; j < count; j += 7) {
                if (prob[i][j + 4] < 0.1) continue; 
                avg_prob += prob[i][j + 4];
                avg_width += prob[i][j + 2];
                avg_area += prob[i][j + 2] * prob[i][j + 3];
            }
            avg_prob /= count;
            layer_prob[i] = avg_prob;
        }
        avg_area /= total_count;
        avg_width /= total_count;
        // choose layer
        if (avg_area < 20 || avg_width < 5) {
            best_index = 0;
        } else if (avg_width < 12) {
            best_index = 1;
        } else {
            for (int i = 1; i < layer_prob.size(); i++) {
                if (layer_prob[i] > layer_prob[best_index]) {
                    best_index = i;
                }
            }
            if (best_index == 2 && layer_prob[2] - layer_prob[1] < 0.8) {
                best_index = 1;
            }
        }
        std::vector<std::vector<Radian::DetectionWithRad>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms<Radian::DetectionWithRad>(res, &prob[best_index][b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH, Radian::MAX_OUTPUT_BBOX_COUNT);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            cv::Mat img = imgs_buffer[b];
            for (size_t j = 0; j < res.size(); j++) {
                cv::RotatedRect r = get_rotated_rect(img, res[j].bbox, res[j].radian);
                cv::Point2f vertices[4];
                r.points(vertices);
                for (int v = 0; v < 4; v++) {
                    cv::line(img, vertices[v], vertices[(v + 1)%4], cv::Scalar(0x27, 0xC1, 0x36), 2);
                }
                // cv::putText(img, std::to_string((int)res[j].class_id), r.center, cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex0]));
    CUDA_CHECK(cudaFree(buffers[outputIndex1]));
    CUDA_CHECK(cudaFree(buffers[outputIndex2]));
    CUDA_CHECK(cudaFree(buffers[outputIndex3]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
