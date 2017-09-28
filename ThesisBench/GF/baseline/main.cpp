#include <opencv2/opencv.hpp>
#include "clut.h"

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#define BLOCK_SIZE 32
#define THROTTLE_TB 1
#include "header.h"
extern uint8_t hImage[1080][1920];
extern int hRows;
extern int hCols;

void gaborFilter(cv::Mat& srcImage, cv::Mat& dstImage, int radius, double sigma, double theta, double lambd, double gamma, double psi) {

    clutInit();
    cl_program program = clutCompileProgram("_0.ptx");
    cl_kernel kernel = clutCreateKernel(program, "_Z17gaborFilterKernelPfS_iiS_i");
#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif
    
    int diameter = 2 * radius + 1;
    cv::Mat gaborKernel = cv::getGaborKernel(cv::Size(diameter, diameter), sigma, theta, lambd, gamma, psi, CV_32F);
    float* filter1d = (float*)clutMalloc(sizeof(float) * diameter * diameter);
    float* i_arr1d = (float*)clutMalloc(sizeof(float) * hRows * hCols);
    float* o_arr1d = (float*)clutMalloc(sizeof(float) * hRows * hCols);
    for(int y = 0; y < diameter; y++) {
        for(int x = 0; x < diameter; x++) {
            filter1d[y * diameter + x] = gaborKernel.at<float>(y, x);
        }
    }
    
    // coarse-grain data preparation
    for(int y = 0; y < hRows; y++) {
        for(int x = 0; x < hCols; x++) {
            i_arr1d[y * hCols + x] = hImage[y][x];
        }
    }
    

    // kernel
    size_t globalWorkSize[3] = {hCols, (hRows/BLOCK_SIZE)*BLOCK_SIZE, 1};
    size_t localWorkSize[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    
    clutLaunchKernel(kernel, globalWorkSize, localWorkSize, \
        o_arr1d, i_arr1d, hRows, hCols, filter1d, radius);

    clutFinish();
    
#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif
    
    dstImage.create(srcImage.rows, srcImage.cols, CV_8UC1);
    for(int y = 0; y < dstImage.rows; y++) {
        for(int x = 0; x < dstImage.cols; x++) {
            dstImage.at<uchar>(y, x) = o_arr1d[y * dstImage.cols + x];
        }
    }
}

int main(int argc, char* argv[]) {
    //cv::Mat srcImage = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat srcImage;;
    cv::Mat dstImage;
    double radius = 2;
    double sigma = 4.0;
    double theta = CV_PI/2;
    double lambd = 10;
    double gamma = 0.5;
    double psi = 0;
    
    gaborFilter(srcImage, dstImage, radius, sigma, theta, lambd, gamma, psi);
    
    cv::imwrite("result.jpg", dstImage);
    return 0;
}
