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
#define THROTTLE_TB 16
#include "header.h"
extern uint8_t hImage1[640][1280];
extern uint8_t hImage2[640][1280];
extern int hRows;
extern int hCols;

void opticalFlow(cv::Mat& srcImage1, cv::Mat& srcImage2, cv::Size window, int iteration) {
    
    clutInit();
    cl_program program = clutCompileProgram("_0.ptx");
    cl_kernel kernel = clutCreateKernel(program, "_Z17opticalFlowKernelPiS_S_S_iiiii");

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    int* arr1d1 = (int*)clutMalloc(sizeof(float) * hRows * hCols);
    int* arr1d2 = (int*)clutMalloc(sizeof(float) * hRows * hCols);
    int* gy = (int*)clutMalloc(sizeof(float) * hRows * hCols);
    int* gx = (int*)clutMalloc(sizeof(float) * hRows * hCols);
    
    //printf("Baseline: preparing whole input data\n");
    for(int y = 0; y < srcImage1.rows; y++) {
        for(int x = 0; x < srcImage1.cols; x++) {
            arr1d1[y * srcImage1.cols + x] = hImage1[y][x];
            arr1d2[y * srcImage1.cols + x] = hImage2[y][x];
        }
    }
    
    //printf("launch kernel\n");
    size_t globalWorkSize[3] = {hCols, hRows, 1};
    size_t localWorkSize[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    
    clutLaunchKernel(kernel, globalWorkSize, localWorkSize, \
        gx, gy, arr1d1, arr1d2, hRows, hCols, window.width, window.height, iteration);
    
    clutFinish();

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

}

int main(int argc, char* argv[]) {
    
    //cv::Mat srcImage1 = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    //cv::Mat srcImage2 = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat srcImage1;
    cv::Mat srcImage2;
    cv::Size window = cv::Size(1, 1);
    int iteration = 1;
    opticalFlow(srcImage1, srcImage2, window, iteration);
    return 0;
}
