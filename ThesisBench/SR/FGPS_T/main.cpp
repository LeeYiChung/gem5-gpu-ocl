#include <opencv2/opencv.hpp>
#include "clut.h"

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#define THROTTLE_TB 1
#define BLOCK_WIDTH 32
//#define BLOCK_HEIGHT 32
#define BLOCK_HEIGHT 16
#define TILE_WIDTH 96
//#define TILE_HEIGHT 96
#define TILE_HEIGHT 48

int sat(float value) {
    int ret;
    if(value > 255)
        ret = 255;
    else if(value < 0)
        ret =  0;
    else
        ret = value;
    return ret;
}

void superResolution(cv::Mat& srcImage, cv::Mat& dstImage) {
    
    clutInit();
    cl_program program = clutCompileProgram("_0.ptx");
    cl_kernel kernel = clutCreateKernel(program, "_Z21superResolutionKernelPfS_ii");
    
#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    float* i_arr1d = (float*)clutMalloc(sizeof(float) * srcImage.rows * srcImage.cols * 3);
    float* o_arr1d = (float*)clutMalloc(sizeof(float) * srcImage.rows * srcImage.cols * 3);
    
    size_t globalWorkSize[3] = {(srcImage.cols/TILE_WIDTH)*BLOCK_WIDTH, (srcImage.rows/TILE_HEIGHT)*BLOCK_HEIGHT, 1};
    size_t localWorkSize[3] = {BLOCK_WIDTH, BLOCK_HEIGHT, 1};
    
    clutLaunchKernel(kernel, globalWorkSize, localWorkSize, \
        o_arr1d, i_arr1d, srcImage.rows, srcImage.cols);
    
    for(int by = 0; by < srcImage.rows/TILE_HEIGHT; by++) {
        for(int bx = 0; bx < srcImage.cols/TILE_WIDTH; bx++) {
            for(int ty = 0; ty < TILE_HEIGHT; ty++) {
                for(int tx = 0; tx < TILE_WIDTH; tx++) { 
                    int yidx = by * TILE_HEIGHT + ty;
                    int xidx = bx * TILE_WIDTH + tx;
                    i_arr1d[yidx * srcImage.cols + xidx] = srcImage.at<uchar>(yidx, xidx);
                }
            }
            while(clReadReadyBlockCounter() == THROTTLE_TB) {
                // wait
            }
            size_t blockIdx[3] = {bx, by, 0};
            clSignalBlockReady(blockIdx);
        }
    }
    
    clutFinish();

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif
    
    for(int y = 0; y < dstImage.rows; y++) {
        for(int x = 0; x < dstImage.cols; x++) {
            dstImage.at<uchar>(y, x) = sat(o_arr1d[y * dstImage.cols + x]);
        }
    }
}

int main(int argc, char* argv[]) {

    cv::Mat srcImage = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat dstImage(srcImage.rows, srcImage.cols, CV_8UC1);
    superResolution(srcImage, dstImage);

    cv::imwrite("srcImage.bmp", srcImage);
    cv::imwrite("dstImage.bmp", dstImage);
    return 0;

}
