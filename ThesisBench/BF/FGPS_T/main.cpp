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
extern uint8_t hImage[1080][1920][3];
extern int hRows;
extern int hCols;

void bilateralFilter(cv::Mat& srcImage, cv::Mat& dstImage, int radius=0, float sigmaColor=0, float sigmaSpace=0) {
    
    clutInit();
    cl_program program = clutCompileProgram("BilateralFilterKernel.cu");
    cl_kernel kernel = clutCreateKernel(program, "_Z21bilateralFilterKernelPfS_iiiff");

    float gauss_color_coeff = -0.5 / (sigmaColor * sigmaColor);    
    float gauss_space_coeff = -0.5 / (sigmaSpace * sigmaSpace);
    if(radius <= 0)
        radius = sigmaSpace * 1.5;
    
    float* i_arr1d = (float*)clutMalloc(sizeof(float) * hRows * hCols * 3);
    float* o_arr1d = (float*)clutMalloc(sizeof(float) * hRows * hCols * 3);
    
    printf("preparing data\n");
    
    m5_work_begin(0, 0);
    
    // kernel
    size_t globalWorkSize[3] = {hCols, (hRows/BLOCK_SIZE)*BLOCK_SIZE, 1};
    size_t localWorkSize[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    
    clutLaunchKernel(kernel, globalWorkSize, localWorkSize, \
        o_arr1d, i_arr1d, hRows, hCols, radius, gauss_color_coeff, gauss_space_coeff);
    
    // finegrain preparation and synchronization
    for(int by = 0; by < hRows/BLOCK_SIZE; by++) {
        for(int bx = 0; bx < hCols/BLOCK_SIZE; bx++) {
            for(int ty = 0; ty < BLOCK_SIZE; ty++) {
                for(int tx = 0; tx < BLOCK_SIZE; tx++) { 
                    int yidx = by * BLOCK_SIZE + ty;
                    int xidx = bx * BLOCK_SIZE + tx;
                    i_arr1d[yidx * hCols + xidx] = hImage[yidx][xidx][0];
                    i_arr1d[hRows * hCols + yidx * hCols + xidx] = hImage[yidx][xidx][1];
                    i_arr1d[2 * hRows * hCols + yidx * hCols + xidx] = hImage[yidx][xidx][2];
                }
            }
            size_t blockIdx[3] = {bx, by, 0};
            clSignalBlockReady(blockIdx);
            while(clReadReadyBlockCounter() == THROTTLE_TB) {
                // wait
            }
        }
    }

    clutFinish();

    m5_work_end(0, 0);
    
    dstImage.create(hRows, hCols, CV_8UC3);
    for(int r = 0; r < hRows; r++) {
        for(int c = 0; c < hCols; c++) {
            dstImage.at<cv::Vec3b>(r, c)[0] = o_arr1d[r * hCols + c];
            dstImage.at<cv::Vec3b>(r, c)[1] = o_arr1d[hRows * hCols + r * hCols + c];
            dstImage.at<cv::Vec3b>(r, c)[2] = o_arr1d[2 * hRows * hCols + r * hCols + c];
        }
    }

}

int main(int argc, char* argv[]) {
    printf("Hello BilateralFilter\n");
    //cv::Mat srcImage = cv::imread(argv[1]);
    cv::Mat srcImage;
    cv::Mat dstImage;
    bilateralFilter(srcImage, dstImage, 3, 150, 150);
    cv::imwrite("result.jpg", dstImage);
    return 0;
}