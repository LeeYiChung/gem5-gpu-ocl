#include <opencv2/opencv.hpp>
#include <vector>
#include "clut.h"
#include "header.h"
#define THROTTLE_TB 1
#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif
using namespace std;

extern float HPYR[5][512][512];
void buildDoGPyramid(float* dogpyr, cv::Mat& image, int nOctaveLayers, double sigma)
{
    int idx = 0;
    for(int i = 0; i < (nOctaveLayers + 2); i++) {
        for(int r = 0; r < image.rows; r++) {
            for(int c = 0; c < image.cols; c+=8) {
                dogpyr[idx] = HPYR[i + 1][r][c] - HPYR[i][r][c];
                dogpyr[idx+1] = HPYR[i + 1][r][c+1] - HPYR[i][r][c+1];
                dogpyr[idx+2] = HPYR[i + 1][r][c+2] - HPYR[i][r][c+2];
                dogpyr[idx+3] = HPYR[i + 1][r][c+3] - HPYR[i][r][c+3];
                dogpyr[idx+4] = HPYR[i + 1][r][c+4] - HPYR[i][r][c+4];
                dogpyr[idx+5] = HPYR[i + 1][r][c+5] - HPYR[i][r][c+5];
                dogpyr[idx+6] = HPYR[i + 1][r][c+6] - HPYR[i][r][c+6];
                dogpyr[idx+7] = HPYR[i + 1][r][c+7] - HPYR[i][r][c+7];
                idx += 8;
            }
        }
    }
}

void localmax(int* flags, float* dogpyr, int rows, int cols, float contrastThreshold, float curvateThreshold)
{
    cl_program program = clutCompileProgram("_0.ptx");
    cl_kernel kernel = clutCreateKernel(program, "_Z20KeypointLocalizationPiPfiiff");
 
    size_t globalWorkSize[3] = {cols, rows, 4};
    size_t localWorkSize[3] = {16, 16, 4};

    clutLaunchKernel(kernel, globalWorkSize, localWorkSize, \
        flags, dogpyr, rows, cols, contrastThreshold, curvateThreshold);
    // fine-grained preparation and synchronization
    for(int by = 0; by < 32; by++) {
        for(int bx = 0; bx < 32; bx++) {
            for(int tz = 0; tz < 4; tz++) {
                for(int ty = 0; ty < 16; ty++) {
                    int yidx = by * 16 + ty;
                    int xidx = bx * 16;
                    int idx = tz * 512 * 512 + yidx * 512 + xidx;
                    dogpyr[idx] = HPYR[tz + 1][yidx][xidx] - HPYR[tz][yidx][xidx];
                    dogpyr[idx+1] = HPYR[tz + 1][yidx][xidx+1] - HPYR[tz][yidx][xidx+1];
                    dogpyr[idx+2] = HPYR[tz + 1][yidx][xidx+2] - HPYR[tz][yidx][xidx+2];
                    dogpyr[idx+3] = HPYR[tz + 1][yidx][xidx+3] - HPYR[tz][yidx][xidx+3];
                    dogpyr[idx+4] = HPYR[tz + 1][yidx][xidx+4] - HPYR[tz][yidx][xidx+4];
                    dogpyr[idx+5] = HPYR[tz + 1][yidx][xidx+5] - HPYR[tz][yidx][xidx+5];
                    dogpyr[idx+6] = HPYR[tz + 1][yidx][xidx+6] - HPYR[tz][yidx][xidx+6];
                    dogpyr[idx+7] = HPYR[tz + 1][yidx][xidx+7] - HPYR[tz][yidx][xidx+7];
                    dogpyr[idx+8] = HPYR[tz + 1][yidx][xidx+8] - HPYR[tz][yidx][xidx+8];
                    dogpyr[idx+9] = HPYR[tz + 1][yidx][xidx+9] - HPYR[tz][yidx][xidx+9];
                    dogpyr[idx+10] = HPYR[tz + 1][yidx][xidx+10] - HPYR[tz][yidx][xidx+10];
                    dogpyr[idx+11] = HPYR[tz + 1][yidx][xidx+11] - HPYR[tz][yidx][xidx+11];
                    dogpyr[idx+12] = HPYR[tz + 1][yidx][xidx+12] - HPYR[tz][yidx][xidx+12];
                    dogpyr[idx+13] = HPYR[tz + 1][yidx][xidx+13] - HPYR[tz][yidx][xidx+13];
                    dogpyr[idx+14] = HPYR[tz + 1][yidx][xidx+14] - HPYR[tz][yidx][xidx+14];
                    dogpyr[idx+15] = HPYR[tz + 1][yidx][xidx+15] - HPYR[tz][yidx][xidx+15];           
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
    m5_work_end(0, 0);
}

void SIFT(vector<cv::KeyPoint>& keypoints, cv::Mat image, int nOctaves, int nOctaveLayers, double sigma, float contrastThreshold, float curvateThreshold)
{
    m5_work_begin(0, 0);
    float* dogpyr = (float*)clutMalloc((nOctaveLayers + 2) * image.rows * image.cols * sizeof(float));
    int* flags = (int*)clutMalloc((nOctaveLayers + 2) * image.rows * image.cols * sizeof(int));
    localmax(flags, dogpyr, image.rows, image.cols, contrastThreshold, curvateThreshold);
    cv::KeyPoint kpoint;
    for(int l = 0; l < nOctaveLayers + 2; l++) {
        for(int r = 0; r < image.rows; r++) {
            for(int c = 0; c < image.cols; c++) {
                if(flags[l * image.rows * image.cols + r * image.cols + c]) {
                    kpoint.pt.x = c;
                    kpoint.pt.y = r;
                    keypoints.push_back(kpoint);
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
	int nOctaves = 1;
	int nOctaveLayers = 2;
	double sigma = 1.6;
	float contrastThreshold = 0.03;
    float curvateThreshold = 12.1;
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    printf("finish read image\n");
    
    vector<cv::KeyPoint> keypoints;
    cv::Mat outImage;

    clutInit();
    SIFT(keypoints, image, nOctaves, nOctaveLayers, sigma, contrastThreshold, curvateThreshold);
    cv::drawKeypoints(image, keypoints, outImage, cv::Scalar(2, 2, 225));
    imwrite("result.bmp", outImage);
}
