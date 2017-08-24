#include <opencv2/opencv.hpp>
#include <vector>
#include "clut.h"
#include "header.h"
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
    m5_work_begin(0, 0);
    cl_program program = clutCompileProgram("KeypointLocalization.cl");
    cl_kernel kernel = clutCreateKernel(program, "_Z20KeypointLocalizationPiPfiiff");
 
    size_t globalWorkSize[3] = {cols, rows, 4};
    size_t localWorkSize[3] = {16, 16, 4};

    clutLaunchKernel(kernel, globalWorkSize, localWorkSize, \
        flags, dogpyr, rows, cols, contrastThreshold, curvateThreshold);
    
    clutFinish();
    m5_work_end(0, 0);
}

void SIFT(vector<cv::KeyPoint>& keypoints, cv::Mat image, int nOctaves, int nOctaveLayers, double sigma, float contrastThreshold, float curvateThreshold)
{
    m5_work_begin(0, 0);
    
    float* dogpyr = (float*)clutMalloc((nOctaveLayers + 2) * image.rows * image.cols * sizeof(float));
    int* flags = (int*)clutMalloc((nOctaveLayers + 2) * image.rows * image.cols * sizeof(int));
    buildDoGPyramid(dogpyr, image, nOctaveLayers, sigma);
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
