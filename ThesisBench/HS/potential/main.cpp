#include <stdio.h>
#include "clut.h"
#include "header.h"

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

//#define BLOCK_SIZE 16
#define BLOCK_SIZE 32
#define WIDTH 32
#define HEIGHT 32
#define THROTTLE_TB 1

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;

extern float tempArr[1024 * 1024];
extern float powerArr[1024 * 1024];

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

int compute_tran_temp(int col, int row, \
    int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows)
{
    clutInit();
    cl_program program = clutCompileProgram("_0.ptx");
    cl_kernel kernel = clutCreateKernel(program, "_Z13hotspotkerneliPfS_S_iiiiffffff");
    printf("cols=%d, rows=%d\n", blockCols, blockRows);
#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    float* MatrixTemp[2];
    MatrixTemp[0] = (float*)clutMalloc(sizeof(float) * col * row);
    MatrixTemp[1] = (float*)clutMalloc(sizeof(float) * col * row);
    float* MatrixPower = (float*)clutMalloc(sizeof(float) * col * row);

	float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	float t = 0;
    float time_elapsed = 0.001;

    int src = 1, dst = 0;

    // coarse-grained data preparation
	
    for(int y = 0; y < row; y++) {
        for(int x = 0; x < col; x++) {
            int index = y * col + x;
            MatrixTemp[0][index] = tempArr[index];
            MatrixPower[index] = powerArr[index];
        }
    }
    
    // kernel
    //size_t global_work[3] = {blockCols * BLOCK_SIZE, blockRows * BLOCK_SIZE, 1};
    //size_t local_work[3] = {BLOCK_SIZE, BLOCK_SIZE, 1};
    size_t global_work[3] = {blockCols * WIDTH, blockRows * HEIGHT, 1};
    size_t local_work[3] = {WIDTH, HEIGHT, 1};
    
    int temp = src;
    src = dst;
    dst = temp;
    int iteration = MIN(num_iterations, total_iterations-t);

    clutLaunchKernel(kernel, global_work, local_work, \
        iteration, MatrixPower, MatrixTemp[src], MatrixTemp[dst], \
        col, row, borderCols, borderRows, Cap, Rx, Ry, Rz, step, time_elapsed);
    
    clutFinish();

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif
        return dst;
}

int main(int argc, char* argv[]) {
    
    printf("Hello Hotspot\n");
    
    int grid_rows,grid_cols;
    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations
    
    if (argc >= 2) {
	    grid_rows = atoi(argv[1]);
	    grid_cols = atoi(argv[1]);
    }

    if (argc >= 3)
        pyramid_height = atoi(argv[2]);
    if (argc >= 4)
        total_iterations = atoi(argv[3]);
    if (argc < 4) {
	    printf("Wrong Usage: grid_rows/cols pyramid_height total_iterations\n");
        exit(0);
    }

    /* --------------- pyramid parameters --------------- */
    //# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
    # define EXPAND_RATE 0// add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    //int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    //int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    //int smallBlockCol = WIDTH-(pyramid_height)*EXPAND_RATE;
    //int smallBlockRow = HEIGHT-(pyramid_height)*EXPAND_RATE;
    int smallBlockCol = WIDTH;
    int smallBlockRow = HEIGHT;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);
    
    int ret = compute_tran_temp(grid_cols,grid_rows, \
	 total_iterations,pyramid_height, blockCols, blockRows, borderCols, borderRows);
    
    return 0;
}
