#define BLOCK_WIDTH 32
//#define BLOCK_HEIGHT 32
#define BLOCK_HEIGHT 16
#define TILE_WIDTH 96
//#define TILE_HEIGHT 96
#define TILE_HEIGHT 48
#define SMALL_TILE_WIDTH 64
//#define SMALL_TILE_HEIGHT 64
#define SMALL_TILE_HEIGHT 32

__global__ void superResolutionKernel(float* o_arr1d, float* i_arr1d, int rows, int cols) {
    
    __shared__ float tile[TILE_HEIGHT][TILE_WIDTH];
    __shared__ float buf[SMALL_TILE_HEIGHT][TILE_WIDTH]; // buffer for down/up sampling
    __shared__ float small_tile[SMALL_TILE_HEIGHT][SMALL_TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for(int offset_y = 0; offset_y < TILE_HEIGHT; offset_y += BLOCK_HEIGHT) {
        for(int offset_x = 0; offset_x < TILE_WIDTH; offset_x += BLOCK_WIDTH) {
            int xidx = blockIdx.x * TILE_WIDTH + offset_x + tx;
            int yidx = blockIdx.y * TILE_HEIGHT + offset_y + ty;
            tile[offset_y + ty][offset_x + tx] = i_arr1d[yidx * cols + xidx];
        }
    }
    __syncthreads();

    // col downsample: (TILE_HEIGHT, TILE_WIDTH) -> (SMALL_TILE_HEIGHT, TILE_WIDTH)
    for(int offset_x = 0; offset_x < TILE_WIDTH; offset_x += BLOCK_WIDTH) {
        buf[2*ty][offset_x + tx] = -0.022 * tile[3*ty][offset_x + tx] + 0.974 * tile[3*ty+1][offset_x + tx] + 0.227 * tile[3*ty+2][offset_x + tx];
        buf[2*ty+1][offset_x + tx] = 0.227 * tile[3*ty][offset_x + tx] + 0.974 * tile[3*ty+1][offset_x + tx] + -0.022 * tile[3*ty+2][offset_x + tx];
    }
    __syncthreads();

    // row downsample: (SMALL_TILE_HEIGHT, TILE_WIDTH) -> (SMALL_TILE_HEIGHT, SMALL_TILE_WIDTH)
    for(int offset_y = 0; offset_y < SMALL_TILE_HEIGHT; offset_y += BLOCK_HEIGHT) {
        small_tile[offset_y + ty][2*tx] = -0.022 * buf[offset_y + ty][3*tx] + 0.974 * buf[offset_y + ty][3*tx+1] + 0.227 * buf[offset_y + ty][3*tx+2];
        small_tile[offset_y + ty][2*tx+1] = 0.227 * buf[offset_y + ty][3*tx] + 0.974 * buf[offset_y + ty][3*tx+1] + -0.022 * buf[offset_y + ty][3*tx+2];
    }
    __syncthreads();

    // row upsample: (SMALL_TILE_HEIGHT, SMALL_TILE_WIDTH) -> (SMALL_TILE_HEIGHT, TILE_WIDTH)
    for(int offset_y = 0; offset_y < SMALL_TILE_HEIGHT; offset_y += BLOCK_HEIGHT) {
        // 3*tx
        if(tx == 0)
            buf[offset_y + ty][0] = 0.927 * small_tile[offset_y + ty][0] + -0.047 * small_tile[offset_y + ty][1];
        else
            buf[offset_y + ty][3*tx] = 0.119 * small_tile[offset_y + ty][2*tx-1] + 0.927 * small_tile[offset_y + ty][2*tx]
                + -0.047 * small_tile[offset_y + ty][2*tx+1];
        // 3*tx+2
        if(tx == BLOCK_WIDTH-1)
            buf[offset_y + ty][TILE_WIDTH-1] = -0.047 * small_tile[offset_y + ty][SMALL_TILE_WIDTH-2] + 0.927 * small_tile[offset_y + ty][SMALL_TILE_WIDTH-1];
        else
            buf[offset_y + ty][3*tx+2] = -0.047 * small_tile[offset_y + ty][2*tx] + 0.927 * small_tile[offset_y + ty][2*tx+1]
                + 0.119 * small_tile[offset_y + ty][2*tx+2];
        // 3*tx+1
        if(tx == 0)
            buf[offset_y + ty][1] = 0.6 * small_tile[offset_y + ty][0] + 0.6 * small_tile[offset_y + ty][1] + -0.1 * small_tile[offset_y + ty][2];
        else if(tx == BLOCK_WIDTH-1)
            buf[offset_y + ty][TILE_WIDTH-2] = -0.1 * small_tile[offset_y + ty][SMALL_TILE_WIDTH-3] + 0.6 * small_tile[offset_y + ty][SMALL_TILE_WIDTH-2] 
                + 0.6 * small_tile[offset_y + ty][SMALL_TILE_WIDTH-1];
        else
            buf[offset_y + ty][3*tx+1] = -0.1 * small_tile[offset_y + ty][2*tx-1] + 0.6 * small_tile[offset_y + ty][2*tx]
                + 0.6 * small_tile[offset_y + ty][2*tx+1] + -0.1 * small_tile[offset_y + ty][2*tx+2];
    }
    __syncthreads();

    // col upsample: (SMALL_TILE_HEIGHT, TILE_WIDTH) -> (TILE_HEIGHT, TILE_WIDTH)
    for(int offset_x = 0; offset_x < TILE_WIDTH; offset_x += BLOCK_WIDTH) {
        int xidx = blockIdx.x * TILE_WIDTH + offset_x + tx;
        // 3*ty
        if(ty == 0) {
            int yidx = blockIdx.y * TILE_HEIGHT;
            o_arr1d[yidx * cols + xidx] = 0.927 * buf[0][offset_x + tx] + -0.047 * buf[1][offset_x + tx];
        
        }
        else {
            int yidx = blockIdx.y * TILE_HEIGHT + 3*ty;
            o_arr1d[yidx * cols + xidx] = 0.119 * buf[2*tx-1][offset_x + tx] + 0.927 * buf[2*tx][offset_x + tx]
                + -0.047 * buf[2*tx+1][offset_x + tx];
        }
        // 3*ty+2
        if(ty == BLOCK_HEIGHT-1) {
            int yidx = blockIdx.y * TILE_HEIGHT + TILE_HEIGHT-1;
            o_arr1d[yidx * cols + xidx] = -0.047 * buf[SMALL_TILE_HEIGHT-2][offset_x + tx] + 0.927 * buf[SMALL_TILE_HEIGHT-1][offset_x + tx];
        }
        else {
            int yidx = blockIdx.y * TILE_HEIGHT + 3*ty+2;
            o_arr1d[yidx * cols + xidx] = -0.047 * buf[2*ty][offset_x + tx] + 0.927 * buf[2*ty+1][offset_x + tx]
                + 0.119 * small_tile[2*ty+2][offset_x + tx];
        }
        // 3*ty+1
        if(ty == 0) {
            int yidx = blockIdx.y * TILE_HEIGHT + 1;
            o_arr1d[yidx * cols + xidx] = 0.6 * buf[0][offset_x + tx] + 0.6 * buf[1][offset_x + tx] + -0.1 * buf[2][offset_x + tx];
        }
        else if(ty == BLOCK_HEIGHT-1) {
            int yidx = blockIdx.y * TILE_HEIGHT + TILE_HEIGHT-2;
            o_arr1d[yidx * cols + xidx] = -0.1 * buf[SMALL_TILE_HEIGHT-3][offset_x + tx] + 0.6 * buf[SMALL_TILE_HEIGHT-2][offset_x + tx] 
                + 0.6 * buf[SMALL_TILE_HEIGHT-1][offset_x + tx];
        }
        else {
            int yidx = blockIdx.y * TILE_HEIGHT + 3*ty+1;
            o_arr1d[yidx * cols + xidx] = -0.1 * buf[2*ty-1][offset_x + tx] + 0.6 * buf[2*ty][offset_x + tx]
                + 0.6 * buf[2*ty+1][offset_x + tx] + -0.1 * buf[2*ty+2][offset_x + tx];
        }
    }


}
