#define BLOCK_SIZE 32
#define PADDING 2

__global__ void gaborFilterKernel(float* o_arr1d, float* i_arr1d, int rows, int cols, float* filter1d, int radius) {
    
    __shared__ float tile[BLOCK_SIZE+PADDING*2][BLOCK_SIZE+PADDING*2];
    __shared__ float filter_buf[2*PADDING+1][2*PADDING+1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    
    tile[ty+PADDING][tx+PADDING] = i_arr1d[yidx * cols + xidx];
    if(tx < 2*PADDING+1 && ty < 2*PADDING+1)
        filter_buf[ty][tx] = filter1d[ty * (2*PADDING+1) + tx];
    /*if(tx < PADDING)
        tile[ty+PADDING][tx] = i_arr1d[yidx * cols + (xidx - tx)];
    if(tx >= BLOCK_SIZE - PADDING)
        tile[ty+PADDING][tx+PADDING*2] = i_arr1d[yidx * cols + (xidx - tx + BLOCK_SIZE - 1)];
    if(ty < PADDING)
        tile[ty][tx+PADDING] = i_arr1d[(yidx - ty) * cols + xidx];
    if(ty >= BLOCK_SIZE - PADDING)
        tile[ty+PADDING*2][tx+PADDING] = i_arr1d[(yidx - ty + BLOCK_SIZE - 1) * cols + xidx];
    if(tx < PADDING && ty < PADDING)
        tile[ty][tx] = i_arr1d[(yidx - ty) * cols + xidx - tx];
    if(tx >= BLOCK_SIZE - PADDING && ty < PADDING)
        tile[ty][tx+PADDING*2] = i_arr1d[(yidx - ty) * cols + (xidx - tx + BLOCK_SIZE - 1)];
    if(tx < PADDING && ty >= BLOCK_SIZE - PADDING)
        tile[ty+PADDING*2][tx] = i_arr1d[(yidx - ty + BLOCK_SIZE - 1) * cols + xidx - tx];
    if(tx >= BLOCK_SIZE - PADDING && ty >= BLOCK_SIZE - PADDING)
        tile[ty+PADDING*2][tx+PADDING*2] = i_arr1d[(yidx - ty + BLOCK_SIZE - 1) * cols + xidx - tx + BLOCK_SIZE - 1];*/
    __syncthreads();

    float Io = 0.0f;
    float accumWeight = 0.0f;
    for(int dy = -radius; dy <= radius; dy++) {
        for(int dx = -radius; dx <= radius; dx++) {
            float weight = filter_buf[dy + radius][dx + radius];
            //float weight = filter1d[(dy + radius) * radius + (dx + radius)];
            float I = tile[ty + dy + PADDING][tx + dx + PADDING];
            Io += I * weight;
            accumWeight += weight;
        }
    }
    
    // normalize weight
    Io /= accumWeight;
    o_arr1d[yidx * cols + xidx] = Io;
    //o_arr1d[yidx * cols + xidx] = tile[ty+PADDING][tx+PADDING];
}
