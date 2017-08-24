#define BLOCK_SIZE 32
#define BLOCK_DEPTH 3
#define POW2(x) ((x) * (x))
__global__ void bilateralFilterKernel(float* o_arr1d, float* i_arr1d, int rows, int cols, int radius, float gauss_color_coeff, float gauss_space_coeff) {
    
    __shared__ float tile[BLOCK_DEPTH][BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;

    tile[0][ty][tx] = i_arr1d[yidx * cols + xidx];
    tile[1][ty][tx] = i_arr1d[rows * cols + yidx * cols + xidx];
    tile[2][ty][tx] = i_arr1d[2 * rows * cols + yidx * cols + xidx];

    __syncthreads();

    float Iir = tile[0][ty][tx];
    float Iig = tile[1][ty][tx];
    float Iib = tile[2][ty][tx];
    float Ior = 0.0f;
    float Iog = 0.0f;
    float Iob = 0.0f;
    float accumWeight = 0.0f;

    for(int dy = -radius; dy <= radius; dy++) {
        for(int dx = -radius; dx <= radius; dx++) {
            // test boundary of tile
            if(tx + dx >= 0 && tx + dx <= BLOCK_SIZE - 1 && ty + dy >= 0 && ty + dy <= BLOCK_SIZE - 1) {
                float Ir = tile[0][ty + dy][tx + dx];
                float Ig = tile[1][ty + dy][tx + dx];
                float Ib = tile[2][ty + dy][tx + dx];
                // intensity weight exponent
                float expi = gauss_color_coeff * (POW2(Ir - Iir) + POW2(Ig - Iig) + POW2(Ib - Iib));
                // spatial weight exponent
                float exps = gauss_space_coeff * (POW2(dx) + POW2(dy));
                
                float weight = expf(expi + exps);
                Ior += Ir * weight;
                Iog += Ig * weight;
                Iob += Ib * weight;
                accumWeight += weight;
            }
        }
    }

    // normalize weight
    Ior /= accumWeight;
    Iog /= accumWeight;
    Iob /= accumWeight;

    // write result to global memory
    o_arr1d[yidx * cols + xidx] = Ior;
    o_arr1d[rows * cols + yidx * cols + xidx] = Iog;
    o_arr1d[2 * rows * cols + yidx * cols + xidx] = Iob;

}
