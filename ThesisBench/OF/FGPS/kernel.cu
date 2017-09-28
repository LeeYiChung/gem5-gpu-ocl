#define BLOCK_SIZE 32
__global__ void opticalFlowKernel(int* o_gx, int* o_gy, int* arr1dI, int* arr1dJ, int rows, int cols, int wx, int wy, int K) {
    
    __shared__ int I_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int J_tile[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
   
    I_tile[ty][tx] = arr1dI[yidx * cols + xidx];
    J_tile[ty][tx] = arr1dJ[yidx * cols + xidx];

    __syncthreads();

    // Initialization of pyramidal guess
    float gx = 0, gy = 0;
    
    // Spatial gradient matrix
    float Gxx = 0, Gxy = 0, Gyy = 0;
    for(int dy = -wy; dy <= wy; dy++) {
        for(int dx = -wx; dx <= wx; dx++) {
            if(ty + dy - 1 >= 0 && ty + dy + 1 < BLOCK_SIZE && tx + dx - 1 >= 0 && tx + dx + 1 < BLOCK_SIZE) {
                int Ix = (I_tile[ty+dy][tx+dx+1] - I_tile[ty+dy][tx+dx-1]) / 2;
                int Iy = (I_tile[ty+dy+1][tx+dx] - I_tile[ty+dy-1][tx+dx]) / 2;
                Gxx += Ix * Ix;
                Gxy += Ix * Iy;
                Gyy += Iy * Iy;
            }
        }
    }

    float det = Gxx * Gyy - Gxy * Gxy;
    if(det > 0.00001f) {

        // Initialization of iterative L-K
        float vx = 0, vy = 0;
        for(int k = 0; k < K; k++) {
            int Jx = tx + gx + vx;
            int Jy = ty + gy + vy;
            if(Jx - 1 >= 0 && Jx + 1 < BLOCK_SIZE && Jy - 1 >= 0 && Jy + 1 < BLOCK_SIZE) {
                // Image mismatch vector
                float bkx = 0, bky = 0;
                for(int dy = -wy; dy <= wy; dy++) {
                    for(int dx = -wx; dx <= wx; dx++) {
                        if(ty + dy - 1 >= 0 && ty + dy + 1 < BLOCK_SIZE && tx + dx - 1 >= 0 && tx + dx + 1 < BLOCK_SIZE) {
                            int Ik = I_tile[ty+dy][tx+dx] - J_tile[Jy][Jx];
                            int Ix = (I_tile[ty+dy][tx+dx+1] - I_tile[ty+dy][tx+dx-1]) / 2;
                            int Iy = (I_tile[ty+dy+1][tx+dx] - I_tile[ty+dy-1][tx+dx]) / 2;
                            bkx += Ik * Ix;
                            bky += Ik * Iy;
                        }
                    }
                }
                // Optical Flow
                float eta_x = (-Gyy * bkx + Gxy * bky) / det;
                float eta_y = (Gxy * bkx - Gxx * bky) / det;
                
                // Guess for next iteration
                vx += eta_x;
                vy += eta_y;
            }
        }
        // Guess for next level
        gx += vx;
        gy += vy;
    }
    o_gx[yidx * cols + xidx] = gx;
    o_gy[yidx * cols + xidx] = gy;
}
