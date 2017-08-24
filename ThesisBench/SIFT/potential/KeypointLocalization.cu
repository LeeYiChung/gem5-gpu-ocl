#define BLOCK_SIZE 16
#define BLOCK_DEPTH 4

__global__ void KeypointLocalization(int* flags, float* DoG, int rows, int cols, float contrastThreshold, float curvateThreshold)
{
    __shared__ float tile[BLOCK_DEPTH][BLOCK_SIZE][BLOCK_SIZE];
    int tz = threadIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int zidx = blockIdx.z * blockDim.z + threadIdx.z;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = zidx * rows * cols + yidx * cols + xidx;
    
    tile[tz][ty][tx] = DoG[idx];
    __syncthreads();
    
    // exclude low contrast and boundary pixels
    if((tile[tz][ty][tx] > contrastThreshold) 
        && tx != 0 && tx != BLOCK_SIZE - 1
        && ty != 0 && ty != BLOCK_SIZE - 1
        && tz != 0 && tz != BLOCK_DEPTH - 1) {
        
        bool nExtrema = tile[tz][ty][tx] > tile[tz - 1][ty - 1][tx - 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz - 1][ty - 1][tx];
        nExtrema ^= tile[tz][ty][tx] > tile[tz - 1][ty - 1][tx + 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz - 1][ty][tx - 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz - 1][ty][tx];
        nExtrema ^= tile[tz][ty][tx] > tile[tz - 1][ty][tx + 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz - 1][ty + 1][tx - 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz - 1][ty + 1][tx];
        nExtrema ^= tile[tz][ty][tx] > tile[tz - 1][ty + 1][tx + 1];

        nExtrema ^= tile[tz][ty][tx] > tile[tz][ty - 1][tx - 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz][ty - 1][tx];
        nExtrema ^= tile[tz][ty][tx] > tile[tz][ty - 1][tx + 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz][ty][tx - 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz][ty][tx + 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz][ty + 1][tx - 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz][ty + 1][tx];
        nExtrema ^= tile[tz][ty][tx] > tile[tz][ty + 1][tx + 1];
        
        nExtrema ^= tile[tz][ty][tx] > tile[tz + 1][ty - 1][tx - 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz + 1][ty - 1][tx];
        nExtrema ^= tile[tz][ty][tx] > tile[tz + 1][ty - 1][tx + 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz + 1][ty][tx - 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz + 1][ty][tx];
        nExtrema ^= tile[tz][ty][tx] > tile[tz + 1][ty][tx + 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz + 1][ty + 1][tx - 1];
        nExtrema ^= tile[tz][ty][tx] > tile[tz + 1][ty + 1][tx];
        nExtrema ^= tile[tz][ty][tx] > tile[tz + 1][ty + 1][tx + 1];
    
        float Dxx = tile[tz][ty][tx - 1] + tile[tz][ty][tx + 1] - 2 * tile[tz][ty][tx];
        float Dyy = tile[tz][ty - 1][tx] + tile[tz][ty + 1][tx] - 2 * tile[tz][ty][tx];
        float Dxy = (tile[tz][ty + 1][tx + 1] - tile[tz][ty - 1][tx + 1] - tile[tz][ty + 1][tx - 1] + tile[tz][ty - 1][tx - 1]) / 4.0f;
        float Tr = Dxx + Dyy;
        float Det = Dxx * Dyy - Dxy * Dxy;
        float curvate = Tr * Tr / Det;
		
        flags[idx] = !nExtrema && Det >= 0 && curvate < curvateThreshold;
    }
}
