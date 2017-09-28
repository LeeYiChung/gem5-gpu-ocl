#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "clut.h"

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#define WIDTH 32
#define HEIGHT 32
#define THROTTLE_TB 1

#define ETA 0.3 // eta value
#define MOMENTUM 0.3 // momentum value

int layer_size = 0;
int hidden_n = WIDTH;
int output_n = 1;
float *input_units;  /* the input units */
float *input_weights;  /* weights from input to hidden layer */

#define ABS(x) (((x) > 0.0) ? (x) : (-(x)))

float squash(float x) {
    float m;
    // x = -x;
    // m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
    // return(1.0 / (1.0 + m));
    return (1.0 / (1.0 + exp(-x)));
}

void bpnn_initialize(int seed) {
    printf("Random number generator seed: %d\n", seed);
    srand(seed);
}

void bpnn_randomize_weights(float *w, int m, int n) {
    int i, j;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            w[i * n + j] = (float)rand() / RAND_MAX;
        }
    }
}

void bpnn_randomize_row(float *w, int m) {
    int i;
    for (i = 0; i <= m; i++) {
        w[i] = 0.1;
    }
}

void bpnn_zero_weights(float *w, int m, int n) {
    int i, j;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            w[i * n + j] = 0.0;
        }
    }
}

void bpnn_layerforward(float *l1, float *l2, float *conn, int n1, int n2) {
    float sum;
    int j, k;

    /*** Set up thresholding unit ***/
    l1[0] = 1.0;
    /*** For each unit in second layer ***/
    for (j = 0; j < n2; j++) {

        /*** Compute weighted sum of its inputs ***/
        sum = 0.0;
        for (k = 0; k < n1; k++) {
            sum += conn[j * n1 + k] * l1[k];
        }
        l2[j] = squash(sum);
    }
}

void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly,
                         float *w, float *oldw) {
    float new_dw;
    int k, j;
    ly[0] = 1.0;

    for (j = 0; j < ndelta; j++) {
        for (k = 0; k < nly; k++) {
            new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k * ndelta + j]));
            w[k * ndelta + j] += new_dw;
            oldw[k * ndelta + j] = new_dw;
        }
    }
}

void bpnn_output_error(float *delta, float *target, float *output, int nj,
                       float *err) {
    int j;
    float o, t, errsum;
    errsum = 0.0;
    for (j = 0; j < nj; j++) {
        o = output[j];
        t = target[j];
        delta[j] = o * (1.0 - o) * (t - o);
        errsum += ABS(delta[j]);
    }
    *err = errsum;
}

void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no,
                       float *who, float *hidden, float *err) {
    int j, k;
    float h, sum, errsum;

    errsum = 0.0;
    for (j = 0; j < nh; j++) {
        h = hidden[j];
        sum = 0.0;
        for (k = 0; k < no; k++) {
            sum += delta_o[k] * who[j * no + k];
        }
        delta_h[j] = h * (1.0 - h) * sum;
        errsum += ABS(delta_h[j]);
    }
    *err = errsum;
}

void backprop_face() {
    
    clutInit();
    cl_program program = clutCompileProgram("_0.ptx");
    cl_kernel kernel = clutCreateKernel(program, "_Z22bpnn_layerforward_CUDAPfS_S_i");
    
#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    int num_blocks = layer_size / HEIGHT;
    input_units = (float*)clutMalloc(sizeof(float) * layer_size);
    input_weights = (float*)clutMalloc(sizeof(float) * layer_size * hidden_n);
    float* partial_sum = (float*)clutMalloc(sizeof(float) * num_blocks * WIDTH);
    float* hidden_units = (float*)clutMalloc(sizeof(float) * hidden_n);
    float* output_units = (float*)clutMalloc(sizeof(float) * output_n);
    float* hidden_weights = (float*)clutMalloc(sizeof(float) * hidden_n * output_n);
    float* output_delta = (float*)clutMalloc(sizeof(float) * output_n);
    float* target = (float*)clutMalloc(sizeof(float) * output_n);
    float* hidden_delta = (float*)clutMalloc(sizeof(float) * hidden_n);
    float* hidden_prev_weights = (float*)clutMalloc(sizeof(float) * hidden_n * output_n);
    float* input_prev_weights = (float*)clutMalloc(sizeof(float) * layer_size * hidden_n);
    
    // init
    for(int i = 0; i < WIDTH; i++)
        partial_sum[i] = 0.0f;

    // kernel
    size_t global_work[3] = {WIDTH, HEIGHT * num_blocks, 1};
    size_t local_work[3] = {WIDTH, HEIGHT, 1};
    
    clutLaunchKernel(kernel, global_work, local_work, \
        input_units, input_weights, partial_sum, hidden_n);

    // finegrain preparation and synchronization
    for(int by = 0; by < num_blocks; by++) {
        for(int ty = 0; ty < HEIGHT; ty++) {
            input_units[by * HEIGHT + ty] = (float)rand() / RAND_MAX;
            for(int tx = 0; tx < WIDTH; tx++) {
                input_weights[by * HEIGHT * WIDTH + ty * WIDTH + tx] = 0.0f;
            }
        }
        size_t blockIdx[3] = {0, by, 0};
        clSignalBlockReady(blockIdx);
    }

    clutFinish();

    for (int j = 0; j < hidden_n; j++) {
        float sum = 0.0;
        for (int k = 0; k < num_blocks; k++) {
            sum += input_weights[k * hidden_n + j];
        }
        hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
    }
    float out_err, hid_err;
    bpnn_randomize_weights(hidden_weights, hidden_n, output_n);
    bpnn_randomize_row(target, output_n);
    bpnn_zero_weights(hidden_prev_weights, hidden_n, output_n);
    bpnn_zero_weights(input_prev_weights, layer_size, hidden_n);
    
    bpnn_layerforward(hidden_units, output_units, hidden_weights, hidden_n, output_n);
    bpnn_output_error(output_delta, target, output_units, output_n, &out_err);
    bpnn_hidden_error(hidden_delta, hidden_n, output_delta, output_n, hidden_weights, hidden_units, &hid_err);
    bpnn_adjust_weights(output_delta, output_n, hidden_units, hidden_n, hidden_weights, hidden_prev_weights);

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif
}

int main(int argc, char* argv[]) {
    
    printf("Hello Backprop\n");
    
    if (argc != 2) {
        fprintf(stderr, "usage: backprop <num of input elements>\n");
        return 1;
    }

    layer_size = atoi(argv[1]);
    if (layer_size % HEIGHT != 0) {
        fprintf(stderr, "The number of input points must be divided by %d\n", HEIGHT);
        return 1;
    }

    int seed = 7;
    bpnn_initialize(seed);
    backprop_face();
    return 0;
}
