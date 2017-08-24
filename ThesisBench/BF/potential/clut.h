#ifndef _CLUT_H_
#define _CLUT_H_

#include <stdio.h>
#include <CL/cl.h>

#define _GET_NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, _A, _B, _C, _D, _E, _F, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _1A , _1B, _1C, _1D, _1E, _1F, N, ...) N

#define _fe_1(KERNEL, N, x) clSetKernelArg(KERNEL, N, sizeof(x), &x);
#define _fe_2(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_1(KERNEL, N+1, __VA_ARGS__)
#define _fe_3(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_2(KERNEL, N+1, __VA_ARGS__)
#define _fe_4(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_3(KERNEL, N+1, __VA_ARGS__)
#define _fe_5(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_4(KERNEL, N+1, __VA_ARGS__)
#define _fe_6(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_5(KERNEL, N+1, __VA_ARGS__)
#define _fe_7(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_6(KERNEL, N+1, __VA_ARGS__)
#define _fe_8(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_7(KERNEL, N+1, __VA_ARGS__)
#define _fe_9(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_8(KERNEL, N+1, __VA_ARGS__)
#define _fe_A(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_9(KERNEL, N+1, __VA_ARGS__)
#define _fe_B(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_A(KERNEL, N+1, __VA_ARGS__)
#define _fe_C(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_B(KERNEL, N+1, __VA_ARGS__)
#define _fe_D(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_C(KERNEL, N+1, __VA_ARGS__)
#define _fe_E(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_D(KERNEL, N+1, __VA_ARGS__)
#define _fe_F(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_E(KERNEL, N+1, __VA_ARGS__)
#define _fe_10(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_F(KERNEL, N+1, __VA_ARGS__)
#define _fe_11(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_10(KERNEL, N+1, __VA_ARGS__)
#define _fe_12(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_11(KERNEL, N+1, __VA_ARGS__)
#define _fe_13(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_12(KERNEL, N+1, __VA_ARGS__)
#define _fe_14(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_13(KERNEL, N+1, __VA_ARGS__)
#define _fe_15(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_14(KERNEL, N+1, __VA_ARGS__)
#define _fe_16(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_15(KERNEL, N+1, __VA_ARGS__)
#define _fe_17(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_16(KERNEL, N+1, __VA_ARGS__)
#define _fe_18(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_17(KERNEL, N+1, __VA_ARGS__)
#define _fe_19(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_18(KERNEL, N+1, __VA_ARGS__)
#define _fe_1A(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_19(KERNEL, N+1, __VA_ARGS__)
#define _fe_1B(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_1A(KERNEL, N+1, __VA_ARGS__)
#define _fe_1C(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_1B(KERNEL, N+1, __VA_ARGS__)
#define _fe_1D(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_1C(KERNEL, N+1, __VA_ARGS__)
#define _fe_1E(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_1D(KERNEL, N+1, __VA_ARGS__)
#define _fe_1F(KERNEL, N, x, ...) clSetKernelArg(KERNEL, N, sizeof(x), &x); _fe_1E(KERNEL, N+1, __VA_ARGS__)

#define clutLaunchKernel(KERNEL, GLOBAL_SIZE, LOCAL_SIZE, ...) \
    _GET_NTH_ARG(__VA_ARGS__, \
    _fe_1F, _fe_1E, _fe_1D, _fe_1C, _fe_1B, _fe_1A, _fe_19, _fe_18, \
    _fe_17, _fe_16, _fe_15, _fe_14, _fe_13, _fe_12, _fe_11, _fe_10, \
    _fe_F, _fe_E, _fe_D, _fe_C, _fe_B, _fe_A, _fe_9, _fe_8, \
    _fe_7, _fe_6, _fe_5, _fe_4, _fe_3, _fe_2, _fe_1)(KERNEL, 0, __VA_ARGS__) \
    clEnqueueNDRangeKernel(command_queue, KERNEL, dim(GLOBAL_SIZE), NULL, GLOBAL_SIZE, LOCAL_SIZE, 0, NULL, NULL);

cl_platform_id platform_id;
cl_device_id device_id;
cl_context context;
cl_command_queue command_queue;


size_t dim(size_t work_size[3]) {
    if(work_size[0] == 1)
        return 0;
    else if(work_size[1] == 1)
        return 1;
    else if(work_size[2] == 1)
        return 2;
    else
        return 3;

}

void clutInit() {
    
    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    command_queue = clCreateCommandQueue(context, device_id, 0, NULL);

}

cl_program clutCompileProgram(char* file) {

    FILE* fp = fopen(file, "r");
    fseek(fp, 0L, SEEK_END);
    size_t sz = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    char* source = new char[sz + 1];
    fread(source, 1, sz, fp);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&sz, NULL);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    
    delete[] source;
    return program;
}

void* clutMalloc(size_t sz) {
    
    void* ptr = clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sz, 0);
    return ptr;

}

cl_kernel clutCreateKernel(cl_program program, char* kernel_name) {
 
    return clCreateKernel(program, kernel_name, NULL);

}

cl_int clutFinish() {
    return clFinish(command_queue);
}


#endif
