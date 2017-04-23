# 1 "/tmp/tmpxft_0000636c_00000000-1_CUDA_interface.cudafe1.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/tmp/tmpxft_0000636c_00000000-1_CUDA_interface.cudafe1.cpp"
# 1 "CUDA_interface.cu"
# 46 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_types.h"
# 149 "/usr/lib/gcc/x86_64-linux-gnu/4.4.7/include/stddef.h" 3
typedef long ptrdiff_t;
# 211 "/usr/lib/gcc/x86_64-linux-gnu/4.4.7/include/stddef.h" 3
typedef unsigned long size_t;
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/crt/host_runtime.h" 1 3
# 69 "/home/bachelor/deicide218/cuda-3.2/bin/../include/crt/host_runtime.h" 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/builtin_types.h" 1 3
# 42 "/home/bachelor/deicide218/cuda-3.2/bin/../include/builtin_types.h" 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_types.h" 1 3
# 46 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_types.h" 3
enum cudaRoundMode
{
  cudaRoundNearest,
  cudaRoundZero,
  cudaRoundPosInf,
  cudaRoundMinInf
};
# 43 "/home/bachelor/deicide218/cuda-3.2/bin/../include/builtin_types.h" 2 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 1 3
# 96 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
enum cudaError
{





  cudaSuccess = 0,





  cudaErrorMissingConfiguration = 1,





  cudaErrorMemoryAllocation = 2,





  cudaErrorInitializationError = 3,
# 131 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorLaunchFailure = 4,
# 140 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorPriorLaunchFailure = 5,
# 150 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorLaunchTimeout = 6,
# 159 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorLaunchOutOfResources = 7,





  cudaErrorInvalidDeviceFunction = 8,
# 174 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorInvalidConfiguration = 9,





  cudaErrorInvalidDevice = 10,





  cudaErrorInvalidValue = 11,





  cudaErrorInvalidPitchValue = 12,





  cudaErrorInvalidSymbol = 13,




  cudaErrorMapBufferObjectFailed = 14,




  cudaErrorUnmapBufferObjectFailed = 15,





  cudaErrorInvalidHostPointer = 16,





  cudaErrorInvalidDevicePointer = 17,





  cudaErrorInvalidTexture = 18,





  cudaErrorInvalidTextureBinding = 19,






  cudaErrorInvalidChannelDescriptor = 20,





  cudaErrorInvalidMemcpyDirection = 21,
# 255 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorAddressOfConstant = 22,
# 264 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorTextureFetchFailed = 23,
# 273 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorTextureNotBound = 24,
# 282 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorSynchronizationError = 25,





  cudaErrorInvalidFilterSetting = 26,





  cudaErrorInvalidNormSetting = 27,







  cudaErrorMixedDeviceExecution = 28,







  cudaErrorCudartUnloading = 29,




  cudaErrorUnknown = 30,





  cudaErrorNotYetImplemented = 31,
# 330 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorMemoryValueTooLarge = 32,






  cudaErrorInvalidResourceHandle = 33,







  cudaErrorNotReady = 34,






  cudaErrorInsufficientDriver = 35,
# 365 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorSetOnActiveProcess = 36,





  cudaErrorInvalidSurface = 37,





  cudaErrorNoDevice = 38,





  cudaErrorECCUncorrectable = 39,




  cudaErrorSharedObjectSymbolNotFound = 40,




  cudaErrorSharedObjectInitFailed = 41,





  cudaErrorUnsupportedLimit = 42,





  cudaErrorDuplicateVariableName = 43,





  cudaErrorDuplicateTextureName = 44,





  cudaErrorDuplicateSurfaceName = 45,
# 426 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorDevicesUnavailable = 46,




  cudaErrorInvalidKernelImage = 47,







  cudaErrorNoKernelImageForDevice = 48,
# 448 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
  cudaErrorIncompatibleDriverContext = 49,




  cudaErrorStartupFailure = 0x7f,





  cudaErrorApiFailureBase = 10000
};





enum cudaChannelFormatKind
{
  cudaChannelFormatKindSigned = 0,
  cudaChannelFormatKindUnsigned = 1,
  cudaChannelFormatKindFloat = 2,
  cudaChannelFormatKindNone = 3
};





struct cudaChannelFormatDesc
{
  int x;
  int y;
  int z;
  int w;
  enum cudaChannelFormatKind f;
};





struct cudaArray;





enum cudaMemcpyKind
{
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
};






struct cudaPitchedPtr
{
  void *ptr;
  size_t pitch;
  size_t xsize;
  size_t ysize;
};






struct cudaExtent
{
  size_t width;
  size_t height;
  size_t depth;
};






struct cudaPos
{
  size_t x;
  size_t y;
  size_t z;
};





struct cudaMemcpy3DParms
{
  struct cudaArray *srcArray;
  struct cudaPos srcPos;
  struct cudaPitchedPtr srcPtr;

  struct cudaArray *dstArray;
  struct cudaPos dstPos;
  struct cudaPitchedPtr dstPtr;

  struct cudaExtent extent;
  enum cudaMemcpyKind kind;
};





struct cudaGraphicsResource;





enum cudaGraphicsRegisterFlags
{
  cudaGraphicsRegisterFlagsNone = 0
};





enum cudaGraphicsMapFlags
{
  cudaGraphicsMapFlagsNone = 0,
  cudaGraphicsMapFlagsReadOnly = 1,
  cudaGraphicsMapFlagsWriteDiscard = 2
};





enum cudaGraphicsCubeFace {
  cudaGraphicsCubeFacePositiveX = 0x00,
  cudaGraphicsCubeFaceNegativeX = 0x01,
  cudaGraphicsCubeFacePositiveY = 0x02,
  cudaGraphicsCubeFaceNegativeY = 0x03,
  cudaGraphicsCubeFacePositiveZ = 0x04,
  cudaGraphicsCubeFaceNegativeZ = 0x05
};





struct cudaFuncAttributes
{





   size_t sharedSizeBytes;





   size_t constSizeBytes;




   size_t localSizeBytes;






   int maxThreadsPerBlock;




   int numRegs;






   int ptxVersion;






   int binaryVersion;

   int __cudaReserved[6];
};





enum cudaFuncCache
{
  cudaFuncCachePreferNone = 0,
  cudaFuncCachePreferShared = 1,
  cudaFuncCachePreferL1 = 2
};





enum cudaComputeMode
{
  cudaComputeModeDefault = 0,
  cudaComputeModeExclusive = 1,
  cudaComputeModeProhibited = 2
};





enum cudaLimit
{
    cudaLimitStackSize = 0x00,
    cudaLimitPrintfFifoSize = 0x01,
    cudaLimitMallocHeapSize = 0x02
};





struct cudaDeviceProp
{
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maxTexture1D;
  int maxTexture2D[2];
  int maxTexture3D[3];
  int maxTexture2DArray[3];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int tccDriver;
  int __cudaReserved[21];
};
# 768 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h" 3
typedef enum cudaError cudaError_t;





typedef struct CUstream_st *cudaStream_t;





typedef struct CUevent_st *cudaEvent_t;





typedef struct cudaGraphicsResource *cudaGraphicsResource_t;





typedef struct CUuuid_st cudaUUID_t;
# 44 "/home/bachelor/deicide218/cuda-3.2/bin/../include/builtin_types.h" 2 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_types.h" 1 3
# 63 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_types.h" 3
enum cudaSurfaceBoundaryMode
{
  cudaBoundaryModeZero = 0,
  cudaBoundaryModeClamp = 1,
  cudaBoundaryModeTrap = 2
};





enum cudaSurfaceFormatMode
{
  cudaFormatModeForced = 0,
  cudaFormatModeAuto = 1
};





struct surfaceReference
{



  struct cudaChannelFormatDesc channelDesc;
};
# 45 "/home/bachelor/deicide218/cuda-3.2/bin/../include/builtin_types.h" 2 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_types.h" 1 3
# 63 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_types.h" 3
enum cudaTextureAddressMode
{
  cudaAddressModeWrap = 0,
  cudaAddressModeClamp = 1,
  cudaAddressModeMirror = 2,
  cudaAddressModeBorder = 3
};





enum cudaTextureFilterMode
{
  cudaFilterModePoint = 0,
  cudaFilterModeLinear = 1
};





enum cudaTextureReadMode
{
  cudaReadModeElementType = 0,
  cudaReadModeNormalizedFloat = 1
};





struct textureReference
{



  int normalized;



  enum cudaTextureFilterMode filterMode;



  enum cudaTextureAddressMode addressMode[3];



  struct cudaChannelFormatDesc channelDesc;
  int __cudaReserved[16];
};
# 46 "/home/bachelor/deicide218/cuda-3.2/bin/../include/builtin_types.h" 2 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h" 1 3
# 45 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h" 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/builtin_types.h" 1 3
# 46 "/home/bachelor/deicide218/cuda-3.2/bin/../include/builtin_types.h" 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h" 1 3
# 46 "/home/bachelor/deicide218/cuda-3.2/bin/../include/builtin_types.h" 2 3
# 46 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h" 2 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/host_defines.h" 1 3
# 47 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h" 2 3
# 75 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h" 3
struct char1
{
  signed char x;
};


struct uchar1
{
  unsigned char x;
};


struct __attribute__((aligned(2))) char2
{
  signed char x, y;
};


struct __attribute__((aligned(2))) uchar2
{
  unsigned char x, y;
};


struct char3
{
  signed char x, y, z;
};


struct uchar3
{
  unsigned char x, y, z;
};


struct __attribute__((aligned(4))) char4
{
  signed char x, y, z, w;
};


struct __attribute__((aligned(4))) uchar4
{
  unsigned char x, y, z, w;
};


struct short1
{
  short x;
};


struct ushort1
{
  unsigned short x;
};


struct __attribute__((aligned(4))) short2
{
  short x, y;
};


struct __attribute__((aligned(4))) ushort2
{
  unsigned short x, y;
};


struct short3
{
  short x, y, z;
};


struct ushort3
{
  unsigned short x, y, z;
};


struct __attribute__((aligned(8))) short4 { short x; short y; short z; short w; };


struct __attribute__((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; };


struct int1
{
  int x;
};


struct uint1
{
  unsigned int x;
};


struct __attribute__((aligned(8))) int2 { int x; int y; };


struct __attribute__((aligned(8))) uint2 { unsigned int x; unsigned int y; };


struct int3
{
  int x, y, z;
};


struct uint3
{
  unsigned int x, y, z;
};


struct __attribute__((aligned(16))) int4
{
  int x, y, z, w;
};


struct __attribute__((aligned(16))) uint4
{
  unsigned int x, y, z, w;
};


struct long1
{
  long int x;
};


struct ulong1
{
  unsigned long x;
};
# 229 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h" 3
struct __attribute__((aligned(2*sizeof(long int)))) long2
{
  long int x, y;
};


struct __attribute__((aligned(2*sizeof(unsigned long int)))) ulong2
{
  unsigned long int x, y;
};




struct long3
{
  long int x, y, z;
};


struct ulong3
{
  unsigned long int x, y, z;
};


struct __attribute__((aligned(16))) long4
{
  long int x, y, z, w;
};


struct __attribute__((aligned(16))) ulong4
{
  unsigned long int x, y, z, w;
};


struct float1
{
  float x;
};


struct __attribute__((aligned(8))) float2 { float x; float y; };


struct float3
{
  float x, y, z;
};


struct __attribute__((aligned(16))) float4
{
  float x, y, z, w;
};


struct longlong1
{
  long long int x;
};


struct ulonglong1
{
  unsigned long long int x;
};


struct __attribute__((aligned(16))) longlong2
{
  long long int x, y;
};


struct __attribute__((aligned(16))) ulonglong2
{
  unsigned long long int x, y;
};


struct longlong3
{
  long long int x, y, z;
};


struct ulonglong3
{
  unsigned long long int x, y, z;
};


struct __attribute__((aligned(16))) longlong4
{
  long long int x, y, z ,w;
};


struct __attribute__((aligned(16))) ulonglong4
{
  unsigned long long int x, y, z, w;
};


struct double1
{
  double x;
};


struct __attribute__((aligned(16))) double2
{
  double x, y;
};


struct double3
{
  double x, y, z;
};


struct __attribute__((aligned(16))) double4
{
  double x, y, z, w;
};
# 366 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h" 3
typedef struct char1 char1;

typedef struct uchar1 uchar1;

typedef struct char2 char2;

typedef struct uchar2 uchar2;

typedef struct char3 char3;

typedef struct uchar3 uchar3;

typedef struct char4 char4;

typedef struct uchar4 uchar4;

typedef struct short1 short1;

typedef struct ushort1 ushort1;

typedef struct short2 short2;

typedef struct ushort2 ushort2;

typedef struct short3 short3;

typedef struct ushort3 ushort3;

typedef struct short4 short4;

typedef struct ushort4 ushort4;

typedef struct int1 int1;

typedef struct uint1 uint1;

typedef struct int2 int2;

typedef struct uint2 uint2;

typedef struct int3 int3;

typedef struct uint3 uint3;

typedef struct int4 int4;

typedef struct uint4 uint4;

typedef struct long1 long1;

typedef struct ulong1 ulong1;

typedef struct long2 long2;

typedef struct ulong2 ulong2;

typedef struct long3 long3;

typedef struct ulong3 ulong3;

typedef struct long4 long4;

typedef struct ulong4 ulong4;

typedef struct float1 float1;

typedef struct float2 float2;

typedef struct float3 float3;

typedef struct float4 float4;

typedef struct longlong1 longlong1;

typedef struct ulonglong1 ulonglong1;

typedef struct longlong2 longlong2;

typedef struct ulonglong2 ulonglong2;

typedef struct longlong3 longlong3;

typedef struct ulonglong3 ulonglong3;

typedef struct longlong4 longlong4;

typedef struct ulonglong4 ulonglong4;

typedef struct double1 double1;

typedef struct double2 double2;

typedef struct double3 double3;

typedef struct double4 double4;
# 469 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h" 3
struct dim3
{
    unsigned int x, y, z;

    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }

};


typedef struct dim3 dim3;
# 46 "/home/bachelor/deicide218/cuda-3.2/bin/../include/builtin_types.h" 2 3
# 70 "/home/bachelor/deicide218/cuda-3.2/bin/../include/crt/host_runtime.h" 2 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/crt/storage_class.h" 1 3
# 71 "/home/bachelor/deicide218/cuda-3.2/bin/../include/crt/host_runtime.h" 2 3
# 213 "/usr/lib/gcc/x86_64-linux-gnu/4.4.7/include/stddef.h" 2 3
# 96 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 466 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 478 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 491 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 497 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 510 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 523 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 535 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 546 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 564 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 570 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 579 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 590 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 603 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 656 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 667 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 678 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 689 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 768 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 774 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 780 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 786 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 792 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_types.h"
# 63 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_types.h"
# 74 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_types.h"
# 84 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_types.h"
# 63 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_types.h"
# 75 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_types.h"
# 85 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_types.h"
# 95 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_types.h"
# 75 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 81 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 87 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 93 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 99 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 105 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 111 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 117 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 123 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 129 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 135 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 141 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 147 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 153 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 159 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 162 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 165 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 171 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 177 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 180 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 183 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 189 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 195 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 201 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 207 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 213 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 229 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 235 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 243 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 249 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 255 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 261 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 267 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 273 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 276 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 282 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 288 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 294 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 300 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 306 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 312 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 318 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 324 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 330 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 336 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 342 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 348 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 354 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 366 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 368 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 370 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 372 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 374 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 376 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 378 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 380 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 382 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 384 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 386 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 388 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 390 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 392 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 394 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 396 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 398 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 400 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 402 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 404 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 406 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 408 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 410 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 412 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 414 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 416 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 418 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 420 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 422 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 424 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 426 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 428 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 430 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 432 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 434 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 436 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 438 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 440 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 442 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 444 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 446 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 448 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 450 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 452 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 454 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 456 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 458 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 460 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 469 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 480 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_types.h"
# 115 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadExit();
# 131 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadSynchronize();
# 183 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadSetLimit(cudaLimit, size_t);
# 207 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadGetLimit(size_t *, cudaLimit);
# 237 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadGetCacheConfig(cudaFuncCache *);
# 278 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaThreadSetCacheConfig(cudaFuncCache);
# 330 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetLastError();
# 373 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaPeekAtLastError();
# 387 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" const char *cudaGetErrorString(cudaError_t);
# 418 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetDeviceCount(int *);
# 536 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp *, int);
# 555 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaChooseDevice(int *, const cudaDeviceProp *);
# 579 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetDevice(int);
# 597 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetDevice(int *);
# 626 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetValidDevices(int *, int);
# 677 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetDeviceFlags(unsigned);
# 703 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamCreate(cudaStream_t *);
# 719 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t);
# 753 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
# 771 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t);
# 789 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaStreamQuery(cudaStream_t);
# 821 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventCreate(cudaEvent_t *);
# 852 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned);
# 885 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0);
# 914 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventQuery(cudaEvent_t);
# 946 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t);
# 966 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t);
# 1007 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
# 1046 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaConfigureCall(dim3, dim3, size_t = (0), cudaStream_t = 0);
# 1073 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetupArgument(const void *, size_t, size_t);
# 1119 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFuncSetCacheConfig(const char *, cudaFuncCache);
# 1154 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaLaunch(const char *);
# 1187 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *, const char *);
# 1209 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetDoubleForDevice(double *);
# 1231 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaSetDoubleForHost(double *);
# 1263 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMalloc(void **, size_t);
# 1292 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMallocHost(void **, size_t);
# 1331 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMallocPitch(void **, size_t *, size_t, size_t);
# 1370 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMallocArray(cudaArray **, const cudaChannelFormatDesc *, size_t, size_t = (0), unsigned = (0));
# 1394 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFree(void *);
# 1414 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFreeHost(void *);
# 1436 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaFreeArray(cudaArray *);
# 1495 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaHostAlloc(void **, size_t, unsigned);
# 1522 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaHostGetDevicePointer(void **, void *, unsigned);
# 1541 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaHostGetFlags(unsigned *, void *);
# 1576 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMalloc3D(cudaPitchedPtr *, cudaExtent);
# 1626 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMalloc3DArray(cudaArray **, const cudaChannelFormatDesc *, cudaExtent, unsigned = (0));
# 1723 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *);
# 1828 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *, cudaStream_t = 0);
# 1847 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemGetInfo(size_t *, size_t *);
# 1880 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
# 1913 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyToArray(cudaArray *, size_t, size_t, const void *, size_t, cudaMemcpyKind);
# 1946 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyFromArray(void *, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind);
# 1981 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyArrayToArray(cudaArray *, size_t, size_t, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind = cudaMemcpyDeviceToDevice);
# 2023 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2D(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind);
# 2064 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DToArray(cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind);
# 2105 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DFromArray(void *, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind);
# 2144 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DArrayToArray(cudaArray *, size_t, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind = cudaMemcpyDeviceToDevice);
# 2179 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyToSymbol(const char *, const void *, size_t, size_t = (0), cudaMemcpyKind = cudaMemcpyHostToDevice);
# 2213 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyFromSymbol(void *, const char *, size_t, size_t = (0), cudaMemcpyKind = cudaMemcpyDeviceToHost);
# 2256 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2298 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyToArrayAsync(cudaArray *, size_t, size_t, const void *, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2340 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyFromArrayAsync(void *, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2391 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2441 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DToArrayAsync(cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2491 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpy2DFromArrayAsync(void *, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2535 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyToSymbolAsync(const char *, const void *, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2578 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemcpyFromSymbolAsync(void *, const char *, size_t, size_t, cudaMemcpyKind, cudaStream_t = 0);
# 2600 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset(void *, int, size_t);
# 2626 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset2D(void *, size_t, int, size_t, size_t);
# 2665 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset3D(cudaPitchedPtr, int, cudaExtent);
# 2692 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t = 0);
# 2724 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset2DAsync(void *, size_t, int, size_t, size_t, cudaStream_t = 0);
# 2769 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMemset3DAsync(cudaPitchedPtr, int, cudaExtent, cudaStream_t = 0);
# 2796 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetSymbolAddress(void **, const char *);
# 2819 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetSymbolSize(size_t *, const char *);
# 2865 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t);
# 2897 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t, unsigned);
# 2932 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGraphicsMapResources(int, cudaGraphicsResource_t *, cudaStream_t = 0);
# 2963 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGraphicsUnmapResources(int, cudaGraphicsResource_t *, cudaStream_t = 0);
# 2992 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGraphicsResourceGetMappedPointer(void **, size_t *, cudaGraphicsResource_t);
# 3026 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray **, cudaGraphicsResource_t, unsigned, unsigned);
# 3059 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *, const cudaArray *);
# 3094 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaChannelFormatDesc cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind);
# 3136 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaBindTexture(size_t *, const textureReference *, const void *, const cudaChannelFormatDesc *, size_t = (((2147483647) * 2U) + 1U));
# 3179 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaBindTexture2D(size_t *, const textureReference *, const void *, const cudaChannelFormatDesc *, size_t, size_t, size_t);
# 3207 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaBindTextureToArray(const textureReference *, const cudaArray *, const cudaChannelFormatDesc *);
# 3228 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaUnbindTexture(const textureReference *);
# 3253 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetTextureAlignmentOffset(size_t *, const textureReference *);
# 3277 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetTextureReference(const textureReference **, const char *);
# 3310 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaBindSurfaceToArray(const surfaceReference *, const cudaArray *, const cudaChannelFormatDesc *);
# 3328 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetSurfaceReference(const surfaceReference **, const char *);
# 3355 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaDriverGetVersion(int *);
# 3372 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaRuntimeGetVersion(int *);
# 3377 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaGetExportTable(const void **, const cudaUUID_t *);
# 93 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template< class T> inline cudaChannelFormatDesc cudaCreateChannelDesc()
# 94 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 95 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
# 96 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 98 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf()
# 99 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 100 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 102 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
# 103 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 105 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf1()
# 106 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 107 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 109 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
# 110 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 112 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf2()
# 113 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 114 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 116 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
# 117 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 119 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf4()
# 120 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 121 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 123 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
# 124 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 126 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> ()
# 127 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 128 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(char)) * 8);
# 133 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 135 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 137 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> ()
# 138 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 139 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(signed char)) * 8);
# 141 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 142 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 144 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> ()
# 145 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 146 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned char)) * 8);
# 148 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 149 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 151 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> ()
# 152 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 153 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(signed char)) * 8);
# 155 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 156 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 158 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> ()
# 159 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 160 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned char)) * 8);
# 162 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 163 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 165 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> ()
# 166 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 167 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(signed char)) * 8);
# 169 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
# 170 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 172 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> ()
# 173 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 174 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned char)) * 8);
# 176 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
# 177 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 179 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> ()
# 180 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 181 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(signed char)) * 8);
# 183 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
# 184 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 186 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> ()
# 187 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 188 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned char)) * 8);
# 190 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
# 191 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 193 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> ()
# 194 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 195 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(short)) * 8);
# 197 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 198 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 200 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> ()
# 201 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 202 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 204 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 205 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 207 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> ()
# 208 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 209 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(short)) * 8);
# 211 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 212 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 214 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> ()
# 215 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 216 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 218 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 219 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 221 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> ()
# 222 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 223 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(short)) * 8);
# 225 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
# 226 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 228 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> ()
# 229 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 230 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 232 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
# 233 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 235 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> ()
# 236 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 237 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(short)) * 8);
# 239 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
# 240 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 242 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> ()
# 243 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 244 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned short)) * 8);
# 246 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
# 247 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 249 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> ()
# 250 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 251 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(int)) * 8);
# 253 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 254 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 256 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> ()
# 257 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 258 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned)) * 8);
# 260 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 261 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 263 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> ()
# 264 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 265 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(int)) * 8);
# 267 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
# 268 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 270 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> ()
# 271 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 272 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned)) * 8);
# 274 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
# 275 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 277 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> ()
# 278 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 279 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(int)) * 8);
# 281 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
# 282 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 284 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> ()
# 285 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 286 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned)) * 8);
# 288 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
# 289 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 291 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> ()
# 292 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 293 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(int)) * 8);
# 295 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
# 296 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 298 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> ()
# 299 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 300 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(unsigned)) * 8);
# 302 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
# 303 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 365 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> ()
# 366 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 367 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(float)) * 8);
# 369 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
# 370 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 372 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> ()
# 373 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 374 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(float)) * 8);
# 376 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
# 377 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 379 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> ()
# 380 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 381 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(float)) * 8);
# 383 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
# 384 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 386 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> ()
# 387 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
{
# 388 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
int e = (((int)sizeof(float)) * 8);
# 390 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
# 391 "/home/bachelor/deicide218/cuda-3.2/bin/../include/channel_descriptor.h"
}
# 65 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz)
# 66 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
{
# 67 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
cudaPitchedPtr s;
# 69 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
(s.ptr) = d;
# 70 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
(s.pitch) = p;
# 71 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
(s.xsize) = xsz;
# 72 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
(s.ysize) = ysz;
# 74 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
return s;
# 75 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
}
# 92 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z)
# 93 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
{
# 94 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
cudaPos p;
# 96 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
(p.x) = x;
# 97 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
(p.y) = y;
# 98 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
(p.z) = z;
# 100 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
return p;
# 101 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
}
# 118 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d)
# 119 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
{
# 120 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
cudaExtent e;
# 122 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
(e.width) = w;
# 123 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
(e.height) = h;
# 124 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
(e.depth) = d;
# 126 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
return e;
# 127 "/home/bachelor/deicide218/cuda-3.2/bin/../include/driver_functions.h"
}
# 55 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline char1 make_char1(signed char x)
# 56 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 57 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
char1 t; (t.x) = x; return t;
# 58 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 60 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline uchar1 make_uchar1(unsigned char x)
# 61 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 62 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
uchar1 t; (t.x) = x; return t;
# 63 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 65 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline char2 make_char2(signed char x, signed char y)
# 66 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 67 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
char2 t; (t.x) = x; (t.y) = y; return t;
# 68 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 70 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline uchar2 make_uchar2(unsigned char x, unsigned char y)
# 71 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 72 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
uchar2 t; (t.x) = x; (t.y) = y; return t;
# 73 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 75 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline char3 make_char3(signed char x, signed char y, signed char z)
# 76 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 77 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 78 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 80 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
# 81 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 82 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 83 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 85 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w)
# 86 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 87 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 88 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 90 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
# 91 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 92 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 93 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 95 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline short1 make_short1(short x)
# 96 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 97 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
short1 t; (t.x) = x; return t;
# 98 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 100 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline ushort1 make_ushort1(unsigned short x)
# 101 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 102 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
ushort1 t; (t.x) = x; return t;
# 103 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 105 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline short2 make_short2(short x, short y)
# 106 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 107 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
short2 t; (t.x) = x; (t.y) = y; return t;
# 108 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 110 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline ushort2 make_ushort2(unsigned short x, unsigned short y)
# 111 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 112 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
ushort2 t; (t.x) = x; (t.y) = y; return t;
# 113 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 115 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline short3 make_short3(short x, short y, short z)
# 116 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 117 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 118 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 120 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z)
# 121 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 122 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 123 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 125 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline short4 make_short4(short x, short y, short z, short w)
# 126 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 127 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 128 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 130 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w)
# 131 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 132 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 133 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 135 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline int1 make_int1(int x)
# 136 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 137 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
int1 t; (t.x) = x; return t;
# 138 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 140 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline uint1 make_uint1(unsigned x)
# 141 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 142 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
uint1 t; (t.x) = x; return t;
# 143 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 145 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline int2 make_int2(int x, int y)
# 146 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 147 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
int2 t; (t.x) = x; (t.y) = y; return t;
# 148 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 150 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline uint2 make_uint2(unsigned x, unsigned y)
# 151 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 152 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
uint2 t; (t.x) = x; (t.y) = y; return t;
# 153 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 155 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline int3 make_int3(int x, int y, int z)
# 156 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 157 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 158 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 160 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z)
# 161 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 162 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 163 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 165 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline int4 make_int4(int x, int y, int z, int w)
# 166 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 167 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 168 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 170 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w)
# 171 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 172 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 173 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 175 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline long1 make_long1(long x)
# 176 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 177 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
long1 t; (t.x) = x; return t;
# 178 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 180 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline ulong1 make_ulong1(unsigned long x)
# 181 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 182 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
ulong1 t; (t.x) = x; return t;
# 183 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 185 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline long2 make_long2(long x, long y)
# 186 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 187 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
long2 t; (t.x) = x; (t.y) = y; return t;
# 188 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 190 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline ulong2 make_ulong2(unsigned long x, unsigned long y)
# 191 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 192 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
ulong2 t; (t.x) = x; (t.y) = y; return t;
# 193 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 195 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline long3 make_long3(long x, long y, long z)
# 196 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 197 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 198 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 200 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z)
# 201 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 202 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 203 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 205 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline long4 make_long4(long x, long y, long z, long w)
# 206 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 207 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 208 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 210 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w)
# 211 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 212 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 213 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 215 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline float1 make_float1(float x)
# 216 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 217 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
float1 t; (t.x) = x; return t;
# 218 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 220 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline float2 make_float2(float x, float y)
# 221 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 222 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
float2 t; (t.x) = x; (t.y) = y; return t;
# 223 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 225 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline float3 make_float3(float x, float y, float z)
# 226 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 227 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 228 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 230 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline float4 make_float4(float x, float y, float z, float w)
# 231 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 232 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 233 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 235 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline longlong1 make_longlong1(long long x)
# 236 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 237 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
longlong1 t; (t.x) = x; return t;
# 238 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 240 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline ulonglong1 make_ulonglong1(unsigned long long x)
# 241 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 242 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
ulonglong1 t; (t.x) = x; return t;
# 243 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 245 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline longlong2 make_longlong2(long long x, long long y)
# 246 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 247 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
longlong2 t; (t.x) = x; (t.y) = y; return t;
# 248 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 250 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y)
# 251 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 252 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
ulonglong2 t; (t.x) = x; (t.y) = y; return t;
# 253 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 255 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline longlong3 make_longlong3(long long x, long long y, long long z)
# 256 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 257 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 258 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 260 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z)
# 261 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 262 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 263 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 265 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w)
# 266 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 267 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 268 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 270 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w)
# 271 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 272 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 273 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 275 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline double1 make_double1(double x)
# 276 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 277 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
double1 t; (t.x) = x; return t;
# 278 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 280 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline double2 make_double2(double x, double y)
# 281 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 282 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
double2 t; (t.x) = x; (t.y) = y; return t;
# 283 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 285 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline double3 make_double3(double x, double y, double z)
# 286 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 287 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t;
# 288 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 290 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
static inline double4 make_double4(double x, double y, double z, double w)
# 291 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
{
# 292 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t;
# 293 "/home/bachelor/deicide218/cuda-3.2/bin/../include/vector_functions.h"
}
# 44 "/usr/include/string.h" 3
extern "C" __attribute__((weak)) void *memcpy(void *__restrict__, const void *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 49 "/usr/include/string.h" 3
extern "C" void *memmove(void *, const void *, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 57 "/usr/include/string.h" 3
extern "C" void *memccpy(void *__restrict__, const void *__restrict__, int, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 65 "/usr/include/string.h" 3
extern "C" __attribute__((weak)) void *memset(void *, int, size_t) throw() __attribute__((nonnull(1)));
# 68 "/usr/include/string.h" 3
extern "C" int memcmp(const void *, const void *, size_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 75 "/usr/include/string.h" 3
extern void *memchr(void *, int, size_t) throw() __asm__("memchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 77 "/usr/include/string.h" 3
extern const void *memchr(const void *, int, size_t) throw() __asm__("memchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 104 "/usr/include/string.h" 3
void *rawmemchr(void *, int) throw() __asm__("rawmemchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 106 "/usr/include/string.h" 3
const void *rawmemchr(const void *, int) throw() __asm__("rawmemchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 115 "/usr/include/string.h" 3
void *memrchr(void *, int, size_t) throw() __asm__("memrchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 117 "/usr/include/string.h" 3
const void *memrchr(const void *, int, size_t) throw() __asm__("memrchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 128 "/usr/include/string.h" 3
extern "C" char *strcpy(char *__restrict__, const char *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 131 "/usr/include/string.h" 3
extern "C" char *strncpy(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 136 "/usr/include/string.h" 3
extern "C" char *strcat(char *__restrict__, const char *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 139 "/usr/include/string.h" 3
extern "C" char *strncat(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 143 "/usr/include/string.h" 3
extern "C" int strcmp(const char *, const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 146 "/usr/include/string.h" 3
extern "C" int strncmp(const char *, const char *, size_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 150 "/usr/include/string.h" 3
extern "C" int strcoll(const char *, const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 153 "/usr/include/string.h" 3
extern "C" size_t strxfrm(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((nonnull(2)));
# 40 "/usr/include/xlocale.h" 3
extern "C" { typedef
# 28 "/usr/include/xlocale.h" 3
struct __locale_struct {
# 31 "/usr/include/xlocale.h" 3
struct __locale_data *__locales[13];
# 34 "/usr/include/xlocale.h" 3
const unsigned short *__ctype_b;
# 35 "/usr/include/xlocale.h" 3
const int *__ctype_tolower;
# 36 "/usr/include/xlocale.h" 3
const int *__ctype_toupper;
# 39 "/usr/include/xlocale.h" 3
const char *__names[13];
# 40 "/usr/include/xlocale.h" 3
} *__locale_t; }
# 43 "/usr/include/xlocale.h" 3
extern "C" { typedef __locale_t locale_t; }
# 165 "/usr/include/string.h" 3
extern "C" int strcoll_l(const char *, const char *, __locale_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2))) __attribute__((nonnull(3)));
# 168 "/usr/include/string.h" 3
extern "C" size_t strxfrm_l(char *, const char *, size_t, __locale_t) throw() __attribute__((nonnull(2))) __attribute__((nonnull(4)));
# 175 "/usr/include/string.h" 3
extern "C" char *strdup(const char *) throw() __attribute__((__malloc__)) __attribute__((nonnull(1)));
# 183 "/usr/include/string.h" 3
extern "C" char *strndup(const char *, size_t) throw() __attribute__((__malloc__)) __attribute__((nonnull(1)));
# 215 "/usr/include/string.h" 3
extern char *strchr(char *, int) throw() __asm__("strchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 217 "/usr/include/string.h" 3
extern const char *strchr(const char *, int) throw() __asm__("strchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 242 "/usr/include/string.h" 3
extern char *strrchr(char *, int) throw() __asm__("strrchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 244 "/usr/include/string.h" 3
extern const char *strrchr(const char *, int) throw() __asm__("strrchr") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 271 "/usr/include/string.h" 3
char *strchrnul(char *, int) throw() __asm__("strchrnul") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 273 "/usr/include/string.h" 3
const char *strchrnul(const char *, int) throw() __asm__("strchrnul") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 284 "/usr/include/string.h" 3
extern "C" size_t strcspn(const char *, const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 288 "/usr/include/string.h" 3
extern "C" size_t strspn(const char *, const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 294 "/usr/include/string.h" 3
extern char *strpbrk(char *, const char *) throw() __asm__("strpbrk") __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 296 "/usr/include/string.h" 3
extern const char *strpbrk(const char *, const char *) throw() __asm__("strpbrk") __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 321 "/usr/include/string.h" 3
extern char *strstr(char *, const char *) throw() __asm__("strstr") __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 323 "/usr/include/string.h" 3
extern const char *strstr(const char *, const char *) throw() __asm__("strstr") __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 348 "/usr/include/string.h" 3
extern "C" char *strtok(char *__restrict__, const char *__restrict__) throw() __attribute__((nonnull(2)));
# 354 "/usr/include/string.h" 3
extern "C" char *__strtok_r(char *__restrict__, const char *__restrict__, char **__restrict__) throw() __attribute__((nonnull(2))) __attribute__((nonnull(3)));
# 359 "/usr/include/string.h" 3
extern "C" char *strtok_r(char *__restrict__, const char *__restrict__, char **__restrict__) throw() __attribute__((nonnull(2))) __attribute__((nonnull(3)));
# 367 "/usr/include/string.h" 3
char *strcasestr(char *, const char *) throw() __asm__("strcasestr") __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 369 "/usr/include/string.h" 3
const char *strcasestr(const char *, const char *) throw() __asm__("strcasestr") __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 382 "/usr/include/string.h" 3
extern "C" void *memmem(const void *, size_t, const void *, size_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(3)));
# 388 "/usr/include/string.h" 3
extern "C" void *__mempcpy(void *__restrict__, const void *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 391 "/usr/include/string.h" 3
extern "C" void *mempcpy(void *__restrict__, const void *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 399 "/usr/include/string.h" 3
extern "C" size_t strlen(const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 406 "/usr/include/string.h" 3
extern "C" size_t strnlen(const char *, size_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 413 "/usr/include/string.h" 3
extern "C" char *strerror(int) throw();
# 438 "/usr/include/string.h" 3
extern "C" char *strerror_r(int, char *, size_t) throw() __attribute__((nonnull(2)));
# 445 "/usr/include/string.h" 3
extern "C" char *strerror_l(int, __locale_t) throw();
# 451 "/usr/include/string.h" 3
extern "C" void __bzero(void *, size_t) throw() __attribute__((nonnull(1)));
# 455 "/usr/include/string.h" 3
extern "C" void bcopy(const void *, void *, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 459 "/usr/include/string.h" 3
extern "C" void bzero(void *, size_t) throw() __attribute__((nonnull(1)));
# 462 "/usr/include/string.h" 3
extern "C" int bcmp(const void *, const void *, size_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 469 "/usr/include/string.h" 3
extern char *index(char *, int) throw() __asm__("index") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 471 "/usr/include/string.h" 3
extern const char *index(const char *, int) throw() __asm__("index") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 497 "/usr/include/string.h" 3
extern char *rindex(char *, int) throw() __asm__("rindex") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 499 "/usr/include/string.h" 3
extern const char *rindex(const char *, int) throw() __asm__("rindex") __attribute__((__pure__)) __attribute__((nonnull(1)));
# 523 "/usr/include/string.h" 3
extern "C" int ffs(int) throw() __attribute__((__const__));
# 528 "/usr/include/string.h" 3
extern "C" int ffsl(long) throw() __attribute__((__const__));
# 530 "/usr/include/string.h" 3
extern "C" int ffsll(long long) throw() __attribute__((__const__));
# 536 "/usr/include/string.h" 3
extern "C" int strcasecmp(const char *, const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 540 "/usr/include/string.h" 3
extern "C" int strncasecmp(const char *, const char *, size_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 547 "/usr/include/string.h" 3
extern "C" int strcasecmp_l(const char *, const char *, __locale_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2))) __attribute__((nonnull(3)));
# 551 "/usr/include/string.h" 3
extern "C" int strncasecmp_l(const char *, const char *, size_t, __locale_t) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2))) __attribute__((nonnull(4)));
# 559 "/usr/include/string.h" 3
extern "C" char *strsep(char **__restrict__, const char *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 566 "/usr/include/string.h" 3
extern "C" char *strsignal(int) throw();
# 569 "/usr/include/string.h" 3
extern "C" char *__stpcpy(char *__restrict__, const char *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 571 "/usr/include/string.h" 3
extern "C" char *stpcpy(char *__restrict__, const char *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 576 "/usr/include/string.h" 3
extern "C" char *__stpncpy(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 579 "/usr/include/string.h" 3
extern "C" char *stpncpy(char *__restrict__, const char *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 586 "/usr/include/string.h" 3
extern "C" int strverscmp(const char *, const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 590 "/usr/include/string.h" 3
extern "C" char *strfry(char *) throw() __attribute__((nonnull(1)));
# 593 "/usr/include/string.h" 3
extern "C" void *memfrob(void *, size_t) throw() __attribute__((nonnull(1)));
# 601 "/usr/include/string.h" 3
char *basename(char *) throw() __asm__("basename") __attribute__((nonnull(1)));
# 603 "/usr/include/string.h" 3
const char *basename(const char *) throw() __asm__("basename") __attribute__((nonnull(1)));
# 31 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned char __u_char; }
# 32 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned short __u_short; }
# 33 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned __u_int; }
# 34 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __u_long; }
# 37 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef signed char __int8_t; }
# 38 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned char __uint8_t; }
# 39 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef signed short __int16_t; }
# 40 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned short __uint16_t; }
# 41 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef signed int __int32_t; }
# 42 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned __uint32_t; }
# 44 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef signed long __int64_t; }
# 45 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __uint64_t; }
# 53 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef long __quad_t; }
# 54 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __u_quad_t; }
# 134 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __dev_t; }
# 135 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned __uid_t; }
# 136 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned __gid_t; }
# 137 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __ino_t; }
# 138 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __ino64_t; }
# 139 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned __mode_t; }
# 140 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __nlink_t; }
# 141 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef long __off_t; }
# 142 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef long __off64_t; }
# 143 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef int __pid_t; }
# 144 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef struct { int __val[2]; } __fsid_t; }
# 145 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef long __clock_t; }
# 146 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __rlim_t; }
# 147 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __rlim64_t; }
# 148 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned __id_t; }
# 149 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef long __time_t; }
# 150 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned __useconds_t; }
# 151 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef long __suseconds_t; }
# 153 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef int __daddr_t; }
# 154 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef long __swblk_t; }
# 155 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef int __key_t; }
# 158 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef int __clockid_t; }
# 161 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef void *__timer_t; }
# 164 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef long __blksize_t; }
# 169 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef long __blkcnt_t; }
# 170 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef long __blkcnt64_t; }
# 173 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __fsblkcnt_t; }
# 174 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __fsblkcnt64_t; }
# 177 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __fsfilcnt_t; }
# 178 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned long __fsfilcnt64_t; }
# 180 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef long __ssize_t; }
# 184 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef __off64_t __loff_t; }
# 185 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef __quad_t *__qaddr_t; }
# 186 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef char *__caddr_t; }
# 189 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef long __intptr_t; }
# 192 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
extern "C" { typedef unsigned __socklen_t; }
# 60 "/usr/include/time.h" 3
extern "C" { typedef __clock_t clock_t; }
# 76 "/usr/include/time.h" 3
extern "C" { typedef __time_t time_t; }
# 92 "/usr/include/time.h" 3
extern "C" { typedef __clockid_t clockid_t; }
# 104 "/usr/include/time.h" 3
extern "C" { typedef __timer_t timer_t; }
# 120 "/usr/include/time.h" 3
extern "C" { struct timespec {
# 122 "/usr/include/time.h" 3
__time_t tv_sec;
# 123 "/usr/include/time.h" 3
long tv_nsec;
# 124 "/usr/include/time.h" 3
}; }
# 133 "/usr/include/time.h" 3
extern "C" { struct tm {
# 135 "/usr/include/time.h" 3
int tm_sec;
# 136 "/usr/include/time.h" 3
int tm_min;
# 137 "/usr/include/time.h" 3
int tm_hour;
# 138 "/usr/include/time.h" 3
int tm_mday;
# 139 "/usr/include/time.h" 3
int tm_mon;
# 140 "/usr/include/time.h" 3
int tm_year;
# 141 "/usr/include/time.h" 3
int tm_wday;
# 142 "/usr/include/time.h" 3
int tm_yday;
# 143 "/usr/include/time.h" 3
int tm_isdst;
# 146 "/usr/include/time.h" 3
long tm_gmtoff;
# 147 "/usr/include/time.h" 3
const char *tm_zone;
# 152 "/usr/include/time.h" 3
}; }
# 161 "/usr/include/time.h" 3
extern "C" { struct itimerspec {
# 163 "/usr/include/time.h" 3
timespec it_interval;
# 164 "/usr/include/time.h" 3
timespec it_value;
# 165 "/usr/include/time.h" 3
}; }
# 168 "/usr/include/time.h" 3
struct sigevent;
# 174 "/usr/include/time.h" 3
extern "C" { typedef __pid_t pid_t; }
# 183 "/usr/include/time.h" 3
extern "C" __attribute__((weak)) clock_t clock() throw();
# 186 "/usr/include/time.h" 3
extern "C" time_t time(time_t *) throw();
# 189 "/usr/include/time.h" 3
extern "C" double difftime(time_t, time_t) throw() __attribute__((__const__));
# 193 "/usr/include/time.h" 3
extern "C" time_t mktime(tm *) throw();
# 199 "/usr/include/time.h" 3
extern "C" size_t strftime(char *__restrict__, size_t, const char *__restrict__, const tm *__restrict__) throw();
# 207 "/usr/include/time.h" 3
extern "C" char *strptime(const char *__restrict__, const char *__restrict__, tm *) throw();
# 217 "/usr/include/time.h" 3
extern "C" size_t strftime_l(char *__restrict__, size_t, const char *__restrict__, const tm *__restrict__, __locale_t) throw();
# 224 "/usr/include/time.h" 3
extern "C" char *strptime_l(const char *__restrict__, const char *__restrict__, tm *, __locale_t) throw();
# 233 "/usr/include/time.h" 3
extern "C" tm *gmtime(const time_t *) throw();
# 237 "/usr/include/time.h" 3
extern "C" tm *localtime(const time_t *) throw();
# 243 "/usr/include/time.h" 3
extern "C" tm *gmtime_r(const time_t *__restrict__, tm *__restrict__) throw();
# 248 "/usr/include/time.h" 3
extern "C" tm *localtime_r(const time_t *__restrict__, tm *__restrict__) throw();
# 255 "/usr/include/time.h" 3
extern "C" char *asctime(const tm *) throw();
# 258 "/usr/include/time.h" 3
extern "C" char *ctime(const time_t *) throw();
# 266 "/usr/include/time.h" 3
extern "C" char *asctime_r(const tm *__restrict__, char *__restrict__) throw();
# 270 "/usr/include/time.h" 3
extern "C" char *ctime_r(const time_t *__restrict__, char *__restrict__) throw();
# 276 "/usr/include/time.h" 3
extern "C" { extern char *__tzname[2]; }
# 277 "/usr/include/time.h" 3
extern "C" { extern int __daylight; }
# 278 "/usr/include/time.h" 3
extern "C" { extern long __timezone; }
# 283 "/usr/include/time.h" 3
extern "C" { extern char *tzname[2]; }
# 287 "/usr/include/time.h" 3
extern "C" void tzset() throw();
# 291 "/usr/include/time.h" 3
extern "C" { extern int daylight; }
# 292 "/usr/include/time.h" 3
extern "C" { extern long timezone; }
# 298 "/usr/include/time.h" 3
extern "C" int stime(const time_t *) throw();
# 313 "/usr/include/time.h" 3
extern "C" time_t timegm(tm *) throw();
# 316 "/usr/include/time.h" 3
extern "C" time_t timelocal(tm *) throw();
# 319 "/usr/include/time.h" 3
extern "C" int dysize(int) throw() __attribute__((__const__));
# 328 "/usr/include/time.h" 3
extern "C" int nanosleep(const timespec *, timespec *);
# 333 "/usr/include/time.h" 3
extern "C" int clock_getres(clockid_t, timespec *) throw();
# 336 "/usr/include/time.h" 3
extern "C" int clock_gettime(clockid_t, timespec *) throw();
# 339 "/usr/include/time.h" 3
extern "C" int clock_settime(clockid_t, const timespec *) throw();
# 347 "/usr/include/time.h" 3
extern "C" int clock_nanosleep(clockid_t, int, const timespec *, timespec *);
# 352 "/usr/include/time.h" 3
extern "C" int clock_getcpuclockid(pid_t, clockid_t *) throw();
# 357 "/usr/include/time.h" 3
extern "C" int timer_create(clockid_t, sigevent *__restrict__, timer_t *__restrict__) throw();
# 362 "/usr/include/time.h" 3
extern "C" int timer_delete(timer_t) throw();
# 365 "/usr/include/time.h" 3
extern "C" int timer_settime(timer_t, int, const itimerspec *__restrict__, itimerspec *__restrict__) throw();
# 370 "/usr/include/time.h" 3
extern "C" int timer_gettime(timer_t, itimerspec *) throw();
# 374 "/usr/include/time.h" 3
extern "C" int timer_getoverrun(timer_t) throw();
# 390 "/usr/include/time.h" 3
extern "C" { extern int getdate_err; }
# 399 "/usr/include/time.h" 3
extern "C" tm *getdate(const char *);
# 413 "/usr/include/time.h" 3
extern "C" int getdate_r(const char *__restrict__, tm *__restrict__);
# 57 "/home/bachelor/deicide218/cuda-3.2/bin/../include/common_functions.h"
extern "C" __attribute__((weak)) clock_t clock() throw();
# 59 "/home/bachelor/deicide218/cuda-3.2/bin/../include/common_functions.h"
extern "C" __attribute__((weak)) void *memset(void *, int, size_t) throw() __attribute__((nonnull(1)));
# 61 "/home/bachelor/deicide218/cuda-3.2/bin/../include/common_functions.h"
extern "C" __attribute__((weak)) void *memcpy(void *, const void *, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 66 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int abs(int) throw() __attribute__((__const__));
# 68 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long labs(long) throw() __attribute__((__const__));
# 70 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llabs(long long) throw() __attribute__((__const__));
# 72 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double fabs(double) throw() __attribute__((__const__));
# 74 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float fabsf(float) throw() __attribute__((__const__));
# 77 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int min(int, int);
# 79 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) unsigned umin(unsigned, unsigned);
# 81 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llmin(long long, long long);
# 83 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) unsigned long long ullmin(unsigned long long, unsigned long long);
# 85 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float fminf(float, float) throw();
# 87 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double fmin(double, double) throw();
# 90 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int max(int, int);
# 92 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) unsigned umax(unsigned, unsigned);
# 94 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llmax(long long, long long);
# 96 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) unsigned long long ullmax(unsigned long long, unsigned long long);
# 98 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float fmaxf(float, float) throw();
# 100 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double fmax(double, double) throw();
# 103 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double sin(double) throw();
# 105 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float sinf(float) throw();
# 108 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double cos(double) throw();
# 110 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float cosf(float) throw();
# 113 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) void sincos(double, double *, double *) throw();
# 115 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) void sincosf(float, float *, float *) throw();
# 118 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double tan(double) throw();
# 120 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float tanf(float) throw();
# 123 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double sqrt(double) throw();
# 125 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float sqrtf(float) throw();
# 128 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double rsqrt(double);
# 130 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float rsqrtf(float);
# 133 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double exp2(double) throw();
# 135 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float exp2f(float) throw();
# 138 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double exp10(double) throw();
# 140 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float exp10f(float) throw();
# 143 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double expm1(double) throw();
# 145 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float expm1f(float) throw();
# 148 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double log2(double) throw();
# 150 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float log2f(float) throw();
# 153 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double log10(double) throw();
# 155 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float log10f(float) throw();
# 158 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double log(double) throw();
# 160 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float logf(float) throw();
# 163 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double log1p(double) throw();
# 165 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float log1pf(float) throw();
# 168 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double floor(double) throw() __attribute__((__const__));
# 170 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float floorf(float) throw() __attribute__((__const__));
# 173 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double exp(double) throw();
# 175 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float expf(float) throw();
# 178 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double cosh(double) throw();
# 180 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float coshf(float) throw();
# 183 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double sinh(double) throw();
# 185 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float sinhf(float) throw();
# 188 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double tanh(double) throw();
# 190 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float tanhf(float) throw();
# 193 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double acosh(double) throw();
# 195 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float acoshf(float) throw();
# 198 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double asinh(double) throw();
# 200 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float asinhf(float) throw();
# 203 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double atanh(double) throw();
# 205 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float atanhf(float) throw();
# 208 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double ldexp(double, int) throw();
# 210 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float ldexpf(float, int) throw();
# 213 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double logb(double) throw();
# 215 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float logbf(float) throw();
# 218 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int ilogb(double) throw();
# 220 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int ilogbf(float) throw();
# 223 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double scalbn(double, int) throw();
# 225 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float scalbnf(float, int) throw();
# 228 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double scalbln(double, long) throw();
# 230 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float scalblnf(float, long) throw();
# 233 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double frexp(double, int *) throw();
# 235 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float frexpf(float, int *) throw();
# 238 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double round(double) throw() __attribute__((__const__));
# 240 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float roundf(float) throw() __attribute__((__const__));
# 243 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long lround(double) throw();
# 245 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long lroundf(float) throw();
# 248 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llround(double) throw();
# 250 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llroundf(float) throw();
# 253 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double rint(double) throw();
# 255 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float rintf(float) throw();
# 258 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long lrint(double) throw();
# 260 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long lrintf(float) throw();
# 263 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llrint(double) throw();
# 265 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) long long llrintf(float) throw();
# 268 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double nearbyint(double) throw();
# 270 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float nearbyintf(float) throw();
# 273 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double ceil(double) throw() __attribute__((__const__));
# 275 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float ceilf(float) throw() __attribute__((__const__));
# 278 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double trunc(double) throw() __attribute__((__const__));
# 280 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float truncf(float) throw() __attribute__((__const__));
# 283 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double fdim(double, double) throw();
# 285 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float fdimf(float, float) throw();
# 288 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double atan2(double, double) throw();
# 290 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float atan2f(float, float) throw();
# 293 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double atan(double) throw();
# 295 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float atanf(float) throw();
# 298 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double asin(double) throw();
# 300 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float asinf(float) throw();
# 303 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double acos(double) throw();
# 305 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float acosf(float) throw();
# 308 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double hypot(double, double) throw();
# 310 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float hypotf(float, float) throw();
# 313 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double cbrt(double) throw();
# 315 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float cbrtf(float) throw();
# 318 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double rcbrt(double);
# 320 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float rcbrtf(float);
# 323 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double sinpi(double);
# 325 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float sinpif(float);
# 328 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double pow(double, double) throw();
# 330 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float powf(float, float) throw();
# 333 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double modf(double, double *) throw();
# 335 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float modff(float, float *) throw();
# 338 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double fmod(double, double) throw();
# 340 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float fmodf(float, float) throw();
# 343 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double remainder(double, double) throw();
# 345 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float remainderf(float, float) throw();
# 348 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double remquo(double, double, int *) throw();
# 350 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float remquof(float, float, int *) throw();
# 353 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double erf(double) throw();
# 355 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float erff(float) throw();
# 358 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double erfinv(double);
# 360 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float erfinvf(float);
# 363 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double erfc(double) throw();
# 365 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float erfcf(float) throw();
# 368 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double erfcinv(double);
# 370 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float erfcinvf(float);
# 373 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double lgamma(double) throw();
# 375 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float lgammaf(float) throw();
# 378 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double tgamma(double) throw();
# 380 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float tgammaf(float) throw();
# 383 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double copysign(double, double) throw() __attribute__((__const__));
# 385 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float copysignf(float, float) throw() __attribute__((__const__));
# 388 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double nextafter(double, double) throw() __attribute__((__const__));
# 390 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float nextafterf(float, float) throw() __attribute__((__const__));
# 393 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double nan(const char *) throw() __attribute__((__const__));
# 395 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float nanf(const char *) throw() __attribute__((__const__));
# 398 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __isinf(double) throw() __attribute__((__const__));
# 400 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __isinff(float) throw() __attribute__((__const__));
# 403 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __isnan(double) throw() __attribute__((__const__));
# 405 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __isnanf(float) throw() __attribute__((__const__));
# 419 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __finite(double) throw() __attribute__((__const__));
# 421 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __finitef(float) throw() __attribute__((__const__));
# 423 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __signbit(double) throw() __attribute__((__const__));
# 428 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __signbitf(float) throw() __attribute__((__const__));
# 431 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) double fma(double, double, double) throw();
# 433 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) float fmaf(float, float, float) throw();
# 441 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __signbitl(long double) throw() __attribute__((__const__));
# 443 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __isinfl(long double) throw() __attribute__((__const__));
# 445 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __isnanl(long double) throw() __attribute__((__const__));
# 455 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern "C" __attribute__((weak)) int __finitel(long double) throw() __attribute__((__const__));
# 31 "/usr/include/x86_64-linux-gnu/bits/mathdef.h" 3
extern "C" { typedef float float_t; }
# 32 "/usr/include/x86_64-linux-gnu/bits/mathdef.h" 3
extern "C" { typedef double double_t; }
# 55 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double acos(double) throw(); extern "C" double __acos(double) throw();
# 57 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double asin(double) throw(); extern "C" double __asin(double) throw();
# 59 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double atan(double) throw(); extern "C" double __atan(double) throw();
# 61 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double atan2(double, double) throw(); extern "C" double __atan2(double, double) throw();
# 64 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double cos(double) throw(); extern "C" double __cos(double) throw();
# 66 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double sin(double) throw(); extern "C" double __sin(double) throw();
# 68 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double tan(double) throw(); extern "C" double __tan(double) throw();
# 73 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double cosh(double) throw(); extern "C" double __cosh(double) throw();
# 75 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double sinh(double) throw(); extern "C" double __sinh(double) throw();
# 77 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double tanh(double) throw(); extern "C" double __tanh(double) throw();
# 82 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) void sincos(double, double *, double *) throw(); extern "C" void __sincos(double, double *, double *) throw();
# 89 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double acosh(double) throw(); extern "C" double __acosh(double) throw();
# 91 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double asinh(double) throw(); extern "C" double __asinh(double) throw();
# 93 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double atanh(double) throw(); extern "C" double __atanh(double) throw();
# 101 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double exp(double) throw(); extern "C" double __exp(double) throw();
# 104 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double frexp(double, int *) throw(); extern "C" double __frexp(double, int *) throw();
# 107 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double ldexp(double, int) throw(); extern "C" double __ldexp(double, int) throw();
# 110 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double log(double) throw(); extern "C" double __log(double) throw();
# 113 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double log10(double) throw(); extern "C" double __log10(double) throw();
# 116 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double modf(double, double *) throw(); extern "C" double __modf(double, double *) throw();
# 121 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double exp10(double) throw(); extern "C" double __exp10(double) throw();
# 123 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double pow10(double) throw(); extern "C" double __pow10(double) throw();
# 129 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double expm1(double) throw(); extern "C" double __expm1(double) throw();
# 132 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double log1p(double) throw(); extern "C" double __log1p(double) throw();
# 135 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double logb(double) throw(); extern "C" double __logb(double) throw();
# 142 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double exp2(double) throw(); extern "C" double __exp2(double) throw();
# 145 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double log2(double) throw(); extern "C" double __log2(double) throw();
# 154 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double pow(double, double) throw(); extern "C" double __pow(double, double) throw();
# 157 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double sqrt(double) throw(); extern "C" double __sqrt(double) throw();
# 163 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double hypot(double, double) throw(); extern "C" double __hypot(double, double) throw();
# 170 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double cbrt(double) throw(); extern "C" double __cbrt(double) throw();
# 179 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double ceil(double) throw() __attribute__((__const__)); extern "C" double __ceil(double) throw() __attribute__((__const__));
# 182 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double fabs(double) throw() __attribute__((__const__)); extern "C" double __fabs(double) throw() __attribute__((__const__));
# 185 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double floor(double) throw() __attribute__((__const__)); extern "C" double __floor(double) throw() __attribute__((__const__));
# 188 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double fmod(double, double) throw(); extern "C" double __fmod(double, double) throw();
# 193 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __isinf(double) throw() __attribute__((__const__));
# 196 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __finite(double) throw() __attribute__((__const__));
# 202 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int isinf(double) throw() __attribute__((__const__));
# 205 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int finite(double) throw() __attribute__((__const__));
# 208 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double drem(double, double) throw(); extern "C" double __drem(double, double) throw();
# 212 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double significand(double) throw(); extern "C" double __significand(double) throw();
# 218 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double copysign(double, double) throw() __attribute__((__const__)); extern "C" double __copysign(double, double) throw() __attribute__((__const__));
# 225 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double nan(const char *) throw() __attribute__((__const__)); extern "C" double __nan(const char *) throw() __attribute__((__const__));
# 231 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __isnan(double) throw() __attribute__((__const__));
# 235 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int isnan(double) throw() __attribute__((__const__));
# 238 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double j0(double) throw(); extern "C" double __j0(double) throw();
# 239 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double j1(double) throw(); extern "C" double __j1(double) throw();
# 240 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double jn(int, double) throw(); extern "C" double __jn(int, double) throw();
# 241 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double y0(double) throw(); extern "C" double __y0(double) throw();
# 242 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double y1(double) throw(); extern "C" double __y1(double) throw();
# 243 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double yn(int, double) throw(); extern "C" double __yn(int, double) throw();
# 250 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double erf(double) throw(); extern "C" double __erf(double) throw();
# 251 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double erfc(double) throw(); extern "C" double __erfc(double) throw();
# 252 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double lgamma(double) throw(); extern "C" double __lgamma(double) throw();
# 259 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double tgamma(double) throw(); extern "C" double __tgamma(double) throw();
# 265 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double gamma(double) throw(); extern "C" double __gamma(double) throw();
# 272 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double lgamma_r(double, int *) throw(); extern "C" double __lgamma_r(double, int *) throw();
# 280 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double rint(double) throw(); extern "C" double __rint(double) throw();
# 283 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double nextafter(double, double) throw() __attribute__((__const__)); extern "C" double __nextafter(double, double) throw() __attribute__((__const__));
# 285 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double nexttoward(double, long double) throw() __attribute__((__const__)); extern "C" double __nexttoward(double, long double) throw() __attribute__((__const__));
# 289 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double remainder(double, double) throw(); extern "C" double __remainder(double, double) throw();
# 293 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double scalbn(double, int) throw(); extern "C" double __scalbn(double, int) throw();
# 297 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int ilogb(double) throw(); extern "C" int __ilogb(double) throw();
# 302 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double scalbln(double, long) throw(); extern "C" double __scalbln(double, long) throw();
# 306 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double nearbyint(double) throw(); extern "C" double __nearbyint(double) throw();
# 310 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double round(double) throw() __attribute__((__const__)); extern "C" double __round(double) throw() __attribute__((__const__));
# 314 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double trunc(double) throw() __attribute__((__const__)); extern "C" double __trunc(double) throw() __attribute__((__const__));
# 319 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double remquo(double, double, int *) throw(); extern "C" double __remquo(double, double, int *) throw();
# 326 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long lrint(double) throw(); extern "C" long __lrint(double) throw();
# 327 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long long llrint(double) throw(); extern "C" long long __llrint(double) throw();
# 331 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long lround(double) throw(); extern "C" long __lround(double) throw();
# 332 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long long llround(double) throw(); extern "C" long long __llround(double) throw();
# 336 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double fdim(double, double) throw(); extern "C" double __fdim(double, double) throw();
# 339 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double fmax(double, double) throw(); extern "C" double __fmax(double, double) throw();
# 342 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double fmin(double, double) throw(); extern "C" double __fmin(double, double) throw();
# 346 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int __fpclassify(double) throw() __attribute__((__const__));
# 350 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __signbit(double) throw() __attribute__((__const__));
# 355 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) double fma(double, double, double) throw(); extern "C" double __fma(double, double, double) throw();
# 364 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" double scalb(double, double) throw(); extern "C" double __scalb(double, double) throw();
# 55 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float acosf(float) throw(); extern "C" float __acosf(float) throw();
# 57 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float asinf(float) throw(); extern "C" float __asinf(float) throw();
# 59 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float atanf(float) throw(); extern "C" float __atanf(float) throw();
# 61 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float atan2f(float, float) throw(); extern "C" float __atan2f(float, float) throw();
# 64 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float cosf(float) throw();
# 66 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float sinf(float) throw();
# 68 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float tanf(float) throw();
# 73 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float coshf(float) throw(); extern "C" float __coshf(float) throw();
# 75 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float sinhf(float) throw(); extern "C" float __sinhf(float) throw();
# 77 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float tanhf(float) throw(); extern "C" float __tanhf(float) throw();
# 82 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) void sincosf(float, float *, float *) throw();
# 89 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float acoshf(float) throw(); extern "C" float __acoshf(float) throw();
# 91 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float asinhf(float) throw(); extern "C" float __asinhf(float) throw();
# 93 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float atanhf(float) throw(); extern "C" float __atanhf(float) throw();
# 101 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float expf(float) throw();
# 104 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float frexpf(float, int *) throw(); extern "C" float __frexpf(float, int *) throw();
# 107 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float ldexpf(float, int) throw(); extern "C" float __ldexpf(float, int) throw();
# 110 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float logf(float) throw();
# 113 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float log10f(float) throw();
# 116 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float modff(float, float *) throw(); extern "C" float __modff(float, float *) throw();
# 121 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float exp10f(float) throw();
# 123 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float pow10f(float) throw(); extern "C" float __pow10f(float) throw();
# 129 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float expm1f(float) throw(); extern "C" float __expm1f(float) throw();
# 132 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float log1pf(float) throw(); extern "C" float __log1pf(float) throw();
# 135 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float logbf(float) throw(); extern "C" float __logbf(float) throw();
# 142 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float exp2f(float) throw(); extern "C" float __exp2f(float) throw();
# 145 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float log2f(float) throw();
# 154 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float powf(float, float) throw();
# 157 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float sqrtf(float) throw(); extern "C" float __sqrtf(float) throw();
# 163 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float hypotf(float, float) throw(); extern "C" float __hypotf(float, float) throw();
# 170 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float cbrtf(float) throw(); extern "C" float __cbrtf(float) throw();
# 179 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float ceilf(float) throw() __attribute__((__const__)); extern "C" float __ceilf(float) throw() __attribute__((__const__));
# 182 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float fabsf(float) throw() __attribute__((__const__)); extern "C" float __fabsf(float) throw() __attribute__((__const__));
# 185 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float floorf(float) throw() __attribute__((__const__)); extern "C" float __floorf(float) throw() __attribute__((__const__));
# 188 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float fmodf(float, float) throw(); extern "C" float __fmodf(float, float) throw();
# 193 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __isinff(float) throw() __attribute__((__const__));
# 196 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __finitef(float) throw() __attribute__((__const__));
# 202 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int isinff(float) throw() __attribute__((__const__));
# 205 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int finitef(float) throw() __attribute__((__const__));
# 208 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float dremf(float, float) throw(); extern "C" float __dremf(float, float) throw();
# 212 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float significandf(float) throw(); extern "C" float __significandf(float) throw();
# 218 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float copysignf(float, float) throw() __attribute__((__const__)); extern "C" float __copysignf(float, float) throw() __attribute__((__const__));
# 225 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float nanf(const char *) throw() __attribute__((__const__)); extern "C" float __nanf(const char *) throw() __attribute__((__const__));
# 231 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __isnanf(float) throw() __attribute__((__const__));
# 235 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int isnanf(float) throw() __attribute__((__const__));
# 238 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float j0f(float) throw(); extern "C" float __j0f(float) throw();
# 239 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float j1f(float) throw(); extern "C" float __j1f(float) throw();
# 240 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float jnf(int, float) throw(); extern "C" float __jnf(int, float) throw();
# 241 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float y0f(float) throw(); extern "C" float __y0f(float) throw();
# 242 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float y1f(float) throw(); extern "C" float __y1f(float) throw();
# 243 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float ynf(int, float) throw(); extern "C" float __ynf(int, float) throw();
# 250 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float erff(float) throw(); extern "C" float __erff(float) throw();
# 251 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float erfcf(float) throw(); extern "C" float __erfcf(float) throw();
# 252 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float lgammaf(float) throw(); extern "C" float __lgammaf(float) throw();
# 259 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float tgammaf(float) throw(); extern "C" float __tgammaf(float) throw();
# 265 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float gammaf(float) throw(); extern "C" float __gammaf(float) throw();
# 272 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float lgammaf_r(float, int *) throw(); extern "C" float __lgammaf_r(float, int *) throw();
# 280 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float rintf(float) throw(); extern "C" float __rintf(float) throw();
# 283 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float nextafterf(float, float) throw() __attribute__((__const__)); extern "C" float __nextafterf(float, float) throw() __attribute__((__const__));
# 285 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float nexttowardf(float, long double) throw() __attribute__((__const__)); extern "C" float __nexttowardf(float, long double) throw() __attribute__((__const__));
# 289 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float remainderf(float, float) throw(); extern "C" float __remainderf(float, float) throw();
# 293 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float scalbnf(float, int) throw(); extern "C" float __scalbnf(float, int) throw();
# 297 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int ilogbf(float) throw(); extern "C" int __ilogbf(float) throw();
# 302 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float scalblnf(float, long) throw(); extern "C" float __scalblnf(float, long) throw();
# 306 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float nearbyintf(float) throw(); extern "C" float __nearbyintf(float) throw();
# 310 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float roundf(float) throw() __attribute__((__const__)); extern "C" float __roundf(float) throw() __attribute__((__const__));
# 314 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float truncf(float) throw() __attribute__((__const__)); extern "C" float __truncf(float) throw() __attribute__((__const__));
# 319 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float remquof(float, float, int *) throw(); extern "C" float __remquof(float, float, int *) throw();
# 326 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long lrintf(float) throw(); extern "C" long __lrintf(float) throw();
# 327 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long long llrintf(float) throw(); extern "C" long long __llrintf(float) throw();
# 331 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long lroundf(float) throw(); extern "C" long __lroundf(float) throw();
# 332 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) long long llroundf(float) throw(); extern "C" long long __llroundf(float) throw();
# 336 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float fdimf(float, float) throw(); extern "C" float __fdimf(float, float) throw();
# 339 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float fmaxf(float, float) throw(); extern "C" float __fmaxf(float, float) throw();
# 342 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float fminf(float, float) throw(); extern "C" float __fminf(float, float) throw();
# 346 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int __fpclassifyf(float) throw() __attribute__((__const__));
# 350 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __signbitf(float) throw() __attribute__((__const__));
# 355 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) float fmaf(float, float, float) throw(); extern "C" float __fmaf(float, float, float) throw();
# 364 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" float scalbf(float, float) throw(); extern "C" float __scalbf(float, float) throw();
# 55 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double acosl(long double) throw(); extern "C" long double __acosl(long double) throw();
# 57 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double asinl(long double) throw(); extern "C" long double __asinl(long double) throw();
# 59 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double atanl(long double) throw(); extern "C" long double __atanl(long double) throw();
# 61 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double atan2l(long double, long double) throw(); extern "C" long double __atan2l(long double, long double) throw();
# 64 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double cosl(long double) throw(); extern "C" long double __cosl(long double) throw();
# 66 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double sinl(long double) throw(); extern "C" long double __sinl(long double) throw();
# 68 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double tanl(long double) throw(); extern "C" long double __tanl(long double) throw();
# 73 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double coshl(long double) throw(); extern "C" long double __coshl(long double) throw();
# 75 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double sinhl(long double) throw(); extern "C" long double __sinhl(long double) throw();
# 77 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double tanhl(long double) throw(); extern "C" long double __tanhl(long double) throw();
# 82 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" void sincosl(long double, long double *, long double *) throw(); extern "C" void __sincosl(long double, long double *, long double *) throw();
# 89 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double acoshl(long double) throw(); extern "C" long double __acoshl(long double) throw();
# 91 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double asinhl(long double) throw(); extern "C" long double __asinhl(long double) throw();
# 93 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double atanhl(long double) throw(); extern "C" long double __atanhl(long double) throw();
# 101 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double expl(long double) throw(); extern "C" long double __expl(long double) throw();
# 104 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double frexpl(long double, int *) throw(); extern "C" long double __frexpl(long double, int *) throw();
# 107 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double ldexpl(long double, int) throw(); extern "C" long double __ldexpl(long double, int) throw();
# 110 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double logl(long double) throw(); extern "C" long double __logl(long double) throw();
# 113 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double log10l(long double) throw(); extern "C" long double __log10l(long double) throw();
# 116 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double modfl(long double, long double *) throw(); extern "C" long double __modfl(long double, long double *) throw();
# 121 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double exp10l(long double) throw(); extern "C" long double __exp10l(long double) throw();
# 123 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double pow10l(long double) throw(); extern "C" long double __pow10l(long double) throw();
# 129 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double expm1l(long double) throw(); extern "C" long double __expm1l(long double) throw();
# 132 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double log1pl(long double) throw(); extern "C" long double __log1pl(long double) throw();
# 135 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double logbl(long double) throw(); extern "C" long double __logbl(long double) throw();
# 142 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double exp2l(long double) throw(); extern "C" long double __exp2l(long double) throw();
# 145 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double log2l(long double) throw(); extern "C" long double __log2l(long double) throw();
# 154 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double powl(long double, long double) throw(); extern "C" long double __powl(long double, long double) throw();
# 157 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double sqrtl(long double) throw(); extern "C" long double __sqrtl(long double) throw();
# 163 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double hypotl(long double, long double) throw(); extern "C" long double __hypotl(long double, long double) throw();
# 170 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double cbrtl(long double) throw(); extern "C" long double __cbrtl(long double) throw();
# 179 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double ceill(long double) throw() __attribute__((__const__)); extern "C" long double __ceill(long double) throw() __attribute__((__const__));
# 182 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double fabsl(long double) throw() __attribute__((__const__)); extern "C" long double __fabsl(long double) throw() __attribute__((__const__));
# 185 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double floorl(long double) throw() __attribute__((__const__)); extern "C" long double __floorl(long double) throw() __attribute__((__const__));
# 188 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double fmodl(long double, long double) throw(); extern "C" long double __fmodl(long double, long double) throw();
# 193 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __isinfl(long double) throw() __attribute__((__const__));
# 196 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __finitel(long double) throw() __attribute__((__const__));
# 202 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int isinfl(long double) throw() __attribute__((__const__));
# 205 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int finitel(long double) throw() __attribute__((__const__));
# 208 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double dreml(long double, long double) throw(); extern "C" long double __dreml(long double, long double) throw();
# 212 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double significandl(long double) throw(); extern "C" long double __significandl(long double) throw();
# 218 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double copysignl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __copysignl(long double, long double) throw() __attribute__((__const__));
# 225 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double nanl(const char *) throw() __attribute__((__const__)); extern "C" long double __nanl(const char *) throw() __attribute__((__const__));
# 231 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __isnanl(long double) throw() __attribute__((__const__));
# 235 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int isnanl(long double) throw() __attribute__((__const__));
# 238 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double j0l(long double) throw(); extern "C" long double __j0l(long double) throw();
# 239 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double j1l(long double) throw(); extern "C" long double __j1l(long double) throw();
# 240 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double jnl(int, long double) throw(); extern "C" long double __jnl(int, long double) throw();
# 241 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double y0l(long double) throw(); extern "C" long double __y0l(long double) throw();
# 242 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double y1l(long double) throw(); extern "C" long double __y1l(long double) throw();
# 243 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double ynl(int, long double) throw(); extern "C" long double __ynl(int, long double) throw();
# 250 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double erfl(long double) throw(); extern "C" long double __erfl(long double) throw();
# 251 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double erfcl(long double) throw(); extern "C" long double __erfcl(long double) throw();
# 252 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double lgammal(long double) throw(); extern "C" long double __lgammal(long double) throw();
# 259 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double tgammal(long double) throw(); extern "C" long double __tgammal(long double) throw();
# 265 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double gammal(long double) throw(); extern "C" long double __gammal(long double) throw();
# 272 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double lgammal_r(long double, int *) throw(); extern "C" long double __lgammal_r(long double, int *) throw();
# 280 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double rintl(long double) throw(); extern "C" long double __rintl(long double) throw();
# 283 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double nextafterl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __nextafterl(long double, long double) throw() __attribute__((__const__));
# 285 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double nexttowardl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __nexttowardl(long double, long double) throw() __attribute__((__const__));
# 289 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double remainderl(long double, long double) throw(); extern "C" long double __remainderl(long double, long double) throw();
# 293 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double scalbnl(long double, int) throw(); extern "C" long double __scalbnl(long double, int) throw();
# 297 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int ilogbl(long double) throw(); extern "C" int __ilogbl(long double) throw();
# 302 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double scalblnl(long double, long) throw(); extern "C" long double __scalblnl(long double, long) throw();
# 306 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double nearbyintl(long double) throw(); extern "C" long double __nearbyintl(long double) throw();
# 310 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double roundl(long double) throw() __attribute__((__const__)); extern "C" long double __roundl(long double) throw() __attribute__((__const__));
# 314 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double truncl(long double) throw() __attribute__((__const__)); extern "C" long double __truncl(long double) throw() __attribute__((__const__));
# 319 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double remquol(long double, long double, int *) throw(); extern "C" long double __remquol(long double, long double, int *) throw();
# 326 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long lrintl(long double) throw(); extern "C" long __lrintl(long double) throw();
# 327 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long long llrintl(long double) throw(); extern "C" long long __llrintl(long double) throw();
# 331 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long lroundl(long double) throw(); extern "C" long __lroundl(long double) throw();
# 332 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long long llroundl(long double) throw(); extern "C" long long __llroundl(long double) throw();
# 336 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double fdiml(long double, long double) throw(); extern "C" long double __fdiml(long double, long double) throw();
# 339 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double fmaxl(long double, long double) throw(); extern "C" long double __fmaxl(long double, long double) throw();
# 342 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double fminl(long double, long double) throw(); extern "C" long double __fminl(long double, long double) throw();
# 346 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" int __fpclassifyl(long double) throw() __attribute__((__const__));
# 350 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" __attribute__((weak)) int __signbitl(long double) throw() __attribute__((__const__));
# 355 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double fmal(long double, long double, long double) throw(); extern "C" long double __fmal(long double, long double, long double) throw();
# 364 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern "C" long double scalbl(long double, long double) throw(); extern "C" long double __scalbl(long double, long double) throw();
# 161 "/usr/include/math.h" 3
extern "C" { extern int signgam; }
# 203 "/usr/include/math.h" 3
enum {
# 204 "/usr/include/math.h" 3
FP_NAN,
# 206 "/usr/include/math.h" 3
FP_INFINITE,
# 208 "/usr/include/math.h" 3
FP_ZERO,
# 210 "/usr/include/math.h" 3
FP_SUBNORMAL,
# 212 "/usr/include/math.h" 3
FP_NORMAL
# 214 "/usr/include/math.h" 3
};
# 302 "/usr/include/math.h" 3
extern "C" { typedef
# 296 "/usr/include/math.h" 3
enum {
# 297 "/usr/include/math.h" 3
_IEEE_ = (-1),
# 298 "/usr/include/math.h" 3
_SVID_ = 0,
# 299 "/usr/include/math.h" 3
_XOPEN_,
# 300 "/usr/include/math.h" 3
_POSIX_,
# 301 "/usr/include/math.h" 3
_ISOC_
# 302 "/usr/include/math.h" 3
} _LIB_VERSION_TYPE; }
# 307 "/usr/include/math.h" 3
extern "C" { extern _LIB_VERSION_TYPE _LIB_VERSION; }
# 318 "/usr/include/math.h" 3
extern "C" { struct __exception {
# 323 "/usr/include/math.h" 3
int type;
# 324 "/usr/include/math.h" 3
char *name;
# 325 "/usr/include/math.h" 3
double arg1;
# 326 "/usr/include/math.h" 3
double arg2;
# 327 "/usr/include/math.h" 3
double retval;
# 328 "/usr/include/math.h" 3
}; }
# 331 "/usr/include/math.h" 3
extern "C" int matherr(__exception *) throw();
# 67 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
extern "C" { union wait {
# 69 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
int w_status;
# 71 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
struct {
# 73 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
unsigned __w_termsig:7;
# 74 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
unsigned __w_coredump:1;
# 75 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
unsigned __w_retcode:8;
# 76 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
unsigned:16;
# 84 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
} __wait_terminated;
# 86 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
struct {
# 88 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
unsigned __w_stopval:8;
# 89 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
unsigned __w_stopsig:8;
# 90 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
unsigned:16;
# 97 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
} __wait_stopped;
# 98 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
}; }
# 102 "/usr/include/stdlib.h" 3
extern "C" { typedef
# 99 "/usr/include/stdlib.h" 3
struct {
# 100 "/usr/include/stdlib.h" 3
int quot;
# 101 "/usr/include/stdlib.h" 3
int rem;
# 102 "/usr/include/stdlib.h" 3
} div_t; }
# 110 "/usr/include/stdlib.h" 3
extern "C" { typedef
# 107 "/usr/include/stdlib.h" 3
struct {
# 108 "/usr/include/stdlib.h" 3
long quot;
# 109 "/usr/include/stdlib.h" 3
long rem;
# 110 "/usr/include/stdlib.h" 3
} ldiv_t; }
# 122 "/usr/include/stdlib.h" 3
extern "C" { typedef
# 119 "/usr/include/stdlib.h" 3
struct {
# 120 "/usr/include/stdlib.h" 3
long long quot;
# 121 "/usr/include/stdlib.h" 3
long long rem;
# 122 "/usr/include/stdlib.h" 3
} lldiv_t; }
# 140 "/usr/include/stdlib.h" 3
extern "C" size_t __ctype_get_mb_cur_max() throw();
# 145 "/usr/include/stdlib.h" 3
extern "C" double atof(const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 148 "/usr/include/stdlib.h" 3
extern "C" int atoi(const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 151 "/usr/include/stdlib.h" 3
extern "C" long atol(const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 158 "/usr/include/stdlib.h" 3
extern "C" long long atoll(const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 165 "/usr/include/stdlib.h" 3
extern "C" double strtod(const char *__restrict__, char **__restrict__) throw() __attribute__((nonnull(1)));
# 173 "/usr/include/stdlib.h" 3
extern "C" float strtof(const char *__restrict__, char **__restrict__) throw() __attribute__((nonnull(1)));
# 176 "/usr/include/stdlib.h" 3
extern "C" long double strtold(const char *__restrict__, char **__restrict__) throw() __attribute__((nonnull(1)));
# 184 "/usr/include/stdlib.h" 3
extern "C" long strtol(const char *__restrict__, char **__restrict__, int) throw() __attribute__((nonnull(1)));
# 188 "/usr/include/stdlib.h" 3
extern "C" unsigned long strtoul(const char *__restrict__, char **__restrict__, int) throw() __attribute__((nonnull(1)));
# 196 "/usr/include/stdlib.h" 3
extern "C" long long strtoq(const char *__restrict__, char **__restrict__, int) throw() __attribute__((nonnull(1)));
# 201 "/usr/include/stdlib.h" 3
extern "C" unsigned long long strtouq(const char *__restrict__, char **__restrict__, int) throw() __attribute__((nonnull(1)));
# 210 "/usr/include/stdlib.h" 3
extern "C" long long strtoll(const char *__restrict__, char **__restrict__, int) throw() __attribute__((nonnull(1)));
# 215 "/usr/include/stdlib.h" 3
extern "C" unsigned long long strtoull(const char *__restrict__, char **__restrict__, int) throw() __attribute__((nonnull(1)));
# 240 "/usr/include/stdlib.h" 3
extern "C" long strtol_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(4)));
# 244 "/usr/include/stdlib.h" 3
extern "C" unsigned long strtoul_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(4)));
# 250 "/usr/include/stdlib.h" 3
extern "C" long long strtoll_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(4)));
# 256 "/usr/include/stdlib.h" 3
extern "C" unsigned long long strtoull_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(4)));
# 261 "/usr/include/stdlib.h" 3
extern "C" double strtod_l(const char *__restrict__, char **__restrict__, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(3)));
# 265 "/usr/include/stdlib.h" 3
extern "C" float strtof_l(const char *__restrict__, char **__restrict__, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(3)));
# 269 "/usr/include/stdlib.h" 3
extern "C" long double strtold_l(const char *__restrict__, char **__restrict__, __locale_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(3)));
# 311 "/usr/include/stdlib.h" 3
extern "C" char *l64a(long) throw();
# 314 "/usr/include/stdlib.h" 3
extern "C" long a64l(const char *) throw() __attribute__((__pure__)) __attribute__((nonnull(1)));
# 34 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __u_char u_char; }
# 35 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __u_short u_short; }
# 36 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __u_int u_int; }
# 37 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __u_long u_long; }
# 38 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __quad_t quad_t; }
# 39 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __u_quad_t u_quad_t; }
# 40 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __fsid_t fsid_t; }
# 45 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __loff_t loff_t; }
# 49 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __ino_t ino_t; }
# 56 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __ino64_t ino64_t; }
# 61 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __dev_t dev_t; }
# 66 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __gid_t gid_t; }
# 71 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __mode_t mode_t; }
# 76 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __nlink_t nlink_t; }
# 81 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __uid_t uid_t; }
# 87 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __off_t off_t; }
# 94 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __off64_t off64_t; }
# 105 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __id_t id_t; }
# 110 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __ssize_t ssize_t; }
# 116 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __daddr_t daddr_t; }
# 117 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __caddr_t caddr_t; }
# 123 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __key_t key_t; }
# 137 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __useconds_t useconds_t; }
# 141 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __suseconds_t suseconds_t; }
# 151 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef unsigned long ulong; }
# 152 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef unsigned short ushort; }
# 153 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef unsigned uint; }
# 195 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef signed char int8_t; }
# 196 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef short int16_t; }
# 197 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef int int32_t; }
# 198 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef long int64_t; }
# 201 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef unsigned char u_int8_t; }
# 202 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef unsigned short u_int16_t; }
# 203 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef unsigned u_int32_t; }
# 204 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef unsigned long u_int64_t; }
# 206 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef long register_t; }
# 24 "/usr/include/x86_64-linux-gnu/bits/sigset.h" 3
extern "C" { typedef int __sig_atomic_t; }
# 32 "/usr/include/x86_64-linux-gnu/bits/sigset.h" 3
extern "C" { typedef
# 30 "/usr/include/x86_64-linux-gnu/bits/sigset.h" 3
struct {
# 31 "/usr/include/x86_64-linux-gnu/bits/sigset.h" 3
unsigned long __val[((1024) / ((8) * sizeof(unsigned long)))];
# 32 "/usr/include/x86_64-linux-gnu/bits/sigset.h" 3
} __sigset_t; }
# 38 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern "C" { typedef __sigset_t sigset_t; }
# 75 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
extern "C" { struct timeval {
# 77 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
__time_t tv_sec;
# 78 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
__suseconds_t tv_usec;
# 79 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
}; }
# 55 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern "C" { typedef long __fd_mask; }
# 78 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern "C" { typedef
# 68 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
struct {
# 72 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
__fd_mask fds_bits[(1024 / (8 * ((int)sizeof(__fd_mask))))];
# 78 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
} fd_set; }
# 85 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern "C" { typedef __fd_mask fd_mask; }
# 109 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern "C" int select(int, fd_set *__restrict__, fd_set *__restrict__, fd_set *__restrict__, timeval *__restrict__);
# 121 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern "C" int pselect(int, fd_set *__restrict__, fd_set *__restrict__, fd_set *__restrict__, const timespec *__restrict__, const __sigset_t *__restrict__);
# 31 "/usr/include/x86_64-linux-gnu/sys/sysmacros.h" 3
extern "C" unsigned gnu_dev_major(unsigned long long) throw();
# 34 "/usr/include/x86_64-linux-gnu/sys/sysmacros.h" 3
extern "C" unsigned gnu_dev_minor(unsigned long long) throw();
# 37 "/usr/include/x86_64-linux-gnu/sys/sysmacros.h" 3
extern "C" unsigned long long gnu_dev_makedev(unsigned, unsigned) throw();
# 229 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __blksize_t blksize_t; }
# 236 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __blkcnt_t blkcnt_t; }
# 240 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __fsblkcnt_t fsblkcnt_t; }
# 244 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __fsfilcnt_t fsfilcnt_t; }
# 263 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __blkcnt64_t blkcnt64_t; }
# 264 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __fsblkcnt64_t fsblkcnt64_t; }
# 265 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" { typedef __fsfilcnt64_t fsfilcnt64_t; }
# 50 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef unsigned long pthread_t; }
# 57 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef
# 54 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union {
# 55 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
char __size[56];
# 56 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
long __align;
# 57 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} pthread_attr_t; }
# 65 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef
# 61 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
struct __pthread_internal_list {
# 63 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
__pthread_internal_list *__prev;
# 64 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
__pthread_internal_list *__next;
# 65 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} __pthread_list_t; }
# 104 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef
# 77 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union {
# 78 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
struct __pthread_mutex_s {
# 80 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
int __lock;
# 81 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned __count;
# 82 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
int __owner;
# 84 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned __nusers;
# 88 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
int __kind;
# 90 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
int __spins;
# 91 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
__pthread_list_t __list;
# 101 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} __data;
# 102 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
char __size[40];
# 103 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
long __align;
# 104 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} pthread_mutex_t; }
# 110 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef
# 107 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union {
# 108 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
char __size[4];
# 109 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
int __align;
# 110 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} pthread_mutexattr_t; }
# 130 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef
# 116 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union {
# 118 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
struct {
# 119 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
int __lock;
# 120 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned __futex;
# 121 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
__extension__ unsigned long long __total_seq;
# 122 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
__extension__ unsigned long long __wakeup_seq;
# 123 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
__extension__ unsigned long long __woken_seq;
# 124 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
void *__mutex;
# 125 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned __nwaiters;
# 126 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned __broadcast_seq;
# 127 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} __data;
# 128 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
char __size[48];
# 129 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
__extension__ long long __align;
# 130 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} pthread_cond_t; }
# 136 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef
# 133 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union {
# 134 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
char __size[4];
# 135 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
int __align;
# 136 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} pthread_condattr_t; }
# 140 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef unsigned pthread_key_t; }
# 144 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef int pthread_once_t; }
# 189 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef
# 151 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union {
# 154 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
struct {
# 155 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
int __lock;
# 156 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned __nr_readers;
# 157 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned __readers_wakeup;
# 158 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned __writer_wakeup;
# 159 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned __nr_readers_queued;
# 160 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned __nr_writers_queued;
# 161 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
int __writer;
# 162 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
int __shared;
# 163 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned long __pad1;
# 164 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned long __pad2;
# 167 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
unsigned __flags;
# 168 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} __data;
# 187 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
char __size[56];
# 188 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
long __align;
# 189 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} pthread_rwlock_t; }
# 195 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef
# 192 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union {
# 193 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
char __size[8];
# 194 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
long __align;
# 195 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} pthread_rwlockattr_t; }
# 201 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef volatile int pthread_spinlock_t; }
# 210 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef
# 207 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union {
# 208 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
char __size[32];
# 209 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
long __align;
# 210 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} pthread_barrier_t; }
# 216 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
extern "C" { typedef
# 213 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union {
# 214 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
char __size[4];
# 215 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
int __align;
# 216 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} pthread_barrierattr_t; }
# 327 "/usr/include/stdlib.h" 3
extern "C" long random() throw();
# 330 "/usr/include/stdlib.h" 3
extern "C" void srandom(unsigned) throw();
# 336 "/usr/include/stdlib.h" 3
extern "C" char *initstate(unsigned, char *, size_t) throw() __attribute__((nonnull(2)));
# 341 "/usr/include/stdlib.h" 3
extern "C" char *setstate(char *) throw() __attribute__((nonnull(1)));
# 349 "/usr/include/stdlib.h" 3
extern "C" { struct random_data {
# 351 "/usr/include/stdlib.h" 3
int32_t *fptr;
# 352 "/usr/include/stdlib.h" 3
int32_t *rptr;
# 353 "/usr/include/stdlib.h" 3
int32_t *state;
# 354 "/usr/include/stdlib.h" 3
int rand_type;
# 355 "/usr/include/stdlib.h" 3
int rand_deg;
# 356 "/usr/include/stdlib.h" 3
int rand_sep;
# 357 "/usr/include/stdlib.h" 3
int32_t *end_ptr;
# 358 "/usr/include/stdlib.h" 3
}; }
# 360 "/usr/include/stdlib.h" 3
extern "C" int random_r(random_data *__restrict__, int32_t *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 363 "/usr/include/stdlib.h" 3
extern "C" int srandom_r(unsigned, random_data *) throw() __attribute__((nonnull(2)));
# 366 "/usr/include/stdlib.h" 3
extern "C" int initstate_r(unsigned, char *__restrict__, size_t, random_data *__restrict__) throw() __attribute__((nonnull(2))) __attribute__((nonnull(4)));
# 371 "/usr/include/stdlib.h" 3
extern "C" int setstate_r(char *__restrict__, random_data *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 380 "/usr/include/stdlib.h" 3
extern "C" int rand() throw();
# 382 "/usr/include/stdlib.h" 3
extern "C" void srand(unsigned) throw();
# 387 "/usr/include/stdlib.h" 3
extern "C" int rand_r(unsigned *) throw();
# 395 "/usr/include/stdlib.h" 3
extern "C" double drand48() throw();
# 396 "/usr/include/stdlib.h" 3
extern "C" double erand48(unsigned short [3]) throw() __attribute__((nonnull(1)));
# 399 "/usr/include/stdlib.h" 3
extern "C" long lrand48() throw();
# 400 "/usr/include/stdlib.h" 3
extern "C" long nrand48(unsigned short [3]) throw() __attribute__((nonnull(1)));
# 404 "/usr/include/stdlib.h" 3
extern "C" long mrand48() throw();
# 405 "/usr/include/stdlib.h" 3
extern "C" long jrand48(unsigned short [3]) throw() __attribute__((nonnull(1)));
# 409 "/usr/include/stdlib.h" 3
extern "C" void srand48(long) throw();
# 410 "/usr/include/stdlib.h" 3
extern "C" unsigned short *seed48(unsigned short [3]) throw() __attribute__((nonnull(1)));
# 412 "/usr/include/stdlib.h" 3
extern "C" void lcong48(unsigned short [7]) throw() __attribute__((nonnull(1)));
# 418 "/usr/include/stdlib.h" 3
extern "C" { struct drand48_data {
# 420 "/usr/include/stdlib.h" 3
unsigned short __x[3];
# 421 "/usr/include/stdlib.h" 3
unsigned short __old_x[3];
# 422 "/usr/include/stdlib.h" 3
unsigned short __c;
# 423 "/usr/include/stdlib.h" 3
unsigned short __init;
# 424 "/usr/include/stdlib.h" 3
unsigned long long __a;
# 425 "/usr/include/stdlib.h" 3
}; }
# 428 "/usr/include/stdlib.h" 3
extern "C" int drand48_r(drand48_data *__restrict__, double *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 430 "/usr/include/stdlib.h" 3
extern "C" int erand48_r(unsigned short [3], drand48_data *__restrict__, double *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 435 "/usr/include/stdlib.h" 3
extern "C" int lrand48_r(drand48_data *__restrict__, long *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 438 "/usr/include/stdlib.h" 3
extern "C" int nrand48_r(unsigned short [3], drand48_data *__restrict__, long *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 444 "/usr/include/stdlib.h" 3
extern "C" int mrand48_r(drand48_data *__restrict__, long *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 447 "/usr/include/stdlib.h" 3
extern "C" int jrand48_r(unsigned short [3], drand48_data *__restrict__, long *__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 453 "/usr/include/stdlib.h" 3
extern "C" int srand48_r(long, drand48_data *) throw() __attribute__((nonnull(2)));
# 456 "/usr/include/stdlib.h" 3
extern "C" int seed48_r(unsigned short [3], drand48_data *) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 459 "/usr/include/stdlib.h" 3
extern "C" int lcong48_r(unsigned short [7], drand48_data *) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 471 "/usr/include/stdlib.h" 3
extern "C" void *malloc(size_t) throw() __attribute__((__malloc__));
# 473 "/usr/include/stdlib.h" 3
extern "C" void *calloc(size_t, size_t) throw() __attribute__((__malloc__));
# 485 "/usr/include/stdlib.h" 3
extern "C" void *realloc(void *, size_t) throw() __attribute__((__warn_unused_result__));
# 488 "/usr/include/stdlib.h" 3
extern "C" void free(void *) throw();
# 493 "/usr/include/stdlib.h" 3
extern "C" void cfree(void *) throw();
# 33 "/usr/include/alloca.h" 3
extern "C" void *alloca(size_t) throw();
# 503 "/usr/include/stdlib.h" 3
extern "C" void *valloc(size_t) throw() __attribute__((__malloc__));
# 508 "/usr/include/stdlib.h" 3
extern "C" int posix_memalign(void **, size_t, size_t) throw() __attribute__((nonnull(1)));
# 514 "/usr/include/stdlib.h" 3
extern "C" void abort() throw() __attribute__((__noreturn__));
# 518 "/usr/include/stdlib.h" 3
extern "C" int atexit(void (*)(void)) throw() __attribute__((nonnull(1)));
# 525 "/usr/include/stdlib.h" 3
int at_quick_exit(void (*)(void)) throw() __asm__("at_quick_exit") __attribute__((nonnull(1)));
# 536 "/usr/include/stdlib.h" 3
extern "C" int on_exit(void (*)(int, void *), void *) throw() __attribute__((nonnull(1)));
# 544 "/usr/include/stdlib.h" 3
extern "C" void exit(int) throw() __attribute__((__noreturn__));
# 552 "/usr/include/stdlib.h" 3
extern "C" void quick_exit(int) throw() __attribute__((__noreturn__));
# 560 "/usr/include/stdlib.h" 3
extern "C" void _Exit(int) throw() __attribute__((__noreturn__));
# 567 "/usr/include/stdlib.h" 3
extern "C" char *getenv(const char *) throw() __attribute__((nonnull(1)));
# 572 "/usr/include/stdlib.h" 3
extern "C" char *__secure_getenv(const char *) throw() __attribute__((nonnull(1)));
# 579 "/usr/include/stdlib.h" 3
extern "C" int putenv(char *) throw() __attribute__((nonnull(1)));
# 585 "/usr/include/stdlib.h" 3
extern "C" int setenv(const char *, const char *, int) throw() __attribute__((nonnull(2)));
# 589 "/usr/include/stdlib.h" 3
extern "C" int unsetenv(const char *) throw() __attribute__((nonnull(1)));
# 596 "/usr/include/stdlib.h" 3
extern "C" int clearenv() throw();
# 606 "/usr/include/stdlib.h" 3
extern "C" char *mktemp(char *) throw() __attribute__((nonnull(1)));
# 620 "/usr/include/stdlib.h" 3
extern "C" int mkstemp(char *) __attribute__((nonnull(1)));
# 630 "/usr/include/stdlib.h" 3
extern "C" int mkstemp64(char *) __attribute__((nonnull(1)));
# 642 "/usr/include/stdlib.h" 3
extern "C" int mkstemps(char *, int) __attribute__((nonnull(1)));
# 652 "/usr/include/stdlib.h" 3
extern "C" int mkstemps64(char *, int) __attribute__((nonnull(1)));
# 663 "/usr/include/stdlib.h" 3
extern "C" char *mkdtemp(char *) throw() __attribute__((nonnull(1)));
# 674 "/usr/include/stdlib.h" 3
extern "C" int mkostemp(char *, int) __attribute__((nonnull(1)));
# 684 "/usr/include/stdlib.h" 3
extern "C" int mkostemp64(char *, int) __attribute__((nonnull(1)));
# 694 "/usr/include/stdlib.h" 3
extern "C" int mkostemps(char *, int, int) __attribute__((nonnull(1)));
# 706 "/usr/include/stdlib.h" 3
extern "C" int mkostemps64(char *, int, int) __attribute__((nonnull(1)));
# 717 "/usr/include/stdlib.h" 3
extern "C" int system(const char *);
# 724 "/usr/include/stdlib.h" 3
extern "C" char *canonicalize_file_name(const char *) throw() __attribute__((nonnull(1)));
# 734 "/usr/include/stdlib.h" 3
extern "C" char *realpath(const char *__restrict__, char *__restrict__) throw();
# 742 "/usr/include/stdlib.h" 3
extern "C" { typedef int (*__compar_fn_t)(const void *, const void *); }
# 745 "/usr/include/stdlib.h" 3
extern "C" { typedef __compar_fn_t comparison_fn_t; }
# 749 "/usr/include/stdlib.h" 3
extern "C" { typedef int (*__compar_d_fn_t)(const void *, const void *, void *); }
# 755 "/usr/include/stdlib.h" 3
extern "C" void *bsearch(const void *, const void *, size_t, size_t, __compar_fn_t) __attribute__((nonnull(1))) __attribute__((nonnull(2))) __attribute__((nonnull(5)));
# 761 "/usr/include/stdlib.h" 3
extern "C" void qsort(void *, size_t, size_t, __compar_fn_t) __attribute__((nonnull(1))) __attribute__((nonnull(4)));
# 764 "/usr/include/stdlib.h" 3
extern "C" void qsort_r(void *, size_t, size_t, __compar_d_fn_t, void *) __attribute__((nonnull(1))) __attribute__((nonnull(4)));
# 771 "/usr/include/stdlib.h" 3
extern "C" __attribute__((weak)) int abs(int) throw() __attribute__((__const__));
# 772 "/usr/include/stdlib.h" 3
extern "C" __attribute__((weak)) long labs(long) throw() __attribute__((__const__));
# 776 "/usr/include/stdlib.h" 3
extern "C" __attribute__((weak)) long long llabs(long long) throw() __attribute__((__const__));
# 785 "/usr/include/stdlib.h" 3
extern "C" div_t div(int, int) throw() __attribute__((__const__));
# 787 "/usr/include/stdlib.h" 3
extern "C" ldiv_t ldiv(long, long) throw() __attribute__((__const__));
# 793 "/usr/include/stdlib.h" 3
extern "C" lldiv_t lldiv(long long, long long) throw() __attribute__((__const__));
# 808 "/usr/include/stdlib.h" 3
extern "C" char *ecvt(double, int, int *__restrict__, int *__restrict__) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4)));
# 814 "/usr/include/stdlib.h" 3
extern "C" char *fcvt(double, int, int *__restrict__, int *__restrict__) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4)));
# 820 "/usr/include/stdlib.h" 3
extern "C" char *gcvt(double, int, char *) throw() __attribute__((nonnull(3)));
# 826 "/usr/include/stdlib.h" 3
extern "C" char *qecvt(long double, int, int *__restrict__, int *__restrict__) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4)));
# 829 "/usr/include/stdlib.h" 3
extern "C" char *qfcvt(long double, int, int *__restrict__, int *__restrict__) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4)));
# 832 "/usr/include/stdlib.h" 3
extern "C" char *qgcvt(long double, int, char *) throw() __attribute__((nonnull(3)));
# 838 "/usr/include/stdlib.h" 3
extern "C" int ecvt_r(double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4))) __attribute__((nonnull(5)));
# 841 "/usr/include/stdlib.h" 3
extern "C" int fcvt_r(double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4))) __attribute__((nonnull(5)));
# 845 "/usr/include/stdlib.h" 3
extern "C" int qecvt_r(long double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4))) __attribute__((nonnull(5)));
# 849 "/usr/include/stdlib.h" 3
extern "C" int qfcvt_r(long double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw() __attribute__((nonnull(3))) __attribute__((nonnull(4))) __attribute__((nonnull(5)));
# 860 "/usr/include/stdlib.h" 3
extern "C" int mblen(const char *, size_t) throw();
# 863 "/usr/include/stdlib.h" 3
extern "C" int mbtowc(wchar_t *__restrict__, const char *__restrict__, size_t) throw();
# 867 "/usr/include/stdlib.h" 3
extern "C" int wctomb(char *, wchar_t) throw();
# 871 "/usr/include/stdlib.h" 3
extern "C" size_t mbstowcs(wchar_t *__restrict__, const char *__restrict__, size_t) throw();
# 874 "/usr/include/stdlib.h" 3
extern "C" size_t wcstombs(char *__restrict__, const wchar_t *__restrict__, size_t) throw();
# 885 "/usr/include/stdlib.h" 3
extern "C" int rpmatch(const char *) throw() __attribute__((nonnull(1)));
# 896 "/usr/include/stdlib.h" 3
extern "C" int getsubopt(char **__restrict__, char *const *__restrict__, char **__restrict__) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2))) __attribute__((nonnull(3)));
# 905 "/usr/include/stdlib.h" 3
extern "C" void setkey(const char *) throw() __attribute__((nonnull(1)));
# 913 "/usr/include/stdlib.h" 3
extern "C" int posix_openpt(int);
# 921 "/usr/include/stdlib.h" 3
extern "C" int grantpt(int) throw();
# 925 "/usr/include/stdlib.h" 3
extern "C" int unlockpt(int) throw();
# 930 "/usr/include/stdlib.h" 3
extern "C" char *ptsname(int) throw();
# 937 "/usr/include/stdlib.h" 3
extern "C" int ptsname_r(int, char *, size_t) throw() __attribute__((nonnull(2)));
# 941 "/usr/include/stdlib.h" 3
extern "C" int getpt();
# 948 "/usr/include/stdlib.h" 3
extern "C" int getloadavg(double [], int) throw() __attribute__((nonnull(1)));
# 69 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {
# 71 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Iterator, class _Container> class __normal_iterator;
# 74 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
}
# 76 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
namespace std __attribute__((visibility("default"))) {
# 78 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __true_type { };
# 79 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __false_type { };
# 81 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< bool __T0>
# 82 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __truth_type {
# 83 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type; };
# 86 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __truth_type< true> {
# 87 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type; };
# 91 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Sp, class _Tp>
# 92 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __traitor {
# 94 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = (((bool)_Sp::__value) || ((bool)_Tp::__value))};
# 95 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef typename __truth_type< __value> ::__type __type;
# 96 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 99 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class , class >
# 100 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __are_same {
# 102 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 103 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 104 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 106 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 107 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __are_same< _Tp, _Tp> {
# 109 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 110 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 111 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 114 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 115 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_void {
# 117 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 118 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 119 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 122 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_void< void> {
# 124 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 125 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 126 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 131 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 132 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_integer {
# 134 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 135 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 136 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 142 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< bool> {
# 144 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 145 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 146 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 149 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< char> {
# 151 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 152 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 153 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 156 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< signed char> {
# 158 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 159 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 160 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 163 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned char> {
# 165 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 166 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 167 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 171 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< wchar_t> {
# 173 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 174 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 175 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 195 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< short> {
# 197 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 198 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 199 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 202 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned short> {
# 204 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 205 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 206 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 209 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< int> {
# 211 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 212 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 213 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 216 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned> {
# 218 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 219 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 220 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 223 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< long> {
# 225 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 226 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 227 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 230 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned long> {
# 232 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 233 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 234 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 237 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< long long> {
# 239 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 240 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 241 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 244 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_integer< unsigned long long> {
# 246 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 247 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 248 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 253 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 254 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_floating {
# 256 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 257 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 258 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 262 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_floating< float> {
# 264 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 265 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 266 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 269 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_floating< double> {
# 271 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 272 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 273 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 276 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_floating< long double> {
# 278 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 279 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 280 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 285 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 286 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_pointer {
# 288 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 289 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 290 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 292 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 293 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_pointer< _Tp *> {
# 295 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 296 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 297 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 302 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 303 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_normal_iterator {
# 305 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 306 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 307 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 309 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Iterator, class _Container>
# 310 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_normal_iterator< __gnu_cxx::__normal_iterator< _Iterator, _Container> > {
# 313 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 314 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 315 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 320 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 321 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> > {
# 323 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 328 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 329 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_fundamental : public __traitor< __is_void< _Tp> , __is_arithmetic< _Tp> > {
# 331 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 336 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 337 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> > {
# 339 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 344 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 345 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_char {
# 347 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 348 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 349 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 352 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_char< char> {
# 354 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 355 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 356 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 360 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_char< wchar_t> {
# 362 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 363 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 364 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 367 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 368 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_byte {
# 370 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 371 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 372 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 375 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_byte< char> {
# 377 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 378 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 379 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 382 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_byte< signed char> {
# 384 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 385 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 386 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 389 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template<> struct __is_byte< unsigned char> {
# 391 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value = 1};
# 392 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __true_type __type;
# 393 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 398 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
template< class _Tp>
# 399 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
struct __is_move_iterator {
# 401 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum { __value};
# 402 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
typedef __false_type __type;
# 403 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
};
# 417 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
}
# 37 "/usr/include/c++/4.4/ext/type_traits.h" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {
# 40 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< bool __T1, class >
# 41 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __enable_if {
# 42 "/usr/include/c++/4.4/ext/type_traits.h" 3
};
# 44 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp>
# 45 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __enable_if< true, _Tp> {
# 46 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef _Tp __type; };
# 50 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< bool _Cond, class _Iftrue, class _Iffalse>
# 51 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __conditional_type {
# 52 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef _Iftrue __type; };
# 54 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Iftrue, class _Iffalse>
# 55 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __conditional_type< false, _Iftrue, _Iffalse> {
# 56 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef _Iffalse __type; };
# 60 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp>
# 61 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __add_unsigned {
# 64 "/usr/include/c++/4.4/ext/type_traits.h" 3
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp> __if_type;
# 67 "/usr/include/c++/4.4/ext/type_traits.h" 3
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type;
# 68 "/usr/include/c++/4.4/ext/type_traits.h" 3
};
# 71 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< char> {
# 72 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef unsigned char __type; };
# 75 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< signed char> {
# 76 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef unsigned char __type; };
# 79 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< short> {
# 80 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef unsigned short __type; };
# 83 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< int> {
# 84 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef unsigned __type; };
# 87 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< long> {
# 88 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef unsigned long __type; };
# 91 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< long long> {
# 92 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef unsigned long long __type; };
# 96 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< bool> ;
# 99 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __add_unsigned< wchar_t> ;
# 103 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp>
# 104 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __remove_unsigned {
# 107 "/usr/include/c++/4.4/ext/type_traits.h" 3
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp> __if_type;
# 110 "/usr/include/c++/4.4/ext/type_traits.h" 3
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type;
# 111 "/usr/include/c++/4.4/ext/type_traits.h" 3
};
# 114 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< char> {
# 115 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef signed char __type; };
# 118 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned char> {
# 119 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef signed char __type; };
# 122 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned short> {
# 123 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef short __type; };
# 126 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned> {
# 127 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef int __type; };
# 130 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned long> {
# 131 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef long __type; };
# 134 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< unsigned long long> {
# 135 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef long long __type; };
# 139 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< bool> ;
# 142 "/usr/include/c++/4.4/ext/type_traits.h" 3
template<> struct __remove_unsigned< wchar_t> ;
# 146 "/usr/include/c++/4.4/ext/type_traits.h" 3
template < typename _Type >
    inline bool
    __is_null_pointer ( _Type * __ptr )
    { return __ptr == 0; }
# 151 "/usr/include/c++/4.4/ext/type_traits.h" 3
template < typename _Type >
    inline bool
    __is_null_pointer ( _Type )
    { return false; }
# 158 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp, bool __T2 = std::__is_integer< _Tp> ::__value>
# 159 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __promote {
# 160 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef double __type; };
# 162 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp>
# 163 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __promote< _Tp, false> {
# 164 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef _Tp __type; };
# 166 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp, class _Up>
# 167 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __promote_2 {
# 170 "/usr/include/c++/4.4/ext/type_traits.h" 3
private: typedef typename __promote< _Tp, std::__is_integer< _Tp> ::__value> ::__type __type1;
# 171 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef typename __promote< _Up, std::__is_integer< _Up> ::__value> ::__type __type2;
# 174 "/usr/include/c++/4.4/ext/type_traits.h" 3
public: typedef __typeof__(__type1() + __type2()) __type;
# 175 "/usr/include/c++/4.4/ext/type_traits.h" 3
};
# 177 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp, class _Up, class _Vp>
# 178 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __promote_3 {
# 181 "/usr/include/c++/4.4/ext/type_traits.h" 3
private: typedef typename __promote< _Tp, std::__is_integer< _Tp> ::__value> ::__type __type1;
# 182 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef typename __promote< _Up, std::__is_integer< _Up> ::__value> ::__type __type2;
# 183 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef typename __promote< _Vp, std::__is_integer< _Vp> ::__value> ::__type __type3;
# 186 "/usr/include/c++/4.4/ext/type_traits.h" 3
public: typedef __typeof__((__type1() + __type2()) + __type3()) __type;
# 187 "/usr/include/c++/4.4/ext/type_traits.h" 3
};
# 189 "/usr/include/c++/4.4/ext/type_traits.h" 3
template< class _Tp, class _Up, class _Vp, class _Wp>
# 190 "/usr/include/c++/4.4/ext/type_traits.h" 3
struct __promote_4 {
# 193 "/usr/include/c++/4.4/ext/type_traits.h" 3
private: typedef typename __promote< _Tp, std::__is_integer< _Tp> ::__value> ::__type __type1;
# 194 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef typename __promote< _Up, std::__is_integer< _Up> ::__value> ::__type __type2;
# 195 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef typename __promote< _Vp, std::__is_integer< _Vp> ::__value> ::__type __type3;
# 196 "/usr/include/c++/4.4/ext/type_traits.h" 3
typedef typename __promote< _Wp, std::__is_integer< _Wp> ::__value> ::__type __type4;
# 199 "/usr/include/c++/4.4/ext/type_traits.h" 3
public: typedef __typeof__(((__type1() + __type2()) + __type3()) + __type4()) __type;
# 200 "/usr/include/c++/4.4/ext/type_traits.h" 3
};
# 202 "/usr/include/c++/4.4/ext/type_traits.h" 3
}
# 77 "/usr/include/c++/4.4/cmath" 3
namespace std __attribute__((visibility("default"))) {
# 81 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    _Tp __cmath_power ( _Tp, unsigned int );
# 84 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline _Tp
    __pow_helper ( _Tp __x, int __n )
    {
      return __n < 0
        ? _Tp ( 1 ) / __cmath_power ( __x, - __n )
        : __cmath_power ( __x, __n );
    }
# 94 "/usr/include/c++/4.4/cmath" 3
inline double abs(double __x)
# 95 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fabs(__x); }
# 98 "/usr/include/c++/4.4/cmath" 3
inline float abs(float __x)
# 99 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fabsf(__x); }
# 102 "/usr/include/c++/4.4/cmath" 3
inline long double abs(long double __x)
# 103 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fabsl(__x); }
# 105 "/usr/include/c++/4.4/cmath" 3
using ::acos;
# 108 "/usr/include/c++/4.4/cmath" 3
inline float acos(float __x)
# 109 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_acosf(__x); }
# 112 "/usr/include/c++/4.4/cmath" 3
inline long double acos(long double __x)
# 113 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_acosl(__x); }
# 115 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    acos ( _Tp __x )
    { return __builtin_acos ( __x ); }
# 121 "/usr/include/c++/4.4/cmath" 3
using ::asin;
# 124 "/usr/include/c++/4.4/cmath" 3
inline float asin(float __x)
# 125 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_asinf(__x); }
# 128 "/usr/include/c++/4.4/cmath" 3
inline long double asin(long double __x)
# 129 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_asinl(__x); }
# 131 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    asin ( _Tp __x )
    { return __builtin_asin ( __x ); }
# 137 "/usr/include/c++/4.4/cmath" 3
using ::atan;
# 140 "/usr/include/c++/4.4/cmath" 3
inline float atan(float __x)
# 141 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_atanf(__x); }
# 144 "/usr/include/c++/4.4/cmath" 3
inline long double atan(long double __x)
# 145 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_atanl(__x); }
# 147 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    atan ( _Tp __x )
    { return __builtin_atan ( __x ); }
# 153 "/usr/include/c++/4.4/cmath" 3
using ::atan2;
# 156 "/usr/include/c++/4.4/cmath" 3
inline float atan2(float __y, float __x)
# 157 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_atan2f(__y, __x); }
# 160 "/usr/include/c++/4.4/cmath" 3
inline long double atan2(long double __y, long double __x)
# 161 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_atan2l(__y, __x); }
# 163 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp, typename _Up >
    inline
    typename __gnu_cxx :: __promote_2 <
    typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value
        && __is_arithmetic < _Up > :: __value,
        _Tp > :: __type, _Up > :: __type
    atan2 ( _Tp __y, _Up __x )
    {
      typedef typename __gnu_cxx :: __promote_2 < _Tp, _Up > :: __type __type;
      return atan2 ( __type ( __y ), __type ( __x ) );
    }
# 175 "/usr/include/c++/4.4/cmath" 3
using ::ceil;
# 178 "/usr/include/c++/4.4/cmath" 3
inline float ceil(float __x)
# 179 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_ceilf(__x); }
# 182 "/usr/include/c++/4.4/cmath" 3
inline long double ceil(long double __x)
# 183 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_ceill(__x); }
# 185 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    ceil ( _Tp __x )
    { return __builtin_ceil ( __x ); }
# 191 "/usr/include/c++/4.4/cmath" 3
using ::cos;
# 194 "/usr/include/c++/4.4/cmath" 3
inline float cos(float __x)
# 195 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_cosf(__x); }
# 198 "/usr/include/c++/4.4/cmath" 3
inline long double cos(long double __x)
# 199 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_cosl(__x); }
# 201 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    cos ( _Tp __x )
    { return __builtin_cos ( __x ); }
# 207 "/usr/include/c++/4.4/cmath" 3
using ::cosh;
# 210 "/usr/include/c++/4.4/cmath" 3
inline float cosh(float __x)
# 211 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_coshf(__x); }
# 214 "/usr/include/c++/4.4/cmath" 3
inline long double cosh(long double __x)
# 215 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_coshl(__x); }
# 217 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    cosh ( _Tp __x )
    { return __builtin_cosh ( __x ); }
# 223 "/usr/include/c++/4.4/cmath" 3
using ::exp;
# 226 "/usr/include/c++/4.4/cmath" 3
inline float exp(float __x)
# 227 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_expf(__x); }
# 230 "/usr/include/c++/4.4/cmath" 3
inline long double exp(long double __x)
# 231 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_expl(__x); }
# 233 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    exp ( _Tp __x )
    { return __builtin_exp ( __x ); }
# 239 "/usr/include/c++/4.4/cmath" 3
using ::fabs;
# 242 "/usr/include/c++/4.4/cmath" 3
inline float fabs(float __x)
# 243 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fabsf(__x); }
# 246 "/usr/include/c++/4.4/cmath" 3
inline long double fabs(long double __x)
# 247 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fabsl(__x); }
# 249 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    fabs ( _Tp __x )
    { return __builtin_fabs ( __x ); }
# 255 "/usr/include/c++/4.4/cmath" 3
using ::floor;
# 258 "/usr/include/c++/4.4/cmath" 3
inline float floor(float __x)
# 259 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_floorf(__x); }
# 262 "/usr/include/c++/4.4/cmath" 3
inline long double floor(long double __x)
# 263 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_floorl(__x); }
# 265 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    floor ( _Tp __x )
    { return __builtin_floor ( __x ); }
# 271 "/usr/include/c++/4.4/cmath" 3
using ::fmod;
# 274 "/usr/include/c++/4.4/cmath" 3
inline float fmod(float __x, float __y)
# 275 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fmodf(__x, __y); }
# 278 "/usr/include/c++/4.4/cmath" 3
inline long double fmod(long double __x, long double __y)
# 279 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_fmodl(__x, __y); }
# 281 "/usr/include/c++/4.4/cmath" 3
using ::frexp;
# 284 "/usr/include/c++/4.4/cmath" 3
inline float frexp(float __x, int *__exp)
# 285 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_frexpf(__x, __exp); }
# 288 "/usr/include/c++/4.4/cmath" 3
inline long double frexp(long double __x, int *__exp)
# 289 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_frexpl(__x, __exp); }
# 291 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    frexp ( _Tp __x, int * __exp )
    { return __builtin_frexp ( __x, __exp ); }
# 297 "/usr/include/c++/4.4/cmath" 3
using ::ldexp;
# 300 "/usr/include/c++/4.4/cmath" 3
inline float ldexp(float __x, int __exp)
# 301 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_ldexpf(__x, __exp); }
# 304 "/usr/include/c++/4.4/cmath" 3
inline long double ldexp(long double __x, int __exp)
# 305 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_ldexpl(__x, __exp); }
# 307 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
  ldexp ( _Tp __x, int __exp )
  { return __builtin_ldexp ( __x, __exp ); }
# 313 "/usr/include/c++/4.4/cmath" 3
using ::log;
# 316 "/usr/include/c++/4.4/cmath" 3
inline float log(float __x)
# 317 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_logf(__x); }
# 320 "/usr/include/c++/4.4/cmath" 3
inline long double log(long double __x)
# 321 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_logl(__x); }
# 323 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    log ( _Tp __x )
    { return __builtin_log ( __x ); }
# 329 "/usr/include/c++/4.4/cmath" 3
using ::log10;
# 332 "/usr/include/c++/4.4/cmath" 3
inline float log10(float __x)
# 333 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_log10f(__x); }
# 336 "/usr/include/c++/4.4/cmath" 3
inline long double log10(long double __x)
# 337 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_log10l(__x); }
# 339 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    log10 ( _Tp __x )
    { return __builtin_log10 ( __x ); }
# 345 "/usr/include/c++/4.4/cmath" 3
using ::modf;
# 348 "/usr/include/c++/4.4/cmath" 3
inline float modf(float __x, float *__iptr)
# 349 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_modff(__x, __iptr); }
# 352 "/usr/include/c++/4.4/cmath" 3
inline long double modf(long double __x, long double *__iptr)
# 353 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_modfl(__x, __iptr); }
# 355 "/usr/include/c++/4.4/cmath" 3
using ::pow;
# 358 "/usr/include/c++/4.4/cmath" 3
inline float pow(float __x, float __y)
# 359 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_powf(__x, __y); }
# 362 "/usr/include/c++/4.4/cmath" 3
inline long double pow(long double __x, long double __y)
# 363 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_powl(__x, __y); }
# 369 "/usr/include/c++/4.4/cmath" 3
inline double pow(double __x, int __i)
# 370 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_powi(__x, __i); }
# 373 "/usr/include/c++/4.4/cmath" 3
inline float pow(float __x, int __n)
# 374 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_powif(__x, __n); }
# 377 "/usr/include/c++/4.4/cmath" 3
inline long double pow(long double __x, int __n)
# 378 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_powil(__x, __n); }
# 381 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp, typename _Up >
    inline
    typename __gnu_cxx :: __promote_2 <
    typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value
        && __is_arithmetic < _Up > :: __value,
        _Tp > :: __type, _Up > :: __type
    pow ( _Tp __x, _Up __y )
    {
      typedef typename __gnu_cxx :: __promote_2 < _Tp, _Up > :: __type __type;
      return pow ( __type ( __x ), __type ( __y ) );
    }
# 393 "/usr/include/c++/4.4/cmath" 3
using ::sin;
# 396 "/usr/include/c++/4.4/cmath" 3
inline float sin(float __x)
# 397 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_sinf(__x); }
# 400 "/usr/include/c++/4.4/cmath" 3
inline long double sin(long double __x)
# 401 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_sinl(__x); }
# 403 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    sin ( _Tp __x )
    { return __builtin_sin ( __x ); }
# 409 "/usr/include/c++/4.4/cmath" 3
using ::sinh;
# 412 "/usr/include/c++/4.4/cmath" 3
inline float sinh(float __x)
# 413 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_sinhf(__x); }
# 416 "/usr/include/c++/4.4/cmath" 3
inline long double sinh(long double __x)
# 417 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_sinhl(__x); }
# 419 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    sinh ( _Tp __x )
    { return __builtin_sinh ( __x ); }
# 425 "/usr/include/c++/4.4/cmath" 3
using ::sqrt;
# 428 "/usr/include/c++/4.4/cmath" 3
inline float sqrt(float __x)
# 429 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_sqrtf(__x); }
# 432 "/usr/include/c++/4.4/cmath" 3
inline long double sqrt(long double __x)
# 433 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_sqrtl(__x); }
# 435 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    sqrt ( _Tp __x )
    { return __builtin_sqrt ( __x ); }
# 441 "/usr/include/c++/4.4/cmath" 3
using ::tan;
# 444 "/usr/include/c++/4.4/cmath" 3
inline float tan(float __x)
# 445 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_tanf(__x); }
# 448 "/usr/include/c++/4.4/cmath" 3
inline long double tan(long double __x)
# 449 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_tanl(__x); }
# 451 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    tan ( _Tp __x )
    { return __builtin_tan ( __x ); }
# 457 "/usr/include/c++/4.4/cmath" 3
using ::tanh;
# 460 "/usr/include/c++/4.4/cmath" 3
inline float tanh(float __x)
# 461 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_tanhf(__x); }
# 464 "/usr/include/c++/4.4/cmath" 3
inline long double tanh(long double __x)
# 465 "/usr/include/c++/4.4/cmath" 3
{ return __builtin_tanhl(__x); }
# 467 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_integer < _Tp > :: __value,
        double > :: __type
    tanh ( _Tp __x )
    { return __builtin_tanh ( __x ); }
# 473 "/usr/include/c++/4.4/cmath" 3
}
# 492 "/usr/include/c++/4.4/cmath" 3
namespace std __attribute__((visibility("default"))) {
# 494 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    fpclassify ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_fpclassify ( FP_NAN, FP_INFINITE, FP_NORMAL,
      FP_SUBNORMAL, FP_ZERO, __type ( __f ) );
    }
# 504 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isfinite ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isfinite ( __type ( __f ) );
    }
# 513 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isinf ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isinf ( __type ( __f ) );
    }
# 522 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isnan ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isnan ( __type ( __f ) );
    }
# 531 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isnormal ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isnormal ( __type ( __f ) );
    }
# 540 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    signbit ( _Tp __f )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_signbit ( __type ( __f ) );
    }
# 549 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isgreater ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isgreater ( __type ( __f1 ), __type ( __f2 ) );
    }
# 558 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isgreaterequal ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isgreaterequal ( __type ( __f1 ), __type ( __f2 ) );
    }
# 567 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isless ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isless ( __type ( __f1 ), __type ( __f2 ) );
    }
# 576 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    islessequal ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_islessequal ( __type ( __f1 ), __type ( __f2 ) );
    }
# 585 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    islessgreater ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_islessgreater ( __type ( __f1 ), __type ( __f2 ) );
    }
# 594 "/usr/include/c++/4.4/cmath" 3
template < typename _Tp >
    inline typename __gnu_cxx :: __enable_if < __is_arithmetic < _Tp > :: __value,
        int > :: __type
    isunordered ( _Tp __f1, _Tp __f2 )
    {
      typedef typename __gnu_cxx :: __promote < _Tp > :: __type __type;
      return __builtin_isunordered ( __type ( __f1 ), __type ( __f2 ) );
    }
# 603 "/usr/include/c++/4.4/cmath" 3
}
# 35 "/usr/include/c++/4.4/bits/cmath.tcc" 3
namespace std __attribute__((visibility("default"))) {
# 37 "/usr/include/c++/4.4/bits/cmath.tcc" 3
template < typename _Tp >
    inline _Tp
    __cmath_power ( _Tp __x, unsigned int __n )
    {
      _Tp __y = __n % 2 ? __x : _Tp ( 1 );

      while ( __n >>= 1 )
        {
          __x = __x * __x;
          if ( __n % 2 )
            __y = __y * __x;
        }

      return __y;
    }
# 53 "/usr/include/c++/4.4/bits/cmath.tcc" 3
}
# 49 "/usr/include/c++/4.4/cstddef" 3
namespace std __attribute__((visibility("default"))) {
# 51 "/usr/include/c++/4.4/cstddef" 3
using ::ptrdiff_t;
# 52 "/usr/include/c++/4.4/cstddef" 3
using ::size_t;
# 54 "/usr/include/c++/4.4/cstddef" 3
}
# 100 "/usr/include/c++/4.4/cstdlib" 3
namespace std __attribute__((visibility("default"))) {
# 102 "/usr/include/c++/4.4/cstdlib" 3
using ::div_t;
# 103 "/usr/include/c++/4.4/cstdlib" 3
using ::ldiv_t;
# 105 "/usr/include/c++/4.4/cstdlib" 3
using ::abort;
# 106 "/usr/include/c++/4.4/cstdlib" 3
using ::abs;
# 107 "/usr/include/c++/4.4/cstdlib" 3
using ::atexit;
# 108 "/usr/include/c++/4.4/cstdlib" 3
using ::atof;
# 109 "/usr/include/c++/4.4/cstdlib" 3
using ::atoi;
# 110 "/usr/include/c++/4.4/cstdlib" 3
using ::atol;
# 111 "/usr/include/c++/4.4/cstdlib" 3
using ::bsearch;
# 112 "/usr/include/c++/4.4/cstdlib" 3
using ::calloc;
# 113 "/usr/include/c++/4.4/cstdlib" 3
using ::div;
# 114 "/usr/include/c++/4.4/cstdlib" 3
using ::exit;
# 115 "/usr/include/c++/4.4/cstdlib" 3
using ::free;
# 116 "/usr/include/c++/4.4/cstdlib" 3
using ::getenv;
# 117 "/usr/include/c++/4.4/cstdlib" 3
using ::labs;
# 118 "/usr/include/c++/4.4/cstdlib" 3
using ::ldiv;
# 119 "/usr/include/c++/4.4/cstdlib" 3
using ::malloc;
# 121 "/usr/include/c++/4.4/cstdlib" 3
using ::mblen;
# 122 "/usr/include/c++/4.4/cstdlib" 3
using ::mbstowcs;
# 123 "/usr/include/c++/4.4/cstdlib" 3
using ::mbtowc;
# 125 "/usr/include/c++/4.4/cstdlib" 3
using ::qsort;
# 126 "/usr/include/c++/4.4/cstdlib" 3
using ::rand;
# 127 "/usr/include/c++/4.4/cstdlib" 3
using ::realloc;
# 128 "/usr/include/c++/4.4/cstdlib" 3
using ::srand;
# 129 "/usr/include/c++/4.4/cstdlib" 3
using ::strtod;
# 130 "/usr/include/c++/4.4/cstdlib" 3
using ::strtol;
# 131 "/usr/include/c++/4.4/cstdlib" 3
using ::strtoul;
# 132 "/usr/include/c++/4.4/cstdlib" 3
using ::system;
# 134 "/usr/include/c++/4.4/cstdlib" 3
using ::wcstombs;
# 135 "/usr/include/c++/4.4/cstdlib" 3
using ::wctomb;
# 139 "/usr/include/c++/4.4/cstdlib" 3
inline long abs(long __i) { return labs(__i); }
# 142 "/usr/include/c++/4.4/cstdlib" 3
inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); }
# 144 "/usr/include/c++/4.4/cstdlib" 3
}
# 157 "/usr/include/c++/4.4/cstdlib" 3
namespace __gnu_cxx __attribute__((visibility("default"))) {
# 160 "/usr/include/c++/4.4/cstdlib" 3
using ::lldiv_t;
# 166 "/usr/include/c++/4.4/cstdlib" 3
using ::_Exit;
# 170 "/usr/include/c++/4.4/cstdlib" 3
inline long long abs(long long __x) { return (__x >= (0)) ? __x : (-__x); }
# 173 "/usr/include/c++/4.4/cstdlib" 3
using ::llabs;
# 176 "/usr/include/c++/4.4/cstdlib" 3
inline lldiv_t div(long long __n, long long __d)
# 177 "/usr/include/c++/4.4/cstdlib" 3
{ lldiv_t __q; (__q.quot) = (__n / __d); (__q.rem) = (__n % __d); return __q; }
# 179 "/usr/include/c++/4.4/cstdlib" 3
using ::lldiv;
# 190 "/usr/include/c++/4.4/cstdlib" 3
using ::atoll;
# 191 "/usr/include/c++/4.4/cstdlib" 3
using ::strtoll;
# 192 "/usr/include/c++/4.4/cstdlib" 3
using ::strtoull;
# 194 "/usr/include/c++/4.4/cstdlib" 3
using ::strtof;
# 195 "/usr/include/c++/4.4/cstdlib" 3
using ::strtold;
# 197 "/usr/include/c++/4.4/cstdlib" 3
}
# 199 "/usr/include/c++/4.4/cstdlib" 3
namespace std __attribute__((visibility("default"))) {
# 202 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::lldiv_t;
# 204 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::_Exit;
# 205 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::abs;
# 207 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::llabs;
# 208 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::div;
# 209 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::lldiv;
# 211 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::atoll;
# 212 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::strtof;
# 213 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::strtoll;
# 214 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::strtoull;
# 215 "/usr/include/c++/4.4/cstdlib" 3
using __gnu_cxx::strtold;
# 217 "/usr/include/c++/4.4/cstdlib" 3
}
# 497 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
namespace __gnu_cxx {
# 499 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline long long abs(long long) __attribute__((visibility("default")));
# 500 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 502 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
namespace std {
# 504 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
template< class T> extern inline T __pow_helper(T, int);
# 505 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
template< class T> extern inline T __cmath_power(T, unsigned);
# 506 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 508 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::abs;
# 509 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::fabs;
# 510 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::ceil;
# 511 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::floor;
# 512 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::sqrt;
# 513 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::pow;
# 514 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::log;
# 515 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::log10;
# 516 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::fmod;
# 517 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::modf;
# 518 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::exp;
# 519 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::frexp;
# 520 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::ldexp;
# 521 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::asin;
# 522 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::sin;
# 523 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::sinh;
# 524 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::acos;
# 525 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::cos;
# 526 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::cosh;
# 527 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::atan;
# 528 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::atan2;
# 529 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::tan;
# 530 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
using std::tanh;
# 584 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
namespace std {
# 587 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline long abs(long) __attribute__((visibility("default")));
# 588 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float abs(float) __attribute__((visibility("default")));
# 589 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline double abs(double) __attribute__((visibility("default")));
# 590 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float fabs(float) __attribute__((visibility("default")));
# 591 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float ceil(float) __attribute__((visibility("default")));
# 592 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float floor(float) __attribute__((visibility("default")));
# 593 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float sqrt(float) __attribute__((visibility("default")));
# 594 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float pow(float, float) __attribute__((visibility("default")));
# 595 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float pow(float, int) __attribute__((visibility("default")));
# 596 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline double pow(double, int) __attribute__((visibility("default")));
# 597 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float log(float) __attribute__((visibility("default")));
# 598 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float log10(float) __attribute__((visibility("default")));
# 599 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float fmod(float, float) __attribute__((visibility("default")));
# 600 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float modf(float, float *) __attribute__((visibility("default")));
# 601 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float exp(float) __attribute__((visibility("default")));
# 602 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float frexp(float, int *) __attribute__((visibility("default")));
# 603 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float ldexp(float, int) __attribute__((visibility("default")));
# 604 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float asin(float) __attribute__((visibility("default")));
# 605 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float sin(float) __attribute__((visibility("default")));
# 606 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float sinh(float) __attribute__((visibility("default")));
# 607 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float acos(float) __attribute__((visibility("default")));
# 608 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float cos(float) __attribute__((visibility("default")));
# 609 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float cosh(float) __attribute__((visibility("default")));
# 610 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float atan(float) __attribute__((visibility("default")));
# 611 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float atan2(float, float) __attribute__((visibility("default")));
# 612 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float tan(float) __attribute__((visibility("default")));
# 613 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
extern inline float tanh(float) __attribute__((visibility("default")));
# 616 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 619 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float logb(float a)
# 620 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 621 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return logbf(a);
# 622 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 624 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline int ilogb(float a)
# 625 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 626 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return ilogbf(a);
# 627 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 629 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float scalbn(float a, int b)
# 630 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 631 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return scalbnf(a, b);
# 632 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 634 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float scalbln(float a, long b)
# 635 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 636 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return scalblnf(a, b);
# 637 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 639 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float exp2(float a)
# 640 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 641 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return exp2f(a);
# 642 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 644 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float exp10(float a)
# 645 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 646 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return exp10f(a);
# 647 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 649 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float expm1(float a)
# 650 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 651 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return expm1f(a);
# 652 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 654 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float log2(float a)
# 655 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 656 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return log2f(a);
# 657 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 659 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float log1p(float a)
# 660 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 661 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return log1pf(a);
# 662 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 664 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float rsqrt(float a)
# 665 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 666 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return rsqrtf(a);
# 667 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 669 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float acosh(float a)
# 670 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 671 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return acoshf(a);
# 672 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 674 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float asinh(float a)
# 675 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 676 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return asinhf(a);
# 677 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 679 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float atanh(float a)
# 680 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 681 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return atanhf(a);
# 682 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 684 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float hypot(float a, float b)
# 685 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 686 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return hypotf(a, b);
# 687 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 689 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float cbrt(float a)
# 690 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 691 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return cbrtf(a);
# 692 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 694 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float rcbrt(float a)
# 695 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 696 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return rcbrtf(a);
# 697 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 699 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float sinpi(float a)
# 700 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 701 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return sinpif(a);
# 702 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 704 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline void sincos(float a, float *sptr, float *cptr)
# 705 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 706 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
sincosf(a, sptr, cptr);
# 707 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 709 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float erf(float a)
# 710 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 711 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return erff(a);
# 712 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 714 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float erfinv(float a)
# 715 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 716 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return erfinvf(a);
# 717 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 719 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float erfc(float a)
# 720 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 721 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return erfcf(a);
# 722 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 724 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float erfcinv(float a)
# 725 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 726 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return erfcinvf(a);
# 727 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 729 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float lgamma(float a)
# 730 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 731 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return lgammaf(a);
# 732 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 734 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float tgamma(float a)
# 735 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 736 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return tgammaf(a);
# 737 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 739 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float copysign(float a, float b)
# 740 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 741 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return copysignf(a, b);
# 742 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 744 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline double copysign(double a, float b)
# 745 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 746 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return copysign(a, (double)b);
# 747 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 749 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float copysign(float a, double b)
# 750 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 751 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return copysignf(a, (float)b);
# 752 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 754 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float nextafter(float a, float b)
# 755 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 756 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return nextafterf(a, b);
# 757 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 759 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float remainder(float a, float b)
# 760 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 761 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return remainderf(a, b);
# 762 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 764 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float remquo(float a, float b, int *quo)
# 765 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 766 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return remquof(a, b, quo);
# 767 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 769 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float round(float a)
# 770 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 771 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return roundf(a);
# 772 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 774 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline long lround(float a)
# 775 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 776 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return lroundf(a);
# 777 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 779 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline long long llround(float a)
# 780 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 781 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return llroundf(a);
# 782 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 784 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float trunc(float a)
# 785 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 786 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return truncf(a);
# 787 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 789 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float rint(float a)
# 790 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 791 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return rintf(a);
# 792 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 794 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline long lrint(float a)
# 795 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 796 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return lrintf(a);
# 797 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 799 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline long long llrint(float a)
# 800 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 801 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return llrintf(a);
# 802 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 804 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float nearbyint(float a)
# 805 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 806 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return nearbyintf(a);
# 807 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 809 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float fdim(float a, float b)
# 810 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 811 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return fdimf(a, b);
# 812 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 814 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float fma(float a, float b, float c)
# 815 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 816 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return fmaf(a, b, c);
# 817 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 819 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float fmax(float a, float b)
# 820 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 821 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return fmaxf(a, b);
# 822 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 824 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float fmin(float a, float b)
# 825 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 826 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return fminf(a, b);
# 827 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 829 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline unsigned min(unsigned a, unsigned b)
# 830 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 831 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return umin(a, b);
# 832 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 834 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline unsigned min(int a, unsigned b)
# 835 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 836 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return umin((unsigned)a, b);
# 837 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 839 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline unsigned min(unsigned a, int b)
# 840 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 841 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return umin(a, (unsigned)b);
# 842 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 844 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline long long min(long long a, long long b)
# 845 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 846 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return llmin(a, b);
# 847 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 849 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline unsigned long long min(unsigned long long a, unsigned long long b)
# 850 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 851 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return ullmin(a, b);
# 852 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 854 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline unsigned long long min(long long a, unsigned long long b)
# 855 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 856 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return ullmin((unsigned long long)a, b);
# 857 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 859 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline unsigned long long min(unsigned long long a, long long b)
# 860 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 861 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return ullmin(a, (unsigned long long)b);
# 862 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 864 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float min(float a, float b)
# 865 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 866 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return fminf(a, b);
# 867 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 869 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline double min(double a, double b)
# 870 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 871 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return fmin(a, b);
# 872 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 874 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline double min(float a, double b)
# 875 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 876 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return fmin((double)a, b);
# 877 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 879 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline double min(double a, float b)
# 880 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 881 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return fmin(a, (double)b);
# 882 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 884 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline unsigned max(unsigned a, unsigned b)
# 885 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 886 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return umax(a, b);
# 887 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 889 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline unsigned max(int a, unsigned b)
# 890 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 891 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return umax((unsigned)a, b);
# 892 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 894 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline unsigned max(unsigned a, int b)
# 895 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 896 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return umax(a, (unsigned)b);
# 897 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 899 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline long long max(long long a, long long b)
# 900 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 901 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return llmax(a, b);
# 902 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 904 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline unsigned long long max(unsigned long long a, unsigned long long b)
# 905 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 906 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return ullmax(a, b);
# 907 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 909 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline unsigned long long max(long long a, unsigned long long b)
# 910 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 911 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return ullmax((unsigned long long)a, b);
# 912 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 914 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline unsigned long long max(unsigned long long a, long long b)
# 915 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 916 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return ullmax(a, (unsigned long long)b);
# 917 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 919 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline float max(float a, float b)
# 920 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 921 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return fmaxf(a, b);
# 922 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 924 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline double max(double a, double b)
# 925 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 926 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return fmax(a, b);
# 927 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 929 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline double max(float a, double b)
# 930 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 931 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return fmax((double)a, b);
# 932 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 934 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
static inline double max(double a, float b)
# 935 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
{
# 936 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
return fmax(a, (double)b);
# 937 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h"
}
# 60 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
template< class T, int dim = 1>
# 61 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
struct surface : public surfaceReference {
# 63 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
surface()
# 64 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
{
# 65 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
(channelDesc) = cudaCreateChannelDesc< T> ();
# 66 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
}
# 68 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
surface(cudaChannelFormatDesc desc)
# 69 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
{
# 70 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
(channelDesc) = desc;
# 71 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
}
# 72 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
};
# 75 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
template< int dim>
# 76 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
struct surface< void, dim> : public surfaceReference {
# 78 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
surface()
# 79 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
{
# 80 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
(channelDesc) = cudaCreateChannelDesc< void> ();
# 81 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
}
# 82 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_surface_types.h"
};
# 60 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
template< class T, int dim = 1, cudaTextureReadMode mode = cudaReadModeElementType>
# 61 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
struct texture : public textureReference {
# 63 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
texture(int norm = 0, cudaTextureFilterMode
# 64 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
fMode = cudaFilterModePoint, cudaTextureAddressMode
# 65 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
aMode = cudaAddressModeClamp)
# 66 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
{
# 67 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
(normalized) = norm;
# 68 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
(filterMode) = fMode;
# 69 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
((addressMode)[0]) = aMode;
# 70 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
((addressMode)[1]) = aMode;
# 71 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
((addressMode)[2]) = aMode;
# 72 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
(channelDesc) = cudaCreateChannelDesc< T> ();
# 73 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
}
# 75 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
texture(int norm, cudaTextureFilterMode
# 76 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
fMode, cudaTextureAddressMode
# 77 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
aMode, cudaChannelFormatDesc
# 78 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
desc)
# 79 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
{
# 80 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
(normalized) = norm;
# 81 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
(filterMode) = fMode;
# 82 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
((addressMode)[0]) = aMode;
# 83 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
((addressMode)[1]) = aMode;
# 84 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
((addressMode)[2]) = aMode;
# 85 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
(channelDesc) = desc;
# 86 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
}
# 87 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_texture_types.h"
};
# 324 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline int mulhi(int a, int b)
# 325 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 327 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 329 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned mulhi(unsigned a, unsigned b)
# 330 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 332 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 334 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned mulhi(int a, unsigned b)
# 335 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 337 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 339 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned mulhi(unsigned a, int b)
# 340 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 342 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 344 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline long long mul64hi(long long a, long long b)
# 345 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 347 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 349 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, unsigned long long b)
# 350 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 352 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 354 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned long long mul64hi(long long a, unsigned long long b)
# 355 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 357 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 359 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, long long b)
# 360 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 362 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 364 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline int float_as_int(float a)
# 365 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 367 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 369 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline float int_as_float(int a)
# 370 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 372 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 374 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline float saturate(float a)
# 375 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 377 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 379 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline int mul24(int a, int b)
# 380 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 382 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 384 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned umul24(unsigned a, unsigned b)
# 385 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 387 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 389 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline void trap()
# 390 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 392 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 394 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline void brkpt(int c)
# 395 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 397 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 399 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline void syncthreads()
# 400 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 402 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 404 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline void prof_trigger(int e)
# 405 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 422 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 424 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline void threadfence(bool global = true)
# 425 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 427 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 429 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline int float2int(float a, cudaRoundMode mode = cudaRoundZero)
# 430 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 435 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 437 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline unsigned float2uint(float a, cudaRoundMode mode = cudaRoundZero)
# 438 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 443 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 445 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline float int2float(int a, cudaRoundMode mode = cudaRoundNearest)
# 446 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 451 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 453 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
__attribute__((unused)) static inline float uint2float(unsigned a, cudaRoundMode mode = cudaRoundNearest)
# 454 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
{int volatile ___ = 1;
# 459 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_functions.h"
exit(___);}
# 102 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd(int *address, int val)
# 103 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 105 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 107 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd(unsigned *address, unsigned val)
# 108 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 110 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 112 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub(int *address, int val)
# 113 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 115 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 117 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub(unsigned *address, unsigned val)
# 118 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 120 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 122 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch(int *address, int val)
# 123 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 125 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 127 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch(unsigned *address, unsigned val)
# 128 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 130 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 132 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch(float *address, float val)
# 133 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 135 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 137 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin(int *address, int val)
# 138 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 140 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 142 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin(unsigned *address, unsigned val)
# 143 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 145 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 147 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax(int *address, int val)
# 148 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 150 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 152 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax(unsigned *address, unsigned val)
# 153 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 155 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 157 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc(unsigned *address, unsigned val)
# 158 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 160 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 162 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec(unsigned *address, unsigned val)
# 163 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 165 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 167 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd(int *address, int val)
# 168 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 170 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 172 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd(unsigned *address, unsigned val)
# 173 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 175 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 177 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr(int *address, int val)
# 178 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 180 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 182 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr(unsigned *address, unsigned val)
# 183 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 185 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 187 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor(int *address, int val)
# 188 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 190 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 192 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor(unsigned *address, unsigned val)
# 193 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 195 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 197 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS(int *address, int compare, int val)
# 198 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 200 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 202 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val)
# 203 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
{int volatile ___ = 1;
# 205 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_11_atomic_functions.h"
exit(___);}
# 75 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd(unsigned long long *address, unsigned long long val)
# 76 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
{int volatile ___ = 1;
# 78 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
exit(___);}
# 80 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch(unsigned long long *address, unsigned long long val)
# 81 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
{int volatile ___ = 1;
# 83 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
exit(___);}
# 85 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val)
# 86 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
{int volatile ___ = 1;
# 88 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
exit(___);}
# 90 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
__attribute__((unused)) static inline bool any(bool cond)
# 91 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
{int volatile ___ = 1;
# 93 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
exit(___);}
# 95 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
__attribute__((unused)) static inline bool all(bool cond)
# 96 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
{int volatile ___ = 1;
# 98 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_12_atomic_functions.h"
exit(___);}
# 170 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode)
# 171 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 176 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
exit(___);}
# 178 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest)
# 179 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 184 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
exit(___);}
# 186 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest)
# 187 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 192 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
exit(___);}
# 194 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode = cudaRoundZero)
# 195 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 200 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
exit(___);}
# 202 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero)
# 203 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 208 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
exit(___);}
# 210 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode = cudaRoundZero)
# 211 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 216 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
exit(___);}
# 218 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode = cudaRoundZero)
# 219 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 224 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
exit(___);}
# 226 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode = cudaRoundNearest)
# 227 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 232 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
exit(___);}
# 234 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode = cudaRoundNearest)
# 235 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 240 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
exit(___);}
# 242 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest)
# 243 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 245 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
exit(___);}
# 247 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest)
# 248 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 250 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
exit(___);}
# 252 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest)
# 253 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
{int volatile ___ = 1;
# 255 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_13_double_functions.h"
exit(___);}
# 66 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd(float *address, float val)
# 67 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_atomic_functions.h"
{int volatile ___ = 1;
# 69 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_atomic_functions.h"
exit(___);}
# 124 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned ballot(bool pred)
# 125 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_intrinsics.h"
{int volatile ___ = 1;
# 127 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_intrinsics.h"
exit(___);}
# 129 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_intrinsics.h"
__attribute__((unused)) static inline int syncthreads_count(bool pred)
# 130 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_intrinsics.h"
{int volatile ___ = 1;
# 132 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_intrinsics.h"
exit(___);}
# 134 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_and(bool pred)
# 135 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_intrinsics.h"
{int volatile ___ = 1;
# 137 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_intrinsics.h"
exit(___);}
# 139 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_or(bool pred)
# 140 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_intrinsics.h"
{int volatile ___ = 1;
# 142 "/home/bachelor/deicide218/cuda-3.2/bin/../include/sm_20_intrinsics.h"
exit(___);}
# 97 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 98 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
surf1Dread(T *res, surface< void, 1> surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 99 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 106 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 108 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline T
# 109 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 110 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 116 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 118 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 119 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
surf1Dread(T *res, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 120 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 122 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 125 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 126 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 128 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 131 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline signed char surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 132 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 134 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 137 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned char surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 138 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 140 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 143 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 144 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 146 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 149 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uchar1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 150 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 152 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 155 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 156 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 160 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 163 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uchar2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 164 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 166 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 169 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 170 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 174 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 177 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uchar4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 178 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 180 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 183 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 184 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 186 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 189 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned short surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 190 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 192 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 195 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 196 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 198 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 201 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ushort1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 202 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 204 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 207 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 208 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 212 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 215 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ushort2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 216 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 218 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 221 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 222 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 226 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 229 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ushort4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 230 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 232 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 235 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 236 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 238 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 241 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 242 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 244 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 247 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 248 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 250 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 253 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uint1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 254 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 256 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 259 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 260 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 264 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 267 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uint2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 268 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 270 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 273 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 274 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 278 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 281 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uint4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 282 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 284 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 287 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline long long surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 288 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 290 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 293 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned long long surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 294 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 296 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 299 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline longlong1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 300 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 302 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 305 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ulonglong1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 306 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 308 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 311 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline longlong2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 312 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 316 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 319 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ulonglong2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 320 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 322 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 385 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 386 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 388 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 391 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float1 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 392 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 394 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 397 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float2 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 398 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 402 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 405 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float4 surf1Dread(surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode)
# 406 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 410 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 457 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 458 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
surf2Dread(T *res, surface< void, 2> surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 459 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 466 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 468 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline T
# 469 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 470 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 476 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 478 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 479 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
surf2Dread(T *res, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 480 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 482 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 485 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 486 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 488 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 491 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline signed char surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 492 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 494 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 497 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned char surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 498 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 500 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 503 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 504 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 506 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 509 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uchar1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 510 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 512 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 515 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 516 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 520 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 523 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uchar2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 524 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 526 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 529 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline char4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 530 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 534 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 537 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uchar4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 538 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 540 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 543 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 544 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 546 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 549 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned short surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 550 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 552 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 555 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 556 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 558 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 561 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ushort1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 562 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 564 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 567 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 568 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 572 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 575 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ushort2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 576 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 578 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 581 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline short4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 582 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 586 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 589 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ushort4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 590 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 592 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 595 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 596 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 598 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 601 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 602 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 604 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 607 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 608 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 610 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 613 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uint1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 614 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 616 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 619 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 620 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 624 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 627 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uint2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 628 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 630 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 633 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline int4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 634 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 638 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 641 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline uint4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 642 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 644 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 647 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline long long surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 648 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 650 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 653 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline unsigned long long surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 654 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 656 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 659 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline longlong1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 660 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 662 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 665 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ulonglong1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 666 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 668 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 671 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline longlong2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 672 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 676 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 679 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline ulonglong2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 680 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 682 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 745 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 746 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 748 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 751 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float1 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 752 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 754 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 757 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float2 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 758 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 762 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 765 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template<> __attribute__((unused)) inline float4 surf2Dread(surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode)
# 766 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 770 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 817 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 818 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
surf1Dwrite(T val, surface< void, 1> surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 819 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 837 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 839 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 840 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
surf1Dwrite(T val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 841 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 843 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 846 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(char val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 847 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 849 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 851 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(signed char val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 852 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 854 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 856 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(unsigned char val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 857 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 859 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 861 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(char1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 862 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 864 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 866 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(uchar1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 867 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 869 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 871 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(char2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 872 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 874 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 876 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(uchar2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 877 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 879 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 881 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(char4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 882 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 884 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 886 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(uchar4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 887 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 889 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 891 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(short val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 892 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 894 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 896 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(unsigned short val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 897 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 899 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 901 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(short1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 902 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 904 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 906 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(ushort1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 907 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 909 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 911 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(short2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 912 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 914 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 916 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(ushort2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 917 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 919 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 921 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(short4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 922 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 924 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 926 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(ushort4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 927 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 929 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 931 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(int val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 932 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 934 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 936 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(unsigned val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 937 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 939 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 941 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(int1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 942 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 944 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 946 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(uint1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 947 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 949 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 951 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(int2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 952 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 954 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 956 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(uint2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 957 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 959 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 961 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(int4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 962 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 964 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 966 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(uint4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 967 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 969 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 971 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(long long val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 972 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 974 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 976 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(unsigned long long val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 977 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 979 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 981 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(longlong1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 982 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 984 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 986 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(ulonglong1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 987 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 989 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 991 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(longlong2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 992 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 994 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 996 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(ulonglong2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 997 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 999 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1045 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(float val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1046 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1048 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1050 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(float1 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1051 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1053 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1055 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(float2 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1056 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1058 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1060 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf1Dwrite(float4 val, surface< void, 1> surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1061 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1063 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1110 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 1111 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
surf2Dwrite(T val, surface< void, 2> surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1112 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1130 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1132 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
template< class T> __attribute__((unused)) static inline void
# 1133 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
surf2Dwrite(T val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1134 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1136 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1139 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(char val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1140 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1142 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1144 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(signed char val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1145 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1147 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1149 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(unsigned char val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1150 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1152 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1154 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(char1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1155 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1157 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1159 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(uchar1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1160 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1162 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1164 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(char2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1165 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1167 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1169 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(uchar2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1170 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1172 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1174 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(char4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1175 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1177 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1179 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(uchar4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1180 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1182 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1184 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(short val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1185 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1187 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1189 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(unsigned short val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1190 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1192 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1194 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(short1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1195 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1197 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1199 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(ushort1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1200 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1202 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1204 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(short2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1205 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1207 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1209 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(ushort2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1210 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1212 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1214 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(short4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1215 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1217 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1219 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(ushort4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1220 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1222 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1224 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(int val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1225 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1227 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1229 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(unsigned val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1230 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1232 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1234 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(int1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1235 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1237 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1239 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(uint1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1240 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1242 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1244 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(int2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1245 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1247 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1249 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(uint2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1250 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1252 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1254 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(int4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1255 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1257 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1259 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(uint4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1260 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1262 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1264 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(long long val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1265 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1267 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1269 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(unsigned long long val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1270 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1272 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1274 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(longlong1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1275 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1277 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1279 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(ulonglong1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1280 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1282 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1284 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(longlong2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1285 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1287 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1289 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(ulonglong2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1290 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1292 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1338 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(float val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1339 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1341 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1343 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(float1 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1344 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1346 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1348 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(float2 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1349 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1351 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 1353 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
__attribute__((unused)) static inline void surf2Dwrite(float4 val, surface< void, 2> surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
# 1354 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
{int volatile ___ = 1;
# 1356 "/home/bachelor/deicide218/cuda-3.2/bin/../include/surface_functions.h"
exit(___);}
# 61 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
template< class T, cudaTextureReadMode readMode> __attribute__((unused)) extern uint4 __utexfetchi(texture< T, 1, readMode> , int4);
# 63 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
template< class T, cudaTextureReadMode readMode> __attribute__((unused)) extern int4 __itexfetchi(texture< T, 1, readMode> , int4);
# 65 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
template< class T, cudaTextureReadMode readMode> __attribute__((unused)) extern float4 __ftexfetchi(texture< T, 1, readMode> , int4);
# 68 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
template< class T, int dim, cudaTextureReadMode readMode> __attribute__((unused)) extern uint4 __utexfetch(texture< T, dim, readMode> , float4, int = dim);
# 70 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
template< class T, int dim, cudaTextureReadMode readMode> __attribute__((unused)) extern int4 __itexfetch(texture< T, dim, readMode> , float4, int = dim);
# 72 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
template< class T, int dim, cudaTextureReadMode readMode> __attribute__((unused)) extern float4 __ftexfetch(texture< T, dim, readMode> , float4, int = dim);
# 80 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char tex1Dfetch(texture< char, 1, cudaReadModeElementType> t, int x)
# 81 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 89 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 91 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline signed char tex1Dfetch(texture< signed char, 1, cudaReadModeElementType> t, int x)
# 92 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 96 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 98 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned char tex1Dfetch(texture< unsigned char, 1, cudaReadModeElementType> t, int x)
# 99 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 103 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 105 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char1 tex1Dfetch(texture< char1, 1, cudaReadModeElementType> t, int x)
# 106 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 110 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 112 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar1 tex1Dfetch(texture< uchar1, 1, cudaReadModeElementType> t, int x)
# 113 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 117 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 119 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char2 tex1Dfetch(texture< char2, 1, cudaReadModeElementType> t, int x)
# 120 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 124 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 126 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar2 tex1Dfetch(texture< uchar2, 1, cudaReadModeElementType> t, int x)
# 127 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 131 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 133 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex1Dfetch(texture< char4, 1, cudaReadModeElementType> t, int x)
# 134 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 138 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 140 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex1Dfetch(texture< uchar4, 1, cudaReadModeElementType> t, int x)
# 141 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 145 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 153 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short tex1Dfetch(texture< short, 1, cudaReadModeElementType> t, int x)
# 154 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 158 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 160 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned short tex1Dfetch(texture< unsigned short, 1, cudaReadModeElementType> t, int x)
# 161 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 165 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 167 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short1 tex1Dfetch(texture< short1, 1, cudaReadModeElementType> t, int x)
# 168 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 172 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 174 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort1 tex1Dfetch(texture< ushort1, 1, cudaReadModeElementType> t, int x)
# 175 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 179 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 181 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short2 tex1Dfetch(texture< short2, 1, cudaReadModeElementType> t, int x)
# 182 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 186 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 188 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort2 tex1Dfetch(texture< ushort2, 1, cudaReadModeElementType> t, int x)
# 189 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 193 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 195 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex1Dfetch(texture< short4, 1, cudaReadModeElementType> t, int x)
# 196 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 200 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 202 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex1Dfetch(texture< ushort4, 1, cudaReadModeElementType> t, int x)
# 203 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 207 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 215 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int tex1Dfetch(texture< int, 1, cudaReadModeElementType> t, int x)
# 216 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 220 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 222 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned tex1Dfetch(texture< unsigned, 1, cudaReadModeElementType> t, int x)
# 223 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 227 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 229 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int1 tex1Dfetch(texture< int1, 1, cudaReadModeElementType> t, int x)
# 230 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 234 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 236 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint1 tex1Dfetch(texture< uint1, 1, cudaReadModeElementType> t, int x)
# 237 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 241 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 243 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int2 tex1Dfetch(texture< int2, 1, cudaReadModeElementType> t, int x)
# 244 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 248 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 250 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint2 tex1Dfetch(texture< uint2, 1, cudaReadModeElementType> t, int x)
# 251 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 255 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 257 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex1Dfetch(texture< int4, 1, cudaReadModeElementType> t, int x)
# 258 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 262 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 264 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex1Dfetch(texture< uint4, 1, cudaReadModeElementType> t, int x)
# 265 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 269 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 343 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1Dfetch(texture< float, 1, cudaReadModeElementType> t, int x)
# 344 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 348 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 350 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1Dfetch(texture< float1, 1, cudaReadModeElementType> t, int x)
# 351 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 355 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 357 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1Dfetch(texture< float2, 1, cudaReadModeElementType> t, int x)
# 358 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 362 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 364 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1Dfetch(texture< float4, 1, cudaReadModeElementType> t, int x)
# 365 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 369 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 377 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1Dfetch(texture< char, 1, cudaReadModeNormalizedFloat> t, int x)
# 378 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 387 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 389 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1Dfetch(texture< signed char, 1, cudaReadModeNormalizedFloat> t, int x)
# 390 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 395 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 397 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1Dfetch(texture< unsigned char, 1, cudaReadModeNormalizedFloat> t, int x)
# 398 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 403 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 405 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1Dfetch(texture< char1, 1, cudaReadModeNormalizedFloat> t, int x)
# 406 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 411 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 413 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1Dfetch(texture< uchar1, 1, cudaReadModeNormalizedFloat> t, int x)
# 414 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 419 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 421 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1Dfetch(texture< char2, 1, cudaReadModeNormalizedFloat> t, int x)
# 422 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 427 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 429 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1Dfetch(texture< uchar2, 1, cudaReadModeNormalizedFloat> t, int x)
# 430 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 435 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 437 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1Dfetch(texture< char4, 1, cudaReadModeNormalizedFloat> t, int x)
# 438 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 443 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 445 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1Dfetch(texture< uchar4, 1, cudaReadModeNormalizedFloat> t, int x)
# 446 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 451 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 459 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1Dfetch(texture< short, 1, cudaReadModeNormalizedFloat> t, int x)
# 460 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 465 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 467 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1Dfetch(texture< unsigned short, 1, cudaReadModeNormalizedFloat> t, int x)
# 468 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 473 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 475 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1Dfetch(texture< short1, 1, cudaReadModeNormalizedFloat> t, int x)
# 476 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 481 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 483 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1Dfetch(texture< ushort1, 1, cudaReadModeNormalizedFloat> t, int x)
# 484 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 489 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 491 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1Dfetch(texture< short2, 1, cudaReadModeNormalizedFloat> t, int x)
# 492 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 497 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 499 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1Dfetch(texture< ushort2, 1, cudaReadModeNormalizedFloat> t, int x)
# 500 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 505 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 507 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1Dfetch(texture< short4, 1, cudaReadModeNormalizedFloat> t, int x)
# 508 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 513 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 515 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1Dfetch(texture< ushort4, 1, cudaReadModeNormalizedFloat> t, int x)
# 516 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 521 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 529 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char tex1D(texture< char, 1, cudaReadModeElementType> t, float x)
# 530 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 538 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 540 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline signed char tex1D(texture< signed char, 1, cudaReadModeElementType> t, float x)
# 541 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 545 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 547 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned char tex1D(texture< unsigned char, 1, cudaReadModeElementType> t, float x)
# 548 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 552 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 554 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char1 tex1D(texture< char1, 1, cudaReadModeElementType> t, float x)
# 555 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 559 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 561 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar1 tex1D(texture< uchar1, 1, cudaReadModeElementType> t, float x)
# 562 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 566 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 568 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char2 tex1D(texture< char2, 1, cudaReadModeElementType> t, float x)
# 569 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 573 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 575 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar2 tex1D(texture< uchar2, 1, cudaReadModeElementType> t, float x)
# 576 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 580 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 582 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex1D(texture< char4, 1, cudaReadModeElementType> t, float x)
# 583 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 587 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 589 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex1D(texture< uchar4, 1, cudaReadModeElementType> t, float x)
# 590 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 594 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 602 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short tex1D(texture< short, 1, cudaReadModeElementType> t, float x)
# 603 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 607 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 609 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned short tex1D(texture< unsigned short, 1, cudaReadModeElementType> t, float x)
# 610 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 614 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 616 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short1 tex1D(texture< short1, 1, cudaReadModeElementType> t, float x)
# 617 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 621 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 623 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort1 tex1D(texture< ushort1, 1, cudaReadModeElementType> t, float x)
# 624 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 628 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 630 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short2 tex1D(texture< short2, 1, cudaReadModeElementType> t, float x)
# 631 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 635 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 637 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort2 tex1D(texture< ushort2, 1, cudaReadModeElementType> t, float x)
# 638 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 642 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 644 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex1D(texture< short4, 1, cudaReadModeElementType> t, float x)
# 645 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 649 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 651 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex1D(texture< ushort4, 1, cudaReadModeElementType> t, float x)
# 652 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 656 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 664 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int tex1D(texture< int, 1, cudaReadModeElementType> t, float x)
# 665 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 669 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 671 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned tex1D(texture< unsigned, 1, cudaReadModeElementType> t, float x)
# 672 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 676 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 678 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int1 tex1D(texture< int1, 1, cudaReadModeElementType> t, float x)
# 679 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 683 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 685 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint1 tex1D(texture< uint1, 1, cudaReadModeElementType> t, float x)
# 686 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 690 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 692 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int2 tex1D(texture< int2, 1, cudaReadModeElementType> t, float x)
# 693 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 697 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 699 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint2 tex1D(texture< uint2, 1, cudaReadModeElementType> t, float x)
# 700 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 704 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 706 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex1D(texture< int4, 1, cudaReadModeElementType> t, float x)
# 707 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 711 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 713 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex1D(texture< uint4, 1, cudaReadModeElementType> t, float x)
# 714 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 718 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 798 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1D(texture< float, 1, cudaReadModeElementType> t, float x)
# 799 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 803 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 805 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1D(texture< float1, 1, cudaReadModeElementType> t, float x)
# 806 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 810 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 812 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1D(texture< float2, 1, cudaReadModeElementType> t, float x)
# 813 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 817 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 819 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1D(texture< float4, 1, cudaReadModeElementType> t, float x)
# 820 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 824 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 832 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1D(texture< char, 1, cudaReadModeNormalizedFloat> t, float x)
# 833 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 842 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 844 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1D(texture< signed char, 1, cudaReadModeNormalizedFloat> t, float x)
# 845 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 850 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 852 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1D(texture< unsigned char, 1, cudaReadModeNormalizedFloat> t, float x)
# 853 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 858 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 860 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1D(texture< char1, 1, cudaReadModeNormalizedFloat> t, float x)
# 861 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 866 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 868 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1D(texture< uchar1, 1, cudaReadModeNormalizedFloat> t, float x)
# 869 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 874 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 876 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1D(texture< char2, 1, cudaReadModeNormalizedFloat> t, float x)
# 877 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 882 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 884 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1D(texture< uchar2, 1, cudaReadModeNormalizedFloat> t, float x)
# 885 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 890 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 892 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1D(texture< char4, 1, cudaReadModeNormalizedFloat> t, float x)
# 893 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 898 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 900 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1D(texture< uchar4, 1, cudaReadModeNormalizedFloat> t, float x)
# 901 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 906 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 914 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1D(texture< short, 1, cudaReadModeNormalizedFloat> t, float x)
# 915 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 920 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 922 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex1D(texture< unsigned short, 1, cudaReadModeNormalizedFloat> t, float x)
# 923 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 928 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 930 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1D(texture< short1, 1, cudaReadModeNormalizedFloat> t, float x)
# 931 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 936 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 938 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex1D(texture< ushort1, 1, cudaReadModeNormalizedFloat> t, float x)
# 939 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 944 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 946 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1D(texture< short2, 1, cudaReadModeNormalizedFloat> t, float x)
# 947 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 952 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 954 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex1D(texture< ushort2, 1, cudaReadModeNormalizedFloat> t, float x)
# 955 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 960 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 962 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1D(texture< short4, 1, cudaReadModeNormalizedFloat> t, float x)
# 963 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 968 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 970 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex1D(texture< ushort4, 1, cudaReadModeNormalizedFloat> t, float x)
# 971 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 976 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 984 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char tex2D(texture< char, 2, cudaReadModeElementType> t, float x, float y)
# 985 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 993 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 995 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline signed char tex2D(texture< signed char, 2, cudaReadModeElementType> t, float x, float y)
# 996 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1000 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1002 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned char tex2D(texture< unsigned char, 2, cudaReadModeElementType> t, float x, float y)
# 1003 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1007 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1009 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char1 tex2D(texture< char1, 2, cudaReadModeElementType> t, float x, float y)
# 1010 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1014 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1016 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar1 tex2D(texture< uchar1, 2, cudaReadModeElementType> t, float x, float y)
# 1017 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1021 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1023 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char2 tex2D(texture< char2, 2, cudaReadModeElementType> t, float x, float y)
# 1024 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1028 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1030 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar2 tex2D(texture< uchar2, 2, cudaReadModeElementType> t, float x, float y)
# 1031 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1035 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1037 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2D(texture< char4, 2, cudaReadModeElementType> t, float x, float y)
# 1038 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1042 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1044 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex2D(texture< uchar4, 2, cudaReadModeElementType> t, float x, float y)
# 1045 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1049 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1057 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short tex2D(texture< short, 2, cudaReadModeElementType> t, float x, float y)
# 1058 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1062 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1064 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned short tex2D(texture< unsigned short, 2, cudaReadModeElementType> t, float x, float y)
# 1065 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1069 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1071 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short1 tex2D(texture< short1, 2, cudaReadModeElementType> t, float x, float y)
# 1072 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1076 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1078 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort1 tex2D(texture< ushort1, 2, cudaReadModeElementType> t, float x, float y)
# 1079 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1083 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1085 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short2 tex2D(texture< short2, 2, cudaReadModeElementType> t, float x, float y)
# 1086 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1090 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1092 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort2 tex2D(texture< ushort2, 2, cudaReadModeElementType> t, float x, float y)
# 1093 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1097 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1099 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex2D(texture< short4, 2, cudaReadModeElementType> t, float x, float y)
# 1100 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1104 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1106 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex2D(texture< ushort4, 2, cudaReadModeElementType> t, float x, float y)
# 1107 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1111 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1119 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int tex2D(texture< int, 2, cudaReadModeElementType> t, float x, float y)
# 1120 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1124 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1126 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned tex2D(texture< unsigned, 2, cudaReadModeElementType> t, float x, float y)
# 1127 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1131 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1133 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int1 tex2D(texture< int1, 2, cudaReadModeElementType> t, float x, float y)
# 1134 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1138 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1140 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint1 tex2D(texture< uint1, 2, cudaReadModeElementType> t, float x, float y)
# 1141 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1145 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1147 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int2 tex2D(texture< int2, 2, cudaReadModeElementType> t, float x, float y)
# 1148 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1152 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1154 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint2 tex2D(texture< uint2, 2, cudaReadModeElementType> t, float x, float y)
# 1155 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1159 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1161 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex2D(texture< int4, 2, cudaReadModeElementType> t, float x, float y)
# 1162 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1166 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1168 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex2D(texture< uint4, 2, cudaReadModeElementType> t, float x, float y)
# 1169 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1173 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1247 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex2D(texture< float, 2, cudaReadModeElementType> t, float x, float y)
# 1248 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1252 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1254 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex2D(texture< float1, 2, cudaReadModeElementType> t, float x, float y)
# 1255 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1259 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1261 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex2D(texture< float2, 2, cudaReadModeElementType> t, float x, float y)
# 1262 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1266 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1268 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2D(texture< float4, 2, cudaReadModeElementType> t, float x, float y)
# 1269 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1273 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1281 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex2D(texture< char, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1282 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1291 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1293 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex2D(texture< signed char, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1294 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1299 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1301 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex2D(texture< unsigned char, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1302 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1307 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1309 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex2D(texture< char1, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1310 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1315 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1317 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex2D(texture< uchar1, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1318 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1323 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1325 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex2D(texture< char2, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1326 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1331 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1333 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex2D(texture< uchar2, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1334 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1339 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1341 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2D(texture< char4, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1342 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1347 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1349 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2D(texture< uchar4, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1350 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1355 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1363 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex2D(texture< short, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1364 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1369 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1371 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex2D(texture< unsigned short, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1372 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1377 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1379 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex2D(texture< short1, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1380 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1385 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1387 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex2D(texture< ushort1, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1388 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1393 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1395 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex2D(texture< short2, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1396 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1401 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1403 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex2D(texture< ushort2, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1404 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1409 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1411 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2D(texture< short4, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1412 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1417 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1419 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2D(texture< ushort4, 2, cudaReadModeNormalizedFloat> t, float x, float y)
# 1420 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1425 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1433 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char tex3D(texture< char, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1434 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1442 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1444 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline signed char tex3D(texture< signed char, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1445 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1449 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1451 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned char tex3D(texture< unsigned char, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1452 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1456 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1458 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char1 tex3D(texture< char1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1459 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1463 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1465 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar1 tex3D(texture< uchar1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1466 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1470 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1472 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char2 tex3D(texture< char2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1473 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1477 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1479 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar2 tex3D(texture< uchar2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1480 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1484 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1486 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex3D(texture< char4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1487 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1491 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1493 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex3D(texture< uchar4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1494 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1498 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1506 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short tex3D(texture< short, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1507 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1511 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1513 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned short tex3D(texture< unsigned short, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1514 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1518 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1520 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short1 tex3D(texture< short1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1521 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1525 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1527 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort1 tex3D(texture< ushort1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1528 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1532 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1534 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short2 tex3D(texture< short2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1535 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1539 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1541 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort2 tex3D(texture< ushort2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1542 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1546 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1548 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex3D(texture< short4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1549 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1553 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1555 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex3D(texture< ushort4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1556 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1560 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1568 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int tex3D(texture< int, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1569 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1573 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1575 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline unsigned tex3D(texture< unsigned, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1576 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1580 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1582 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int1 tex3D(texture< int1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1583 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1587 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1589 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint1 tex3D(texture< uint1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1590 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1594 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1596 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int2 tex3D(texture< int2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1597 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1601 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1603 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint2 tex3D(texture< uint2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1604 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1608 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1610 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex3D(texture< int4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1611 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1615 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1617 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex3D(texture< uint4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1618 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1622 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1696 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex3D(texture< float, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1697 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1701 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1703 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex3D(texture< float1, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1704 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1708 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1710 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex3D(texture< float2, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1711 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1715 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1717 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex3D(texture< float4, 3, cudaReadModeElementType> t, float x, float y, float z)
# 1718 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1722 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1730 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex3D(texture< char, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1731 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1740 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1742 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex3D(texture< signed char, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1743 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1748 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1750 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex3D(texture< unsigned char, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1751 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1756 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1758 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex3D(texture< char1, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1759 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1764 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1766 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex3D(texture< uchar1, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1767 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1772 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1774 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex3D(texture< char2, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1775 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1780 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1782 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex3D(texture< uchar2, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1783 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1788 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1790 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex3D(texture< char4, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1791 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1796 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1798 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex3D(texture< uchar4, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1799 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1804 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1812 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex3D(texture< short, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1813 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1818 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1820 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float tex3D(texture< unsigned short, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1821 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1826 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1828 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex3D(texture< short1, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1829 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1834 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1836 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float1 tex3D(texture< ushort1, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1837 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1842 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1844 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex3D(texture< short2, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1845 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1850 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1852 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float2 tex3D(texture< ushort2, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1853 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1858 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1860 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex3D(texture< short4, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1861 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1866 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1868 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex3D(texture< ushort4, 3, cudaReadModeNormalizedFloat> t, float x, float y, float z)
# 1869 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1874 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1930 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
template< int comp, class T> __attribute__((unused)) extern int4 __itex2Dgather(texture< T, 2, cudaReadModeElementType> , float2, int = comp);
# 1932 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
template< int comp, class T> __attribute__((unused)) extern uint4 __utex2Dgather(texture< T, 2, cudaReadModeElementType> , float2, int = comp);
# 1934 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
template< int comp, class T> __attribute__((unused)) extern float4 __ftex2Dgather(texture< T, 2, cudaReadModeElementType> , float2, int = comp);
# 1954 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2Dgather(texture< char, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1955 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1957 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1959 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2Dgather(texture< signed char, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1960 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1962 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1964 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex2Dgather(texture< unsigned char, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1965 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1967 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1969 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2Dgather(texture< char1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1970 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1972 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1974 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex2Dgather(texture< uchar1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1975 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1977 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1979 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2Dgather(texture< char2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1980 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1982 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1984 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex2Dgather(texture< uchar2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1985 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1987 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1989 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2Dgather(texture< char3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1990 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1992 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1994 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex2Dgather(texture< uchar3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 1995 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 1997 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 1999 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline char4 tex2Dgather(texture< char4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2000 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2002 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2004 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uchar4 tex2Dgather(texture< uchar4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2005 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2007 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2009 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex2Dgather(texture< short, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2010 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2012 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2014 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex2Dgather(texture< unsigned short, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2015 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2017 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2019 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex2Dgather(texture< short1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2020 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2022 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2024 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex2Dgather(texture< ushort1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2025 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2027 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2029 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex2Dgather(texture< short2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2030 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2032 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2034 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex2Dgather(texture< ushort2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2035 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2037 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2039 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex2Dgather(texture< short3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2040 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2042 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2044 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex2Dgather(texture< ushort3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2045 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2047 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2049 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline short4 tex2Dgather(texture< short4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2050 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2052 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2054 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline ushort4 tex2Dgather(texture< ushort4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2055 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2057 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2059 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex2Dgather(texture< int, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2060 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2062 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2064 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex2Dgather(texture< unsigned, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2065 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2067 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2069 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex2Dgather(texture< int1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2070 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2072 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2074 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex2Dgather(texture< uint1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2075 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2077 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2079 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex2Dgather(texture< int2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2080 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2082 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2084 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex2Dgather(texture< uint2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2085 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2087 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2089 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex2Dgather(texture< int3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2090 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2092 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2094 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex2Dgather(texture< uint3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2095 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2097 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2099 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline int4 tex2Dgather(texture< int4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2100 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2102 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2104 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline uint4 tex2Dgather(texture< uint4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2105 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2107 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2109 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2Dgather(texture< float, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2110 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2112 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2114 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2Dgather(texture< float1, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2115 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2117 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2119 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2Dgather(texture< float2, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2120 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2122 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2124 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2Dgather(texture< float3, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2125 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2127 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 2129 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
__attribute__((unused)) static inline float4 tex2Dgather(texture< float4, 2, cudaReadModeElementType> t, float x, float y, int comp = 0)
# 2130 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
{int volatile ___ = 1;
# 2132 "/home/bachelor/deicide218/cuda-3.2/bin/../include/texture_fetch_functions.h"
exit(___);}
# 53 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_launch_parameters.h"
extern "C" { extern const uint3 threadIdx; }
# 55 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_launch_parameters.h"
extern "C" { extern const uint3 blockIdx; }
# 57 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_launch_parameters.h"
extern "C" { extern const dim3 blockDim; }
# 59 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_launch_parameters.h"
extern "C" { extern const dim3 gridDim; }
# 61 "/home/bachelor/deicide218/cuda-3.2/bin/../include/device_launch_parameters.h"
extern "C" { extern const int warpSize; }
# 106 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 107 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaSetupArgument(T
# 108 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
arg, size_t
# 109 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset)
# 111 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 112 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaSetupArgument((const void *)(&arg), sizeof(T), offset);
# 113 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 145 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaEventCreate(cudaEvent_t *
# 146 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
event, unsigned
# 147 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
flags)
# 149 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 150 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaEventCreateWithFlags(event, 0);
# 151 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 208 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMallocHost(void **
# 209 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
ptr, size_t
# 210 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
size, unsigned
# 211 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
flags)
# 213 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 214 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaHostAlloc(ptr, size, flags);
# 215 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 217 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 218 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaHostAlloc(T **
# 219 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
ptr, size_t
# 220 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
size, unsigned
# 221 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
flags)
# 223 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 224 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaHostAlloc((void **)((void *)ptr), size, flags);
# 225 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 227 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 228 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaHostGetDevicePointer(T **
# 229 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
pDevice, void *
# 230 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
pHost, unsigned
# 231 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
flags)
# 233 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 234 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags);
# 235 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 237 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 238 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaMalloc(T **
# 239 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
devPtr, size_t
# 240 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
size)
# 242 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 243 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaMalloc((void **)((void *)devPtr), size);
# 244 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 246 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 247 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaMallocHost(T **
# 248 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
ptr, size_t
# 249 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
size, unsigned
# 250 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
flags = (0))
# 252 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 253 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaMallocHost((void **)((void *)ptr), size, flags);
# 254 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 256 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 257 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaMallocPitch(T **
# 258 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
devPtr, size_t *
# 259 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
pitch, size_t
# 260 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
width, size_t
# 261 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
height)
# 263 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 264 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaMallocPitch((void **)((void *)devPtr), pitch, width, height);
# 265 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 275 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyToSymbol(char *
# 276 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
symbol, const void *
# 277 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
src, size_t
# 278 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
count, size_t
# 279 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 280 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
kind = cudaMemcpyHostToDevice)
# 282 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 283 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaMemcpyToSymbol((const char *)symbol, src, count, offset, kind);
# 284 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 286 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 287 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaMemcpyToSymbol(const T &
# 288 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
symbol, const void *
# 289 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
src, size_t
# 290 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
count, size_t
# 291 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 292 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
kind = cudaMemcpyHostToDevice)
# 294 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 295 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaMemcpyToSymbol((const char *)(&symbol), src, count, offset, kind);
# 296 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 298 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyToSymbolAsync(char *
# 299 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
symbol, const void *
# 300 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
src, size_t
# 301 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
count, size_t
# 302 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 303 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
kind = cudaMemcpyHostToDevice, cudaStream_t
# 304 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
stream = 0)
# 306 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 307 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaMemcpyToSymbolAsync((const char *)symbol, src, count, offset, kind, stream);
# 308 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 310 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 311 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaMemcpyToSymbolAsync(const T &
# 312 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
symbol, const void *
# 313 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
src, size_t
# 314 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
count, size_t
# 315 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 316 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
kind = cudaMemcpyHostToDevice, cudaStream_t
# 317 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
stream = 0)
# 319 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 320 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaMemcpyToSymbolAsync((const char *)(&symbol), src, count, offset, kind, stream);
# 321 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 329 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyFromSymbol(void *
# 330 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
dst, char *
# 331 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
symbol, size_t
# 332 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
count, size_t
# 333 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 334 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
kind = cudaMemcpyDeviceToHost)
# 336 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 337 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaMemcpyFromSymbol(dst, (const char *)symbol, count, offset, kind);
# 338 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 340 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 341 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaMemcpyFromSymbol(void *
# 342 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
dst, const T &
# 343 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
symbol, size_t
# 344 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
count, size_t
# 345 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 346 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
kind = cudaMemcpyDeviceToHost)
# 348 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 349 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaMemcpyFromSymbol(dst, (const char *)(&symbol), count, offset, kind);
# 350 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 352 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaMemcpyFromSymbolAsync(void *
# 353 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
dst, char *
# 354 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
symbol, size_t
# 355 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
count, size_t
# 356 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 357 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
kind = cudaMemcpyDeviceToHost, cudaStream_t
# 358 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
stream = 0)
# 360 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 361 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaMemcpyFromSymbolAsync(dst, (const char *)symbol, count, offset, kind, stream);
# 362 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 364 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 365 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaMemcpyFromSymbolAsync(void *
# 366 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
dst, const T &
# 367 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
symbol, size_t
# 368 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
count, size_t
# 369 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset = (0), cudaMemcpyKind
# 370 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
kind = cudaMemcpyDeviceToHost, cudaStream_t
# 371 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
stream = 0)
# 373 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 374 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaMemcpyFromSymbolAsync(dst, (const char *)(&symbol), count, offset, kind, stream);
# 375 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 377 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaGetSymbolAddress(void **
# 378 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
devPtr, char *
# 379 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
symbol)
# 381 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 382 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaGetSymbolAddress(devPtr, (const char *)symbol);
# 383 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 410 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 411 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaGetSymbolAddress(void **
# 412 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
devPtr, const T &
# 413 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
symbol)
# 415 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 416 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaGetSymbolAddress(devPtr, (const char *)(&symbol));
# 417 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 425 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
static inline cudaError_t cudaGetSymbolSize(size_t *
# 426 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
size, char *
# 427 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
symbol)
# 429 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 430 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaGetSymbolSize(size, (const char *)symbol);
# 431 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 458 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 459 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaGetSymbolSize(size_t *
# 460 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
size, const T &
# 461 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
symbol)
# 463 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 464 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaGetSymbolSize(size, (const char *)(&symbol));
# 465 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 507 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 508 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaBindTexture(size_t *
# 509 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 510 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
tex, const void *
# 511 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
devPtr, const cudaChannelFormatDesc &
# 512 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
desc, size_t
# 513 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
size = (((2147483647) * 2U) + 1U))
# 515 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 516 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaBindTexture(offset, &tex, devPtr, &desc, size);
# 517 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 552 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 553 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaBindTexture(size_t *
# 554 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 555 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
tex, const void *
# 556 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
devPtr, size_t
# 557 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
size = (((2147483647) * 2U) + 1U))
# 559 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 560 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size);
# 561 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 608 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 609 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaBindTexture2D(size_t *
# 610 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 611 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
tex, const void *
# 612 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
devPtr, const cudaChannelFormatDesc &
# 613 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
desc, size_t
# 614 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
width, size_t
# 615 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
height, size_t
# 616 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
pitch)
# 618 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 619 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch);
# 620 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 666 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 667 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaBindTexture2D(size_t *
# 668 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 669 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
tex, const void *
# 670 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
devPtr, size_t
# 671 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
width, size_t
# 672 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
height, size_t
# 673 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
pitch)
# 675 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 676 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaBindTexture2D(offset, &tex, devPtr, &(tex.texture< T, dim, readMode> ::channelDesc), width, height, pitch);
# 677 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 708 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 709 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaBindTextureToArray(const texture< T, dim, readMode> &
# 710 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
tex, const cudaArray *
# 711 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
array, const cudaChannelFormatDesc &
# 712 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
desc)
# 714 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 715 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaBindTextureToArray(&tex, array, &desc);
# 716 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 746 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 747 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaBindTextureToArray(const texture< T, dim, readMode> &
# 748 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
tex, const cudaArray *
# 749 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
array)
# 751 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 752 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaChannelFormatDesc desc;
# 753 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaError_t err = cudaGetChannelDesc(&desc, array);
# 755 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err;
# 756 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 785 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 786 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaUnbindTexture(const texture< T, dim, readMode> &
# 787 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
tex)
# 789 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 790 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaUnbindTexture(&tex);
# 791 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 825 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> inline cudaError_t
# 826 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaGetTextureAlignmentOffset(size_t *
# 827 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
offset, const texture< T, dim, readMode> &
# 828 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
tex)
# 830 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 831 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaGetTextureAlignmentOffset(offset, &tex);
# 832 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 886 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 887 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaFuncSetCacheConfig(T *
# 888 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
func, cudaFuncCache
# 889 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cacheConfig)
# 891 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 892 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaFuncSetCacheConfig((const char *)func, cacheConfig);
# 893 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 930 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 931 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaLaunch(T *
# 932 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
entry)
# 934 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 935 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaLaunch((const char *)entry);
# 936 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 970 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T> inline cudaError_t
# 971 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaFuncGetAttributes(cudaFuncAttributes *
# 972 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
attr, T *
# 973 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
entry)
# 975 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 976 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaFuncGetAttributes(attr, (const char *)entry);
# 977 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 999 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T, int dim> inline cudaError_t
# 1000 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaBindSurfaceToArray(const surface< T, dim> &
# 1001 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
surf, const cudaArray *
# 1002 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
array, const cudaChannelFormatDesc &
# 1003 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
desc)
# 1005 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 1006 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return cudaBindSurfaceToArray(&surf, array, &desc);
# 1007 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 1028 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
template< class T, int dim> inline cudaError_t
# 1029 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaBindSurfaceToArray(const surface< T, dim> &
# 1030 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
surf, const cudaArray *
# 1031 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
array)
# 1033 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
{
# 1034 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaChannelFormatDesc desc;
# 1035 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
cudaError_t err = cudaGetChannelDesc(&desc, array);
# 1037 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err;
# 1038 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda_runtime.h"
}
# 45 "/usr/include/stdio.h" 3
struct _IO_FILE;
# 49 "/usr/include/stdio.h" 3
extern "C" { typedef _IO_FILE FILE; }
# 65 "/usr/include/stdio.h" 3
extern "C" { typedef _IO_FILE __FILE; }
# 95 "/usr/include/wchar.h" 3
extern "C" { typedef
# 84 "/usr/include/wchar.h" 3
struct {
# 85 "/usr/include/wchar.h" 3
int __count;
# 87 "/usr/include/wchar.h" 3
union {
# 89 "/usr/include/wchar.h" 3
unsigned __wch;
# 93 "/usr/include/wchar.h" 3
char __wchb[4];
# 94 "/usr/include/wchar.h" 3
} __value;
# 95 "/usr/include/wchar.h" 3
} __mbstate_t; }
# 26 "/usr/include/_G_config.h" 3
extern "C" { typedef
# 23 "/usr/include/_G_config.h" 3
struct {
# 24 "/usr/include/_G_config.h" 3
__off_t __pos;
# 25 "/usr/include/_G_config.h" 3
__mbstate_t __state;
# 26 "/usr/include/_G_config.h" 3
} _G_fpos_t; }
# 31 "/usr/include/_G_config.h" 3
extern "C" { typedef
# 28 "/usr/include/_G_config.h" 3
struct {
# 29 "/usr/include/_G_config.h" 3
__off64_t __pos;
# 30 "/usr/include/_G_config.h" 3
__mbstate_t __state;
# 31 "/usr/include/_G_config.h" 3
} _G_fpos64_t; }
# 53 "/usr/include/_G_config.h" 3
extern "C" { typedef short _G_int16_t; }
# 54 "/usr/include/_G_config.h" 3
extern "C" { typedef int _G_int32_t; }
# 55 "/usr/include/_G_config.h" 3
extern "C" { typedef unsigned short _G_uint16_t; }
# 56 "/usr/include/_G_config.h" 3
extern "C" { typedef unsigned _G_uint32_t; }
# 40 "/usr/lib/gcc/x86_64-linux-gnu/4.4.7/include/stdarg.h" 3
extern "C" { typedef __builtin_va_list __gnuc_va_list; }
# 170 "/usr/include/libio.h" 3
struct _IO_jump_t; struct _IO_FILE;
# 180 "/usr/include/libio.h" 3
extern "C" { typedef void _IO_lock_t; }
# 186 "/usr/include/libio.h" 3
extern "C" { struct _IO_marker {
# 187 "/usr/include/libio.h" 3
_IO_marker *_next;
# 188 "/usr/include/libio.h" 3
_IO_FILE *_sbuf;
# 192 "/usr/include/libio.h" 3
int _pos;
# 203 "/usr/include/libio.h" 3
}; }
# 206 "/usr/include/libio.h" 3
enum __codecvt_result {
# 208 "/usr/include/libio.h" 3
__codecvt_ok,
# 209 "/usr/include/libio.h" 3
__codecvt_partial,
# 210 "/usr/include/libio.h" 3
__codecvt_error,
# 211 "/usr/include/libio.h" 3
__codecvt_noconv
# 212 "/usr/include/libio.h" 3
};
# 271 "/usr/include/libio.h" 3
extern "C" { struct _IO_FILE {
# 272 "/usr/include/libio.h" 3
int _flags;
# 277 "/usr/include/libio.h" 3
char *_IO_read_ptr;
# 278 "/usr/include/libio.h" 3
char *_IO_read_end;
# 279 "/usr/include/libio.h" 3
char *_IO_read_base;
# 280 "/usr/include/libio.h" 3
char *_IO_write_base;
# 281 "/usr/include/libio.h" 3
char *_IO_write_ptr;
# 282 "/usr/include/libio.h" 3
char *_IO_write_end;
# 283 "/usr/include/libio.h" 3
char *_IO_buf_base;
# 284 "/usr/include/libio.h" 3
char *_IO_buf_end;
# 286 "/usr/include/libio.h" 3
char *_IO_save_base;
# 287 "/usr/include/libio.h" 3
char *_IO_backup_base;
# 288 "/usr/include/libio.h" 3
char *_IO_save_end;
# 290 "/usr/include/libio.h" 3
_IO_marker *_markers;
# 292 "/usr/include/libio.h" 3
_IO_FILE *_chain;
# 294 "/usr/include/libio.h" 3
int _fileno;
# 298 "/usr/include/libio.h" 3
int _flags2;
# 300 "/usr/include/libio.h" 3
__off_t _old_offset;
# 304 "/usr/include/libio.h" 3
unsigned short _cur_column;
# 305 "/usr/include/libio.h" 3
signed char _vtable_offset;
# 306 "/usr/include/libio.h" 3
char _shortbuf[1];
# 310 "/usr/include/libio.h" 3
_IO_lock_t *_lock;
# 319 "/usr/include/libio.h" 3
__off64_t _offset;
# 328 "/usr/include/libio.h" 3
void *__pad1;
# 329 "/usr/include/libio.h" 3
void *__pad2;
# 330 "/usr/include/libio.h" 3
void *__pad3;
# 331 "/usr/include/libio.h" 3
void *__pad4;
# 332 "/usr/include/libio.h" 3
size_t __pad5;
# 334 "/usr/include/libio.h" 3
int _mode;
# 336 "/usr/include/libio.h" 3
char _unused2[((((15) * sizeof(int)) - ((4) * sizeof(void *))) - sizeof(size_t))];
# 338 "/usr/include/libio.h" 3
}; }
# 344 "/usr/include/libio.h" 3
struct _IO_FILE_plus;
# 346 "/usr/include/libio.h" 3
extern "C" { extern _IO_FILE_plus _IO_2_1_stdin_; }
# 347 "/usr/include/libio.h" 3
extern "C" { extern _IO_FILE_plus _IO_2_1_stdout_; }
# 348 "/usr/include/libio.h" 3
extern "C" { extern _IO_FILE_plus _IO_2_1_stderr_; }
# 364 "/usr/include/libio.h" 3
extern "C" { typedef __ssize_t __io_read_fn(void *, char *, size_t); }
# 372 "/usr/include/libio.h" 3
extern "C" { typedef __ssize_t __io_write_fn(void *, const char *, size_t); }
# 381 "/usr/include/libio.h" 3
extern "C" { typedef int __io_seek_fn(void *, __off64_t *, int); }
# 384 "/usr/include/libio.h" 3
extern "C" { typedef int __io_close_fn(void *); }
# 389 "/usr/include/libio.h" 3
extern "C" { typedef __io_read_fn cookie_read_function_t; }
# 390 "/usr/include/libio.h" 3
extern "C" { typedef __io_write_fn cookie_write_function_t; }
# 391 "/usr/include/libio.h" 3
extern "C" { typedef __io_seek_fn cookie_seek_function_t; }
# 392 "/usr/include/libio.h" 3
extern "C" { typedef __io_close_fn cookie_close_function_t; }
# 401 "/usr/include/libio.h" 3
extern "C" { typedef
# 396 "/usr/include/libio.h" 3
struct {
# 397 "/usr/include/libio.h" 3
__io_read_fn *read;
# 398 "/usr/include/libio.h" 3
__io_write_fn *write;
# 399 "/usr/include/libio.h" 3
__io_seek_fn *seek;
# 400 "/usr/include/libio.h" 3
__io_close_fn *close;
# 401 "/usr/include/libio.h" 3
} _IO_cookie_io_functions_t; }
# 402 "/usr/include/libio.h" 3
extern "C" { typedef _IO_cookie_io_functions_t cookie_io_functions_t; }
# 404 "/usr/include/libio.h" 3
struct _IO_cookie_file;
# 407 "/usr/include/libio.h" 3
extern "C" void _IO_cookie_init(_IO_cookie_file *, int, void *, _IO_cookie_io_functions_t);
# 416 "/usr/include/libio.h" 3
extern "C" int __underflow(_IO_FILE *);
# 417 "/usr/include/libio.h" 3
extern "C" int __uflow(_IO_FILE *);
# 418 "/usr/include/libio.h" 3
extern "C" int __overflow(_IO_FILE *, int);
# 460 "/usr/include/libio.h" 3
extern "C" int _IO_getc(_IO_FILE *);
# 461 "/usr/include/libio.h" 3
extern "C" int _IO_putc(int, _IO_FILE *);
# 462 "/usr/include/libio.h" 3
extern "C" int _IO_feof(_IO_FILE *) throw();
# 463 "/usr/include/libio.h" 3
extern "C" int _IO_ferror(_IO_FILE *) throw();
# 465 "/usr/include/libio.h" 3
extern "C" int _IO_peekc_locked(_IO_FILE *);
# 471 "/usr/include/libio.h" 3
extern "C" void _IO_flockfile(_IO_FILE *) throw();
# 472 "/usr/include/libio.h" 3
extern "C" void _IO_funlockfile(_IO_FILE *) throw();
# 473 "/usr/include/libio.h" 3
extern "C" int _IO_ftrylockfile(_IO_FILE *) throw();
# 490 "/usr/include/libio.h" 3
extern "C" int _IO_vfscanf(_IO_FILE *__restrict__, const char *__restrict__, __gnuc_va_list, int *__restrict__);
# 492 "/usr/include/libio.h" 3
extern "C" int _IO_vfprintf(_IO_FILE *__restrict__, const char *__restrict__, __gnuc_va_list);
# 494 "/usr/include/libio.h" 3
extern "C" __ssize_t _IO_padn(_IO_FILE *, int, __ssize_t);
# 495 "/usr/include/libio.h" 3
extern "C" size_t _IO_sgetn(_IO_FILE *, void *, size_t);
# 497 "/usr/include/libio.h" 3
extern "C" __off64_t _IO_seekoff(_IO_FILE *, __off64_t, int, int);
# 498 "/usr/include/libio.h" 3
extern "C" __off64_t _IO_seekpos(_IO_FILE *, __off64_t, int);
# 500 "/usr/include/libio.h" 3
extern "C" void _IO_free_backup_area(_IO_FILE *) throw();
# 80 "/usr/include/stdio.h" 3
extern "C" { typedef __gnuc_va_list va_list; }
# 111 "/usr/include/stdio.h" 3
extern "C" { typedef _G_fpos_t fpos_t; }
# 117 "/usr/include/stdio.h" 3
extern "C" { typedef _G_fpos64_t fpos64_t; }
# 165 "/usr/include/stdio.h" 3
extern "C" { extern _IO_FILE *stdin; }
# 166 "/usr/include/stdio.h" 3
extern "C" { extern _IO_FILE *stdout; }
# 167 "/usr/include/stdio.h" 3
extern "C" { extern _IO_FILE *stderr; }
# 175 "/usr/include/stdio.h" 3
extern "C" int remove(const char *) throw();
# 177 "/usr/include/stdio.h" 3
extern "C" int rename(const char *, const char *) throw();
# 182 "/usr/include/stdio.h" 3
extern "C" int renameat(int, const char *, int, const char *) throw();
# 192 "/usr/include/stdio.h" 3
extern "C" FILE *tmpfile();
# 202 "/usr/include/stdio.h" 3
extern "C" FILE *tmpfile64();
# 206 "/usr/include/stdio.h" 3
extern "C" char *tmpnam(char *) throw();
# 212 "/usr/include/stdio.h" 3
extern "C" char *tmpnam_r(char *) throw();
# 224 "/usr/include/stdio.h" 3
extern "C" char *tempnam(const char *, const char *) throw() __attribute__((__malloc__));
# 234 "/usr/include/stdio.h" 3
extern "C" int fclose(FILE *);
# 239 "/usr/include/stdio.h" 3
extern "C" int fflush(FILE *);
# 249 "/usr/include/stdio.h" 3
extern "C" int fflush_unlocked(FILE *);
# 259 "/usr/include/stdio.h" 3
extern "C" int fcloseall();
# 269 "/usr/include/stdio.h" 3
extern "C" FILE *fopen(const char *__restrict__, const char *__restrict__);
# 275 "/usr/include/stdio.h" 3
extern "C" FILE *freopen(const char *__restrict__, const char *__restrict__, FILE *__restrict__);
# 294 "/usr/include/stdio.h" 3
extern "C" FILE *fopen64(const char *__restrict__, const char *__restrict__);
# 296 "/usr/include/stdio.h" 3
extern "C" FILE *freopen64(const char *__restrict__, const char *__restrict__, FILE *__restrict__);
# 303 "/usr/include/stdio.h" 3
extern "C" FILE *fdopen(int, const char *) throw();
# 309 "/usr/include/stdio.h" 3
extern "C" FILE *fopencookie(void *__restrict__, const char *__restrict__, _IO_cookie_io_functions_t) throw();
# 316 "/usr/include/stdio.h" 3
extern "C" FILE *fmemopen(void *, size_t, const char *) throw();
# 322 "/usr/include/stdio.h" 3
extern "C" FILE *open_memstream(char **, size_t *) throw();
# 329 "/usr/include/stdio.h" 3
extern "C" void setbuf(FILE *__restrict__, char *__restrict__) throw();
# 333 "/usr/include/stdio.h" 3
extern "C" int setvbuf(FILE *__restrict__, char *__restrict__, int, size_t) throw();
# 340 "/usr/include/stdio.h" 3
extern "C" void setbuffer(FILE *__restrict__, char *__restrict__, size_t) throw();
# 344 "/usr/include/stdio.h" 3
extern "C" void setlinebuf(FILE *) throw();
# 353 "/usr/include/stdio.h" 3
extern "C" int fprintf(FILE *__restrict__, const char *__restrict__, ...);
# 359 "/usr/include/stdio.h" 3
extern "C" int printf(const char *__restrict__, ...);
# 361 "/usr/include/stdio.h" 3
extern "C" int sprintf(char *__restrict__, const char *__restrict__, ...) throw();
# 368 "/usr/include/stdio.h" 3
extern "C" int vfprintf(FILE *__restrict__, const char *__restrict__, __gnuc_va_list);
# 374 "/usr/include/stdio.h" 3
extern "C" int vprintf(const char *__restrict__, __gnuc_va_list);
# 376 "/usr/include/stdio.h" 3
extern "C" int vsprintf(char *__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 383 "/usr/include/stdio.h" 3
extern "C" int snprintf(char *__restrict__, size_t, const char *__restrict__, ...) throw();
# 387 "/usr/include/stdio.h" 3
extern "C" int vsnprintf(char *__restrict__, size_t, const char *__restrict__, __gnuc_va_list) throw();
# 396 "/usr/include/stdio.h" 3
extern "C" int vasprintf(char **__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 399 "/usr/include/stdio.h" 3
extern "C" int __asprintf(char **__restrict__, const char *__restrict__, ...) throw();
# 402 "/usr/include/stdio.h" 3
extern "C" int asprintf(char **__restrict__, const char *__restrict__, ...) throw();
# 414 "/usr/include/stdio.h" 3
extern "C" int vdprintf(int, const char *__restrict__, __gnuc_va_list);
# 417 "/usr/include/stdio.h" 3
extern "C" int dprintf(int, const char *__restrict__, ...);
# 427 "/usr/include/stdio.h" 3
extern "C" int fscanf(FILE *__restrict__, const char *__restrict__, ...);
# 433 "/usr/include/stdio.h" 3
extern "C" int scanf(const char *__restrict__, ...);
# 435 "/usr/include/stdio.h" 3
extern "C" int sscanf(const char *__restrict__, const char *__restrict__, ...) throw();
# 473 "/usr/include/stdio.h" 3
extern "C" int vfscanf(FILE *__restrict__, const char *__restrict__, __gnuc_va_list);
# 481 "/usr/include/stdio.h" 3
extern "C" int vscanf(const char *__restrict__, __gnuc_va_list);
# 485 "/usr/include/stdio.h" 3
extern "C" int vsscanf(const char *__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 533 "/usr/include/stdio.h" 3
extern "C" int fgetc(FILE *);
# 534 "/usr/include/stdio.h" 3
extern "C" int getc(FILE *);
# 540 "/usr/include/stdio.h" 3
extern "C" int getchar();
# 552 "/usr/include/stdio.h" 3
extern "C" int getc_unlocked(FILE *);
# 553 "/usr/include/stdio.h" 3
extern "C" int getchar_unlocked();
# 563 "/usr/include/stdio.h" 3
extern "C" int fgetc_unlocked(FILE *);
# 575 "/usr/include/stdio.h" 3
extern "C" int fputc(int, FILE *);
# 576 "/usr/include/stdio.h" 3
extern "C" int putc(int, FILE *);
# 582 "/usr/include/stdio.h" 3
extern "C" int putchar(int);
# 596 "/usr/include/stdio.h" 3
extern "C" int fputc_unlocked(int, FILE *);
# 604 "/usr/include/stdio.h" 3
extern "C" int putc_unlocked(int, FILE *);
# 605 "/usr/include/stdio.h" 3
extern "C" int putchar_unlocked(int);
# 612 "/usr/include/stdio.h" 3
extern "C" int getw(FILE *);
# 615 "/usr/include/stdio.h" 3
extern "C" int putw(int, FILE *);
# 624 "/usr/include/stdio.h" 3
extern "C" char *fgets(char *__restrict__, int, FILE *__restrict__);
# 632 "/usr/include/stdio.h" 3
extern "C" char *gets(char *);
# 642 "/usr/include/stdio.h" 3
extern "C" char *fgets_unlocked(char *__restrict__, int, FILE *__restrict__);
# 658 "/usr/include/stdio.h" 3
extern "C" __ssize_t __getdelim(char **__restrict__, size_t *__restrict__, int, FILE *__restrict__);
# 661 "/usr/include/stdio.h" 3
extern "C" __ssize_t getdelim(char **__restrict__, size_t *__restrict__, int, FILE *__restrict__);
# 671 "/usr/include/stdio.h" 3
extern "C" __ssize_t getline(char **__restrict__, size_t *__restrict__, FILE *__restrict__);
# 682 "/usr/include/stdio.h" 3
extern "C" int fputs(const char *__restrict__, FILE *__restrict__);
# 688 "/usr/include/stdio.h" 3
extern "C" int puts(const char *);
# 695 "/usr/include/stdio.h" 3
extern "C" int ungetc(int, FILE *);
# 702 "/usr/include/stdio.h" 3
extern "C" size_t fread(void *__restrict__, size_t, size_t, FILE *__restrict__);
# 708 "/usr/include/stdio.h" 3
extern "C" size_t fwrite(const void *__restrict__, size_t, size_t, FILE *__restrict__);
# 719 "/usr/include/stdio.h" 3
extern "C" int fputs_unlocked(const char *__restrict__, FILE *__restrict__);
# 730 "/usr/include/stdio.h" 3
extern "C" size_t fread_unlocked(void *__restrict__, size_t, size_t, FILE *__restrict__);
# 732 "/usr/include/stdio.h" 3
extern "C" size_t fwrite_unlocked(const void *__restrict__, size_t, size_t, FILE *__restrict__);
# 742 "/usr/include/stdio.h" 3
extern "C" int fseek(FILE *, long, int);
# 747 "/usr/include/stdio.h" 3
extern "C" long ftell(FILE *);
# 752 "/usr/include/stdio.h" 3
extern "C" void rewind(FILE *);
# 766 "/usr/include/stdio.h" 3
extern "C" int fseeko(FILE *, __off_t, int);
# 771 "/usr/include/stdio.h" 3
extern "C" __off_t ftello(FILE *);
# 791 "/usr/include/stdio.h" 3
extern "C" int fgetpos(FILE *__restrict__, fpos_t *__restrict__);
# 796 "/usr/include/stdio.h" 3
extern "C" int fsetpos(FILE *, const fpos_t *);
# 811 "/usr/include/stdio.h" 3
extern "C" int fseeko64(FILE *, __off64_t, int);
# 812 "/usr/include/stdio.h" 3
extern "C" __off64_t ftello64(FILE *);
# 813 "/usr/include/stdio.h" 3
extern "C" int fgetpos64(FILE *__restrict__, fpos64_t *__restrict__);
# 814 "/usr/include/stdio.h" 3
extern "C" int fsetpos64(FILE *, const fpos64_t *);
# 819 "/usr/include/stdio.h" 3
extern "C" void clearerr(FILE *) throw();
# 821 "/usr/include/stdio.h" 3
extern "C" int feof(FILE *) throw();
# 823 "/usr/include/stdio.h" 3
extern "C" int ferror(FILE *) throw();
# 828 "/usr/include/stdio.h" 3
extern "C" void clearerr_unlocked(FILE *) throw();
# 829 "/usr/include/stdio.h" 3
extern "C" int feof_unlocked(FILE *) throw();
# 830 "/usr/include/stdio.h" 3
extern "C" int ferror_unlocked(FILE *) throw();
# 839 "/usr/include/stdio.h" 3
extern "C" void perror(const char *);
# 27 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h" 3
extern "C" { extern int sys_nerr; }
# 28 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h" 3
extern "C" { extern const char *const sys_errlist[]; }
# 31 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h" 3
extern "C" { extern int _sys_nerr; }
# 32 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h" 3
extern "C" { extern const char *const _sys_errlist[]; }
# 851 "/usr/include/stdio.h" 3
extern "C" int fileno(FILE *) throw();
# 856 "/usr/include/stdio.h" 3
extern "C" int fileno_unlocked(FILE *) throw();
# 866 "/usr/include/stdio.h" 3
extern "C" FILE *popen(const char *, const char *);
# 872 "/usr/include/stdio.h" 3
extern "C" int pclose(FILE *);
# 878 "/usr/include/stdio.h" 3
extern "C" char *ctermid(char *) throw();
# 884 "/usr/include/stdio.h" 3
extern "C" char *cuserid(char *);
# 889 "/usr/include/stdio.h" 3
struct obstack;
# 892 "/usr/include/stdio.h" 3
extern "C" int obstack_printf(obstack *__restrict__, const char *__restrict__, ...) throw();
# 895 "/usr/include/stdio.h" 3
extern "C" int obstack_vprintf(obstack *__restrict__, const char *__restrict__, __gnuc_va_list) throw();
# 906 "/usr/include/stdio.h" 3
extern "C" void flockfile(FILE *) throw();
# 910 "/usr/include/stdio.h" 3
extern "C" int ftrylockfile(FILE *) throw();
# 913 "/usr/include/stdio.h" 3
extern "C" void funlockfile(FILE *) throw();
# 268 "/usr/include/unistd.h" 3
extern "C" { typedef __intptr_t intptr_t; }
# 275 "/usr/include/unistd.h" 3
extern "C" { typedef __socklen_t socklen_t; }
# 288 "/usr/include/unistd.h" 3
extern "C" int access(const char *, int) throw() __attribute__((nonnull(1)));
# 293 "/usr/include/unistd.h" 3
extern "C" int euidaccess(const char *, int) throw() __attribute__((nonnull(1)));
# 297 "/usr/include/unistd.h" 3
extern "C" int eaccess(const char *, int) throw() __attribute__((nonnull(1)));
# 305 "/usr/include/unistd.h" 3
extern "C" int faccessat(int, const char *, int, int) throw() __attribute__((nonnull(2)));
# 331 "/usr/include/unistd.h" 3
extern "C" __off_t lseek(int, __off_t, int) throw();
# 342 "/usr/include/unistd.h" 3
extern "C" __off64_t lseek64(int, __off64_t, int) throw();
# 350 "/usr/include/unistd.h" 3
extern "C" int close(int);
# 357 "/usr/include/unistd.h" 3
extern "C" ssize_t read(int, void *, size_t);
# 363 "/usr/include/unistd.h" 3
extern "C" ssize_t write(int, const void *, size_t);
# 373 "/usr/include/unistd.h" 3
extern "C" ssize_t pread(int, void *, size_t, __off_t);
# 381 "/usr/include/unistd.h" 3
extern "C" ssize_t pwrite(int, const void *, size_t, __off_t);
# 401 "/usr/include/unistd.h" 3
extern "C" ssize_t pread64(int, void *, size_t, __off64_t);
# 405 "/usr/include/unistd.h" 3
extern "C" ssize_t pwrite64(int, const void *, size_t, __off64_t);
# 414 "/usr/include/unistd.h" 3
extern "C" int pipe(int [2]) throw();
# 419 "/usr/include/unistd.h" 3
extern "C" int pipe2(int [2], int) throw();
# 429 "/usr/include/unistd.h" 3
extern "C" unsigned alarm(unsigned) throw();
# 441 "/usr/include/unistd.h" 3
extern "C" unsigned sleep(unsigned);
# 449 "/usr/include/unistd.h" 3
extern "C" __useconds_t ualarm(__useconds_t, __useconds_t) throw();
# 457 "/usr/include/unistd.h" 3
extern "C" int usleep(__useconds_t);
# 466 "/usr/include/unistd.h" 3
extern "C" int pause();
# 470 "/usr/include/unistd.h" 3
extern "C" int chown(const char *, __uid_t, __gid_t) throw() __attribute__((nonnull(1)));
# 475 "/usr/include/unistd.h" 3
extern "C" int fchown(int, __uid_t, __gid_t) throw();
# 480 "/usr/include/unistd.h" 3
extern "C" int lchown(const char *, __uid_t, __gid_t) throw() __attribute__((nonnull(1)));
# 488 "/usr/include/unistd.h" 3
extern "C" int fchownat(int, const char *, __uid_t, __gid_t, int) throw() __attribute__((nonnull(2)));
# 494 "/usr/include/unistd.h" 3
extern "C" int chdir(const char *) throw() __attribute__((nonnull(1)));
# 498 "/usr/include/unistd.h" 3
extern "C" int fchdir(int) throw();
# 508 "/usr/include/unistd.h" 3
extern "C" char *getcwd(char *, size_t) throw();
# 514 "/usr/include/unistd.h" 3
extern "C" char *get_current_dir_name() throw();
# 522 "/usr/include/unistd.h" 3
extern "C" char *getwd(char *) throw() __attribute__((__deprecated__)) __attribute__((nonnull(1)));
# 528 "/usr/include/unistd.h" 3
extern "C" int dup(int) throw();
# 531 "/usr/include/unistd.h" 3
extern "C" int dup2(int, int) throw();
# 536 "/usr/include/unistd.h" 3
extern "C" int dup3(int, int, int) throw();
# 540 "/usr/include/unistd.h" 3
extern "C" { extern char **__environ; }
# 542 "/usr/include/unistd.h" 3
extern "C" { extern char **environ; }
# 548 "/usr/include/unistd.h" 3
extern "C" int execve(const char *, char *const [], char *const []) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 554 "/usr/include/unistd.h" 3
extern "C" int fexecve(int, char *const [], char *const []) throw() __attribute__((nonnull(2)));
# 560 "/usr/include/unistd.h" 3
extern "C" int execv(const char *, char *const []) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 565 "/usr/include/unistd.h" 3
extern "C" int execle(const char *, const char *, ...) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 570 "/usr/include/unistd.h" 3
extern "C" int execl(const char *, const char *, ...) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 575 "/usr/include/unistd.h" 3
extern "C" int execvp(const char *, char *const []) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 581 "/usr/include/unistd.h" 3
extern "C" int execlp(const char *, const char *, ...) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 587 "/usr/include/unistd.h" 3
extern "C" int execvpe(const char *, char *const [], char *const []) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 595 "/usr/include/unistd.h" 3
extern "C" int nice(int) throw();
# 600 "/usr/include/unistd.h" 3
extern "C" void _exit(int) __attribute__((__noreturn__));
# 27 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
enum {
# 28 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_LINK_MAX,
# 30 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_MAX_CANON,
# 32 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_MAX_INPUT,
# 34 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_NAME_MAX,
# 36 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_PATH_MAX,
# 38 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_PIPE_BUF,
# 40 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_CHOWN_RESTRICTED,
# 42 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_NO_TRUNC,
# 44 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_VDISABLE,
# 46 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_SYNC_IO,
# 48 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_ASYNC_IO,
# 50 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_PRIO_IO,
# 52 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_SOCK_MAXBUF,
# 54 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_FILESIZEBITS,
# 56 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_REC_INCR_XFER_SIZE,
# 58 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_REC_MAX_XFER_SIZE,
# 60 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_REC_MIN_XFER_SIZE,
# 62 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_REC_XFER_ALIGN,
# 64 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_ALLOC_SIZE_MIN,
# 66 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_SYMLINK_MAX,
# 68 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_PC_2_SYMLINKS
# 70 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
};
# 74 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
enum {
# 75 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_ARG_MAX,
# 77 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_CHILD_MAX,
# 79 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_CLK_TCK,
# 81 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_NGROUPS_MAX,
# 83 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_OPEN_MAX,
# 85 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_STREAM_MAX,
# 87 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TZNAME_MAX,
# 89 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_JOB_CONTROL,
# 91 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SAVED_IDS,
# 93 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_REALTIME_SIGNALS,
# 95 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PRIORITY_SCHEDULING,
# 97 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TIMERS,
# 99 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_ASYNCHRONOUS_IO,
# 101 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PRIORITIZED_IO,
# 103 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SYNCHRONIZED_IO,
# 105 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_FSYNC,
# 107 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_MAPPED_FILES,
# 109 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_MEMLOCK,
# 111 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_MEMLOCK_RANGE,
# 113 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_MEMORY_PROTECTION,
# 115 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_MESSAGE_PASSING,
# 117 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SEMAPHORES,
# 119 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SHARED_MEMORY_OBJECTS,
# 121 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_AIO_LISTIO_MAX,
# 123 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_AIO_MAX,
# 125 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_AIO_PRIO_DELTA_MAX,
# 127 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_DELAYTIMER_MAX,
# 129 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_MQ_OPEN_MAX,
# 131 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_MQ_PRIO_MAX,
# 133 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_VERSION,
# 135 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PAGESIZE,
# 138 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_RTSIG_MAX,
# 140 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SEM_NSEMS_MAX,
# 142 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SEM_VALUE_MAX,
# 144 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SIGQUEUE_MAX,
# 146 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TIMER_MAX,
# 151 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_BC_BASE_MAX,
# 153 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_BC_DIM_MAX,
# 155 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_BC_SCALE_MAX,
# 157 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_BC_STRING_MAX,
# 159 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_COLL_WEIGHTS_MAX,
# 161 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_EQUIV_CLASS_MAX,
# 163 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_EXPR_NEST_MAX,
# 165 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LINE_MAX,
# 167 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_RE_DUP_MAX,
# 169 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_CHARCLASS_NAME_MAX,
# 172 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_VERSION,
# 174 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_C_BIND,
# 176 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_C_DEV,
# 178 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_FORT_DEV,
# 180 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_FORT_RUN,
# 182 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_SW_DEV,
# 184 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_LOCALEDEF,
# 187 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PII,
# 189 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PII_XTI,
# 191 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PII_SOCKET,
# 193 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PII_INTERNET,
# 195 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PII_OSI,
# 197 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_POLL,
# 199 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SELECT,
# 201 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_UIO_MAXIOV,
# 203 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_IOV_MAX = 60,
# 205 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PII_INTERNET_STREAM,
# 207 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PII_INTERNET_DGRAM,
# 209 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PII_OSI_COTS,
# 211 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PII_OSI_CLTS,
# 213 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PII_OSI_M,
# 215 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_T_IOV_MAX,
# 219 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREADS,
# 221 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_SAFE_FUNCTIONS,
# 223 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_GETGR_R_SIZE_MAX,
# 225 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_GETPW_R_SIZE_MAX,
# 227 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LOGIN_NAME_MAX,
# 229 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TTY_NAME_MAX,
# 231 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_DESTRUCTOR_ITERATIONS,
# 233 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_KEYS_MAX,
# 235 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_STACK_MIN,
# 237 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_THREADS_MAX,
# 239 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_ATTR_STACKADDR,
# 241 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_ATTR_STACKSIZE,
# 243 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_PRIORITY_SCHEDULING,
# 245 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_PRIO_INHERIT,
# 247 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_PRIO_PROTECT,
# 249 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_PROCESS_SHARED,
# 252 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_NPROCESSORS_CONF,
# 254 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_NPROCESSORS_ONLN,
# 256 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PHYS_PAGES,
# 258 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_AVPHYS_PAGES,
# 260 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_ATEXIT_MAX,
# 262 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PASS_MAX,
# 265 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_VERSION,
# 267 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_XCU_VERSION,
# 269 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_UNIX,
# 271 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_CRYPT,
# 273 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_ENH_I18N,
# 275 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_SHM,
# 278 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_CHAR_TERM,
# 280 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_C_VERSION,
# 282 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_UPE,
# 285 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_XPG2,
# 287 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_XPG3,
# 289 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_XPG4,
# 292 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_CHAR_BIT,
# 294 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_CHAR_MAX,
# 296 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_CHAR_MIN,
# 298 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_INT_MAX,
# 300 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_INT_MIN,
# 302 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LONG_BIT,
# 304 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_WORD_BIT,
# 306 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_MB_LEN_MAX,
# 308 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_NZERO,
# 310 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SSIZE_MAX,
# 312 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SCHAR_MAX,
# 314 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SCHAR_MIN,
# 316 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SHRT_MAX,
# 318 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SHRT_MIN,
# 320 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_UCHAR_MAX,
# 322 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_UINT_MAX,
# 324 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_ULONG_MAX,
# 326 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_USHRT_MAX,
# 329 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_NL_ARGMAX,
# 331 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_NL_LANGMAX,
# 333 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_NL_MSGMAX,
# 335 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_NL_NMAX,
# 337 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_NL_SETMAX,
# 339 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_NL_TEXTMAX,
# 342 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XBS5_ILP32_OFF32,
# 344 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XBS5_ILP32_OFFBIG,
# 346 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XBS5_LP64_OFF64,
# 348 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XBS5_LPBIG_OFFBIG,
# 351 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_LEGACY,
# 353 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_REALTIME,
# 355 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_REALTIME_THREADS,
# 358 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_ADVISORY_INFO,
# 360 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_BARRIERS,
# 362 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_BASE,
# 364 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_C_LANG_SUPPORT,
# 366 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_C_LANG_SUPPORT_R,
# 368 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_CLOCK_SELECTION,
# 370 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_CPUTIME,
# 372 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_CPUTIME,
# 374 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_DEVICE_IO,
# 376 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_DEVICE_SPECIFIC,
# 378 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_DEVICE_SPECIFIC_R,
# 380 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_FD_MGMT,
# 382 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_FIFO,
# 384 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_PIPE,
# 386 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_FILE_ATTRIBUTES,
# 388 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_FILE_LOCKING,
# 390 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_FILE_SYSTEM,
# 392 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_MONOTONIC_CLOCK,
# 394 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_MULTI_PROCESS,
# 396 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SINGLE_PROCESS,
# 398 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_NETWORKING,
# 400 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_READER_WRITER_LOCKS,
# 402 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SPIN_LOCKS,
# 404 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_REGEXP,
# 406 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_REGEX_VERSION,
# 408 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SHELL,
# 410 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SIGNALS,
# 412 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SPAWN,
# 414 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SPORADIC_SERVER,
# 416 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_SPORADIC_SERVER,
# 418 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SYSTEM_DATABASE,
# 420 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SYSTEM_DATABASE_R,
# 422 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TIMEOUTS,
# 424 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TYPED_MEMORY_OBJECTS,
# 426 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_USER_GROUPS,
# 428 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_USER_GROUPS_R,
# 430 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_PBS,
# 432 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_PBS_ACCOUNTING,
# 434 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_PBS_LOCATE,
# 436 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_PBS_MESSAGE,
# 438 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_PBS_TRACK,
# 440 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SYMLOOP_MAX,
# 442 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_STREAMS,
# 444 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_2_PBS_CHECKPOINT,
# 447 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_V6_ILP32_OFF32,
# 449 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_V6_ILP32_OFFBIG,
# 451 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_V6_LP64_OFF64,
# 453 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_V6_LPBIG_OFFBIG,
# 456 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_HOST_NAME_MAX,
# 458 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TRACE,
# 460 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TRACE_EVENT_FILTER,
# 462 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TRACE_INHERIT,
# 464 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TRACE_LOG,
# 467 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL1_ICACHE_SIZE,
# 469 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL1_ICACHE_ASSOC,
# 471 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL1_ICACHE_LINESIZE,
# 473 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL1_DCACHE_SIZE,
# 475 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL1_DCACHE_ASSOC,
# 477 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL1_DCACHE_LINESIZE,
# 479 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL2_CACHE_SIZE,
# 481 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL2_CACHE_ASSOC,
# 483 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL2_CACHE_LINESIZE,
# 485 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL3_CACHE_SIZE,
# 487 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL3_CACHE_ASSOC,
# 489 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL3_CACHE_LINESIZE,
# 491 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL4_CACHE_SIZE,
# 493 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL4_CACHE_ASSOC,
# 495 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_LEVEL4_CACHE_LINESIZE,
# 499 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_IPV6 = 235,
# 501 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_RAW_SOCKETS,
# 504 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_V7_ILP32_OFF32,
# 506 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_V7_ILP32_OFFBIG,
# 508 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_V7_LP64_OFF64,
# 510 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_V7_LPBIG_OFFBIG,
# 513 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_SS_REPL_MAX,
# 516 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TRACE_EVENT_NAME_MAX,
# 518 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TRACE_NAME_MAX,
# 520 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TRACE_SYS_MAX,
# 522 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_TRACE_USER_EVENT_MAX,
# 525 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_XOPEN_STREAMS,
# 528 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_ROBUST_PRIO_INHERIT,
# 530 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_SC_THREAD_ROBUST_PRIO_PROTECT
# 532 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
};
# 536 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
enum {
# 537 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_PATH,
# 540 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_V6_WIDTH_RESTRICTED_ENVS,
# 544 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_GNU_LIBC_VERSION,
# 546 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_GNU_LIBPTHREAD_VERSION,
# 549 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_V5_WIDTH_RESTRICTED_ENVS,
# 553 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_V7_WIDTH_RESTRICTED_ENVS,
# 557 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_LFS_CFLAGS = 1000,
# 559 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_LFS_LDFLAGS,
# 561 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_LFS_LIBS,
# 563 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_LFS_LINTFLAGS,
# 565 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_LFS64_CFLAGS,
# 567 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_LFS64_LDFLAGS,
# 569 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_LFS64_LIBS,
# 571 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_LFS64_LINTFLAGS,
# 574 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_ILP32_OFF32_CFLAGS = 1100,
# 576 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_ILP32_OFF32_LDFLAGS,
# 578 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_ILP32_OFF32_LIBS,
# 580 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_ILP32_OFF32_LINTFLAGS,
# 582 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_ILP32_OFFBIG_CFLAGS,
# 584 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_ILP32_OFFBIG_LDFLAGS,
# 586 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_ILP32_OFFBIG_LIBS,
# 588 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_ILP32_OFFBIG_LINTFLAGS,
# 590 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_LP64_OFF64_CFLAGS,
# 592 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_LP64_OFF64_LDFLAGS,
# 594 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_LP64_OFF64_LIBS,
# 596 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_LP64_OFF64_LINTFLAGS,
# 598 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_LPBIG_OFFBIG_CFLAGS,
# 600 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_LPBIG_OFFBIG_LDFLAGS,
# 602 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_LPBIG_OFFBIG_LIBS,
# 604 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_XBS5_LPBIG_OFFBIG_LINTFLAGS,
# 607 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_ILP32_OFF32_CFLAGS,
# 609 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_ILP32_OFF32_LDFLAGS,
# 611 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_ILP32_OFF32_LIBS,
# 613 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_ILP32_OFF32_LINTFLAGS,
# 615 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_ILP32_OFFBIG_CFLAGS,
# 617 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS,
# 619 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_ILP32_OFFBIG_LIBS,
# 621 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS,
# 623 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_LP64_OFF64_CFLAGS,
# 625 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_LP64_OFF64_LDFLAGS,
# 627 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_LP64_OFF64_LIBS,
# 629 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_LP64_OFF64_LINTFLAGS,
# 631 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS,
# 633 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS,
# 635 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_LPBIG_OFFBIG_LIBS,
# 637 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS,
# 640 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_ILP32_OFF32_CFLAGS,
# 642 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_ILP32_OFF32_LDFLAGS,
# 644 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_ILP32_OFF32_LIBS,
# 646 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_ILP32_OFF32_LINTFLAGS,
# 648 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_ILP32_OFFBIG_CFLAGS,
# 650 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_ILP32_OFFBIG_LDFLAGS,
# 652 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_ILP32_OFFBIG_LIBS,
# 654 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_ILP32_OFFBIG_LINTFLAGS,
# 656 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_LP64_OFF64_CFLAGS,
# 658 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_LP64_OFF64_LDFLAGS,
# 660 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_LP64_OFF64_LIBS,
# 662 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_LP64_OFF64_LINTFLAGS,
# 664 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_LPBIG_OFFBIG_CFLAGS,
# 666 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_LPBIG_OFFBIG_LDFLAGS,
# 668 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_LPBIG_OFFBIG_LIBS,
# 670 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_POSIX_V7_LPBIG_OFFBIG_LINTFLAGS,
# 673 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_V6_ENV,
# 675 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
_CS_V7_ENV
# 677 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
};
# 609 "/usr/include/unistd.h" 3
extern "C" long pathconf(const char *, int) throw() __attribute__((nonnull(1)));
# 613 "/usr/include/unistd.h" 3
extern "C" long fpathconf(int, int) throw();
# 616 "/usr/include/unistd.h" 3
extern "C" long sysconf(int) throw();
# 620 "/usr/include/unistd.h" 3
extern "C" size_t confstr(int, char *, size_t) throw();
# 625 "/usr/include/unistd.h" 3
extern "C" __pid_t getpid() throw();
# 628 "/usr/include/unistd.h" 3
extern "C" __pid_t getppid() throw();
# 633 "/usr/include/unistd.h" 3
extern "C" __pid_t getpgrp() throw();
# 643 "/usr/include/unistd.h" 3
extern "C" __pid_t __getpgid(__pid_t) throw();
# 645 "/usr/include/unistd.h" 3
extern "C" __pid_t getpgid(__pid_t) throw();
# 652 "/usr/include/unistd.h" 3
extern "C" int setpgid(__pid_t, __pid_t) throw();
# 669 "/usr/include/unistd.h" 3
extern "C" int setpgrp() throw();
# 686 "/usr/include/unistd.h" 3
extern "C" __pid_t setsid() throw();
# 690 "/usr/include/unistd.h" 3
extern "C" __pid_t getsid(__pid_t) throw();
# 694 "/usr/include/unistd.h" 3
extern "C" __uid_t getuid() throw();
# 697 "/usr/include/unistd.h" 3
extern "C" __uid_t geteuid() throw();
# 700 "/usr/include/unistd.h" 3
extern "C" __gid_t getgid() throw();
# 703 "/usr/include/unistd.h" 3
extern "C" __gid_t getegid() throw();
# 708 "/usr/include/unistd.h" 3
extern "C" int getgroups(int, __gid_t []) throw();
# 712 "/usr/include/unistd.h" 3
extern "C" int group_member(__gid_t) throw();
# 719 "/usr/include/unistd.h" 3
extern "C" int setuid(__uid_t) throw();
# 724 "/usr/include/unistd.h" 3
extern "C" int setreuid(__uid_t, __uid_t) throw();
# 729 "/usr/include/unistd.h" 3
extern "C" int seteuid(__uid_t) throw();
# 736 "/usr/include/unistd.h" 3
extern "C" int setgid(__gid_t) throw();
# 741 "/usr/include/unistd.h" 3
extern "C" int setregid(__gid_t, __gid_t) throw();
# 746 "/usr/include/unistd.h" 3
extern "C" int setegid(__gid_t) throw();
# 752 "/usr/include/unistd.h" 3
extern "C" int getresuid(__uid_t *, __uid_t *, __uid_t *) throw();
# 757 "/usr/include/unistd.h" 3
extern "C" int getresgid(__gid_t *, __gid_t *, __gid_t *) throw();
# 762 "/usr/include/unistd.h" 3
extern "C" int setresuid(__uid_t, __uid_t, __uid_t) throw();
# 767 "/usr/include/unistd.h" 3
extern "C" int setresgid(__gid_t, __gid_t, __gid_t) throw();
# 775 "/usr/include/unistd.h" 3
extern "C" __pid_t fork() throw();
# 783 "/usr/include/unistd.h" 3
extern "C" __pid_t vfork() throw();
# 789 "/usr/include/unistd.h" 3
extern "C" char *ttyname(int) throw();
# 793 "/usr/include/unistd.h" 3
extern "C" int ttyname_r(int, char *, size_t) throw() __attribute__((nonnull(2)));
# 798 "/usr/include/unistd.h" 3
extern "C" int isatty(int) throw();
# 804 "/usr/include/unistd.h" 3
extern "C" int ttyslot() throw();
# 809 "/usr/include/unistd.h" 3
extern "C" int link(const char *, const char *) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 815 "/usr/include/unistd.h" 3
extern "C" int linkat(int, const char *, int, const char *, int) throw() __attribute__((nonnull(2))) __attribute__((nonnull(4)));
# 822 "/usr/include/unistd.h" 3
extern "C" int symlink(const char *, const char *) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 828 "/usr/include/unistd.h" 3
extern "C" ssize_t readlink(const char *__restrict__, char *__restrict__, size_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 835 "/usr/include/unistd.h" 3
extern "C" int symlinkat(const char *, int, const char *) throw() __attribute__((nonnull(1))) __attribute__((nonnull(3)));
# 839 "/usr/include/unistd.h" 3
extern "C" ssize_t readlinkat(int, const char *__restrict__, char *__restrict__, size_t) throw() __attribute__((nonnull(2))) __attribute__((nonnull(3)));
# 845 "/usr/include/unistd.h" 3
extern "C" int unlink(const char *) throw() __attribute__((nonnull(1)));
# 849 "/usr/include/unistd.h" 3
extern "C" int unlinkat(int, const char *, int) throw() __attribute__((nonnull(2)));
# 854 "/usr/include/unistd.h" 3
extern "C" int rmdir(const char *) throw() __attribute__((nonnull(1)));
# 858 "/usr/include/unistd.h" 3
extern "C" __pid_t tcgetpgrp(int) throw();
# 861 "/usr/include/unistd.h" 3
extern "C" int tcsetpgrp(int, __pid_t) throw();
# 868 "/usr/include/unistd.h" 3
extern "C" char *getlogin();
# 876 "/usr/include/unistd.h" 3
extern "C" int getlogin_r(char *, size_t) __attribute__((nonnull(1)));
# 881 "/usr/include/unistd.h" 3
extern "C" int setlogin(const char *) throw() __attribute__((nonnull(1)));
# 59 "/usr/include/getopt.h" 3
extern "C" { extern char *optarg; }
# 73 "/usr/include/getopt.h" 3
extern "C" { extern int optind; }
# 78 "/usr/include/getopt.h" 3
extern "C" { extern int opterr; }
# 82 "/usr/include/getopt.h" 3
extern "C" { extern int optopt; }
# 152 "/usr/include/getopt.h" 3
extern "C" int getopt(int, char *const *, const char *) throw();
# 898 "/usr/include/unistd.h" 3
extern "C" int gethostname(char *, size_t) throw() __attribute__((nonnull(1)));
# 905 "/usr/include/unistd.h" 3
extern "C" int sethostname(const char *, size_t) throw() __attribute__((nonnull(1)));
# 910 "/usr/include/unistd.h" 3
extern "C" int sethostid(long) throw();
# 916 "/usr/include/unistd.h" 3
extern "C" int getdomainname(char *, size_t) throw() __attribute__((nonnull(1)));
# 918 "/usr/include/unistd.h" 3
extern "C" int setdomainname(const char *, size_t) throw() __attribute__((nonnull(1)));
# 925 "/usr/include/unistd.h" 3
extern "C" int vhangup() throw();
# 928 "/usr/include/unistd.h" 3
extern "C" int revoke(const char *) throw() __attribute__((nonnull(1)));
# 936 "/usr/include/unistd.h" 3
extern "C" int profil(unsigned short *, size_t, size_t, unsigned) throw() __attribute__((nonnull(1)));
# 944 "/usr/include/unistd.h" 3
extern "C" int acct(const char *) throw();
# 948 "/usr/include/unistd.h" 3
extern "C" char *getusershell() throw();
# 949 "/usr/include/unistd.h" 3
extern "C" void endusershell() throw();
# 950 "/usr/include/unistd.h" 3
extern "C" void setusershell() throw();
# 956 "/usr/include/unistd.h" 3
extern "C" int daemon(int, int) throw();
# 963 "/usr/include/unistd.h" 3
extern "C" int chroot(const char *) throw() __attribute__((nonnull(1)));
# 967 "/usr/include/unistd.h" 3
extern "C" char *getpass(const char *) __attribute__((nonnull(1)));
# 976 "/usr/include/unistd.h" 3
extern "C" int fsync(int);
# 983 "/usr/include/unistd.h" 3
extern "C" long gethostid();
# 986 "/usr/include/unistd.h" 3
extern "C" void sync() throw();
# 992 "/usr/include/unistd.h" 3
extern "C" int getpagesize() throw() __attribute__((__const__));
# 997 "/usr/include/unistd.h" 3
extern "C" int getdtablesize() throw();
# 1007 "/usr/include/unistd.h" 3
extern "C" int truncate(const char *, __off_t) throw() __attribute__((nonnull(1)));
# 1019 "/usr/include/unistd.h" 3
extern "C" int truncate64(const char *, __off64_t) throw() __attribute__((nonnull(1)));
# 1029 "/usr/include/unistd.h" 3
extern "C" int ftruncate(int, __off_t) throw();
# 1039 "/usr/include/unistd.h" 3
extern "C" int ftruncate64(int, __off64_t) throw();
# 1050 "/usr/include/unistd.h" 3
extern "C" int brk(void *) throw();
# 1056 "/usr/include/unistd.h" 3
extern "C" void *sbrk(intptr_t) throw();
# 1071 "/usr/include/unistd.h" 3
extern "C" long syscall(long, ...) throw();
# 1094 "/usr/include/unistd.h" 3
extern "C" int lockf(int, int, __off_t);
# 1104 "/usr/include/unistd.h" 3
extern "C" int lockf64(int, int, __off64_t);
# 1125 "/usr/include/unistd.h" 3
extern "C" int fdatasync(int);
# 1133 "/usr/include/unistd.h" 3
extern "C" char *crypt(const char *, const char *) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 1138 "/usr/include/unistd.h" 3
extern "C" void encrypt(char *, int) throw() __attribute__((nonnull(1)));
# 1145 "/usr/include/unistd.h" 3
extern "C" void swab(const void *__restrict__, void *__restrict__, ssize_t) throw() __attribute__((nonnull(1))) __attribute__((nonnull(2)));
# 1154 "/usr/include/unistd.h" 3
extern "C" char *ctermid(char *) throw();
# 15 "../common/parboil.h"
extern "C" { struct pb_Parameters {
# 16 "../common/parboil.h"
char *outFile;
# 19 "../common/parboil.h"
char **inpFiles;
# 23 "../common/parboil.h"
int synchronizeGpu;
# 30 "../common/parboil.h"
}; }
# 42 "../common/parboil.h"
extern "C" pb_Parameters *pb_ReadParameters(int *, char **);
# 47 "../common/parboil.h"
extern "C" void pb_FreeParameters(pb_Parameters *);
# 52 "../common/parboil.h"
extern "C" int pb_Parameters_CountInputs(pb_Parameters *);
# 56 "../common/parboil.h"
extern "C" { typedef unsigned long long pb_Timestamp; }
# 61 "../common/parboil.h"
enum pb_TimerState {
# 62 "../common/parboil.h"
pb_Timer_STOPPED,
# 63 "../common/parboil.h"
pb_Timer_RUNNING
# 64 "../common/parboil.h"
};
# 66 "../common/parboil.h"
extern "C" { struct pb_Timer {
# 67 "../common/parboil.h"
pb_TimerState state;
# 68 "../common/parboil.h"
pb_Timestamp elapsed;
# 69 "../common/parboil.h"
pb_Timestamp init;
# 72 "../common/parboil.h"
}; }
# 79 "../common/parboil.h"
extern "C" void pb_ResetTimer(pb_Timer *);
# 87 "../common/parboil.h"
extern "C" void pb_StartTimer(pb_Timer *);
# 94 "../common/parboil.h"
extern "C" void pb_StopTimer(pb_Timer *);
# 98 "../common/parboil.h"
extern "C" double pb_GetElapsedTime(pb_Timer *);
# 101 "../common/parboil.h"
enum pb_TimerID {
# 102 "../common/parboil.h"
pb_TimerID_NONE,
# 103 "../common/parboil.h"
pb_TimerID_IO,
# 104 "../common/parboil.h"
pb_TimerID_GPU,
# 105 "../common/parboil.h"
pb_TimerID_COPY,
# 107 "../common/parboil.h"
pb_TimerID_COMPUTE,
# 110 "../common/parboil.h"
pb_TimerID_LAST
# 111 "../common/parboil.h"
};
# 114 "../common/parboil.h"
extern "C" { struct pb_TimerSet {
# 115 "../common/parboil.h"
pb_TimerID current;
# 116 "../common/parboil.h"
pb_Timer timers[pb_TimerID_LAST];
# 117 "../common/parboil.h"
}; }
# 121 "../common/parboil.h"
extern "C" void pb_InitializeTimerSet(pb_TimerSet *);
# 127 "../common/parboil.h"
extern "C" void pb_SwitchToTimer(pb_TimerSet *, pb_TimerID);
# 131 "../common/parboil.h"
extern "C" void pb_PrintTimerSet(pb_TimerSet *);
# 22 "UDTypes.h"
typedef
# 12 "UDTypes.h"
struct {
# 13 "UDTypes.h"
int numSamples;
# 14 "UDTypes.h"
int aquisitionMatrixSize[3];
# 15 "UDTypes.h"
int reconstructionMatrixSize[3];
# 16 "UDTypes.h"
float kMax[3];
# 17 "UDTypes.h"
int gridSize[3];
# 18 "UDTypes.h"
float oversample;
# 19 "UDTypes.h"
float kernelWidth;
# 20 "UDTypes.h"
int binsize;
# 21 "UDTypes.h"
int useLUT;
# 22 "UDTypes.h"
} parameters;
# 31 "UDTypes.h"
typedef
# 24 "UDTypes.h"
struct {
# 25 "UDTypes.h"
float real;
# 26 "UDTypes.h"
float imag;
# 27 "UDTypes.h"
float kX;
# 28 "UDTypes.h"
float kY;
# 29 "UDTypes.h"
float kZ;
# 30 "UDTypes.h"
float sdc;
# 31 "UDTypes.h"
} ReconstructionSample;
# 36 "UDTypes.h"
typedef
# 33 "UDTypes.h"
struct {
# 34 "UDTypes.h"
float real;
# 35 "UDTypes.h"
float imag;
# 36 "UDTypes.h"
} cmplx;
# 9 "scanLargeArray.h"
void scanLargeArray(unsigned, unsigned *);
# 128 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef unsigned long long CUdeviceptr; }
# 135 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef int CUdevice; }
# 136 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef struct CUctx_st *CUcontext; }
# 137 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef struct CUmod_st *CUmodule; }
# 138 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef struct CUfunc_st *CUfunction; }
# 139 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef struct CUarray_st *CUarray; }
# 140 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef struct CUtexref_st *CUtexref; }
# 141 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef struct CUsurfref_st *CUsurfref; }
# 142 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef CUevent_st *CUevent; }
# 143 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef CUstream_st *CUstream; }
# 144 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef struct CUgraphicsResource_st *CUgraphicsResource; }
# 148 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 146 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
struct CUuuid_st {
# 147 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
char bytes[16];
# 148 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUuuid; }
# 162 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 153 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUctx_flags_enum {
# 154 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CTX_SCHED_AUTO,
# 155 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CTX_SCHED_SPIN,
# 156 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CTX_SCHED_YIELD,
# 157 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CTX_SCHED_MASK,
# 158 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CTX_BLOCKING_SYNC,
# 159 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CTX_MAP_HOST = 8,
# 160 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CTX_LMEM_RESIZE_TO_MAX = 16,
# 161 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CTX_FLAGS_MASK = 31
# 162 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUctx_flags; }
# 171 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 167 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUevent_flags_enum {
# 168 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_EVENT_DEFAULT,
# 169 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_EVENT_BLOCKING_SYNC,
# 170 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_EVENT_DISABLE_TIMING
# 171 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUevent_flags; }
# 185 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 176 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUarray_format_enum {
# 177 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_AD_FORMAT_UNSIGNED_INT8 = 1,
# 178 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_AD_FORMAT_UNSIGNED_INT16,
# 179 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_AD_FORMAT_UNSIGNED_INT32,
# 180 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_AD_FORMAT_SIGNED_INT8 = 8,
# 181 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_AD_FORMAT_SIGNED_INT16,
# 182 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_AD_FORMAT_SIGNED_INT32,
# 183 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_AD_FORMAT_HALF = 16,
# 184 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_AD_FORMAT_FLOAT = 32
# 185 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUarray_format; }
# 195 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 190 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUaddress_mode_enum {
# 191 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_TR_ADDRESS_MODE_WRAP,
# 192 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_TR_ADDRESS_MODE_CLAMP,
# 193 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_TR_ADDRESS_MODE_MIRROR,
# 194 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_TR_ADDRESS_MODE_BORDER
# 195 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUaddress_mode; }
# 203 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 200 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUfilter_mode_enum {
# 201 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_TR_FILTER_MODE_POINT,
# 202 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_TR_FILTER_MODE_LINEAR
# 203 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUfilter_mode; }
# 246 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 208 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUdevice_attribute_enum {
# 209 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
# 210 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
# 211 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
# 212 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
# 213 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
# 214 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
# 215 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
# 216 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
# 217 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
# 218 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
# 219 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_WARP_SIZE,
# 220 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_PITCH,
# 221 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
# 222 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
# 223 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
# 224 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
# 225 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
# 226 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
# 227 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
# 228 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_INTEGRATED,
# 229 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
# 230 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
# 231 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
# 232 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
# 233 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
# 234 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
# 235 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
# 236 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
# 237 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH,
# 238 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT,
# 239 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES,
# 240 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
# 241 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
# 242 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
# 243 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
# 244 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
# 245 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_DEVICE_ATTRIBUTE_TCC_DRIVER
# 246 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUdevice_attribute; }
# 262 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 251 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
struct CUdevprop_st {
# 252 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
int maxThreadsPerBlock;
# 253 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
int maxThreadsDim[3];
# 254 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
int maxGridSize[3];
# 255 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
int sharedMemPerBlock;
# 256 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
int totalConstantMemory;
# 257 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
int SIMDWidth;
# 258 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
int memPitch;
# 259 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
int regsPerBlock;
# 260 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
int clockRate;
# 261 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
int textureAlign;
# 262 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUdevprop; }
# 317 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 267 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUfunction_attribute_enum {
# 273 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
# 280 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
# 286 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
# 291 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
# 296 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_FUNC_ATTRIBUTE_NUM_REGS,
# 305 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_FUNC_ATTRIBUTE_PTX_VERSION,
# 314 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_FUNC_ATTRIBUTE_BINARY_VERSION,
# 316 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_FUNC_ATTRIBUTE_MAX
# 317 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUfunction_attribute; }
# 326 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 322 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUfunc_cache_enum {
# 323 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_FUNC_CACHE_PREFER_NONE,
# 324 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_FUNC_CACHE_PREFER_SHARED,
# 325 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_FUNC_CACHE_PREFER_L1
# 326 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUfunc_cache; }
# 335 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 331 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUmemorytype_enum {
# 332 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_MEMORYTYPE_HOST = 1,
# 333 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_MEMORYTYPE_DEVICE,
# 334 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_MEMORYTYPE_ARRAY
# 335 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUmemorytype; }
# 344 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 340 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUcomputemode_enum {
# 341 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_COMPUTEMODE_DEFAULT,
# 342 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_COMPUTEMODE_EXCLUSIVE,
# 343 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_COMPUTEMODE_PROHIBITED
# 344 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUcomputemode; }
# 436 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 349 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUjit_option_enum {
# 355 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_JIT_MAX_REGISTERS,
# 368 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_JIT_THREADS_PER_BLOCK,
# 375 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_JIT_WALL_TIME,
# 383 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_JIT_INFO_LOG_BUFFER,
# 391 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
# 399 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_JIT_ERROR_LOG_BUFFER,
# 407 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
# 414 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_JIT_OPTIMIZATION_LEVEL,
# 421 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_JIT_TARGET_FROM_CUCONTEXT,
# 427 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_JIT_TARGET,
# 434 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_JIT_FALLBACK_STRATEGY
# 436 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUjit_option; }
# 449 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 441 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUjit_target_enum {
# 443 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_TARGET_COMPUTE_10,
# 444 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_TARGET_COMPUTE_11,
# 445 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_TARGET_COMPUTE_12,
# 446 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_TARGET_COMPUTE_13,
# 447 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_TARGET_COMPUTE_20,
# 448 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_TARGET_COMPUTE_21
# 449 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUjit_target; }
# 460 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 454 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUjit_fallback_enum {
# 456 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_PREFER_PTX,
# 458 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_PREFER_BINARY
# 460 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUjit_fallback; }
# 467 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 465 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUgraphicsRegisterFlags_enum {
# 466 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_NONE
# 467 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUgraphicsRegisterFlags; }
# 476 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 472 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUgraphicsMapResourceFlags_enum {
# 473 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE,
# 474 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY,
# 475 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD
# 476 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUgraphicsMapResourceFlags; }
# 488 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 481 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUarray_cubemap_face_enum {
# 482 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CUBEMAP_FACE_POSITIVE_X,
# 483 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CUBEMAP_FACE_NEGATIVE_X,
# 484 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CUBEMAP_FACE_POSITIVE_Y,
# 485 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CUBEMAP_FACE_NEGATIVE_Y,
# 486 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CUBEMAP_FACE_POSITIVE_Z,
# 487 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_CUBEMAP_FACE_NEGATIVE_Z
# 488 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUarray_cubemap_face; }
# 497 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 493 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum CUlimit_enum {
# 494 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_LIMIT_STACK_SIZE,
# 495 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_LIMIT_PRINTF_FIFO_SIZE,
# 496 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CU_LIMIT_MALLOC_HEAP_SIZE
# 497 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUlimit; }
# 728 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 502 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
enum cudaError_enum {
# 508 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_SUCCESS,
# 514 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_INVALID_VALUE,
# 520 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_OUT_OF_MEMORY,
# 526 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_NOT_INITIALIZED,
# 531 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_DEINITIALIZED,
# 538 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_NO_DEVICE = 100,
# 544 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_INVALID_DEVICE,
# 551 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_INVALID_IMAGE = 200,
# 561 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_INVALID_CONTEXT,
# 570 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
# 575 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_MAP_FAILED = 205,
# 580 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_UNMAP_FAILED,
# 586 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_ARRAY_IS_MAPPED,
# 591 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_ALREADY_MAPPED,
# 599 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_NO_BINARY_FOR_GPU,
# 604 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_ALREADY_ACQUIRED,
# 609 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_NOT_MAPPED,
# 615 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
# 621 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_NOT_MAPPED_AS_POINTER,
# 627 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_ECC_UNCORRECTABLE,
# 633 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_UNSUPPORTED_LIMIT,
# 639 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_INVALID_SOURCE = 300,
# 644 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_FILE_NOT_FOUND,
# 649 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
# 654 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
# 659 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_OPERATING_SYSTEM,
# 666 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_INVALID_HANDLE = 400,
# 673 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_NOT_FOUND = 500,
# 682 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_NOT_READY = 600,
# 693 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_LAUNCH_FAILED = 700,
# 704 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
# 715 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_LAUNCH_TIMEOUT,
# 721 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
# 727 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUDA_ERROR_UNKNOWN = 999
# 728 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUresult; }
# 777 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 756 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
struct CUDA_MEMCPY2D_st {
# 757 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t srcXInBytes;
# 758 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t srcY;
# 760 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUmemorytype srcMemoryType;
# 761 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
const void *srcHost;
# 762 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUdeviceptr srcDevice;
# 763 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUarray srcArray;
# 764 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t srcPitch;
# 766 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t dstXInBytes;
# 767 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t dstY;
# 769 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUmemorytype dstMemoryType;
# 770 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
void *dstHost;
# 771 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUdeviceptr dstDevice;
# 772 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUarray dstArray;
# 773 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t dstPitch;
# 775 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t WidthInBytes;
# 776 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t Height;
# 777 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUDA_MEMCPY2D; }
# 810 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 782 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
struct CUDA_MEMCPY3D_st {
# 783 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t srcXInBytes;
# 784 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t srcY;
# 785 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t srcZ;
# 786 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t srcLOD;
# 787 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUmemorytype srcMemoryType;
# 788 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
const void *srcHost;
# 789 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUdeviceptr srcDevice;
# 790 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUarray srcArray;
# 791 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
void *reserved0;
# 792 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t srcPitch;
# 793 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t srcHeight;
# 795 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t dstXInBytes;
# 796 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t dstY;
# 797 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t dstZ;
# 798 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t dstLOD;
# 799 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUmemorytype dstMemoryType;
# 800 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
void *dstHost;
# 801 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUdeviceptr dstDevice;
# 802 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUarray dstArray;
# 803 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
void *reserved1;
# 804 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t dstPitch;
# 805 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t dstHeight;
# 807 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t WidthInBytes;
# 808 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t Height;
# 809 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t Depth;
# 810 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUDA_MEMCPY3D; }
# 822 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 815 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
struct CUDA_ARRAY_DESCRIPTOR_st {
# 817 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t Width;
# 818 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t Height;
# 820 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUarray_format Format;
# 821 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
unsigned NumChannels;
# 822 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUDA_ARRAY_DESCRIPTOR; }
# 836 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" { typedef
# 827 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
struct CUDA_ARRAY3D_DESCRIPTOR_st {
# 829 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t Width;
# 830 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t Height;
# 831 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
size_t Depth;
# 833 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
CUarray_format Format;
# 834 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
unsigned NumChannels;
# 835 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
unsigned Flags;
# 836 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
} CUDA_ARRAY3D_DESCRIPTOR; }
# 917 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuInit(unsigned);
# 944 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuDriverGetVersion(int *);
# 982 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuDeviceGet(CUdevice *, int);
# 1008 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuDeviceGetCount(int *);
# 1037 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuDeviceGetName(char *, int, CUdevice);
# 1066 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuDeviceComputeCapability(int *, int *, CUdevice);
# 1094 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuDeviceTotalMem_v2(size_t *, CUdevice);
# 1154 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuDeviceGetProperties(CUdevprop *, CUdevice);
# 1235 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuDeviceGetAttribute(int *, CUdevice_attribute, CUdevice);
# 1333 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
# 1367 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxDestroy(CUcontext);
# 1403 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxAttach(CUcontext *, unsigned);
# 1435 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxDetach(CUcontext);
# 1473 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxPushCurrent(CUcontext);
# 1515 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxPopCurrent(CUcontext *);
# 1545 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxGetDevice(CUdevice *);
# 1575 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxSynchronize();
# 1638 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxSetLimit(CUlimit, size_t);
# 1673 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxGetLimit(size_t *, CUlimit);
# 1716 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxGetCacheConfig(CUfunc_cache *);
# 1766 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxSetCacheConfig(CUfunc_cache);
# 1803 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuCtxGetApiVersion(CUcontext, unsigned *);
# 1851 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuModuleLoad(CUmodule *, const char *);
# 1885 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuModuleLoadData(CUmodule *, const void *);
# 1963 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuModuleLoadDataEx(CUmodule *, const void *, unsigned, CUjit_option *, void **);
# 2000 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuModuleLoadFatBinary(CUmodule *, const void *);
# 2025 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuModuleUnload(CUmodule);
# 2055 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *);
# 2089 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuModuleGetGlobal_v2(CUdeviceptr *, size_t *, CUmodule, const char *);
# 2123 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuModuleGetTexRef(CUtexref *, CUmodule, const char *);
# 2154 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuModuleGetSurfRef(CUsurfref *, CUmodule, const char *);
# 2197 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemGetInfo_v2(size_t *, size_t *);
# 2230 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemAlloc_v2(CUdeviceptr *, size_t);
# 2291 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemAllocPitch_v2(CUdeviceptr *, size_t *, size_t, size_t, unsigned);
# 2320 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemFree_v2(CUdeviceptr);
# 2353 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemGetAddressRange_v2(CUdeviceptr *, size_t *, CUdeviceptr);
# 2392 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemAllocHost_v2(void **, size_t);
# 2422 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemFreeHost(void *);
# 2494 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemHostAlloc(void **, size_t, unsigned);
# 2532 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *, void *, unsigned);
# 2557 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemHostGetFlags(unsigned *, void *);
# 2591 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
# 2624 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t);
# 2657 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyDtoD_v2(CUdeviceptr, CUdeviceptr, size_t);
# 2691 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyDtoA_v2(CUarray, size_t, CUdeviceptr, size_t);
# 2727 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyAtoD_v2(CUdeviceptr, CUarray, size_t, size_t);
# 2761 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyHtoA_v2(CUarray, size_t, const void *, size_t);
# 2795 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyAtoH_v2(void *, CUarray, size_t, size_t);
# 2833 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyAtoA_v2(CUarray, size_t, CUarray, size_t, size_t);
# 2978 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *);
# 3121 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *);
# 3273 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *);
# 3313 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr, const void *, size_t, CUstream);
# 3353 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyDtoHAsync_v2(void *, CUdeviceptr, size_t, CUstream);
# 3390 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr, CUdeviceptr, size_t, CUstream);
# 3432 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyHtoAAsync_v2(CUarray, size_t, const void *, size_t, CUstream);
# 3474 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpyAtoHAsync_v2(void *, CUarray, size_t, size_t, CUstream);
# 3630 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *, CUstream);
# 3790 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *, CUstream);
# 3823 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemsetD8_v2(CUdeviceptr, unsigned char, size_t);
# 3856 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemsetD16_v2(CUdeviceptr, unsigned short, size_t);
# 3889 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemsetD32_v2(CUdeviceptr, unsigned, size_t);
# 3927 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemsetD2D8_v2(CUdeviceptr, size_t, unsigned char, size_t, size_t);
# 3965 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemsetD2D16_v2(CUdeviceptr, size_t, unsigned short, size_t, size_t);
# 4003 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemsetD2D32_v2(CUdeviceptr, size_t, unsigned, size_t, size_t);
# 4040 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemsetD8Async(CUdeviceptr, unsigned char, size_t, CUstream);
# 4077 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemsetD16Async(CUdeviceptr, unsigned short, size_t, CUstream);
# 4113 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemsetD32Async(CUdeviceptr, unsigned, size_t, CUstream);
# 4155 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemsetD2D8Async(CUdeviceptr, size_t, unsigned char, size_t, size_t, CUstream);
# 4197 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemsetD2D16Async(CUdeviceptr, size_t, unsigned short, size_t, size_t, CUstream);
# 4239 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuMemsetD2D32Async(CUdeviceptr, size_t, unsigned, size_t, size_t, CUstream);
# 4342 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuArrayCreate_v2(CUarray *, const CUDA_ARRAY_DESCRIPTOR *);
# 4375 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *, CUarray);
# 4406 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuArrayDestroy(CUarray);
# 4508 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuArray3DCreate_v2(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
# 4544 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *, CUarray);
# 4582 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuStreamCreate(CUstream *, unsigned);
# 4622 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuStreamWaitEvent(CUstream, CUevent, unsigned);
# 4646 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuStreamQuery(CUstream);
# 4671 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuStreamSynchronize(CUstream);
# 4693 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuStreamDestroy(CUstream);
# 4741 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuEventCreate(CUevent *, unsigned);
# 4777 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuEventRecord(CUevent, CUstream);
# 4809 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuEventQuery(CUevent);
# 4843 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuEventSynchronize(CUevent);
# 4866 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuEventDestroy(CUevent);
# 4909 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuEventElapsedTime(float *, CUevent, CUevent);
# 4954 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuFuncSetBlockShape(CUfunction, int, int, int);
# 4985 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuFuncSetSharedSize(CUfunction, unsigned);
# 5042 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuFuncGetAttribute(int *, CUfunction_attribute, CUfunction);
# 5090 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuFuncSetCacheConfig(CUfunction, CUfunc_cache);
# 5119 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuParamSetSize(CUfunction, unsigned);
# 5149 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuParamSeti(CUfunction, int, unsigned);
# 5179 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuParamSetf(CUfunction, int, float);
# 5211 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuParamSetv(CUfunction, int, void *, unsigned);
# 5245 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuLaunch(CUfunction);
# 5281 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuLaunchGrid(CUfunction, int, int);
# 5321 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuLaunchGridAsync(CUfunction, int, int, CUstream);
# 5354 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuParamSetTexRef(CUfunction, int, CUtexref);
# 5395 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefSetArray(CUtexref, CUarray, unsigned);
# 5433 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefSetAddress_v2(size_t *, CUtexref, CUdeviceptr, size_t);
# 5474 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefSetAddress2D_v2(CUtexref, const CUDA_ARRAY_DESCRIPTOR *, CUdeviceptr, size_t);
# 5503 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefSetFormat(CUtexref, CUarray_format, int);
# 5541 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefSetAddressMode(CUtexref, int, CUaddress_mode);
# 5574 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefSetFilterMode(CUtexref, CUfilter_mode);
# 5606 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefSetFlags(CUtexref, unsigned);
# 5632 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefGetAddress_v2(CUdeviceptr *, CUtexref);
# 5658 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefGetArray(CUarray *, CUtexref);
# 5684 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefGetAddressMode(CUaddress_mode *, CUtexref, int);
# 5708 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefGetFilterMode(CUfilter_mode *, CUtexref);
# 5734 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefGetFormat(CUarray_format *, int *, CUtexref);
# 5757 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefGetFlags(unsigned *, CUtexref);
# 5791 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefCreate(CUtexref *);
# 5811 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuTexRefDestroy(CUtexref);
# 5849 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuSurfRefSetArray(CUsurfref, CUarray, unsigned);
# 5870 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuSurfRefGetArray(CUarray *, CUsurfref);
# 5911 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuGraphicsUnregisterResource(CUgraphicsResource);
# 5949 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuGraphicsSubResourceGetMappedArray(CUarray *, CUgraphicsResource, unsigned, unsigned);
# 5983 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *, size_t *, CUgraphicsResource);
# 6024 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource, unsigned);
# 6062 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuGraphicsMapResources(unsigned, CUgraphicsResource *, CUstream);
# 6097 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuGraphicsUnmapResources(unsigned, CUgraphicsResource *, CUstream);
# 6101 "/home/bachelor/deicide218/cuda-3.2/bin/../include/cuda.h"
extern "C" CUresult cuGetExportTable(const void **, const CUuuid *);
# 24 "GPU_kernels.cu"
typedef
# 21 "GPU_kernels.cu"
struct {
# 22 "GPU_kernels.cu"
float2 *data;
# 23 "GPU_kernels.cu"
float4 *loc;
# 24 "GPU_kernels.cu"
} sampleArrayStruct;
# 26 "GPU_kernels.cu"
static float cutoff2_c;
# 27 "GPU_kernels.cu"
static float cutoff_c;
# 28 "GPU_kernels.cu"
static int gridSize_c[3];
# 29 "GPU_kernels.cu"
static int size_xy_c;
# 30 "GPU_kernels.cu"
static float _1overCutoff2_c;
# 32 "GPU_kernels.cu"
void binning_kernel(unsigned n, ReconstructionSample *sample_g, unsigned *idxKey_g, unsigned *
# 33 "GPU_kernels.cu"
idxValue_g, unsigned *binCount_g, unsigned binsize, unsigned gridNumElems) ;
# 61 "GPU_kernels.cu"
void reorder_kernel(int n, unsigned *idxValue_g, ReconstructionSample *samples_g, sampleArrayStruct sortedSampleSoA_g) ;
# 85 "GPU_kernels.cu"
__attribute__((unused)) float kernel_value(float v) {int volatile ___ = 1;
# 107 "GPU_kernels.cu"
exit(___);}
# 109 "GPU_kernels.cu"
void gridding_GPU(sampleArrayStruct sortedSampleSoA_g, unsigned *binStartAddr_g, float2 *gridData_g, float *sampleDensity_g, float beta) ;
# 15 "CPU_kernels.h"
extern "C" void calculateLUT(float, float, float **, unsigned *);
# 17 "CPU_kernels.h"
extern "C" int gridding_Gold(unsigned, parameters, ReconstructionSample *, float *, unsigned, cmplx *, float *);
# 19 "CPU_kernels.h"
extern "C" int gridding_CPU(unsigned, parameters, ReconstructionSample *, int *, int, float *, int, cmplx *[], float *[], int *[]);
# 9 "sort.h"
void sort(int, unsigned, unsigned *&, unsigned *&);
# 9 "scanLargeArray.h"
void scanLargeArray(unsigned, unsigned *);
# 34 "CUDA_interface.cu"
int compare(const void *a, const void *b)
# 35 "CUDA_interface.cu"
{
# 36 "CUDA_interface.cu"
return (*((int *)a)) - (*((int *)b));
# 37 "CUDA_interface.cu"
}
# 51 "CUDA_interface.cu"
void CUDA_interface(pb_TimerSet *
# 52 "CUDA_interface.cu"
timers, unsigned
# 53 "CUDA_interface.cu"
n, parameters
# 54 "CUDA_interface.cu"
params, ReconstructionSample *
# 55 "CUDA_interface.cu"
sample, float *
# 56 "CUDA_interface.cu"
LUT, int
# 58 "CUDA_interface.cu"
sizeLUT, cmplx *
# 59 "CUDA_interface.cu"
gridData, float *
# 60 "CUDA_interface.cu"
sampleDensity)
# 62 "CUDA_interface.cu"
{
# 65 "CUDA_interface.cu"
dim3 dims(8, 4, 2);
# 68 "CUDA_interface.cu"
int size_x = ((params.gridSize)[0]);
# 69 "CUDA_interface.cu"
int size_y = ((params.gridSize)[1]);
# 70 "CUDA_interface.cu"
int size_z = ((params.gridSize)[2]);
# 71 "CUDA_interface.cu"
int size_xy = (size_y * size_x);
# 73 "CUDA_interface.cu"
int gridNumElems = ((size_x * size_y) * size_z);
# 75 "CUDA_interface.cu"
float beta = ((3.141592653590000062) * sqrt(((((((4) * (params.kernelWidth)) * (params.kernelWidth)) / ((params.oversample) * (params.oversample))) * ((params.oversample) - (0.5))) * ((params.oversample) - (0.5))) - (0.8000000000000000444)));
# 77 "CUDA_interface.cu"
float cutoff = (((float)(params.kernelWidth)) / (2.0));
# 78 "CUDA_interface.cu"
float cutoff2 = (cutoff * cutoff);
# 79 "CUDA_interface.cu"
float _1overCutoff2 = ((1) / cutoff2);
# 82 "CUDA_interface.cu"
int npad = 0;
# 83 "CUDA_interface.cu"
if ((n % (64)) != (0)) {
# 84 "CUDA_interface.cu"
npad = ((64) - (n % (64)));
# 85 "CUDA_interface.cu"
}
# 88 "CUDA_interface.cu"
cmplx *gridData_CPU;
# 89 "CUDA_interface.cu"
float *sampleDensity_CPU;
# 90 "CUDA_interface.cu"
int *indices_CPU;
# 93 "CUDA_interface.cu"
ReconstructionSample *sample_d = (__null);
# 94 "CUDA_interface.cu"
float *sortedSample_d = (__null);
# 97 "CUDA_interface.cu"
float2 *gridData_d = (__null);
# 98 "CUDA_interface.cu"
float *sampleDensity_d = (__null);
# 99 "CUDA_interface.cu"
unsigned *idxKey_d = (__null);
# 102 "CUDA_interface.cu"
unsigned *idxValue_d = (__null);
# 107 "CUDA_interface.cu"
sampleArrayStruct sortedSampleSoA_d;
# 110 "CUDA_interface.cu"
unsigned *binCount_d = (__null);
# 113 "CUDA_interface.cu"
unsigned *binStartAddr_d = (__null);
# 116 "CUDA_interface.cu"
pb_SwitchToTimer(timers, pb_TimerID_COPY);
# 118 "CUDA_interface.cu"
cudaMalloc((void **)(&sortedSample_d), (n + npad) * sizeof(ReconstructionSample));
# 119 "CUDA_interface.cu"
cudaMalloc((void **)(&binStartAddr_d), (gridNumElems + 1) * sizeof(unsigned));
# 120 "CUDA_interface.cu"
cudaMalloc((void **)(&sample_d), n * sizeof(ReconstructionSample));
# 121 "CUDA_interface.cu"
cudaMalloc((void **)(&idxKey_d), (((n + (3)) / (4)) * (4)) * sizeof(unsigned));
# 122 "CUDA_interface.cu"
cudaMalloc((void **)(&idxValue_d), (((n + (3)) / (4)) * (4)) * sizeof(unsigned));
# 135 "CUDA_interface.cu"
binCount_d = binStartAddr_d;
# 139 "CUDA_interface.cu"
cudaMemcpyToSymbol(cutoff2_c, &cutoff2, sizeof(float), 0);
# 140 "CUDA_interface.cu"
cudaMemcpyToSymbol(cutoff_c, &cutoff, sizeof(float), 0);
# 141 "CUDA_interface.cu"
cudaMemcpyToSymbol(gridSize_c, params.gridSize, (3) * sizeof(int), 0);
# 142 "CUDA_interface.cu"
cudaMemcpyToSymbol(size_xy_c, &size_xy, sizeof(int), 0);
# 143 "CUDA_interface.cu"
cudaMemcpyToSymbol(_1overCutoff2_c, &_1overCutoff2, sizeof(float), 0);
# 144 "CUDA_interface.cu"
cudaMemcpy(sample_d, sample, n * sizeof(ReconstructionSample), cudaMemcpyHostToDevice);
# 145 "CUDA_interface.cu"
cudaMemset(binCount_d, 0, (gridNumElems + 1) * sizeof(unsigned));
# 149 "CUDA_interface.cu"
cudaMemset(idxKey_d + n, 255, (((n + (3)) & (~3)) - n) * sizeof(unsigned));
# 151 "CUDA_interface.cu"
(sortedSampleSoA_d.data) = ((float2 *)sortedSample_d);
# 152 "CUDA_interface.cu"
(sortedSampleSoA_d.loc) = ((float4 *)(sortedSample_d + ((2) * (n + npad))));
# 154 "CUDA_interface.cu"
pb_SwitchToTimer(timers, pb_TimerID_GPU);
# 159 "CUDA_interface.cu"
dim3 block1(512);
# 160 "CUDA_interface.cu"
dim3 grid1(((n + (512)) - (1)) / (512));
# 162 "CUDA_interface.cu"
cudaConfigureCall(grid1, block1) ? ((void)0) : binning_kernel(n, sample_d, idxKey_d, idxValue_d, binCount_d, params.binsize, gridNumElems);
# 184 "CUDA_interface.cu"
sort(n, gridNumElems + 1, idxKey_d, idxValue_d);
# 195 "CUDA_interface.cu"
cudaConfigureCall(grid1, block1) ? ((void)0) : reorder_kernel(n, idxValue_d, sample_d, sortedSampleSoA_d);
# 197 "CUDA_interface.cu"
pb_SwitchToTimer(timers, pb_TimerID_COPY);
# 199 "CUDA_interface.cu"
cudaFree(idxKey_d);
# 200 "CUDA_interface.cu"
cudaFree(sample_d);
# 202 "CUDA_interface.cu"
pb_SwitchToTimer(timers, pb_TimerID_GPU);
# 220 "CUDA_interface.cu"
scanLargeArray(gridNumElems + 1, binCount_d);
# 223 "CUDA_interface.cu"
pb_SwitchToTimer(timers, pb_TimerID_COPY);
# 226 "CUDA_interface.cu"
int cpuStart;
# 227 "CUDA_interface.cu"
cudaMemcpy(&cpuStart, binCount_d + gridNumElems, sizeof(unsigned), cudaMemcpyDeviceToHost);
# 229 "CUDA_interface.cu"
int CPUbin_size = (((int)n) - ((int)cpuStart));
# 231 "CUDA_interface.cu"
int *CPUbin;
# 232 "CUDA_interface.cu"
cudaMallocHost((void **)(&CPUbin), CPUbin_size * sizeof(unsigned));
# 233 "CUDA_interface.cu"
cudaMemcpy(CPUbin, idxValue_d + cpuStart, CPUbin_size * sizeof(unsigned), cudaMemcpyDeviceToHost);
# 235 "CUDA_interface.cu"
cudaFree(idxValue_d);
# 244 "CUDA_interface.cu"
cudaMalloc((void **)(&gridData_d), gridNumElems * sizeof(float2));
# 245 "CUDA_interface.cu"
cudaMalloc((void **)(&sampleDensity_d), gridNumElems * sizeof(float));
# 247 "CUDA_interface.cu"
cudaMemset(gridData_d, 0, gridNumElems * sizeof(float2));
# 248 "CUDA_interface.cu"
cudaMemset(sampleDensity_d, 0, gridNumElems * sizeof(float));
# 250 "CUDA_interface.cu"
pb_SwitchToTimer(timers, pb_TimerID_GPU);
# 252 "CUDA_interface.cu"
dim3 block2(dims.x, dims.y, dims.z);
# 253 "CUDA_interface.cu"
dim3 grid2(size_x / (dims.x), (size_y * size_z) / (((4) * (dims.y)) * (dims.z)));
# 255 "CUDA_interface.cu"
cudaConfigureCall(grid2, block2) ? ((void)0) : gridding_GPU(sortedSampleSoA_d, binStartAddr_d, gridData_d, sampleDensity_d, beta);
# 257 "CUDA_interface.cu"
pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);
# 259 "CUDA_interface.cu"
qsort(CPUbin, CPUbin_size, sizeof(int), compare);
# 260 "CUDA_interface.cu"
int num = gridding_CPU(n, params, sample, CPUbin, CPUbin_size, LUT, sizeLUT, &gridData_CPU, &sampleDensity_CPU, &indices_CPU);
# 262 "CUDA_interface.cu"
pb_SwitchToTimer(timers, pb_TimerID_COPY);
# 265 "CUDA_interface.cu"
cudaMemcpy(sampleDensity, sampleDensity_d, gridNumElems * sizeof(float), cudaMemcpyDeviceToHost);
# 266 "CUDA_interface.cu"
cudaMemcpy(gridData, gridData_d, gridNumElems * sizeof(float2), cudaMemcpyDeviceToHost);
# 268 "CUDA_interface.cu"
pb_SwitchToTimer(timers, pb_TimerID_COMPUTE);
# 273 "CUDA_interface.cu"
for (int i = 0; i < num; i++) {
# 274 "CUDA_interface.cu"
((gridData[indices_CPU[i]]).real) += ((gridData_CPU[i]).real);
# 275 "CUDA_interface.cu"
((gridData[indices_CPU[i]]).imag) += ((gridData_CPU[i]).imag);
# 276 "CUDA_interface.cu"
(sampleDensity[indices_CPU[i]]) += (sampleDensity_CPU[i]);
# 277 "CUDA_interface.cu"
}
# 279 "CUDA_interface.cu"
if (gridData_CPU != (__null)) {
# 280 "CUDA_interface.cu"
free(indices_CPU);
# 281 "CUDA_interface.cu"
free(gridData_CPU);
# 282 "CUDA_interface.cu"
free(sampleDensity_CPU);
# 283 "CUDA_interface.cu"
}
# 285 "CUDA_interface.cu"
pb_SwitchToTimer(timers, pb_TimerID_COPY);
# 287 "CUDA_interface.cu"
cudaFreeHost(CPUbin);
# 288 "CUDA_interface.cu"
cudaFree(gridData_d);
# 289 "CUDA_interface.cu"
cudaFree(sampleDensity_d);
# 290 "CUDA_interface.cu"
cudaFree(binCount_d);
# 291 "CUDA_interface.cu"
cudaFree(sortedSample_d);
# 293 "CUDA_interface.cu"
pb_SwitchToTimer(timers, pb_TimerID_NONE);
# 296 "CUDA_interface.cu"
}
# 1 "/tmp/tmpxft_0000636c_00000000-1_CUDA_interface.cudafe1.stub.c"
# 1 "/tmp/tmpxft_0000636c_00000000-1_CUDA_interface.cudafe1.stub.c" 1
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/crt/host_runtime.h" 1
# 91 "/home/bachelor/deicide218/cuda-3.2/bin/../include/crt/host_runtime.h"
extern "C" {

extern void** __cudaRegisterFatBinary(
  void *fatCubin
);

extern void __cudaUnregisterFatBinary(
  void **fatCubinHandle
);

extern void __cudaRegisterVar(
        void **fatCubinHandle,
        char *hostVar,
        char *deviceAddress,
  const char *deviceName,
        int ext,
        int size,
        int constant,
        int global
);

extern void __cudaRegisterTexture(
        void **fatCubinHandle,
  const struct textureReference *hostVar,
  const void **deviceAddress,
  const char *deviceName,
        int dim,
        int norm,
        int ext
);

extern void __cudaRegisterSurface(
        void **fatCubinHandle,
  const struct surfaceReference *hostVar,
  const void **deviceAddress,
  const char *deviceName,
        int dim,
        int ext
);

extern void __cudaRegisterFunction(
        void **fatCubinHandle,
  const char *hostFun,
        char *deviceFun,
  const char *deviceName,
        int thread_limit,
        uint3 *tid,
        uint3 *bid,
        dim3 *bDim,
        dim3 *gDim,
        int *wSize
);



extern int atexit(void(*)(void)) throw();







}

static void **__cudaFatCubinHandle;

static void __cudaUnregisterBinaryUtil(void)
{
  __cudaUnregisterFatBinary(__cudaFatCubinHandle);
}

# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/common_functions.h" 1
# 90 "/home/bachelor/deicide218/cuda-3.2/bin/../include/common_functions.h"
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h" 1 3
# 948 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h" 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_constants.h" 1 3
# 949 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h" 2 3
# 2973 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h" 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/crt/func_macro.h" 1 3
# 2974 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h" 2 3
# 4683 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h" 3
extern __attribute__((__weak__)) double rsqrt(double a); double rsqrt(double a)
{
  return 1.0 / sqrt(a);
}

extern __attribute__((__weak__)) double rcbrt(double a); double rcbrt(double a)
{
  double s, t;

  if (__isnan(a)) {
    return a + a;
  }
  if (a == 0.0 || __isinf(a)) {
    return 1.0 / a;
  }
  s = fabs(a);
  t = exp2(-3.3333333333333333e-1 * log2(s));
  t = ((t*t) * (-s*t) + 1.0) * (3.3333333333333333e-1*t) + t;



  if (__signbit(a))

  {
    t = -t;
  }
  return t;
}

extern __attribute__((__weak__)) double sinpi(double a); double sinpi(double a)
{
  int n;

  if (__isnan(a)) {
    return a + a;
  }
  if (a == 0.0 || __isinf(a)) {
    return sin (a);
  }
  if (a == floor(a)) {
    return ((a / 1.0e308) / 1.0e308) / 1.0e308;
  }
  a = remquo (a, 0.5, &n);
  a = a * 3.1415926535897931e+0;
  if (n & 1) {
    a = cos (a);
  } else {
    a = sin (a);
  }
  if (n & 2) {
    a = -a;
  }
  return a;
}

extern __attribute__((__weak__)) double erfinv(double a); double erfinv(double a)
{
  double p, q, t, fa;
  volatile union {
    double d;
    unsigned long long int l;
  } cvt;

  fa = fabs(a);
  if (fa >= 1.0) {
    cvt.l = 0xfff8000000000000ull;
    t = cvt.d;
    if (fa == 1.0) {
      t = a * exp(1000.0);
    }
  } else if (fa >= 0.9375) {




    t = log1p(-fa);
    t = 1.0 / sqrt(-t);
    p = 2.7834010353747001060e-3;
    p = p * t + 8.6030097526280260580e-1;
    p = p * t + 2.1371214997265515515e+0;
    p = p * t + 3.1598519601132090206e+0;
    p = p * t + 3.5780402569085996758e+0;
    p = p * t + 1.5335297523989890804e+0;
    p = p * t + 3.4839207139657522572e-1;
    p = p * t + 5.3644861147153648366e-2;
    p = p * t + 4.3836709877126095665e-3;
    p = p * t + 1.3858518113496718808e-4;
    p = p * t + 1.1738352509991666680e-6;
    q = t + 2.2859981272422905412e+0;
    q = q * t + 4.3859045256449554654e+0;
    q = q * t + 4.6632960348736635331e+0;
    q = q * t + 3.9846608184671757296e+0;
    q = q * t + 1.6068377709719017609e+0;
    q = q * t + 3.5609087305900265560e-1;
    q = q * t + 5.3963550303200816744e-2;
    q = q * t + 4.3873424022706935023e-3;
    q = q * t + 1.3858762165532246059e-4;
    q = q * t + 1.1738313872397777529e-6;
    t = p / (q * t);
    if (a < 0.0) t = -t;
  } else if (fa >= 0.75) {




    t = a * a - .87890625;
    p = .21489185007307062000e+0;
    p = p * t - .64200071507209448655e+1;
    p = p * t + .29631331505876308123e+2;
    p = p * t - .47644367129787181803e+2;
    p = p * t + .34810057749357500873e+2;
    p = p * t - .12954198980646771502e+2;
    p = p * t + .25349389220714893917e+1;
    p = p * t - .24758242362823355486e+0;
    p = p * t + .94897362808681080020e-2;
    q = t - .12831383833953226499e+2;
    q = q * t + .41409991778428888716e+2;
    q = q * t - .53715373448862143349e+2;
    q = q * t + .33880176779595142685e+2;
    q = q * t - .11315360624238054876e+2;
    q = q * t + .20369295047216351160e+1;
    q = q * t - .18611650627372178511e+0;
    q = q * t + .67544512778850945940e-2;
    p = p / q;
    t = a * p;
  } else {




    t = a * a - .5625;
    p = - .23886240104308755900e+2;
    p = p * t + .45560204272689128170e+3;
    p = p * t - .22977467176607144887e+4;
    p = p * t + .46631433533434331287e+4;
    p = p * t - .43799652308386926161e+4;
    p = p * t + .19007153590528134753e+4;
    p = p * t - .30786872642313695280e+3;
    q = t - .83288327901936570000e+2;
    q = q * t + .92741319160935318800e+3;
    q = q * t - .35088976383877264098e+4;
    q = q * t + .59039348134843665626e+4;
    q = q * t - .48481635430048872102e+4;
    q = q * t + .18997769186453057810e+4;
    q = q * t - .28386514725366621129e+3;
    p = p / q;
    t = a * p;
  }
  return t;
}

extern __attribute__((__weak__)) double erfcinv(double a); double erfcinv(double a)
{
  double t;
  volatile union {
    double d;
    unsigned long long int l;
  } cvt;

  if (__isnan(a)) {
    return a + a;
  }
  if (a <= 0.0) {
    cvt.l = 0xfff8000000000000ull;
    t = cvt.d;
    if (a == 0.0) {
        t = (1.0 - a) * exp(1000.0);
    }
  }
  else if (a >= 0.0625) {
    t = erfinv (1.0 - a);
  }
  else if (a >= 1e-100) {




    double p, q;
    t = log(a);
    t = 1.0 / sqrt(-t);
    p = 2.7834010353747001060e-3;
    p = p * t + 8.6030097526280260580e-1;
    p = p * t + 2.1371214997265515515e+0;
    p = p * t + 3.1598519601132090206e+0;
    p = p * t + 3.5780402569085996758e+0;
    p = p * t + 1.5335297523989890804e+0;
    p = p * t + 3.4839207139657522572e-1;
    p = p * t + 5.3644861147153648366e-2;
    p = p * t + 4.3836709877126095665e-3;
    p = p * t + 1.3858518113496718808e-4;
    p = p * t + 1.1738352509991666680e-6;
    q = t + 2.2859981272422905412e+0;
    q = q * t + 4.3859045256449554654e+0;
    q = q * t + 4.6632960348736635331e+0;
    q = q * t + 3.9846608184671757296e+0;
    q = q * t + 1.6068377709719017609e+0;
    q = q * t + 3.5609087305900265560e-1;
    q = q * t + 5.3963550303200816744e-2;
    q = q * t + 4.3873424022706935023e-3;
    q = q * t + 1.3858762165532246059e-4;
    q = q * t + 1.1738313872397777529e-6;
    t = p / (q * t);
  }
  else {




    double p, q;
    t = log(a);
    t = 1.0 / sqrt(-t);
    p = 6.9952990607058154858e-1;
    p = p * t + 1.9507620287580568829e+0;
    p = p * t + 8.2810030904462690216e-1;
    p = p * t + 1.1279046353630280005e-1;
    p = p * t + 6.0537914739162189689e-3;
    p = p * t + 1.3714329569665128933e-4;
    p = p * t + 1.2964481560643197452e-6;
    p = p * t + 4.6156006321345332510e-9;
    p = p * t + 4.5344689563209398450e-12;
    q = t + 1.5771922386662040546e+0;
    q = q * t + 2.1238242087454993542e+0;
    q = q * t + 8.4001814918178042919e-1;
    q = q * t + 1.1311889334355782065e-1;
    q = q * t + 6.0574830550097140404e-3;
    q = q * t + 1.3715891988350205065e-4;
    q = q * t + 1.2964671850944981713e-6;
    q = q * t + 4.6156017600933592558e-9;
    q = q * t + 4.5344687377088206783e-12;
    t = p / (q * t);
  }
  return t;
}

extern __attribute__((__weak__)) float rsqrtf(float a); float rsqrtf(float a)
{
  return (float)rsqrt((double)a);
}

extern __attribute__((__weak__)) float rcbrtf(float a); float rcbrtf(float a)
{
  return (float)rcbrt((double)a);
}

extern __attribute__((__weak__)) float sinpif(float a); float sinpif(float a)
{
  return (float)sinpi((double)a);
}

extern __attribute__((__weak__)) float erfinvf(float a); float erfinvf(float a)
{
  return (float)erfinv((double)a);
}

extern __attribute__((__weak__)) float erfcinvf(float a); float erfcinvf(float a)
{
  return (float)erfcinv((double)a);
}







extern __attribute__((__weak__)) int min(int a, int b); int min(int a, int b)
{
  return a < b ? a : b;
}

extern __attribute__((__weak__)) unsigned int umin(unsigned int a, unsigned int b); unsigned int umin(unsigned int a, unsigned int b)
{
  return a < b ? a : b;
}

extern __attribute__((__weak__)) long long int llmin(long long int a, long long int b); long long int llmin(long long int a, long long int b)
{
  return a < b ? a : b;
}

extern __attribute__((__weak__)) unsigned long long int ullmin(unsigned long long int a, unsigned long long int b); unsigned long long int ullmin(unsigned long long int a, unsigned long long int b)
{
  return a < b ? a : b;
}

extern __attribute__((__weak__)) int max(int a, int b); int max(int a, int b)
{
  return a > b ? a : b;
}

extern __attribute__((__weak__)) unsigned int umax(unsigned int a, unsigned int b); unsigned int umax(unsigned int a, unsigned int b)
{
  return a > b ? a : b;
}

extern __attribute__((__weak__)) long long int llmax(long long int a, long long int b); long long int llmax(long long int a, long long int b)
{
  return a > b ? a : b;
}

extern __attribute__((__weak__)) unsigned long long int ullmax(unsigned long long int a, unsigned long long int b); unsigned long long int ullmax(unsigned long long int a, unsigned long long int b)
{
  return a > b ? a : b;
}
# 5006 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h" 3
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions_dbl_ptx3.h" 1 3
# 5007 "/home/bachelor/deicide218/cuda-3.2/bin/../include/math_functions.h" 2 3
# 91 "/home/bachelor/deicide218/cuda-3.2/bin/../include/common_functions.h" 2
# 164 "/home/bachelor/deicide218/cuda-3.2/bin/../include/crt/host_runtime.h" 2







#pragma pack()
# 2 "/tmp/tmpxft_0000636c_00000000-1_CUDA_interface.cudafe1.stub.c" 2
# 1 "/tmp/tmpxft_0000636c_00000000-3_CUDA_interface.fatbin.c" 1
# 1 "/home/bachelor/deicide218/cuda-3.2/bin/../include/__cudaFatFormat.h" 1
# 83 "/home/bachelor/deicide218/cuda-3.2/bin/../include/__cudaFatFormat.h"
extern "C" {
# 97 "/home/bachelor/deicide218/cuda-3.2/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* gpuProfileName;
    char* cubin;
} __cudaFatCubinEntry;
# 113 "/home/bachelor/deicide218/cuda-3.2/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* gpuProfileName;
    char* ptx;
} __cudaFatPtxEntry;
# 125 "/home/bachelor/deicide218/cuda-3.2/bin/../include/__cudaFatFormat.h"
typedef struct __cudaFatDebugEntryRec {
    char* gpuProfileName;
    char* debug;
    struct __cudaFatDebugEntryRec *next;
    unsigned int size;
} __cudaFatDebugEntry;

typedef struct __cudaFatElfEntryRec {
    char* gpuProfileName;
    char* elf;
    struct __cudaFatElfEntryRec *next;
    unsigned int size;
} __cudaFatElfEntry;

typedef enum {
      __cudaFatDontSearchFlag = (1 << 0),
      __cudaFatDontCacheFlag = (1 << 1),
      __cudaFatSassDebugFlag = (1 << 2)
} __cudaFatCudaBinaryFlag;
# 152 "/home/bachelor/deicide218/cuda-3.2/bin/../include/__cudaFatFormat.h"
typedef struct {
    char* name;
} __cudaFatSymbol;
# 166 "/home/bachelor/deicide218/cuda-3.2/bin/../include/__cudaFatFormat.h"
typedef struct __cudaFatCudaBinaryRec {
    unsigned long magic;
    unsigned long version;
    unsigned long gpuInfoVersion;
    char* key;
    char* ident;
    char* usageMode;
    __cudaFatPtxEntry *ptx;
    __cudaFatCubinEntry *cubin;
    __cudaFatDebugEntry *debug;
    void* debugInfo;
    unsigned int flags;
    __cudaFatSymbol *exported;
    __cudaFatSymbol *imported;
    struct __cudaFatCudaBinaryRec *dependends;
    unsigned int characteristic;
    __cudaFatElfEntry *elf;
} __cudaFatCudaBinary;
# 203 "/home/bachelor/deicide218/cuda-3.2/bin/../include/__cudaFatFormat.h"
    typedef enum {
        __cudaFatAvoidPTX,
        __cudaFatPreferBestCode,
        __cudaFatForcePTX
    } __cudaFatCompilationPolicy;
# 227 "/home/bachelor/deicide218/cuda-3.2/bin/../include/__cudaFatFormat.h"
void fatGetCubinForGpuWithPolicy( __cudaFatCudaBinary *binary, __cudaFatCompilationPolicy policy, char* gpuName, char* *cubin, char* *dbgInfoFile );
# 240 "/home/bachelor/deicide218/cuda-3.2/bin/../include/__cudaFatFormat.h"
unsigned char fatCheckJitForGpuWithPolicy( __cudaFatCudaBinary *binary, __cudaFatCompilationPolicy policy, char* gpuName, char* *ptx );
# 250 "/home/bachelor/deicide218/cuda-3.2/bin/../include/__cudaFatFormat.h"
void fatFreeCubin( char* cubin, char* dbgInfoFile );





void __cudaFatFreePTX( char* ptx );


}
# 2 "/tmp/tmpxft_0000636c_00000000-3_CUDA_interface.fatbin.c" 2

asm(
".section .rodata\n"
".align 32\n"
"__deviceText_$sm_20$:\n"
".quad 0x33010102464c457f,0x0000000000000004,0x0000000100be0002,0x0000000000000000\n"
".quad 0x00000000000020e4,0x0000000000000040,0x0038004000140114,0x0001001100400006\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000300000001,0x0000000000000000,0x0000000000000000,0x0000000000000480\n"
".quad 0x0000000000000312,0x0000000000000000,0x0000000000000004,0x0000000000000000\n"
".quad 0x000000030000000b,0x0000000000000000,0x0000000000000000,0x0000000000000792\n"
".quad 0x00000000000000f7,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x0000000200000013,0x0000000000000000,0x0000000000000000,0x0000000000000889\n"
".quad 0x00000000000002a0,0x0000001300000002,0x0000000000000001,0x0000000000000018\n"
".quad 0x000000010000013f,0x0000000000000006,0x0000000000000000,0x0000000000000b29\n"
".quad 0x0000000000000160,0x0800000b00000003,0x0000000000000004,0x0000000000000000\n"
".quad 0x00000001000001b8,0x0000000000000002,0x0000000000000000,0x0000000000000c89\n"
".quad 0x0000000000000050,0x0000000400000000,0x0000000000000004,0x0000000000000000\n"
".quad 0x000000010000017a,0x0000000000000002,0x0000000000000000,0x0000000000000cd9\n"
".quad 0x00000000000000ac,0x0000000400000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x00000001000000b0,0x0000000000000006,0x0000000000000000,0x0000000000000d85\n"
".quad 0x0000000000000100,0x0e00000900000003,0x0000000000000004,0x0000000000000000\n"
".quad 0x00000001000002b6,0x0000000000000002,0x0000000000000000,0x0000000000000e85\n"
".quad 0x0000000000000048,0x0000000700000000,0x0000000000000004,0x0000000000000000\n"
".quad 0x00000001000000f6,0x0000000000000002,0x0000000000000000,0x0000000000000ecd\n"
".quad 0x0000000000000070,0x0000000700000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x0000000100000036,0x0000000000100006,0x0000000000000000,0x0000000000000f3d\n"
".quad 0x0000000000001010,0x3f00000600000003,0x0000000000000004,0x0000000000000000\n"
".quad 0x0000000100000277,0x0000000000000002,0x0000000000000000,0x0000000000001f4d\n"
".quad 0x000000000000004c,0x0000000a00000000,0x0000000000000004,0x0000000000000000\n"
".quad 0x0000000100000237,0x0000000000000002,0x0000000000000000,0x0000000000001f99\n"
".quad 0x0000000000000048,0x0000000a00000000,0x0000000000000004,0x0000000000000000\n"
".quad 0x000000010000006d,0x0000000000000002,0x0000000000000000,0x0000000000001fe1\n"
".quad 0x0000000000000084,0x0000000a00000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x00000008000001fb,0x0000000000000003,0x0000000000000000,0x0000000000002065\n"
".quad 0x0000000000000600,0x0000000a00000000,0x0000000000000004,0x0000000000000000\n"
".quad 0x0000000100000304,0x0000000000000002,0x0000000000000000,0x0000000000002065\n"
".quad 0x000000000000001c,0x0000000000000000,0x0000000000000004,0x0000000000000000\n"
".quad 0x00000001000000a7,0x0000000000000002,0x0000000000000000,0x0000000000002081\n"
".quad 0x0000000000000060,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x7472747368732e00,0x747274732e006261,0x746d79732e006261,0x672e766e2e006261\n"
".quad 0x6e692e6c61626f6c,0x672e766e2e007469,0x742e006c61626f6c,0x32315a5f2e747865\n"
".quad 0x676e696464697267,0x617337315550475f,0x61727241656c706d,0x5074637572745379\n"
".quad 0x74616f6c6636506a,0x766e2e0066665032,0x5a5f2e6f666e692e,0x6964646972673231\n"
".quad 0x37315550475f676e,0x7241656c706d6173,0x6375727453796172,0x6f6c6636506a5074\n"
".quad 0x2e00666650327461,0x006f666e692e766e,0x5a5f2e747865742e,0x6564726f65723431\n"
".quad 0x6c656e72656b5f72,0x65523032506a5069,0x63757274736e6f63,0x706d61536e6f6974\n"
".quad 0x706d61733731656c,0x537961727241656c,0x6e2e007463757274,0x5f2e6f666e692e76\n"
".quad 0x64726f657234315a,0x656e72656b5f7265,0x523032506a50696c,0x757274736e6f6365\n"
".quad 0x6d61536e6f697463,0x6d61733731656c70,0x7961727241656c70,0x2e00746375727453\n"
".quad 0x315a5f2e74786574,0x676e696e6e696234,0x6a6c656e72656b5f,0x6e6f636552303250\n"
".quad 0x6f69746375727473,0x50656c706d61536e,0x6a5f31535f31536a,0x6e692e766e2e006a\n"
".quad 0x6234315a5f2e6f66,0x6b5f676e696e6e69,0x32506a6c656e7265,0x74736e6f63655230\n"
".quad 0x536e6f6974637572,0x536a50656c706d61,0x006a6a5f31535f31,0x736e6f632e766e2e\n"
".quad 0x5a5f2e30746e6174,0x6e696e6e69623431,0x6c656e72656b5f67,0x6f6365523032506a\n"
".quad 0x697463757274736e,0x656c706d61536e6f,0x5f31535f31536a50,0x732e766e2e006a6a\n"
".quad 0x5a5f2e6465726168,0x6964646972673231,0x37315550475f676e,0x7241656c706d6173\n"
".quad 0x6375727453796172,0x6f6c6636506a5074,0x2e00666650327461,0x74736e6f632e766e\n"
".quad 0x5a5f2e3631746e61,0x6964646972673231,0x37315550475f676e,0x7241656c706d6173\n"
".quad 0x6375727453796172,0x6f6c6636506a5074,0x2e00666650327461,0x74736e6f632e766e\n"
".quad 0x315a5f2e30746e61,0x6e69646469726732,0x7337315550475f67,0x727241656c706d61\n"
".quad 0x7463757274537961,0x616f6c6636506a50,0x6e2e006666503274,0x6174736e6f632e76\n"
".quad 0x34315a5f2e30746e,0x5f726564726f6572,0x50696c656e72656b,0x6f6365523032506a\n"
".quad 0x697463757274736e,0x656c706d61536e6f,0x656c706d61733731,0x7274537961727241\n"
".quad 0x2e766e2e00746375,0x746e6174736e6f63,0x6732315a5f000032,0x5f676e6964646972\n"
".quad 0x6d61733731555047,0x7961727241656c70,0x6a50746375727453,0x3274616f6c663650\n"
".quad 0x75635f5f00666650,0x5f30326d735f6164,0x5f6e725f74727173,0x34315a5f00323366\n"
".quad 0x5f726564726f6572,0x50696c656e72656b,0x6f6365523032506a,0x697463757274736e\n"
".quad 0x656c706d61536e6f,0x656c706d61733731,0x7274537961727241,0x34315a5f00746375\n"
".quad 0x5f676e696e6e6962,0x506a6c656e72656b,0x736e6f6365523032,0x6e6f697463757274\n"
".quad 0x6a50656c706d6153,0x6a6a5f31535f3153,0x7a69536469726700,0x6f74756300635f65\n"
".quad 0x697300635f326666,0x00635f79785f657a,0x75437265766f315f,0x00635f3266666f74\n"
".quad 0x635f66666f747563,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0100030000000000,0x0000000000000000,0x0000000000000000,0x0200030000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0300030000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000030000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000030000000000,0x0000000000000000,0x0000000000000000,0x0a00030000000000\n"
".quad 0x0000000000000000,0x0000000000101000,0x0d00030000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x1000030000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0700030000000000,0x0000000000000000,0x0000000000010000,0x0900030000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0400030000000000,0x0000000000000000\n"
".quad 0x0000000000016000,0x0600030000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0500030000000000,0x0000000000000000,0x0000000000000000,0x0e00030000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0c00030000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0b00030000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0800030000000000,0x0000000000000000,0x0000000000000000,0x0f00030000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0a10120000000100,0x0000000000000000\n"
".quad 0x00000000000f1800,0x0a00120000003200,0x00000000000f1800,0x000000000000f800\n"
".quad 0x0710120000004a00,0x0000000000000000,0x0000000000010000,0x0410120000008a00\n"
".quad 0x0000000000000000,0x0000000000016000,0x0f0011000000bf00,0x0000000000000400\n"
".quad 0x0000000000000c00,0x0f0011000000ca00,0x0000000000001000,0x0000000000000400\n"
".quad 0x0f0011000000d400,0x0000000000000000,0x0000000000000400,0x0f0011000000de00\n"
".quad 0x0000000000001800,0x0000000000000400,0x0f0011000000ee00,0x0000000000001400\n"
".quad 0x0000000000000400,0x00440400005de400,0x00000094001c0428,0x00000084009c042c\n"
".quad 0x04400020001c032c,0x0e40008001dc0320,0x000000000001e71b,0x004000a000dde480\n"
".quad 0x00c00060009c4328,0x0000034000000750,0x07c00060019c0360,0x004000b021dc4320\n"
".quad 0x0000003060dc8548,0x00000040609c8584,0x00000020611c8584,0x0600000d20dc0484\n"
".quad 0x06000009209c0414,0x00480010315c0314,0x0600001120dc0450,0x00400100011de414\n"
".quad 0x0a480000209c0328,0x00000008309c0320,0x09c00010211c0348,0x00c0001020dc4320\n"
".quad 0x00400110315c4350,0x0000000040dc8548,0x0e40012031dc0384,0x000000e00001e71b\n"
".quad 0x0000000400dde240,0x7e18000040dc0518,0x0e40012031dc0354,0x000000000021f41b\n"
".quad 0xfffffffc00dde240,0x00400130009de41b,0x0000000040dc1528,0x00400130009df414\n"
".quad 0x00000010019de228,0x00c0007800dc0318,0x0d8000c0011c0358,0x004000d0315c4320\n"
".quad 0x0d8000e0019c0348,0x00000000409c8520,0x004000f031dc4394,0x00000000601c8548\n"
".quad 0x00000000001de794,0x0000000000000080,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x000000001c0c0400\n"
".quad 0x0000100000000800,0x0000200000001800,0x00002c0000002800,0x00000d00080a0400\n"
".quad 0x3019030030002000,0xffffff000c170400,0x11f000002c0006ff,0xffffff000c170400\n"
".quad 0x11f00000280005ff,0xffffff000c170400,0x21f00000200004ff,0xffffff000c170400\n"
".quad 0x21f00000180003ff,0xffffff000c170400,0x21f00000100002ff,0xffffff000c170400\n"
".quad 0x21f00000080001ff,0xffffff000c170400,0x11f00000000000ff,0x04003000080d0400\n"
".quad 0x005de40000000300,0x001c042800440400,0x009c042c00000094,0x001c032c00000084\n"
".quad 0x01dc032004400020,0x0001e71b0e400080,0x00dde48000000000,0x009c4328004000a0\n"
".quad 0x021de45000c00010,0x011c0328004000e0,0x00dde42007c00010,0x025c4328004000c0\n"
".quad 0x215c435000c00020,0x409c8548004000b0,0x229c038400000000,0x209c432007c00060\n"
".quad 0x22dc435000c00060,0x021c0348004000d0,0xa0dc852011c00020,0xa09c858400000010\n"
".quad 0xa11c858400000000,0xa15c858400000020,0xa19c858400000030,0xa1dc858400000040\n"
".quad 0x925c438400000050,0x02dde448004000f0,0x029c432800400100,0x031c035000c00040\n"
".quad 0xa35c432017c00040,0x809ca54800400110,0xc11cc59400000000,0x001de79400000000\n"
".quad 0x0000008000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x100c040000000000,0x0000080000000000,0x0000180000001000\n"
".quad 0x00001100080a0400,0x2819030028002000,0xffffff000c170400,0x41f00000180003ff\n"
".quad 0xffffff000c170400,0x21f00000100002ff,0xffffff000c170400,0x21f00000080001ff\n"
".quad 0xffffff000c170400,0x11f00000000000ff,0x01000c00080d0400,0x005de40000000100\n"
".quad 0x201c042800440400,0x00dde41804400031,0xf3dc032800480020,0x001c001a8e400033\n"
".quad 0x031c04c800000010,0x105d032c00000098,0x001c024800c00080,0x02dc04083ffffff8\n"
".quad 0x029c042c0000008c,0x201c042c00000088,0x0000071486000001,0x009c0360000035a0\n"
".quad 0x209f845000400030,0x001c431c00000009,0x009c432000000008,0x201e035000480020\n"
".quad 0x01dc032006400030,0x0001031b0e400030,0x2080034800400030,0x01dc034800c00004\n"
".quad 0xf01dc31b0e400030,0x2080036800400033,0x025c044800c00004,0x009c042012000008\n"
".quad 0x201c042c00000094,0x00dc001804000025,0x201c84c800000010,0x311c021004480051\n"
".quad 0x20dc04083ffffff8,0x219c041804000025,0x30dc001486000011,0x011c04c800000010\n"
".quad 0x615c032c00000084,0x30dc025000000024,0x21df84083ffffff8,0x215c041c00000015\n"
".quad 0x20dc03148600000d,0x61dc435000400020,0x519c03200c00001c,0x309c035000000024\n"
".quad 0x721c434800400020,0x219f845000000030,0x209c041c00000019,0x81de031800000009\n"
".quad 0x515c432018000024,0x209c00200a000018,0x71dc035000480050,0x515c431b0e000024\n"
".quad 0xf19dc35000000030,0x71c1036800000027,0x515e034800000024,0x8200032018000024\n"
".quad 0x73dc034800c00004,0x51dc031b0e000024,0x231c841b0e000024,0x8204031406000009\n"
".quad 0x5141034800c00004,0xf1ddc34800000024,0x609c036800000027,0x51dc033112000020\n"
".quad 0x359c031b0e000024,0x221c034800000010,0x5141035000400040,0xc1dc234800000024\n"
".quad 0x819c03188e480010,0x709c034800400040,0x01e1e43112000014,0x619e032800480010\n"
".quad 0x215c036000c00008,0x7320035000400030,0x219c044800fffffc,0x509c031800000019\n"
".quad 0x821e034800400030,0x61dc006000c00008,0x219c045000480050,0x225e041800000009\n"
".quad 0x209c841800000021,0x619c00140600001d,0x91dd005000480050,0x21dc235000000000\n"
".quad 0x225c84188e480030,0x131c851406000019,0x219c84c800000070,0x01e1e4140600001d\n"
".quad 0x93dc232800480030,0x631c23188e480020,0x70a003081e0000fc,0x01a5e44800fffffc\n"
".quad 0x131c852800480020,0x21dc23c800000060,0x626403188e000030,0x55dc034800fffffc\n"
".quad 0x861c034800000028,0x125c85480000002c,0x0001e7c800000050,0x21de0440000029a0\n"
".quad 0x215e041800000015,0xb19c03180000000d,0x71dd002014400030,0x501d005000000000\n"
".quad 0x665c035000000000,0x211c842008400020,0x201c84140600001d,0x115c851406000001\n"
".quad 0x411c23c000000050,0x001c23081e0000fc,0x209c03081e0000fc,0x8b1c034800c00004\n"
".quad 0x101c854800400041,0x51dc23c800000020,0x109c851b0e000010,0xcb5c03c800000040\n"
".quad 0x001c044800400042,0x06dde4080e00001c,0x00dde428000000fc,0x009de428000000fc\n"
".quad 0x071de428000000fc,0x045de428000000fc,0x041de428000000fc,0x075de428000000fc\n"
".quad 0x04dde428000000fc,0x049de428000000fc,0x079de428000000fc,0x055de428000000fc\n"
".quad 0x051de428000000fc,0x111c8528000000fc,0xd69c03c800000030,0x101c854800400042\n"
".quad 0x111c85c800000010,0x101c85c000000010,0x000007c000000030,0x41dc036000002440\n"
".quad 0x0e5de41a8e0000fc,0x0021e72800000000,0x101c8540000023c0,0x115c85c000000060\n"
".quad 0x11dc85c000000030,0x025de2c000000070,0x129c851800000010,0x011ca3c000000020\n"
".quad 0x501ca35000480000,0x415c235000480010,0x019c037000c0047c,0x011c234801000010\n"
".quad 0x701ce37000c0047c,0x621c035000c00010,0x411c435800c00078,0x715ca34800000014\n"
".quad 0x01dc4320138000c0,0x6e9c0348004000d0,0x415c43200bc00010,0x121c854000000020\n"
".quad 0xa01ce3c000000050,0x5edc435000c00010,0x611c03480000001c,0x8f9c0320138000c0\n"
".quad 0x515c434800c00004,0xaf1ca348004000d0,0x5f5c432009c00010,0xcb9c854800000000\n"
".quad 0xa01c858c00000003,0x0000078c00000013,0x0bdd036000001f00,0xf01c0348000000b8\n"
".quad 0x0e1c034800c000fe,0x81dc035800c00018,0x0001e7190e0000ff,0x07dde44000001e40\n"
".quad 0x9fdc0328000000fc,0xffdc04207e000069,0xf01c0350ee0000ff,0xf11d036000c00019\n"
".quad 0x4c1c034800000002,0x01dc23080ec00100,0x0081e7198e000067,0x0020034000000260\n"
".quad 0x0121e44800000064,0x0161e42800400080,0x00200328004000a0,0x03200348000000b8\n"
".quad 0x0120432009c00020,0x4360435000c00020,0x02a0034800400090,0x002043200bc00040\n"
".quad 0xc220a55000c00040,0x02e0438400000000,0x90220348004000b0,0xa120c56000c00009\n"
".quad 0x0220858400000000,0x026085c900001400,0x012085c900000800,0x016085c900001000\n"
".quad 0x01a085c900000400,0x01e085c900000c00,0x000007c900000000,0xffdc046000001a80\n"
".quad 0x01dc2350ee0000ff,0x0001e7198e0000ff,0x0ddde44000001a00,0x0c5de428000000fc\n"
".quad 0x00000728000000fc,0x7d1c857000001920,0x7cdc85c100001403,0x7c9c85c100000803\n"
".quad 0x311c04c100000003,0x41dc0110000000d1,0x0001e71e8e0000fc,0x311c044000000060\n"
".quad 0x41dc0110000000cd,0x0021e71e8e0000fc,0x311c04b000000000,0x41dc0110000000c9\n"
".quad 0x0021e71e8e0000fc,0x711c85b000000000,0x215e04c100001003,0x701c851800000059\n"
".quad 0x219e04c100000403,0x7d5c85180000005d,0x000007c100000c03,0x411d006000000600\n"
".quad 0x215e045000000014,0x001d001800000061,0x411c005000000018,0x515d005800000010\n"
".quad 0x0d9c005000000017,0x501c003008000000,0x01dc00306c000014,0x0021f4208e480040\n"
".quad 0x001c004000000000,0x311c045800480060,0x411e011000000001,0xa11c044800cffc00\n"
".quad 0x0100071000000011,0x401c005000001f20,0x011de25800400120,0x001c00187ccf2698\n"
".quad 0x011c005800000000,0x015c023008400014,0x011c002b15011e08,0x015c003008800024\n"
".quad 0x011c00300a8000f4,0x015d003008800034,0x011c00300a8000e4,0x011c003008800044\n"
".quad 0x011c003008800054,0x011c003008800064,0x011c003008800074,0x011c003008800084\n"
".quad 0x011c003008800094,0x011c0030088000a4,0x011c0030088000b4,0x011c0030088000c4\n"
".quad 0x001c0030088000d4,0x51dc8030088000e4,0x511c00208ec02000,0x019c005800d2e000\n"
".quad 0x21dd045a00d2e000,0x401c041000000001,0x311c042000000014,0x619c041000000079\n"
".quad 0x001c00200000001c,0x411c01c800000010,0x601c004800cffc00,0xa79c045800000000\n"
".quad 0x001c001000000011,0x451c0058000000c8,0x355c103028000003,0x201e04302a000003\n"
".quad 0x00000718000000b1,0x501d006000000560,0x001c005000000003,0x01dc00306c000000\n"
".quad 0x0021f4208e480040,0x001c004000000000,0x311c045800480060,0x411e011000000001\n"
".quad 0xa11c044800cffc00,0x0100071000000011,0x401c005000001980,0x011de25800400120\n"
".quad 0x001c00187ccf2698,0x011c005800000000,0x015c023008400014,0x011c002b15011e08\n"
".quad 0x015c003008800024,0x011c00300a8000f4,0x015d003008800034,0x011c00300a8000e4\n"
".quad 0x011c003008800044,0x011c003008800054,0x011c003008800064,0x011c003008800074\n"
".quad 0x011c003008800084,0x011c003008800094,0x011c0030088000a4,0x011c0030088000b4\n"
".quad 0x011c0030088000c4,0x001c0030088000d4,0x51dc8030088000e4,0x511c00208ec02000\n"
".quad 0x019c005800d2e000,0x21dd045a00d2e000,0x401c041000000001,0x311c042000000014\n"
".quad 0x619c041000000075,0x001c00200000001c,0x411c01c800000010,0x601c004800cffc00\n"
".quad 0xa75c045800000000,0x001c001000000011,0x449c0058000000c8,0x34dc103024000003\n"
".quad 0x201e043026000003,0x00000718000000b5,0x501d006000000560,0x001c005000000003\n"
".quad 0x01dc00306c000000,0x0021f4208e480040,0x001c004000000000,0x311c045800480060\n"
".quad 0x411e011000000001,0xa11c044800cffc00,0x0100071000000011,0x401c0050000013e0\n"
".quad 0x011de25800400120,0x001c00187ccf2698,0x011c005800000000,0x015c023008400014\n"
".quad 0x011c002b15011e08,0x015c003008800024,0x011c00300a8000f4,0x015d003008800034\n"
".quad 0x011c00300a8000e4,0x011c003008800044,0x011c003008800054,0x011c003008800064\n"
".quad 0x011c003008800074,0x011c003008800084,0x011c003008800094,0x011c0030088000a4\n"
".quad 0x011c0030088000b4,0x011c0030088000c4,0x001c0030088000d4,0x51dc8030088000e4\n"
".quad 0x511c00208ec02000,0x019c005800d2e000,0x21dd045a00d2e000,0x401c041000000001\n"
".quad 0x311c042000000014,0x619c041000000071,0x001c00200000001c,0x411c01c800000010\n"
".quad 0x601c004800cffc00,0xa71c045800000000,0x001c001000000011,0x441c0058000000c8\n"
".quad 0x345c103020000003,0x201e043022000003,0x501d001800000069,0x001c005000000003\n"
".quad 0x01dc00306c000000,0x0021e7208e480040,0x001c00b000000000,0x311c045800480060\n"
".quad 0x411e011000000001,0xa11c044800cffc00,0x0100071000000011,0x401c005000000e60\n"
".quad 0x011de25800400120,0x001c00187ccf2698,0x011c005800000000,0x015c023008400014\n"
".quad 0x011c002b15011e08,0x015c003008800024,0x011c00300a8000f4,0x015d003008800034\n"
".quad 0x011c00300a8000e4,0x011c003008800044,0x011c003008800054,0x011c003008800064\n"
".quad 0x011c003008800074,0x011c003008800084,0x011c003008800094,0x011c0030088000a4\n"
".quad 0x011c0030088000b4,0x011c0030088000c4,0x001c0030088000d4,0x51dc8030088000e4\n"
".quad 0x511c00208ec02000,0x019c005800d2e000,0x21dd045a00d2e000,0x401c041000000001\n"
".quad 0x311c042000000014,0x619c04100000006d,0x001c00200000001c,0x411c01c800000010\n"
".quad 0x601c004800cffc00,0xa6dc045800000000,0x001c001000000011,0x409c0058000000c8\n"
".quad 0x30dc003004000003,0x001de73006000003,0x1c5c03b000000000,0x7ddc034800c00007\n"
".quad 0x11dc234800c00013,0x0001e71a8e0000c3,0xf7dc134003ffe640,0xf1dc034800c00005\n"
".quad 0x0001e71a8e0000e1,0x001df44003ffe1e0,0x001e032800480010,0x011de46000c00008\n"
".quad 0x9e5c032800480010,0xae9c034800c00007,0x411ce34801000003,0xe1dc235000c00010\n"
".quad 0xbedc431a8e0000e7,0x0f1c034800000013,0x4f5c4348010000f0,0x0001e748000000f4\n"
".quad 0x001df44003ffdf60,0x101e854000000000,0x111c85c000000060,0x001c03c000000040\n"
".quad 0x41dc234800c00004,0x101c851a8e000000,0x0001e7c800000060,0x001df44003ffdaa0\n"
".quad 0x06dde44000000000,0x00dde428000000fc,0x009de428000000fc,0x071de428000000fc\n"
".quad 0x045de428000000fc,0x041de428000000fc,0x075de428000000fc,0x04dde428000000fc\n"
".quad 0x049de428000000fc,0x079de428000000fc,0x055de428000000fc,0x051df428000000fc\n"
".quad 0x701ca328000000fc,0x05dde45000480011,0x801ca32800400100,0x061de42000480001\n"
".quad 0x001c032800480000,0x059de44800000058,0x015ce328004000e0,0x011ca35000c00020\n"
".quad 0x01dce3202dc00020,0x515c435000c00010,0x019ca348004000f0,0x801c03202fc00010\n"
".quad 0x451ca52000400041,0x71dc439400000000,0x025ce34800400110,0x021ca35000c00020\n"
".quad 0x02dce3202dc00020,0x679c855000c00010,0x925c439400000000,0x029ca348004000f0\n"
".quad 0x801c03202fc00010,0x849ca52000400041,0xb2dc439400000000,0x035ce34800400110\n"
".quad 0x031ca35000c00020,0x03dce3202dc00020,0xa75c855000c00010,0xd35c439400000000\n"
".quad 0x039ca348004000f0,0x801c03202fc00010,0xc41ca52000400041,0xf3dc439400000000\n"
".quad 0x061ca34800400110,0x059ce3202dc00020,0xe71c855000c00020,0x665c439400000000\n"
".quad 0x059ca348004000f1,0x001ce3202fc00010,0x809ca55000c00010,0x05dc439400000001\n"
".quad 0x66dc854800400110,0x001de79400000001,0x401c028000000000,0x01dc030bcc000000\n"
".quad 0x0001e7198e400104,0x401c024000000220,0x01dc2339fffffffc,0x0021e71a8e0000fc\n"
".quad 0x41dc004000000300,0x0021e7288e0000fc,0x011de24000000040,0x001de719fffffffc\n"
".quad 0x41dc804000000280,0x201c44298edfe000,0x0001e71080000011,0x411c204000000040\n"
".quad 0x001de75000cfe000,0x01dc0040000001e0,0x0001e7290edfe000,0x411c0040000001a0\n"
".quad 0x71dc04307ed7e000,0x001de70c0e00001c,0xf1dc044000000020,0x401c000c0e00001c\n"
".quad 0x015c40c800000014,0x001c405800000010,0x519e005800cfc000,0x515c003000800114\n"
".quad 0x019c00300a000018,0x501e003000000018,0x011c003008000014,0x410040300a000018\n"
".quad 0x001de75800cbe000,0x0000009000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0xcba9430000000100\n"
".quad 0x43f69c245d1a5119,0x89a1c82e0459d529,0x04c52536dbbbe832,0x96de123eed18563b\n"
".quad 0x0b90d24601cfb442,0xaa0fb44ba51cab49,0x542c9d4eabb80d4d,0x000000727fffff4a\n"
".quad 0x00000000140c043f,0x0000180000001000,0x0000280000002000,0x00001000080a0400\n"
".quad 0x2c1903002c002000,0xffffff000c170400,0x11f00000280004ff,0xffffff000c170400\n"
".quad 0x21f00000200003ff,0xffffff000c170400,0x21f00000180002ff,0xffffff000c170400\n"
".quad 0x21f00000100001ff,0xffffff000c170400,0x41f00000000000ff,0x2701d400080d0400\n"
".quad 0x0000000000002000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000160008120400,0x0811040000000000,0x0000000000001600,0x0000150008120400\n"
".quad 0x0811040000000000,0x0000000000001500,0x0000140008120400,0x0811040000000000\n"
".quad 0x0000000000001400,0x0000130008120400,0x0811040000002000,0x0000200000001300\n"
".quad 0x0000000600000000,0x000020e400000005,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000015000000000,0x0000015000000000,0x0000000400000000,0x6000000000000000\n"
".quad 0x00000b2900001605,0x0000000000000000,0x0000000000000000,0x0000025c00000000\n"
".quad 0x0000025c00000000,0x0000000400000000,0x6000000000000000,0x00000d8500001505\n"
".quad 0x0000000000000000,0x0000000000000000,0x000001b800000000,0x000001b800000000\n"
".quad 0x0000000400000000,0x6000000000000000,0x00000f3d00001305,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000112800000000,0x0000112800000000,0x0000000400000000\n"
".quad 0x6000000000000000,0x0000206500001306,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000060000000000,0x0000000400000000,0x0000000100000000\n"
".quad 0x0000206500000005,0x0000000000000000,0x0000000000000000,0x0000001c00000000\n"
".quad 0x0000001c00000000,0x0000000400000000,0x0000000000000000\n"
".text");

extern "C" {

extern const unsigned long long __deviceText_$sm_20$[1095];

}

asm(
".section .rodata\n"
".align 32\n"
"__deviceText_$compute_20$:\n"
".quad 0x6f69737265762e09,0x2e090a322e32206e,0x7320746567726174,0x2f2f090a30325f6d\n"
".quad 0x656c69706d6f6320,0x2f20687469772064,0x6361622f656d6f68,0x65642f726f6c6568\n"
".quad 0x3831326564696369,0x2e332d616475632f,0x34366e65706f2f32,0x65622f2f62696c2f\n"
".quad 0x6f766e202f2f090a,0x2e332063636e6570,0x20746c6975622032,0x2d30313032206e6f\n"
".quad 0x090a0a33302d3131,0x656c62697369762e,0x2820636e75662e20,0x2e206d617261702e\n"
".quad 0x75635f5f20323366,0x5f5f667465726164,0x656e72656b32315a,0x6665756c61765f6c\n"
".quad 0x656b32315a5f2029,0x6c61765f6c656e72,0x61702e2820666575,0x3233662e206d6172\n"
".quad 0x70616475635f5f20,0x5a5f5f31666d7261,0x6c656e72656b3231,0x296665756c61765f\n"
".quad 0x2d2d2d2f2f090a0a,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x6d6f43202f2f090a,0x2f20676e696c6970,0x78706d742f706d74,0x36303030305f7466\n"
".quad 0x303030305f633633,0x5f33312d30303030,0x746e695f41445543,0x632e656361667265\n"
".quad 0x2f2820692e337070,0x494263632f706d74,0x517556584d502e23,0x2d2d2d2f2f090a29\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2f2f090a0a\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x74704f202f2f090a\n"
".quad 0x2f090a3a736e6f69,0x2d2d2d2d2d2d2d2f,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2f2f090a2d2d2d2d,0x7465677261542020,0x5349202c7874703a,0x2c30325f6d733a41\n"
".quad 0x3a6e6169646e4520,0x202c656c7474696c,0x207265746e696f50,0x0a34363a657a6953\n"
".quad 0x334f2d20202f2f09,0x696d6974704f2809,0x6c206e6f6974617a,0x2f090a296c657665\n"
".quad 0x280930672d20202f,0x656c206775626544,0x2f2f090a296c6576,0x522809326d2d2020\n"
".quad 0x64612074726f7065,0x736569726f736976,0x2d2d2d2f2f090a29,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d\n"
".quad 0x2d2d2d2d2d2d2d2d,0x2d2d2d2d2d2d2d2d,0x656c69662e090a0a,0x6d6f633c22093109\n"
".quad 0x6e696c2d646e616d,0x69662e090a223e65,0x742f22093209656c,0x6678706d742f706d\n"
".quad 0x3336303030305f74,0x30303030305f6336,0x435f32312d303030,0x65746e695f414455\n"
".quad 0x75632e6563616672,0x70672e3265666164,0x6c69662e090a2275,0x5444552209330965\n"
".quad 0x0a22682e73657079,0x3409656c69662e09,0x656b5f5550472209,0x75632e736c656e72\n"
".quad 0x656c69662e090a22,0x7273752f22093509,0x6363672f62696c2f,0x2d34365f3638782f\n"
".quad 0x6e672d78756e696c,0x2f372e342e342f75,0x2f6564756c636e69,0x682e666564647473\n"
".quad 0x656c69662e090a22,0x6d6f682f22093609,0x6c65686361622f65,0x69636965642f726f\n"
".quad 0x75632f3831326564,0x622f322e332d6164,0x6e692f2e2e2f6e69,0x72632f6564756c63\n"
".quad 0x6563697665642f74,0x656d69746e75725f,0x69662e090a22682e,0x682f22093709656c\n"
".quad 0x686361622f656d6f,0x6965642f726f6c65,0x2f38313265646963,0x322e332d61647563\n"
".quad 0x2f2e2e2f6e69622f,0x2f6564756c636e69,0x6665645f74736f68,0x0a22682e73656e69\n"
".quad 0x3809656c69662e09,0x2f656d6f682f2209,0x726f6c6568636162,0x656469636965642f\n"
".quad 0x616475632f383132,0x6e69622f322e332d,0x6c636e692f2e2e2f,0x6c6975622f656475\n"
".quad 0x657079745f6e6974,0x662e090a22682e73,0x2f22093909656c69,0x6361622f656d6f68\n"
".quad 0x65642f726f6c6568,0x3831326564696369,0x2e332d616475632f,0x2e2e2f6e69622f32\n"
".quad 0x6564756c636e692f,0x5f6563697665642f,0x22682e7365707974,0x09656c69662e090a\n"
".quad 0x6d6f682f22093031,0x6c65686361622f65,0x69636965642f726f,0x75632f3831326564\n"
".quad 0x622f322e332d6164,0x6e692f2e2e2f6e69,0x72642f6564756c63,0x7079745f72657669\n"
".quad 0x2e090a22682e7365,0x09313109656c6966,0x622f656d6f682f22,0x2f726f6c65686361\n"
".quad 0x3265646963696564,0x2d616475632f3831,0x2f6e69622f322e33,0x756c636e692f2e2e\n"
".quad 0x61667275732f6564,0x73657079745f6563,0x69662e090a22682e,0x2f2209323109656c\n"
".quad 0x6361622f656d6f68,0x65642f726f6c6568,0x3831326564696369,0x2e332d616475632f\n"
".quad 0x2e2e2f6e69622f32,0x6564756c636e692f,0x657275747865742f,0x682e73657079745f\n"
".quad 0x656c69662e090a22,0x6f682f2209333109,0x65686361622f656d,0x636965642f726f6c\n"
".quad 0x632f383132656469,0x2f322e332d616475,0x692f2e2e2f6e6962,0x762f6564756c636e\n"
".quad 0x79745f726f746365,0x090a22682e736570,0x343109656c69662e,0x2f656d6f682f2209\n"
".quad 0x726f6c6568636162,0x656469636965642f,0x616475632f383132,0x6e69622f322e332d\n"
".quad 0x6c636e692f2e2e2f,0x697665642f656475,0x636e75616c5f6563,0x656d617261705f68\n"
".quad 0x0a22682e73726574,0x3109656c69662e09,0x656d6f682f220935,0x6f6c65686361622f\n"
".quad 0x6469636965642f72,0x6475632f38313265,0x69622f322e332d61,0x636e692f2e2e2f6e\n"
".quad 0x7472632f6564756c,0x656761726f74732f,0x682e7373616c635f,0x656c69662e090a22\n"
".quad 0x73752f2209363109,0x64756c636e692f72,0x34365f3638782f65,0x672d78756e696c2d\n"
".quad 0x2f737469622f756e,0x22682e7365707974,0x09656c69662e090a,0x7273752f22093731\n"
".quad 0x6564756c636e692f,0x22682e656d69742f,0x09656c69662e090a,0x6d6f682f22093831\n"
".quad 0x6c65686361622f65,0x69636965642f726f,0x75632f3831326564,0x622f322e332d6164\n"
".quad 0x6e692f2e2e2f6e69,0x6d732f6564756c63,0x6d6f74615f31315f,0x74636e75665f6369\n"
".quad 0x0a22682e736e6f69,0x3109656c69662e09,0x656d6f682f220939,0x6f6c65686361622f\n"
".quad 0x6469636965642f72,0x6475632f38313265,0x69622f322e332d61,0x636e692f2e2e2f6e\n"
".quad 0x6d6f632f6564756c,0x636e75665f6e6f6d,0x22682e736e6f6974,0x09656c69662e090a\n"
".quad 0x6d6f682f22093032,0x6c65686361622f65,0x69636965642f726f,0x75632f3831326564\n"
".quad 0x622f322e332d6164,0x6e692f2e2e2f6e69,0x616d2f6564756c63,0x74636e75665f6874\n"
".quad 0x0a22682e736e6f69,0x3209656c69662e09,0x656d6f682f220931,0x6f6c65686361622f\n"
".quad 0x6469636965642f72,0x6475632f38313265,0x69622f322e332d61,0x636e692f2e2e2f6e\n"
".quad 0x74616d2f6564756c,0x6174736e6f635f68,0x090a22682e73746e,0x323209656c69662e\n"
".quad 0x2f656d6f682f2209,0x726f6c6568636162,0x656469636965642f,0x616475632f383132\n"
".quad 0x6e69622f322e332d,0x6c636e692f2e2e2f,0x697665642f656475,0x74636e75665f6563\n"
".quad 0x0a22682e736e6f69,0x3209656c69662e09,0x656d6f682f220933,0x6f6c65686361622f\n"
".quad 0x6469636965642f72,0x6475632f38313265,0x69622f322e332d61,0x636e692f2e2e2f6e\n"
".quad 0x5f6d732f6564756c,0x696d6f74615f3231,0x6974636e75665f63,0x090a22682e736e6f\n"
".quad 0x343209656c69662e,0x2f656d6f682f2209,0x726f6c6568636162,0x656469636965642f\n"
".quad 0x616475632f383132,0x6e69622f322e332d,0x6c636e692f2e2e2f,0x315f6d732f656475\n"
".quad 0x656c62756f645f33,0x6f6974636e75665f,0x2e090a22682e736e,0x09353209656c6966\n"
".quad 0x622f656d6f682f22,0x2f726f6c65686361,0x3265646963696564,0x2d616475632f3831\n"
".quad 0x2f6e69622f322e33,0x756c636e692f2e2e,0x30325f6d732f6564,0x5f63696d6f74615f\n"
".quad 0x6e6f6974636e7566,0x662e090a22682e73,0x2209363209656c69,0x61622f656d6f682f\n"
".quad 0x642f726f6c656863,0x3132656469636965,0x332d616475632f38,0x2e2f6e69622f322e\n"
".quad 0x64756c636e692f2e,0x5f30325f6d732f65,0x69736e6972746e69,0x2e090a22682e7363\n"
".quad 0x09373209656c6966,0x622f656d6f682f22,0x2f726f6c65686361,0x3265646963696564\n"
".quad 0x2d616475632f3831,0x2f6e69622f322e33,0x756c636e692f2e2e,0x61667275732f6564\n"
".quad 0x74636e75665f6563,0x0a22682e736e6f69,0x3209656c69662e09,0x656d6f682f220938\n"
".quad 0x6f6c65686361622f,0x6469636965642f72,0x6475632f38313265,0x69622f322e332d61\n"
".quad 0x636e692f2e2e2f6e,0x7865742f6564756c,0x7465665f65727574,0x74636e75665f6863\n"
".quad 0x0a22682e736e6f69,0x3209656c69662e09,0x656d6f682f220939,0x6f6c65686361622f\n"
".quad 0x6469636965642f72,0x6475632f38313265,0x69622f322e332d61,0x636e692f2e2e2f6e\n"
".quad 0x74616d2f6564756c,0x6974636e75665f68,0x5f6c62645f736e6f,0x0a22682e33787470\n"
".quad 0x697369762e090a0a,0x6e75662e20656c62,0x617261702e282063,0x5f203233662e206d\n"
".quad 0x746572616475635f,0x656b32315a5f5f66,0x6c61765f6c656e72,0x315a5f2029666575\n"
".quad 0x5f6c656e72656b32,0x28206665756c6176,0x2e206d617261702e,0x75635f5f20323366\n"
".quad 0x31666d7261706164,0x72656b32315a5f5f,0x756c61765f6c656e,0x090a7b090a296665\n"
".quad 0x33662e206765722e,0x3e32343c66252032,0x09636f6c2e090a3b,0x240a300935380934\n"
".quad 0x6e6967656257444c,0x72656b32315a5f5f,0x756c61765f6c656e,0x2e646c090a3a6665\n"
".quad 0x33662e6d61726170,0x202c316625092032,0x70616475635f5f5b,0x5a5f5f31666d7261\n"
".quad 0x6c656e72656b3231,0x5d6665756c61765f,0x662e766f6d090a3b,0x2c32662509203233\n"
".quad 0x2e090a3b31662520,0x3239093409636f6c,0x2e6c756d090a3009,0x3366250920323366\n"
".quad 0x25202c326625202c,0x6f6c2e090a3b3266,0x0936303109340963,0x662e766f6d090a30\n"
".quad 0x2c34662509203233,0x6262616534663020,0x202020203b643038,0x342e31202f2f0920\n"
".quad 0x39302b6538343034,0x33662e766f6d090a,0x202c356625092032,0x6630616164346630\n"
".quad 0x20202020203b3462,0x36352e33202f2f09,0x0a38302b65343436,0x3233662e766f6d09\n"
".quad 0x30202c3666250920,0x6163313561623466,0x0920202020203b62,0x3436312e32202f2f\n"
".quad 0x090a37302b653631,0x203233662e766f6d,0x6630202c37662509,0x3264303962303934\n"
".quad 0x2f0920202020203b,0x313636313735202f,0x33662e766f6d090a,0x202c386625092032\n"
".quad 0x6663313036346630,0x20202020203b3462,0x37303338202f2f09,0x766f6d090a33392e\n"
".quad 0x662509203233662e,0x3932346630202c39,0x20203b3231656436,0x37202f2f09202020\n"
".quad 0x090a373333342e35,0x203233662e766f6d,0x30202c3031662509,0x3538316465653366\n"
".quad 0x2f09202020203b36,0x303336342e30202f,0x2e766f6d090a3637,0x3166250920323366\n"
".quad 0x3062336630202c31,0x20203b3532356334,0x2e30202f2f092020,0x3139353230323030\n"
".quad 0x33662e766f6d090a,0x2c32316625092032,0x6262643633663020,0x202020203b386562\n"
".quad 0x34352e36202f2f09,0x0a36302d65383538,0x3233662e766f6d09,0x202c333166250920\n"
".quad 0x3161393832336630,0x09202020203b3863,0x3230362e31202f2f,0x090a38302d653532\n"
".quad 0x203233662e766f6d,0x30202c3431662509,0x6439353430653266,0x2f09202020203b35\n"
".quad 0x333930302e33202f,0x6d090a31312d6531,0x09203233662e766f,0x6630202c35316625\n"
".quad 0x6339366633343932,0x2f2f09202020203b,0x36323135332e3420,0x6f6d090a34312d65\n"
".quad 0x2509203233662e76,0x326630202c363166,0x3b31356131643534,0x202f2f0920202020\n"
".quad 0x2d65343439372e34,0x2e766f6d090a3731,0x3166250920323366,0x3366316630202c37\n"
".quad 0x20203b3661396333,0x2e33202f2f092020,0x322d653531373038,0x662e766f6d090a30\n"
".quad 0x3831662509203233,0x626339316630202c,0x2020203b33343961,0x312e32202f2f0920\n"
".quad 0x33322d6531383530,0x6e722e616d66090a,0x662509203233662e,0x38316625202c3931\n"
".quad 0x25202c336625202c,0x6d66090a3b373166,0x3233662e6e722e61,0x202c303266250920\n"
".quad 0x316625202c336625,0x3b36316625202c39,0x6e722e616d66090a,0x662509203233662e\n"
".quad 0x2c336625202c3132,0x25202c3032662520,0x6d66090a3b353166,0x3233662e6e722e61\n"
".quad 0x202c323266250920,0x326625202c336625,0x3b34316625202c31,0x6e722e616d66090a\n"
".quad 0x662509203233662e,0x2c336625202c3332,0x25202c3232662520,0x6d66090a3b333166\n"
".quad 0x3233662e6e722e61,0x202c343266250920,0x326625202c336625,0x3b32316625202c33\n"
".quad 0x6e722e616d66090a,0x662509203233662e,0x2c336625202c3532,0x25202c3432662520\n"
".quad 0x6d66090a3b313166,0x3233662e6e722e61,0x202c363266250920,0x326625202c336625\n"
".quad 0x3b30316625202c35,0x6e722e616d66090a,0x662509203233662e,0x2c336625202c3732\n"
".quad 0x25202c3632662520,0x616d66090a3b3966,0x203233662e6e722e,0x25202c3832662509\n"
".quad 0x37326625202c3366,0x090a3b386625202c,0x662e6e722e616d66,0x3932662509203233\n"
".quad 0x25202c336625202c,0x376625202c383266,0x722e616d66090a3b,0x2509203233662e6e\n"
".quad 0x336625202c303366,0x202c39326625202c,0x6d66090a3b366625,0x3233662e6e722e61\n"
".quad 0x202c313366250920,0x336625202c336625,0x0a3b356625202c30,0x2e6e722e616d6609\n"
".quad 0x3366250920323366,0x202c336625202c32,0x6625202c31336625,0x2e67656e090a3b34\n"
".quad 0x3366250920323366,0x3b32336625202c33,0x33662e766f6d090a,0x2c34336625092032\n"
".quad 0x6262616563663020,0x202020203b643038,0x342e312d202f2f09,0x39302b6538343034\n"
".quad 0x33662e766f6d090a,0x2c35336625092032,0x3234356134663020,0x202020203b643963\n"
".quad 0x37342e33202f2f09,0x0a36302b65363236,0x3233662e766f6d09,0x202c363366250920\n"
".quad 0x3734303435636630,0x09202020203b3238,0x363730332d202f2f,0x646461090a37342e\n"
".quad 0x662509203233662e,0x2c336625202c3733,0x090a3b3633662520,0x662e6e722e616d66\n"
".quad 0x3833662509203233,0x25202c336625202c,0x336625202c373366,0x2e616d66090a3b35\n"
".quad 0x09203233662e6e72,0x6625202c39336625,0x2c38336625202c33,0x090a3b3433662520\n"
".quad 0x727070612e766964,0x09203233662e786f,0x6625202c30346625,0x39336625202c3333\n"
".quad 0x61702e7473090a3b,0x203233662e6d6172,0x616475635f5f5b09,0x315a5f5f66746572\n"
".quad 0x5f6c656e72656b32,0x2c5d6665756c6176,0x090a3b3034662520,0x444c240a3b746572\n"
".quad 0x315a5f5f646e6557,0x5f6c656e72656b32,0x0a3a6665756c6176,0x5a5f202f2f207d09\n"
".quad 0x6c656e72656b3231,0x0a6665756c61765f,0x2074736e6f632e09,0x7a6973203233732e\n"
".quad 0x0a3b635f79785f65,0x2074736e6f632e09,0x34206e67696c612e,0x6972672038622e20\n"
".quad 0x5b635f657a695364,0x2e090a0a3b5d3231,0x5a5f207972746e65,0x6e696e6e69623431\n"
".quad 0x6c656e72656b5f67,0x6f6365523032506a,0x697463757274736e,0x656c706d61536e6f\n"
".quad 0x5f31535f31536a50,0x2e09090a28206a6a,0x752e206d61726170,0x6475635f5f203233\n"
".quad 0x5a5f5f6d72617061,0x6e696e6e69623431,0x6c656e72656b5f67,0x6f6365523032506a\n"
".quad 0x697463757274736e,0x656c706d61536e6f,0x5f31535f31536a50,0x09090a2c6e5f6a6a\n"
".quad 0x2e206d617261702e,0x75635f5f20343675,0x5f5f6d7261706164,0x696e6e696234315a\n"
".quad 0x656e72656b5f676e,0x6365523032506a6c,0x7463757274736e6f,0x6c706d61536e6f69\n"
".quad 0x31535f31536a5065,0x706d61735f6a6a5f,0x09090a2c675f656c,0x2e206d617261702e\n"
".quad 0x75635f5f20343675,0x5f5f6d7261706164,0x696e6e696234315a,0x656e72656b5f676e\n"
".quad 0x6365523032506a6c,0x7463757274736e6f,0x6c706d61536e6f69,0x31535f31536a5065\n"
".quad 0x4b7864695f6a6a5f,0x09090a2c675f7965,0x2e206d617261702e,0x75635f5f20343675\n"
".quad 0x5f5f6d7261706164,0x696e6e696234315a,0x656e72656b5f676e,0x6365523032506a6c\n"
".quad 0x7463757274736e6f,0x6c706d61536e6f69,0x31535f31536a5065,0x567864695f6a6a5f\n"
".quad 0x0a2c675f65756c61,0x6d617261702e0909,0x5f5f203436752e20,0x6d72617061647563\n"
".quad 0x6e696234315a5f5f,0x72656b5f676e696e,0x523032506a6c656e,0x757274736e6f6365\n"
".quad 0x6d61536e6f697463,0x5f31536a50656c70,0x69625f6a6a5f3153,0x675f746e756f436e\n"
".quad 0x7261702e09090a2c,0x203233752e206d61,0x6170616475635f5f,0x6234315a5f5f6d72\n"
".quad 0x6b5f676e696e6e69,0x32506a6c656e7265,0x74736e6f63655230,0x536e6f6974637572\n"
".quad 0x536a50656c706d61,0x5f6a6a5f31535f31,0x2c657a69736e6962,0x617261702e09090a\n"
".quad 0x5f203233752e206d,0x726170616475635f,0x696234315a5f5f6d,0x656b5f676e696e6e\n"
".quad 0x3032506a6c656e72,0x7274736e6f636552,0x61536e6f69746375,0x31536a50656c706d\n"
".quad 0x675f6a6a5f31535f,0x6c456d754e646972,0x0a7b090a29736d65,0x752e206765722e09\n"
".quad 0x35323c7225203233,0x6765722e090a3b3e,0x7225203436752e20,0x090a3b3e35313c64\n"
".quad 0x33662e206765722e,0x3b3e353c66252032,0x2e206765722e090a,0x3c70252064657270\n"
".quad 0x6f6c2e090a3b3e35,0x3009333309340963,0x67656257444c240a,0x6234315a5f5f6e69\n"
".quad 0x6b5f676e696e6e69,0x32506a6c656e7265,0x74736e6f63655230,0x536e6f6974637572\n"
".quad 0x536a50656c706d61,0x3a6a6a5f31535f31,0x33752e766f6d090a,0x202c317225092032\n"
".quad 0x782e646961746325,0x752e766f6d090a3b,0x2c32722509203233,0x782e6469746e2520\n"
".quad 0x6c2e6c756d090a3b,0x2509203233752e6f,0x2c317225202c3372,0x6d090a3b32722520\n"
".quad 0x09203233752e766f,0x697425202c347225,0x6461090a3b782e64,0x2509203233752e64\n"
".quad 0x2c347225202c3572,0x6c090a3b33722520,0x2e6d617261702e64,0x3672250920323375\n"
".quad 0x6475635f5f5b202c,0x5a5f5f6d72617061,0x6e696e6e69623431,0x6c656e72656b5f67\n"
".quad 0x6f6365523032506a,0x697463757274736e,0x656c706d61536e6f,0x5f31535f31536a50\n"
".quad 0x090a3b5d6e5f6a6a,0x2e656c2e70746573,0x3170250920323375,0x25202c367225202c\n"
".quad 0x702540090a3b3572,0x2409206172622031,0x3635325f315f744c,0x636f6c2e090a3b32\n"
".quad 0x0a30093134093409,0x3436752e74766309,0x722509203233752e,0x3b357225202c3164\n"
".quad 0x7261702e646c090a,0x09203436752e6d61,0x5f5b202c32647225,0x726170616475635f\n"
".quad 0x696234315a5f5f6d,0x656b5f676e696e6e,0x3032506a6c656e72,0x7274736e6f636552\n"
".quad 0x61536e6f69746375,0x31536a50656c706d,0x735f6a6a5f31535f,0x5d675f656c706d61\n"
".quad 0x772e6c756d090a3b,0x203233752e656469,0x25202c3364722509,0x0a3b3432202c3572\n"
".quad 0x3436752e64646109,0x202c346472250920,0x7225202c32647225,0x2e646c090a3b3364\n"
".quad 0x662e6c61626f6c67,0x2c31662509203233,0x382b346472255b20,0x672e646c090a3b5d\n"
".quad 0x33662e6c61626f6c,0x202c326625092032,0x32312b346472255b,0x672e646c090a3b5d\n"
".quad 0x33662e6c61626f6c,0x202c336625092032,0x36312b346472255b,0x2e747663090a3b5d\n"
".quad 0x2e3233752e697a72,0x3772250920323366,0x090a3b326625202c,0x2e697a722e747663\n"
".quad 0x203233662e323375,0x6625202c38722509,0x2e747663090a3b33,0x2e3233752e697a72\n"
".quad 0x3972250920323366,0x090a3b316625202c,0x74736e6f632e646c,0x722509203233752e\n"
".quad 0x6972675b202c3031,0x2b635f657a695364,0x6c756d090a3b5d30,0x203233752e6f6c2e\n"
".quad 0x25202c3131722509,0x377225202c303172,0x6f632e646c090a3b,0x203233752e74736e\n"
".quad 0x5b202c3231722509,0x5f79785f657a6973,0x6c756d090a3b5d63,0x203233752e6f6c2e\n"
".quad 0x25202c3331722509,0x387225202c323172,0x752e646461090a3b,0x3431722509203233\n"
".quad 0x202c31317225202c,0x61090a3b33317225,0x09203233752e6464,0x7225202c35317225\n"
".quad 0x3b34317225202c39,0x7261702e646c090a,0x09203436752e6d61,0x5f5b202c35647225\n"
".quad 0x726170616475635f,0x696234315a5f5f6d,0x656b5f676e696e6e,0x3032506a6c656e72\n"
".quad 0x7274736e6f636552,0x61536e6f69746375,0x31536a50656c706d,0x625f6a6a5f31535f\n"
".quad 0x5f746e756f436e69,0x747663090a3b5d67,0x3233752e3436752e,0x202c366472250920\n"
".quad 0x6d090a3b35317225,0x2e656469772e6c75,0x6472250920323375,0x2c35317225202c37\n"
".quad 0x646461090a3b3420,0x722509203436752e,0x35647225202c3864,0x0a3b37647225202c\n"
".quad 0x617261702e646c09,0x2509203233752e6d,0x5f5f5b202c363172,0x6d72617061647563\n"
".quad 0x6e696234315a5f5f,0x72656b5f676e696e,0x523032506a6c656e,0x757274736e6f6365\n"
".quad 0x6d61536e6f697463,0x5f31536a50656c70,0x69625f6a6a5f3153,0x0a3b5d657a69736e\n"
".quad 0x626f6c672e646c09,0x09203233752e6c61,0x255b202c37317225,0x0a3b5d302b386472\n"
".quad 0x65672e7074657309,0x702509203233752e,0x2c37317225202c32,0x090a3b3631722520\n"
".quad 0x6172622032702540,0x5f315f744c240920,0x2e090a3b30333333,0x3534093409636f6c\n"
".quad 0x2e766f6d090a3009,0x3172250920323375,0x61090a3b31202c38,0x626f6c672e6d6f74\n"
".quad 0x752e6464612e6c61,0x3931722509203233,0x5d386472255b202c,0x0a3b38317225202c\n"
".quad 0x3233732e766f6d09,0x202c303272250920,0x73090a3b39317225,0x752e65672e707465\n"
".quad 0x2c33702509203233,0x25202c3032722520,0x2540090a3b363172,0x0920617262203370\n"
".quad 0x38335f315f744c24,0x6f6c2e090a3b3234,0x3009373409340963,0x33732e766f6d090a\n"
".quad 0x2c31327225092032,0x090a3b3531722520,0x20696e752e617262,0x335f315f744c2409\n"
".quad 0x744c240a3b343730,0x3a323438335f315f,0x3109636f6c2e090a,0x0a30093931310938\n"
".quad 0x3233732e766f6d09,0x202c323272250920,0x6f7461090a3b312d,0x6c61626f6c672e6d\n"
".quad 0x3233752e6464612e,0x202c333272250920,0x202c5d386472255b,0x2e090a3b32327225\n"
".quad 0x3035093409636f6c,0x702e646c090a3009,0x3233752e6d617261,0x202c313272250920\n"
".quad 0x70616475635f5f5b,0x34315a5f5f6d7261,0x5f676e696e6e6962,0x506a6c656e72656b\n"
".quad 0x736e6f6365523032,0x6e6f697463757274,0x6a50656c706d6153,0x6a6a5f31535f3153\n"
".quad 0x6d754e646972675f,0x0a3b5d736d656c45,0x696e752e61726209,0x5f315f744c240920\n"
".quad 0x4c240a3b34373033,0x303333335f315f74,0x09636f6c2e090a3a,0x090a300933350934\n"
".quad 0x6d617261702e646c,0x722509203233752e,0x635f5f5b202c3132,0x5f6d726170616475\n"
".quad 0x6e6e696234315a5f,0x6e72656b5f676e69,0x65523032506a6c65,0x63757274736e6f63\n"
".quad 0x706d61536e6f6974,0x535f31536a50656c,0x6972675f6a6a5f31,0x6d656c456d754e64\n"
".quad 0x5f744c240a3b5d73,0x0a3a343730335f31,0x093409636f6c2e09,0x756d090a30093635\n"
".quad 0x3436752e6f6c2e6c,0x202c396472250920,0x3b34202c31647225,0x7261702e646c090a\n"
".quad 0x09203436752e6d61,0x5b202c3031647225,0x6170616475635f5f,0x6234315a5f5f6d72\n"
".quad 0x6b5f676e696e6e69,0x32506a6c656e7265,0x74736e6f63655230,0x536e6f6974637572\n"
".quad 0x536a50656c706d61,0x5f6a6a5f31535f31,0x675f79654b786469,0x2e646461090a3b5d\n"
".quad 0x6472250920343675,0x31647225202c3131,0x3b39647225202c30,0x6f6c672e7473090a\n"
".quad 0x203233752e6c6162,0x2b31316472255b09,0x31327225202c5d30,0x09636f6c2e090a3b\n"
".quad 0x090a300937350934,0x6d617261702e646c,0x722509203436752e,0x5f5f5b202c323164\n"
".quad 0x6d72617061647563,0x6e696234315a5f5f,0x72656b5f676e696e,0x523032506a6c656e\n"
".quad 0x757274736e6f6365,0x6d61536e6f697463,0x5f31536a50656c70,0x64695f6a6a5f3153\n"
".quad 0x675f65756c615678,0x2e646461090a3b5d,0x6472250920343675,0x31647225202c3331\n"
".quad 0x3b39647225202c32,0x6f6c672e7473090a,0x203233752e6c6162,0x2b33316472255b09\n"
".quad 0x3b357225202c5d30,0x325f315f744c240a,0x6c2e090a3a323635,0x093935093409636f\n"
".quad 0x3b74697865090a30,0x646e6557444c240a,0x6e696234315a5f5f,0x72656b5f676e696e\n"
".quad 0x523032506a6c656e,0x757274736e6f6365,0x6d61536e6f697463,0x5f31536a50656c70\n"
".quad 0x090a3a6a6a5f3153,0x315a5f202f2f207d,0x676e696e6e696234,0x6a6c656e72656b5f\n"
".quad 0x6e6f636552303250,0x6f69746375727473,0x50656c706d61536e,0x6a5f31535f31536a\n"
".quad 0x746e652e090a0a6a,0x7234315a5f207972,0x6b5f726564726f65,0x6a50696c656e7265\n"
".quad 0x6e6f636552303250,0x6f69746375727473,0x31656c706d61536e,0x41656c706d617337\n"
".quad 0x7572745379617272,0x2e09090a28207463,0x732e206d61726170,0x6475635f5f203233\n"
".quad 0x5a5f5f6d72617061,0x6564726f65723431,0x6c656e72656b5f72,0x65523032506a5069\n"
".quad 0x63757274736e6f63,0x706d61536e6f6974,0x706d61733731656c,0x537961727241656c\n"
".quad 0x2c6e5f7463757274,0x617261702e09090a,0x5f203436752e206d,0x726170616475635f\n"
".quad 0x657234315a5f5f6d,0x656b5f726564726f,0x506a50696c656e72,0x736e6f6365523032\n"
".quad 0x6e6f697463757274,0x3731656c706d6153,0x7241656c706d6173,0x6375727453796172\n"
".quad 0x6c61567864695f74,0x09090a2c675f6575,0x2e206d617261702e,0x75635f5f20343675\n"
".quad 0x5f5f6d7261706164,0x64726f657234315a,0x656e72656b5f7265,0x523032506a50696c\n"
".quad 0x757274736e6f6365,0x6d61536e6f697463,0x6d61733731656c70,0x7961727241656c70\n"
".quad 0x735f746375727453,0x675f73656c706d61,0x7261702e09090a2c,0x67696c612e206d61\n"
".quad 0x2038622e2038206e,0x6170616475635f5f,0x7234315a5f5f6d72,0x6b5f726564726f65\n"
".quad 0x6a50696c656e7265,0x6e6f636552303250,0x6f69746375727473,0x31656c706d61536e\n"
".quad 0x41656c706d617337,0x7572745379617272,0x6574726f735f7463,0x53656c706d615364\n"
".quad 0x5d36315b675f416f,0x722e090a7b090a29,0x203233752e206765,0x090a3b3e393c7225\n"
".quad 0x36752e206765722e,0x36313c6472252034,0x6765722e090a3b3e,0x6625203233662e20\n"
".quad 0x722e090a3b3e383c,0x646572702e206765,0x0a3b3e333c702520,0x093409636f6c2e09\n"
".quad 0x444c240a30093136,0x5f5f6e6967656257,0x64726f657234315a,0x656e72656b5f7265\n"
".quad 0x523032506a50696c,0x757274736e6f6365,0x6d61536e6f697463,0x6d61733731656c70\n"
".quad 0x7961727241656c70,0x0a3a746375727453,0x3233752e766f6d09,0x25202c3172250920\n"
".quad 0x3b782e6469617463,0x33752e766f6d090a,0x202c327225092032,0x3b782e6469746e25\n"
".quad 0x6f6c2e6c756d090a,0x722509203233752e,0x202c317225202c33,0x6f6d090a3b327225\n"
".quad 0x2509203233752e76,0x64697425202c3472,0x646461090a3b782e,0x722509203233752e\n"
".quad 0x202c347225202c35,0x646c090a3b337225,0x752e6d617261702e,0x2c36722509203233\n"
".quad 0x616475635f5f5b20,0x315a5f5f6d726170,0x726564726f657234,0x696c656e72656b5f\n"
".quad 0x6365523032506a50,0x7463757274736e6f,0x6c706d61536e6f69,0x6c706d6173373165\n"
".quad 0x7453796172724165,0x3b5d6e5f74637572,0x6c2e70746573090a,0x2509203233752e65\n"
".quad 0x2c367225202c3170,0x40090a3b35722520,0x2061726220317025,0x315f325f744c2409\n"
".quad 0x6c2e090a3b363230,0x093736093409636f,0x752e747663090a30,0x09203233752e3436\n"
".quad 0x7225202c31647225,0x702e646c090a3b35,0x3436752e6d617261,0x202c326472250920\n"
".quad 0x70616475635f5f5b,0x34315a5f5f6d7261,0x5f726564726f6572,0x50696c656e72656b\n"
".quad 0x6f6365523032506a,0x697463757274736e,0x656c706d61536e6f,0x656c706d61733731\n"
".quad 0x7274537961727241,0x567864695f746375,0x3b5d675f65756c61,0x69772e6c756d090a\n"
".quad 0x09203233752e6564,0x7225202c33647225,0x61090a3b34202c35,0x09203436752e6464\n"
".quad 0x7225202c34647225,0x33647225202c3264,0x6c672e646c090a3b,0x3233752e6c61626f\n"
".quad 0x5b202c3772250920,0x3b5d302b34647225,0x3409636f6c2e090a,0x6c090a3009383609\n"
".quad 0x2e6d617261702e64,0x6472250920343675,0x75635f5f5b202c35,0x5f5f6d7261706164\n"
".quad 0x64726f657234315a,0x656e72656b5f7265,0x523032506a50696c,0x757274736e6f6365\n"
".quad 0x6d61536e6f697463,0x6d61733731656c70,0x7961727241656c70,0x735f746375727453\n"
".quad 0x675f73656c706d61,0x2e747663090a3b5d,0x203233752e343675,0x25202c3664722509\n"
".quad 0x6c756d090a3b3772,0x33752e656469772e,0x2c37647225092032,0x3432202c37722520\n"
".quad 0x752e646461090a3b,0x3864722509203436,0x202c35647225202c,0x6c090a3b37647225\n"
".quad 0x6c61626f6c672e64,0x662509203233662e,0x386472255b202c31,0x646c090a3b5d342b\n"
".quad 0x2e6c61626f6c672e,0x3266250920323366,0x2b386472255b202c,0x2e646c090a3b5d38\n"
".quad 0x662e6c61626f6c67,0x2c33662509203233,0x312b386472255b20,0x2e646c090a3b5d32\n"
".quad 0x662e6c61626f6c67,0x2c34662509203233,0x312b386472255b20,0x2e646c090a3b5d36\n"
".quad 0x662e6c61626f6c67,0x2c35662509203233,0x322b386472255b20,0x6f6c2e090a3b5d30\n"
".quad 0x3009303809340963,0x7261702e646c090a,0x09203436752e6d61,0x5f5b202c39647225\n"
".quad 0x726170616475635f,0x657234315a5f5f6d,0x656b5f726564726f,0x506a50696c656e72\n"
".quad 0x736e6f6365523032,0x6e6f697463757274,0x3731656c706d6153,0x7241656c706d6173\n"
".quad 0x6375727453796172,0x646574726f735f74,0x6f53656c706d6153,0x0a3b5d302b675f41\n"
".quad 0x6469772e6c756d09,0x2509203233752e65,0x7225202c30316472,0x61090a3b38202c35\n"
".quad 0x09203436752e6464,0x25202c3131647225,0x647225202c396472,0x2e646c090a3b3031\n"
".quad 0x662e6c61626f6c67,0x2c36662509203233,0x302b386472255b20,0x672e7473090a3b5d\n"
".quad 0x32762e6c61626f6c,0x255b09203233662e,0x2c5d302b31316472,0x66252c3666257b20\n"
".quad 0x6f6c2e090a3b7d31,0x3009313809340963,0x7261702e646c090a,0x09203436752e6d61\n"
".quad 0x5b202c3231647225,0x6170616475635f5f,0x7234315a5f5f6d72,0x6b5f726564726f65\n"
".quad 0x6a50696c656e7265,0x6e6f636552303250,0x6f69746375727473,0x31656c706d61536e\n"
".quad 0x41656c706d617337,0x7572745379617272,0x6574726f735f7463,0x53656c706d615364\n"
".quad 0x3b5d382b675f416f,0x69772e6c756d090a,0x09203233752e6564,0x25202c3331647225\n"
".quad 0x0a3b3631202c3572,0x3436752e64646109,0x2c34316472250920,0x202c323164722520\n"
".quad 0x090a3b3331647225,0x61626f6c672e7473,0x3233662e34762e6c,0x34316472255b0920\n"
".quad 0x66257b202c5d302b,0x66252c3366252c32,0x0a3b7d3566252c34,0x30315f325f744c24\n"
".quad 0x6f6c2e090a3a3632,0x3009333809340963,0x0a3b74697865090a,0x5f646e6557444c24\n"
".quad 0x726f657234315a5f,0x6e72656b5f726564,0x3032506a50696c65,0x7274736e6f636552\n"
".quad 0x61536e6f69746375,0x61733731656c706d,0x61727241656c706d,0x3a74637572745379\n"
".quad 0x5f202f2f207d090a,0x64726f657234315a,0x656e72656b5f7265,0x523032506a50696c\n"
".quad 0x757274736e6f6365,0x6d61536e6f697463,0x6d61733731656c70,0x7961727241656c70\n"
".quad 0x090a746375727453,0x2e2074736e6f632e,0x6f74756320323366,0x090a3b635f326666\n"
".quad 0x2e2074736e6f632e,0x6f74756320323366,0x2e090a3b635f6666,0x662e2074736e6f63\n"
".quad 0x65766f315f203233,0x3266666f74754372,0x652e090a0a3b635f,0x315a5f207972746e\n"
".quad 0x6e69646469726732,0x7337315550475f67,0x727241656c706d61,0x7463757274537961\n"
".quad 0x616f6c6636506a50,0x0a28206666503274,0x6d617261702e0909,0x206e67696c612e20\n"
".quad 0x5f5f2038622e2038,0x6d72617061647563,0x69726732315a5f5f,0x50475f676e696464\n"
".quad 0x6c706d6173373155,0x7453796172724165,0x36506a5074637572,0x66503274616f6c66\n"
".quad 0x646574726f735f66,0x6f53656c706d6153,0x2c5d36315b675f41,0x617261702e09090a\n"
".quad 0x5f203436752e206d,0x726170616475635f,0x726732315a5f5f6d,0x475f676e69646469\n"
".quad 0x706d617337315550,0x537961727241656c,0x506a507463757274,0x503274616f6c6636\n"
".quad 0x74536e69625f6666,0x5f72646441747261,0x61702e09090a2c67,0x3436752e206d6172\n"
".quad 0x70616475635f5f20,0x32315a5f5f6d7261,0x676e696464697267,0x617337315550475f\n"
".quad 0x61727241656c706d,0x5074637572745379,0x74616f6c6636506a,0x6972675f66665032\n"
".quad 0x2c675f6174614464,0x617261702e09090a,0x5f203436752e206d,0x726170616475635f\n"
".quad 0x726732315a5f5f6d,0x475f676e69646469,0x706d617337315550,0x537961727241656c\n"
".quad 0x506a507463757274,0x503274616f6c6636,0x6c706d61735f6666,0x797469736e654465\n"
".quad 0x702e09090a2c675f,0x33662e206d617261,0x616475635f5f2032,0x315a5f5f6d726170\n"
".quad 0x6e69646469726732,0x7337315550475f67,0x727241656c706d61,0x7463757274537961\n"
".quad 0x616f6c6636506a50,0x65625f6666503274,0x090a7b090a296174,0x33752e206765722e\n"
".quad 0x3e39383c72252032,0x206765722e090a3b,0x647225203436752e,0x2e090a3b3e37363c\n"
".quad 0x3233662e20676572,0x3e3333323c662520,0x206765722e090a3b,0x646625203436662e\n"
".quad 0x2e090a3b3e32333c,0x6572702e20676572,0x3e31323c70252064,0x726168732e090a3b\n"
".quad 0x67696c612e206465,0x2038622e2034206e,0x5f5f616475635f5f,0x6f6c5f616475635f\n"
".quad 0x5f7261765f6c6163,0x33335f3137383633,0x6e6f635f6e6f6e5f,0x735f6364735f7473\n"
".quad 0x5d3635325b303431,0x726168732e090a3b,0x67696c612e206465,0x2038622e2034206e\n"
".quad 0x5f5f616475635f5f,0x6f6c5f616475635f,0x5f7261765f6c6163,0x33335f3936383633\n"
".quad 0x6e6f635f6e6f6e5f,0x33735f796b5f7473,0x3b5d3635325b3639,0x65726168732e090a\n"
".quad 0x6e67696c612e2064,0x5f2038622e203420,0x5f5f5f616475635f,0x636f6c5f61647563\n"
".quad 0x335f7261765f6c61,0x5f33335f37363836,0x736e6f635f6e6f6e,0x735f67616d695f74\n"
".quad 0x5d3635325b323536,0x726168732e090a3b,0x67696c612e206465,0x2038622e2034206e\n"
".quad 0x5f5f616475635f5f,0x6f6c5f616475635f,0x5f7261765f6c6163,0x33335f3037383633\n"
".quad 0x6e6f635f6e6f6e5f,0x39735f7a6b5f7473,0x3b5d3635325b3830,0x65726168732e090a\n"
".quad 0x6e67696c612e2064,0x5f2038622e203420,0x5f5f5f616475635f,0x636f6c5f61647563\n"
".quad 0x335f7261765f6c61,0x5f33335f38363836,0x736e6f635f6e6f6e,0x3131735f786b5f74\n"
".quad 0x3b5d3635325b3436,0x65726168732e090a,0x6e67696c612e2064,0x5f2038622e203420\n"
".quad 0x5f5f5f616475635f,0x636f6c5f61647563,0x335f7261765f6c61,0x5f33335f36363836\n"
".quad 0x736e6f635f6e6f6e,0x735f6c6165725f74,0x3635325b30323431,0x636f6c2e090a3b5d\n"
".quad 0x3009393031093409,0x67656257444c240a,0x6732315a5f5f6e69,0x5f676e6964646972\n"
".quad 0x6d61733731555047,0x7961727241656c70,0x6a50746375727453,0x3274616f6c663650\n"
".quad 0x6f6d090a3a666650,0x2509203233752e76,0x69746e25202c3172,0x6f6d090a3b782e64\n"
".quad 0x2509203233752e76,0x61746325202c3272,0x6d090a3b782e6469,0x33752e6f6c2e6c75\n"
".quad 0x202c337225092032,0x317225202c327225,0x752e646461090a3b,0x2c34722509203233\n"
".quad 0x7225202c33722520,0x2e747663090a3b31,0x752e3233662e6e72,0x2c31662509203233\n"
".quad 0x6c090a3b34722520,0x2e74736e6f632e64,0x3266250920323366,0x666f7475635b202c\n"
".quad 0x61090a3b5d635f66,0x09203233662e6464,0x316625202c336625,0x090a3b326625202c\n"
".quad 0x2e697a722e747663,0x203233662e323373,0x6625202c35722509,0x632e646c090a3b33\n"
".quad 0x3233732e74736e6f,0x5b202c3672250920,0x657a695364697267,0x090a3b5d302b635f\n"
".quad 0x2e746c2e70746573,0x3170250920323373,0x25202c357225202c,0x702540090a3b3672\n"
".quad 0x2409206172622031,0x3832315f335f744c,0x6f6c2e090a3b3230,0x0934333109340963\n"
".quad 0x732e627573090a30,0x2c37722509203233,0x3b31202c36722520,0x6e752e617262090a\n"
".quad 0x335f744c24092069,0x0a3b36343532315f,0x32315f335f744c24,0x6f6d090a3a323038\n"
".quad 0x2509203233732e76,0x3b357225202c3772,0x315f335f744c240a,0x6d090a3a36343532\n"
".quad 0x09203233752e766f,0x746e25202c387225,0x6c090a3b792e6469,0x2e74736e6f632e64\n"
".quad 0x3972250920323375,0x53646972675b202c,0x5d342b635f657a69,0x752e766964090a3b\n"
".quad 0x3031722509203233,0x25202c397225202c,0x766f6d090a3b3872,0x722509203233752e\n"
".quad 0x61746325202c3131,0x72090a3b792e6469,0x09203233752e6d65,0x7225202c32317225\n"
".quad 0x30317225202c3131,0x6c2e6c756d090a3b,0x2509203233752e6f,0x317225202c333172\n"
".quad 0x0a3b387225202c32,0x3233752e64646109,0x202c343172250920,0x7225202c33317225\n"
".quad 0x2e747663090a3b38,0x752e3233662e6e72,0x2c34662509203233,0x090a3b3431722520\n"
".quad 0x203233662e646461,0x6625202c35662509,0x0a3b326625202c34,0x697a722e74766309\n"
".quad 0x3233662e3233732e,0x202c353172250920,0x6573090a3b356625,0x33732e746c2e7074\n"
".quad 0x202c327025092032,0x7225202c35317225,0x32702540090a3b39,0x4c24092061726220\n"
".quad 0x313333315f335f74,0x636f6c2e090a3b34,0x3009393331093409,0x33732e627573090a\n"
".quad 0x2c36317225092032,0x3b31202c39722520,0x6e752e617262090a,0x335f744c24092069\n"
".quad 0x0a3b38353033315f,0x33315f335f744c24,0x6f6d090a3a343133,0x2509203233732e76\n"
".quad 0x317225202c363172,0x335f744c240a3b35,0x0a3a38353033315f,0x3233752e76696409\n"
".quad 0x202c373172250920,0x7225202c31317225,0x766f6d090a3b3031,0x722509203233752e\n"
".quad 0x69746e25202c3831,0x756d090a3b7a2e64,0x3233752e6f6c2e6c,0x202c393172250920\n"
".quad 0x7225202c37317225,0x646461090a3b3831,0x722509203233752e,0x39317225202c3032\n"
".quad 0x0a3b38317225202c,0x2e6f6c2e6c756d09,0x3272250920323375,0x2c30327225202c31\n"
".quad 0x747663090a3b3420,0x2e3233662e6e722e,0x3666250920323375,0x0a3b31327225202c\n"
".quad 0x3233662e64646109,0x25202c3766250920,0x3b326625202c3666,0x7a722e747663090a\n"
".quad 0x33662e3233732e69,0x2c32327225092032,0x6c090a3b37662520,0x2e74736e6f632e64\n"
".quad 0x3272250920323373,0x646972675b202c33,0x382b635f657a6953,0x70746573090a3b5d\n"
".quad 0x203233732e746c2e,0x7225202c33702509,0x33327225202c3232,0x2033702540090a3b\n"
".quad 0x744c240920617262,0x36323833315f335f,0x09636f6c2e090a3b,0x0a30093434310934\n"
".quad 0x3233732e62757309,0x202c343272250920,0x3b31202c33327225,0x6e752e617262090a\n"
".quad 0x335f744c24092069,0x0a3b30373533315f,0x33315f335f744c24,0x6f6d090a3a363238\n"
".quad 0x2509203233732e76,0x327225202c343272,0x335f744c240a3b32,0x0a3a30373533315f\n"
".quad 0x093409636f6c2e09,0x63090a3009313731,0x662e6970722e7476,0x09203233662e3233\n"
".quad 0x326625202c386625,0x6c2e6c756d090a3b,0x2509203233752e6f,0x317225202c353272\n"
".quad 0x63090a3b34202c39,0x33662e6e722e7476,0x2509203233732e32,0x35327225202c3966\n"
".quad 0x662e627573090a3b,0x3031662509203233,0x25202c396625202c,0x747663090a3b3866\n"
".quad 0x3233732e697a722e,0x722509203233662e,0x30316625202c3632,0x732e766f6d090a3b\n"
".quad 0x3732722509203233,0x616d090a3b30202c,0x2509203233732e78,0x327225202c383272\n"
".quad 0x3b37327225202c36,0x33732e766f6d090a,0x2c39327225092032,0x090a3b3832722520\n"
".quad 0x203233752e766f6d,0x25202c3033722509,0x090a3b782e646974,0x203233752e646461\n"
".quad 0x25202c3133722509,0x30337225202c3372,0x752e766f6d090a3b,0x3233722509203233\n"
".quad 0x792e64697425202c,0x752e646461090a3b,0x3333722509203233,0x202c33317225202c\n"
".quad 0x6d090a3b32337225,0x09203233752e766f,0x7425202c34337225,0x61090a3b7a2e6469\n"
".quad 0x09203233752e6464,0x7225202c35337225,0x34337225202c3532,0x2e70746573090a3b\n"
".quad 0x09203233732e746c,0x327225202c347025,0x3b38327225202c34,0x622034702540090a\n"
".quad 0x5f744c2409206172,0x3b38333730325f33,0x6e722e747663090a,0x3233732e3233662e\n"
".quad 0x202c313166250920,0x7573090a3b337225,0x2509203233662e62,0x316625202c323166\n"
".quad 0x0a3b386625202c31,0x697a722e74766309,0x3233662e3233732e,0x202c363372250920\n"
".quad 0x6d090a3b32316625,0x09203233732e766f,0x3b30202c37337225,0x33732e78616d090a\n"
".quad 0x2c38337225092032,0x25202c3633722520,0x756d090a3b373372,0x3233752e6f6c2e6c\n"
".quad 0x202c393372250920,0x7225202c34337225,0x2e646461090a3b38,0x3472250920323375\n"
".quad 0x2c39337225202c30,0x090a3b3233722520,0x752e6f6c2e6c756d,0x3134722509203233\n"
".quad 0x202c30347225202c,0x6461090a3b317225,0x2509203233752e64,0x347225202c323472\n"
".quad 0x3b30337225202c31,0x33752e646461090a,0x2c33347225092032,0x25202c3533722520\n"
".quad 0x6461090a3b383172,0x2509203233752e64,0x347225202c343472,0x3b38317225202c33\n"
".quad 0x33752e646461090a,0x2c35347225092032,0x25202c3434722520,0x7573090a3b383172\n"
".quad 0x2509203233732e62,0x327225202c363472,0x3b38327225202c34,0x33732e646461090a\n"
".quad 0x2c37347225092032,0x31202c3634722520,0x732e646461090a3b,0x3834722509203233\n"
".quad 0x202c34327225202c,0x2e747663090a3b31,0x732e3233662e6e72,0x3331662509203233\n"
".quad 0x0a3b33317225202c,0x3233662e62757309,0x202c343166250920,0x6625202c33316625\n"
".quad 0x2e747663090a3b38,0x2e3233732e697a72,0x3472250920323366,0x3b34316625202c39\n"
".quad 0x33732e766f6d090a,0x2c30357225092032,0x78616d090a3b3020,0x722509203233732e\n"
".quad 0x39347225202c3135,0x0a3b30357225202c,0x65672e7074657309,0x702509203233732e\n"
".quad 0x2c36317225202c35,0x090a3b3135722520,0x74736e6f632e646c,0x722509203233732e\n"
".quad 0x7a69735b202c3235,0x3b5d635f79785f65,0x33662e766f6d090a,0x2c35316625092032\n"
".quad 0x3030303030663020,0x202020203b303030,0x6d090a30202f2f09,0x09203233662e766f\n"
".quad 0x6630202c36316625,0x3030303030303030,0x2f2f09202020203b,0x2e766f6d090a3020\n"
".quad 0x3166250920323366,0x3030306630202c37,0x20203b3030303030,0x0a30202f2f092020\n"
".quad 0x3233662e766f6d09,0x202c383166250920,0x3030303030306630,0x09202020203b3030\n"
".quad 0x6f6d090a30202f2f,0x2509203233662e76,0x306630202c393166,0x3b30303030303030\n"
".quad 0x202f2f0920202020,0x662e766f6d090a30,0x3032662509203233,0x303030306630202c\n"
".quad 0x2020203b30303030,0x090a30202f2f0920,0x203233662e766f6d,0x30202c3132662509\n"
".quad 0x3030303030303066,0x2f09202020203b30,0x766f6d090a30202f,0x662509203233662e\n"
".quad 0x30306630202c3232,0x203b303030303030,0x30202f2f09202020,0x33662e766f6d090a\n"
".quad 0x2c33326625092032,0x3030303030663020,0x202020203b303030,0x6d090a30202f2f09\n"
".quad 0x09203233662e766f,0x6630202c34326625,0x3030303030303030,0x2f2f09202020203b\n"
".quad 0x2e766f6d090a3020,0x3266250920323366,0x3030306630202c35,0x20203b3030303030\n"
".quad 0x0a30202f2f092020,0x3233662e766f6d09,0x202c363266250920,0x3030303030306630\n"
".quad 0x09202020203b3030,0x6f6d090a30202f2f,0x2509203436752e76,0x635f5f202c316472\n"
".quad 0x75635f5f5f616475,0x6c61636f6c5f6164,0x3836335f7261765f,0x6f6e5f33335f3137\n"
".quad 0x5f74736e6f635f6e,0x303431735f636473,0x752e766f6d090a3b,0x3264722509203436\n"
".quad 0x616475635f5f202c,0x5f616475635f5f5f,0x61765f6c61636f6c,0x5f39363836335f72\n"
".quad 0x635f6e6f6e5f3333,0x5f796b5f74736e6f,0x6d090a3b36393373,0x09203436752e766f\n"
".quad 0x5f5f202c33647225,0x635f5f5f61647563,0x61636f6c5f616475,0x36335f7261765f6c\n"
".quad 0x6e5f33335f373638,0x74736e6f635f6e6f,0x36735f67616d695f,0x766f6d090a3b3235\n"
".quad 0x722509203436752e,0x75635f5f202c3464,0x6475635f5f5f6164,0x5f6c61636f6c5f61\n"
".quad 0x373836335f726176,0x6e6f6e5f33335f30,0x6b5f74736e6f635f,0x0a3b383039735f7a\n"
".quad 0x3436752e766f6d09,0x202c356472250920,0x5f5f616475635f5f,0x6f6c5f616475635f\n"
".quad 0x5f7261765f6c6163,0x33335f3836383633,0x6e6f635f6e6f6e5f,0x31735f786b5f7473\n"
".quad 0x6f6d090a3b343631,0x2509203436752e76,0x635f5f202c366472,0x75635f5f5f616475\n"
".quad 0x6c61636f6c5f6164,0x3836335f7261765f,0x6f6e5f33335f3636,0x5f74736e6f635f6e\n"
".quad 0x3431735f6c616572,0x766f6d090a3b3032,0x722509203233732e,0x37347225202c3335\n"
".quad 0x5f335f744c240a3b,0x200a3a3439353431,0x3e706f6f6c3c2f2f,0x6f6220706f6f4c20\n"
".quad 0x20656e696c207964,0x73656e202c313731,0x70656420676e6974,0x65202c31203a6874\n"
".quad 0x646574616d697473,0x6974617265746920,0x6b6e75203a736e6f,0x6c2e090a6e776f6e\n"
".quad 0x323731093409636f,0x2e766f6d090a3009,0x3572250920323373,0x3b31357225202c34\n"
".quad 0x203570252140090a,0x744c240920617262,0x30353834315f335f,0x6f6c3c2f2f200a3b\n"
".quad 0x74726150203e706f,0x706f6f6c20666f20,0x696c2079646f6220,0x202c31373120656e\n"
".quad 0x62616c2064616568,0x744c242064656c65,0x34393534315f335f,0x33732e627573090a\n"
".quad 0x2c35357225092032,0x25202c3631722520,0x6461090a3b313572,0x2509203233732e64\n"
".quad 0x357225202c363572,0x61090a3b31202c35,0x09203233732e6464,0x7225202c37357225\n"
".quad 0x090a3b31202c3631,0x6d617261702e646c,0x722509203436752e,0x635f5f5b202c3764\n"
".quad 0x5f6d726170616475,0x6469726732315a5f,0x5550475f676e6964,0x656c706d61733731\n"
".quad 0x7274537961727241,0x6636506a50746375,0x6666503274616f6c,0x726174536e69625f\n"
".quad 0x5d675f7264644174,0x6c2e6c756d090a3b,0x2509203233732e6f,0x357225202c383572\n"
".quad 0x0a3b367225202c31,0x3436732e74766309,0x722509203233732e,0x38357225202c3864\n"
".quad 0x6c2e6c756d090a3b,0x2509203233732e6f,0x327225202c393572,0x3b32357225202c39\n"
".quad 0x36732e747663090a,0x2509203233732e34,0x357225202c396472,0x2e646461090a3b39\n"
".quad 0x6472250920343675,0x38647225202c3031,0x0a3b39647225202c,0x2e6f6c2e6c756d09\n"
".quad 0x6472250920343675,0x31647225202c3131,0x63090a3b34202c30,0x732e3436732e7476\n"
".quad 0x3164722509203233,0x0a3b367225202c32,0x6469772e6c756d09,0x2509203233732e65\n"
".quad 0x7225202c33316472,0x63090a3b34202c36,0x732e3436732e7476,0x3164722509203233\n"
".quad 0x0a3b377225202c34,0x6469772e6c756d09,0x2509203233732e65,0x7225202c35316472\n"
".quad 0x61090a3b34202c37,0x09203436732e6464,0x25202c3631647225,0x647225202c376472\n"
".quad 0x646461090a3b3531,0x722509203436732e,0x647225202c373164,0x31647225202c3131\n"
".quad 0x2e646461090a3b36,0x6472250920343675,0x31647225202c3831,0x3b37647225202c31\n"
".quad 0x36732e747663090a,0x2509203233732e34,0x7225202c39316472,0x6c756d090a3b3833\n"
".quad 0x33732e656469772e,0x3032647225092032,0x202c38337225202c,0x2e646461090a3b34\n"
".quad 0x6472250920343675,0x31647225202c3132,0x3032647225202c38,0x6f6c3c2f2f200a3b\n"
".quad 0x74726150203e706f,0x706f6f6c20666f20,0x696c2079646f6220,0x202c31373120656e\n"
".quad 0x62616c2064616568,0x744c242064656c65,0x34393534315f335f,0x33732e766f6d090a\n"
".quad 0x2c30367225092032,0x240a3b3635722520,0x3335315f335f744c,0x3c2f2f200a3a3236\n"
".quad 0x6f4c203e706f6f6c,0x2079646f6220706f,0x32373120656e696c,0x6e697473656e202c\n"
".quad 0x3a68747065642067,0x69747365202c3220,0x746920646574616d,0x736e6f6974617265\n"
".quad 0x776f6e6b6e75203a,0x09636f6c2e090a6e,0x0a30093437310934,0x6f6c672e75646c09\n"
".quad 0x203233752e6c6162,0x5b202c3136722509,0x5d302b3132647225,0x09636f6c2e090a3b\n"
".quad 0x0a30093537310934,0x6f6c672e75646c09,0x203233752e6c6162,0x5b202c3236722509\n"
".quad 0x5d342b3731647225,0x752e627573090a3b,0x3336722509203233,0x202c32367225202c\n"
".quad 0x61090a3b31367225,0x09203233752e6464,0x7225202c34367225,0x0a3b3336202c3336\n"
".quad 0x3233752e72687309,0x202c353672250920,0x3b36202c34367225,0x33752e766f6d090a\n"
".quad 0x2c36367225092032,0x746573090a3b3020,0x3233752e71652e70,0x25202c3670250920\n"
".quad 0x367225202c353672,0x36702540090a3b36,0x4c24092061726220,0x313635315f335f74\n"
".quad 0x6c3c2f2f200a3b38,0x726150203e706f6f,0x6f6f6c20666f2074,0x6c2079646f622070\n"
".quad 0x2c32373120656e69,0x616c206461656820,0x4c242064656c6562,0x363335315f335f74\n"
".quad 0x732e766f6d090a32,0x3736722509203233,0x0a3b35367225202c,0x3233732e766f6d09\n"
".quad 0x202c383672250920,0x6c3c2f2f200a3b30,0x726150203e706f6f,0x6f6f6c20666f2074\n"
".quad 0x6c2079646f622070,0x2c32373120656e69,0x616c206461656820,0x4c242064656c6562\n"
".quad 0x363335315f335f74,0x732e766f6d090a32,0x3936722509203233,0x0a3b37367225202c\n"
".quad 0x36315f335f744c24,0x2f2f200a3a303331,0x4c203e706f6f6c3c,0x79646f6220706f6f\n"
".quad 0x373120656e696c20,0x697473656e202c35,0x687470656420676e,0x747365202c33203a\n"
".quad 0x6920646574616d69,0x6e6f697461726574,0x6f6e6b6e75203a73,0x636f6c2e090a6e77\n"
".quad 0x3009303831093409,0x79732e726162090a,0x090a3b300920636e,0x203233622e6c6873\n"
".quad 0x25202c3037722509,0x0a3b36202c383672,0x3233732e62757309,0x202c313772250920\n"
".quad 0x7225202c33367225,0x766f6d090a3b3037,0x722509203233752e,0x0a3b3436202c3237\n"
".quad 0x3233752e6e696d09,0x202c333772250920,0x7225202c31377225,0x746573090a3b3237\n"
".quad 0x3233732e656c2e70,0x25202c3770250920,0x347225202c333772,0x37702540090a3b32\n"
".quad 0x4c24092061726220,0x383336315f335f74,0x6c3c2f2f200a3b36,0x726150203e706f6f\n"
".quad 0x6f6f6c20666f2074,0x6c2079646f622070,0x2c35373120656e69,0x616c206461656820\n"
".quad 0x4c242064656c6562,0x333136315f335f74,0x09636f6c2e090a30,0x0a30093538310934\n"
".quad 0x3233732e64646109,0x202c343772250920,0x7225202c30377225,0x747663090a3b3234\n"
".quad 0x3233732e3436732e,0x2c32326472250920,0x090a3b3234722520,0x656469772e6c756d\n"
".quad 0x722509203233732e,0x347225202c333264,0x61090a3b34202c32,0x09203233752e6464\n"
".quad 0x7225202c35377225,0x31367225202c3437,0x752e747663090a3b,0x09203233752e3436\n"
".quad 0x25202c3432647225,0x646c090a3b353772,0x752e6d617261702e,0x3264722509203436\n"
".quad 0x75635f5f5b202c35,0x5f5f6d7261706164,0x646469726732315a,0x315550475f676e69\n"
".quad 0x41656c706d617337,0x7572745379617272,0x6c6636506a507463,0x5f6666503274616f\n"
".quad 0x6153646574726f73,0x5f416f53656c706d,0x6d090a3b5d302b67,0x2e656469772e6c75\n"
".quad 0x6472250920323375,0x35377225202c3632,0x6461090a3b38202c,0x2509203436752e64\n"
".quad 0x7225202c37326472,0x647225202c353264,0x2e646c090a3b3632,0x762e6c61626f6c67\n"
".quad 0x7b09203233662e32,0x3266252c37326625,0x6472255b202c7d38,0x090a3b5d302b3732\n"
".quad 0x203436752e646461,0x202c383264722509,0x25202c3332647225,0x7473090a3b366472\n"
".quad 0x2e6465726168732e,0x72255b0920323366,0x202c5d302b383264,0x2e090a3b37326625\n"
".quad 0x3831093409636f6c,0x646461090a300936,0x722509203436752e,0x647225202c393264\n"
".quad 0x33647225202c3332,0x68732e7473090a3b,0x3233662e64657261,0x39326472255b0920\n"
".quad 0x326625202c5d302b,0x636f6c2e090a3b38,0x3009373831093409,0x7261702e646c090a\n"
".quad 0x09203436752e6d61,0x5b202c3033647225,0x6170616475635f5f,0x6732315a5f5f6d72\n"
".quad 0x5f676e6964646972,0x6d61733731555047,0x7961727241656c70,0x6a50746375727453\n"
".quad 0x3274616f6c663650,0x74726f735f666650,0x656c706d61536465,0x5d382b675f416f53\n"
".quad 0x772e6c756d090a3b,0x203233752e656469,0x202c313364722509,0x3631202c35377225\n"
".quad 0x752e646461090a3b,0x3364722509203436,0x3033647225202c32,0x3b3133647225202c\n"
".quad 0x6f6c672e646c090a,0x662e34762e6c6162,0x3266257b09203233,0x252c303366252c39\n"
".quad 0x323366252c313366,0x336472255b202c7d,0x61090a3b5d302b32,0x09203436752e6464\n"
".quad 0x25202c3333647225,0x7225202c33326472,0x2e7473090a3b3564,0x662e646572616873\n"
".quad 0x6472255b09203233,0x25202c5d302b3333,0x6c2e090a3b393266,0x383831093409636f\n"
".quad 0x2e646461090a3009,0x6472250920343675,0x32647225202c3433,0x3b32647225202c33\n"
".quad 0x6168732e7473090a,0x203233662e646572,0x2b34336472255b09,0x30336625202c5d30\n"
".quad 0x09636f6c2e090a3b,0x0a30093938310934,0x3436752e64646109,0x2c35336472250920\n"
".quad 0x202c333264722520,0x73090a3b34647225,0x6465726168732e74,0x255b09203233662e\n"
".quad 0x2c5d302b35336472,0x090a3b3133662520,0x31093409636f6c2e,0x6461090a30093039\n"
".quad 0x2509203436752e64,0x7225202c36336472,0x647225202c333264,0x732e7473090a3b31\n"
".quad 0x33662e6465726168,0x336472255b092032,0x6625202c5d302b36,0x5f744c240a3b3233\n"
".quad 0x3a36383336315f33,0x6f6f6c3c2f2f200a,0x2074726150203e70,0x20706f6f6c20666f\n"
".quad 0x6e696c2079646f62,0x68202c3537312065,0x6562616c20646165,0x5f744c242064656c\n"
".quad 0x0a30333136315f33,0x093409636f6c2e09,0x62090a3009323931,0x20636e79732e7261\n"
".quad 0x766f6d090a3b3009,0x722509203233752e,0x090a3b30202c3637,0x2e656c2e70746573\n"
".quad 0x3870250920323373,0x202c33377225202c,0x40090a3b36377225,0x2061726220387025\n"
".quad 0x315f335f744c2409,0x2f200a3b38393836,0x203e706f6f6c3c2f,0x20666f2074726150\n"
".quad 0x646f6220706f6f6c,0x3120656e696c2079,0x64616568202c3537,0x64656c6562616c20\n"
".quad 0x315f335f744c2420,0x6f6d090a30333136,0x2509203233732e76,0x377225202c373772\n"
".quad 0x2e766f6d090a3b33,0x6472250920343673,0x090a3b30202c3733,0x203233732e766f6d\n"
".quad 0x30202c3837722509,0x6f6c3c2f2f200a3b,0x74726150203e706f,0x706f6f6c20666f20\n"
".quad 0x696c2079646f6220,0x202c35373120656e,0x62616c2064616568,0x744c242064656c65\n"
".quad 0x30333136315f335f,0x33732e766f6d090a,0x2c39377225092032,0x240a3b3737722520\n"
".quad 0x3437315f335f744c,0x3c2f2f200a3a3031,0x6f4c203e706f6f6c,0x2079646f6220706f\n"
".quad 0x32393120656e696c,0x6e697473656e202c,0x3a68747065642067,0x69747365202c3420\n"
".quad 0x746920646574616d,0x736e6f6974617265,0x776f6e6b6e75203a,0x09636f6c2e090a6e\n"
".quad 0x0a30093539310934,0x3436752e64646109,0x2c38336472250920,0x202c373364722520\n"
".quad 0x6c090a3b36647225,0x6465726168732e64,0x662509203233662e,0x6472255b202c3333\n"
".quad 0x090a3b5d302b3833,0x31093409636f6c2e,0x6461090a30093639,0x2509203436752e64\n"
".quad 0x7225202c39336472,0x647225202c373364,0x732e646c090a3b33,0x33662e6465726168\n"
".quad 0x2c34336625092032,0x2b39336472255b20,0x6f6c2e090a3b5d30,0x0937393109340963\n"
".quad 0x752e646461090a30,0x3464722509203436,0x3733647225202c30,0x0a3b31647225202c\n"
".quad 0x726168732e646c09,0x09203233662e6465,0x255b202c35336625,0x3b5d302b30346472\n"
".quad 0x3409636f6c2e090a,0x090a300935393109,0x2e3436662e747663,0x6466250920323366\n"
".quad 0x3b33336625202c31,0x36662e766f6d090a,0x2c32646625092034,0x3030303030643020\n"
".quad 0x3030303030303030,0x202f2f093b303030,0x2e70746573090a30,0x203436662e75656e\n"
".quad 0x6625202c39702509,0x32646625202c3164,0x2039702540090a3b,0x5f4c240920617262\n"
".quad 0x3b30393232315f33,0x6f6f6c3c2f2f200a,0x2074726150203e70,0x20706f6f6c20666f\n"
".quad 0x6e696c2079646f62,0x68202c3239312065,0x6562616c20646165,0x5f744c242064656c\n"
".quad 0x0a30313437315f33,0x3436662e74766309,0x662509203233662e,0x34336625202c3364\n"
".quad 0x662e766f6d090a3b,0x3464662509203436,0x303030306430202c,0x3030303030303030\n"
".quad 0x2f2f093b30303030,0x70746573090a3020,0x3436662e75656e2e,0x202c303170250920\n"
".quad 0x6625202c33646625,0x252140090a3b3464,0x2061726220303170,0x325f335f744c2409\n"
".quad 0x4c240a3b38313032,0x30393232315f335f,0x6f6c3c2f2f200a3a,0x74726150203e706f\n"
".quad 0x706f6f6c20666f20,0x696c2079646f6220,0x202c32393120656e,0x62616c2064616568\n"
".quad 0x744c242064656c65,0x30313437315f335f,0x36662e747663090a,0x2509203233662e34\n"
".quad 0x336625202c356466,0x2e766f6d090a3b35,0x6466250920343666,0x3030306430202c36\n"
".quad 0x3030303030303030,0x2f093b3030303030,0x746573090a30202f,0x36662e75656e2e70\n"
".quad 0x2c31317025092034,0x25202c3564662520,0x2140090a3b366466,0x6172622031317025\n"
".quad 0x5f335f744c240920,0x200a3b3831303232,0x3e706f6f6c3c2f2f,0x666f207472615020\n"
".quad 0x6f6220706f6f6c20,0x20656e696c207964,0x616568202c323931,0x656c6562616c2064\n"
".quad 0x5f335f744c242064,0x63090a3031343731,0x33662e6e722e7476,0x2509203233732e32\n"
".quad 0x337225202c363366,0x2e646461090a3b31,0x6472250920343675,0x33647225202c3134\n"
".quad 0x3b35647225202c37,0x36752e646461090a,0x3234647225092034,0x2c3733647225202c\n"
".quad 0x090a3b3264722520,0x203436752e646461,0x202c333464722509,0x25202c3733647225\n"
".quad 0x646c090a3b346472,0x2e6465726168732e,0x3366250920323366,0x346472255b202c37\n"
".quad 0x6c090a3b5d302b31,0x6465726168732e64,0x662509203233662e,0x6472255b202c3833\n"
".quad 0x090a3b5d302b3234,0x65726168732e646c,0x2509203233662e64,0x72255b202c393366\n"
".quad 0x0a3b5d302b333464,0x2e6e722e74766309,0x203233732e323366,0x25202c3034662509\n"
".quad 0x7573090a3b333372,0x2509203233662e62,0x336625202c313466,0x3b36336625202c37\n"
".quad 0x33662e627573090a,0x2c32346625092032,0x25202c3833662520,0x7663090a3b303466\n"
".quad 0x3233662e6e722e74,0x662509203233732e,0x35337225202c3334,0x662e6c756d090a3b\n"
".quad 0x3434662509203233,0x202c31346625202c,0x73090a3b31346625,0x09203233662e6275\n"
".quad 0x6625202c35346625,0x33346625202c3933,0x722e616d66090a3b,0x2509203233662e6e\n"
".quad 0x346625202c363466,0x2c32346625202c32,0x090a3b3434662520,0x662e6e722e616d66\n"
".quad 0x3734662509203233,0x202c35346625202c,0x6625202c35346625,0x2e646c090a3b3634\n"
".quad 0x33662e74736e6f63,0x2c38346625092032,0x66666f7475635b20,0x73090a3b5d635f32\n"
".quad 0x662e746c2e707465,0x3231702509203233,0x202c37346625202c,0x40090a3b38346625\n"
".quad 0x7262203231702521,0x335f744c24092061,0x0a3b36363637315f,0x706f6f6c3c2f2f20\n"
".quad 0x6f2074726150203e,0x6220706f6f6c2066,0x656e696c2079646f,0x6568202c32393120\n"
".quad 0x6c6562616c206461,0x335f744c24206465,0x090a30313437315f,0x32093409636f6c2e\n"
".quad 0x6f6d090a30093530,0x2509203436662e76,0x336430202c376466,0x3030303030306666\n"
".quad 0x3b30303030303030,0x6c090a31202f2f09,0x2e74736e6f632e64,0x3466250920323366\n"
".quad 0x766f315f5b202c39,0x66666f7475437265,0x6d090a3b5d635f32,0x09203233662e6c75\n"
".quad 0x6625202c30356625,0x37346625202c3934,0x662e747663090a3b,0x09203233662e3436\n"
".quad 0x6625202c38646625,0x627573090a3b3035,0x662509203436662e,0x37646625202c3964\n"
".quad 0x0a3b38646625202c,0x2e6e722e74766309,0x203436662e323366,0x25202c3135662509\n"
".quad 0x6c2e090a3b396466,0x093239093409636f,0x2e74727173090a30,0x09203233662e6e72\n"
".quad 0x6625202c32356625,0x2e646c090a3b3135,0x33662e6d61726170,0x2c33356625092032\n"
".quad 0x616475635f5f5b20,0x315a5f5f6d726170,0x6e69646469726732,0x7337315550475f67\n"
".quad 0x727241656c706d61,0x7463757274537961,0x616f6c6636506a50,0x65625f6666503274\n"
".quad 0x756d090a3b5d6174,0x2509203233662e6c,0x356625202c343566,0x3b32356625202c33\n"
".quad 0x33662e6c756d090a,0x2c35356625092032,0x25202c3435662520,0x6f6d090a3b343566\n"
".quad 0x2509203233662e76,0x346630202c363566,0x3b64303862626165,0x202f2f0920202020\n"
".quad 0x6538343034342e31,0x766f6d090a39302b,0x662509203233662e,0x64346630202c3735\n"
".quad 0x203b346266306161,0x33202f2f09202020,0x2b6534343636352e,0x2e766f6d090a3830\n"
".quad 0x3566250920323366,0x6162346630202c38,0x20203b6261633135,0x2e32202f2f092020\n"
".quad 0x302b653631343631,0x662e766f6d090a37,0x3935662509203233,0x623039346630202c\n"
".quad 0x2020203b32643039,0x313735202f2f0920,0x766f6d090a313636,0x662509203233662e\n"
".quad 0x36346630202c3036,0x203b346266633130,0x38202f2f09202020,0x090a33392e373033\n"
".quad 0x203233662e766f6d,0x30202c3136662509,0x3165643639323466,0x2f09202020203b32\n"
".quad 0x3333342e3537202f,0x662e766f6d090a37,0x3236662509203233,0x646565336630202c\n"
".quad 0x2020203b36353831,0x342e30202f2f0920,0x6d090a3637303336,0x09203233662e766f\n"
".quad 0x6630202c33366625,0x3532356334306233,0x2f2f09202020203b,0x32303230302e3020\n"
".quad 0x766f6d090a313935,0x662509203233662e,0x36336630202c3436,0x203b386562626264\n"
".quad 0x36202f2f09202020,0x2d6538353834352e,0x2e766f6d090a3630,0x3666250920323366\n"
".quad 0x3832336630202c35,0x20203b3863316139,0x2e31202f2f092020,0x302d653532323036\n"
".quad 0x662e766f6d090a38,0x3636662509203233,0x343065326630202c,0x2020203b35643935\n"
".quad 0x302e33202f2f0920,0x31312d6531333930,0x33662e766f6d090a,0x2c37366625092032\n"
".quad 0x6633343932663020,0x202020203b633936,0x35332e34202f2f09,0x0a34312d65363231\n"
".quad 0x3233662e766f6d09,0x202c383666250920,0x6131643534326630,0x09202020203b3135\n"
".quad 0x3439372e34202f2f,0x6d090a37312d6534,0x09203233662e766f,0x6630202c39366625\n"
".quad 0x3661396333336631,0x2f2f09202020203b,0x35313730382e3320,0x6f6d090a30322d65\n"
".quad 0x2509203233662e76,0x316630202c303766,0x3b33343961626339,0x202f2f0920202020\n"
".quad 0x6531383530312e32,0x616d66090a33322d,0x203233662e6e722e,0x25202c3137662509\n"
".quad 0x356625202c303766,0x3b39366625202c35,0x6e722e616d66090a,0x662509203233662e\n"
".quad 0x35356625202c3237,0x202c31376625202c,0x66090a3b38366625,0x33662e6e722e616d\n"
".quad 0x2c33376625092032,0x25202c3535662520,0x366625202c323766,0x2e616d66090a3b37\n"
".quad 0x09203233662e6e72,0x6625202c34376625,0x33376625202c3535,0x0a3b36366625202c\n"
".quad 0x2e6e722e616d6609,0x3766250920323366,0x2c35356625202c35,0x25202c3437662520\n"
".quad 0x6d66090a3b353666,0x3233662e6e722e61,0x202c363766250920,0x6625202c35356625\n"
".quad 0x34366625202c3537,0x722e616d66090a3b,0x2509203233662e6e,0x356625202c373766\n"
".quad 0x2c36376625202c35,0x090a3b3336662520,0x662e6e722e616d66,0x3837662509203233\n"
".quad 0x202c35356625202c,0x6625202c37376625,0x616d66090a3b3236,0x203233662e6e722e\n"
".quad 0x25202c3937662509,0x376625202c353566,0x3b31366625202c38,0x6e722e616d66090a\n"
".quad 0x662509203233662e,0x35356625202c3038,0x202c39376625202c,0x66090a3b30366625\n"
".quad 0x33662e6e722e616d,0x2c31386625092032,0x25202c3535662520,0x356625202c303866\n"
".quad 0x2e616d66090a3b39,0x09203233662e6e72,0x6625202c32386625,0x31386625202c3535\n"
".quad 0x0a3b38356625202c,0x2e6e722e616d6609,0x3866250920323366,0x2c35356625202c33\n"
".quad 0x25202c3238662520,0x6d66090a3b373566,0x3233662e6e722e61,0x202c343866250920\n"
".quad 0x6625202c35356625,0x36356625202c3338,0x09636f6c2e090a3b,0x0a30093630320934\n"
".quad 0x3233662e67656e09,0x202c353866250920,0x6d090a3b34386625,0x09203233662e766f\n"
".quad 0x6630202c36386625,0x3238373430343563,0x2f2f09202020203b,0x342e363730332d20\n"
".quad 0x662e646461090a37,0x3738662509203233,0x202c35356625202c,0x6d090a3b36386625\n"
".quad 0x09203233662e766f,0x6630202c38386625,0x6439633234356134,0x2f2f09202020203b\n"
".quad 0x36323637342e3320,0x6d66090a36302b65,0x3233662e6e722e61,0x202c393866250920\n"
".quad 0x6625202c35356625,0x38386625202c3738,0x662e766f6d090a3b,0x3039662509203233\n"
".quad 0x626165636630202c,0x2020203b64303862,0x2e312d202f2f0920,0x302b653834303434\n"
".quad 0x722e616d66090a39,0x2509203233662e6e,0x356625202c313966,0x2c39386625202c35\n"
".quad 0x090a3b3039662520,0x727070612e766964,0x09203233662e786f,0x6625202c32396625\n"
".quad 0x31396625202c3538,0x662e6c756d090a3b,0x3339662509203233,0x202c32396625202c\n"
".quad 0x66090a3b35336625,0x33662e6e722e616d,0x2c36326625092032,0x25202c3333662520\n"
".quad 0x326625202c333966,0x636f6c2e090a3b36,0x3009373032093409,0x6e722e616d66090a\n"
".quad 0x662509203233662e,0x34336625202c3532,0x202c33396625202c,0x2e090a3b35326625\n"
".quad 0x3032093409636f6c,0x747663090a300938,0x3233662e3436662e,0x2c30316466250920\n"
".quad 0x090a3b3432662520,0x203436662e766f6d,0x202c313164662509,0x3030306666336430\n"
".quad 0x3030303030303030,0x31202f2f093b3030,0x36662e646461090a,0x3231646625092034\n"
".quad 0x2c3031646625202c,0x0a3b313164662520,0x2e6e722e74766309,0x203436662e323366\n"
".quad 0x25202c3432662509,0x4c240a3b32316466,0x363637315f335f74,0x6c3c2f2f200a3a36\n"
".quad 0x726150203e706f6f,0x6f6f6c20666f2074,0x6c2079646f622070,0x2c32393120656e69\n"
".quad 0x616c206461656820,0x4c242064656c6562,0x313437315f335f74,0x722e747663090a30\n"
".quad 0x33732e3233662e6e,0x2c34396625092032,0x090a3b3334722520,0x203233662e627573\n"
".quad 0x25202c3539662509,0x396625202c393366,0x2e616d66090a3b34,0x09203233662e6e72\n"
".quad 0x6625202c36396625,0x35396625202c3539,0x0a3b36346625202c,0x746c2e7074657309\n"
".quad 0x702509203233662e,0x36396625202c3331,0x0a3b38346625202c,0x2033317025214009\n"
".quad 0x744c240920617262,0x38373138315f335f,0x6f6c3c2f2f200a3b,0x74726150203e706f\n"
".quad 0x706f6f6c20666f20,0x696c2079646f6220,0x202c32393120656e,0x62616c2064616568\n"
".quad 0x744c242064656c65,0x30313437315f335f,0x3409636f6c2e090a,0x090a300933313209\n"
".quad 0x203436662e766f6d,0x202c333164662509,0x3030306666336430,0x3030303030303030\n"
".quad 0x31202f2f093b3030,0x6e6f632e646c090a,0x09203233662e7473,0x5f5b202c37396625\n"
".quad 0x7475437265766f31,0x3b5d635f3266666f,0x33662e6c756d090a,0x2c38396625092032\n"
".quad 0x25202c3739662520,0x7663090a3b363966,0x33662e3436662e74,0x3431646625092032\n"
".quad 0x0a3b38396625202c,0x3436662e62757309,0x2c35316466250920,0x202c333164662520\n"
".quad 0x090a3b3431646625,0x662e6e722e747663,0x09203436662e3233,0x6625202c39396625\n"
".quad 0x6c2e090a3b353164,0x093239093409636f,0x2e74727173090a30,0x09203233662e6e72\n"
".quad 0x25202c3030316625,0x646c090a3b393966,0x662e6d617261702e,0x3031662509203233\n"
".quad 0x75635f5f5b202c31,0x5f5f6d7261706164,0x646469726732315a,0x315550475f676e69\n"
".quad 0x41656c706d617337,0x7572745379617272,0x6c6636506a507463,0x5f6666503274616f\n"
".quad 0x090a3b5d61746562,0x203233662e6c756d,0x202c323031662509,0x25202c3130316625\n"
".quad 0x6d090a3b30303166,0x09203233662e6c75,0x25202c3330316625,0x6625202c32303166\n"
".quad 0x6f6d090a3b323031,0x2509203233662e76,0x6630202c34303166,0x6430386262616534\n"
".quad 0x202f2f092020203b,0x6538343034342e31,0x766f6d090a39302b,0x662509203233662e\n"
".quad 0x346630202c353031,0x3b34626630616164,0x33202f2f09202020,0x2b6534343636352e\n"
".quad 0x2e766f6d090a3830,0x3166250920323366,0x62346630202c3630,0x203b626163313561\n"
".quad 0x2e32202f2f092020,0x302b653631343631,0x662e766f6d090a37,0x3031662509203233\n"
".quad 0x3039346630202c37,0x20203b3264303962,0x313735202f2f0920,0x766f6d090a313636\n"
".quad 0x662509203233662e,0x346630202c383031,0x3b34626663313036,0x38202f2f09202020\n"
".quad 0x090a33392e373033,0x203233662e766f6d,0x202c393031662509,0x6564363932346630\n"
".quad 0x2f092020203b3231,0x3333342e3537202f,0x662e766f6d090a37,0x3131662509203233\n"
".quad 0x6565336630202c30,0x20203b3635383164,0x342e30202f2f0920,0x6d090a3637303336\n"
".quad 0x09203233662e766f,0x30202c3131316625,0x3235633430623366,0x2f2f092020203b35\n"
".quad 0x32303230302e3020,0x766f6d090a313935,0x662509203233662e,0x336630202c323131\n"
".quad 0x3b38656262626436,0x36202f2f09202020,0x2d6538353834352e,0x2e766f6d090a3630\n"
".quad 0x3166250920323366,0x32336630202c3331,0x203b386331613938,0x2e31202f2f092020\n"
".quad 0x302d653532323036,0x662e766f6d090a38,0x3131662509203233,0x3065326630202c34\n"
".quad 0x20203b3564393534,0x302e33202f2f0920,0x31312d6531333930,0x33662e766f6d090a\n"
".quad 0x3531316625092032,0x333439326630202c,0x2020203b63393666,0x35332e34202f2f09\n"
".quad 0x0a34312d65363231,0x3233662e766f6d09,0x2c36313166250920,0x3164353432663020\n"
".quad 0x092020203b313561,0x3439372e34202f2f,0x6d090a37312d6534,0x09203233662e766f\n"
".quad 0x30202c3731316625,0x6139633333663166,0x2f2f092020203b36,0x35313730382e3320\n"
".quad 0x6f6d090a30322d65,0x2509203233662e76,0x6630202c38313166,0x3334396162633931\n"
".quad 0x202f2f092020203b,0x6531383530312e32,0x616d66090a33322d,0x203233662e6e722e\n"
".quad 0x202c393131662509,0x25202c3831316625,0x6625202c33303166,0x6d66090a3b373131\n"
".quad 0x3233662e6e722e61,0x2c30323166250920,0x202c333031662520,0x25202c3931316625\n"
".quad 0x66090a3b36313166,0x33662e6e722e616d,0x3132316625092032,0x2c3330316625202c\n"
".quad 0x202c303231662520,0x090a3b3531316625,0x662e6e722e616d66,0x3231662509203233\n"
".quad 0x3330316625202c32,0x2c3132316625202c,0x0a3b343131662520,0x2e6e722e616d6609\n"
".quad 0x3166250920323366,0x30316625202c3332,0x3232316625202c33,0x3b3331316625202c\n"
".quad 0x6e722e616d66090a,0x662509203233662e,0x316625202c343231,0x32316625202c3330\n"
".quad 0x3231316625202c33,0x722e616d66090a3b,0x2509203233662e6e,0x6625202c35323166\n"
".quad 0x316625202c333031,0x31316625202c3432,0x2e616d66090a3b31,0x09203233662e6e72\n"
".quad 0x25202c3632316625,0x6625202c33303166,0x316625202c353231,0x616d66090a3b3031\n"
".quad 0x203233662e6e722e,0x202c373231662509,0x25202c3330316625,0x6625202c36323166\n"
".quad 0x6d66090a3b393031,0x3233662e6e722e61,0x2c38323166250920,0x202c333031662520\n"
".quad 0x25202c3732316625,0x66090a3b38303166,0x33662e6e722e616d,0x3932316625092032\n"
".quad 0x2c3330316625202c,0x202c383231662520,0x090a3b3730316625,0x662e6e722e616d66\n"
".quad 0x3331662509203233,0x3330316625202c30,0x2c3932316625202c,0x0a3b363031662520\n"
".quad 0x2e6e722e616d6609,0x3166250920323366,0x30316625202c3133,0x3033316625202c33\n"
".quad 0x3b3530316625202c,0x6e722e616d66090a,0x662509203233662e,0x30316625202c3438\n"
".quad 0x3133316625202c33,0x3b3430316625202c,0x3409636f6c2e090a,0x090a300934313209\n"
".quad 0x203233662e67656e,0x25202c3538662509,0x6f6d090a3b343866,0x2509203233662e76\n"
".quad 0x6630202c32333166,0x3238373430343563,0x202f2f092020203b,0x37342e363730332d\n"
".quad 0x33662e646461090a,0x3333316625092032,0x2c3330316625202c,0x0a3b323331662520\n"
".quad 0x3233662e766f6d09,0x2c34333166250920,0x3234356134663020,0x092020203b643963\n"
".quad 0x3637342e33202f2f,0x090a36302b653632,0x662e6e722e616d66,0x3331662509203233\n"
".quad 0x3330316625202c35,0x2c3333316625202c,0x0a3b343331662520,0x3233662e766f6d09\n"
".quad 0x2c36333166250920,0x6262616563663020,0x092020203b643038,0x34342e312d202f2f\n"
".quad 0x0a39302b65383430,0x2e6e722e616d6609,0x3166250920323366,0x30316625202c3733\n"
".quad 0x3533316625202c33,0x3b3633316625202c,0x70612e766964090a,0x3233662e786f7270\n"
".quad 0x2c38333166250920,0x25202c3538662520,0x6d090a3b37333166,0x09203233662e6c75\n"
".quad 0x25202c3933316625,0x6625202c38333166,0x616d66090a3b3533,0x203233662e6e722e\n"
".quad 0x25202c3332662509,0x316625202c333366,0x33326625202c3933,0x09636f6c2e090a3b\n"
".quad 0x0a30093531320934,0x2e6e722e616d6609,0x3266250920323366,0x2c34336625202c32\n"
".quad 0x202c393331662520,0x2e090a3b32326625,0x3132093409636f6c,0x747663090a300936\n"
".quad 0x3233662e3436662e,0x2c36316466250920,0x090a3b3132662520,0x203436662e766f6d\n"
".quad 0x202c373164662509,0x3030306666336430,0x3030303030303030,0x31202f2f093b3030\n"
".quad 0x36662e646461090a,0x3831646625092034,0x2c3631646625202c,0x0a3b373164662520\n"
".quad 0x2e6e722e74766309,0x203436662e323366,0x25202c3132662509,0x4c240a3b38316466\n"
".quad 0x373138315f335f74,0x6c3c2f2f200a3a38,0x726150203e706f6f,0x6f6f6c20666f2074\n"
".quad 0x6c2079646f622070,0x2c32393120656e69,0x616c206461656820,0x4c242064656c6562\n"
".quad 0x313437315f335f74,0x722e747663090a30,0x33732e3233662e6e,0x3034316625092032\n"
".quad 0x0a3b34347225202c,0x3233662e62757309,0x2c31343166250920,0x25202c3933662520\n"
".quad 0x66090a3b30343166,0x33662e6e722e616d,0x3234316625092032,0x2c3134316625202c\n"
".quad 0x202c313431662520,0x73090a3b36346625,0x662e746c2e707465,0x3431702509203233\n"
".quad 0x2c3234316625202c,0x090a3b3834662520,0x6220343170252140,0x5f744c2409206172\n"
".quad 0x3b30393638315f33,0x6f6f6c3c2f2f200a,0x2074726150203e70,0x20706f6f6c20666f\n"
".quad 0x6e696c2079646f62,0x68202c3239312065,0x6562616c20646165,0x5f744c242064656c\n"
".quad 0x0a30313437315f33,0x093409636f6c2e09,0x6d090a3009313232,0x09203436662e766f\n"
".quad 0x30202c3931646625,0x3030303066663364,0x3030303030303030,0x0a31202f2f093b30\n"
".quad 0x736e6f632e646c09,0x2509203233662e74,0x5f5b202c33343166,0x7475437265766f31\n"
".quad 0x3b5d635f3266666f,0x33662e6c756d090a,0x3434316625092032,0x2c3334316625202c\n"
".quad 0x0a3b323431662520,0x3436662e74766309,0x662509203233662e,0x316625202c303264\n"
".quad 0x627573090a3b3434,0x662509203436662e,0x646625202c313264,0x32646625202c3931\n"
".quad 0x2e747663090a3b30,0x662e3233662e6e72,0x3431662509203436,0x3132646625202c35\n"
".quad 0x09636f6c2e090a3b,0x090a300932390934,0x2e6e722e74727173,0x3166250920323366\n"
".quad 0x34316625202c3634,0x702e646c090a3b35,0x3233662e6d617261,0x2c37343166250920\n"
".quad 0x616475635f5f5b20,0x315a5f5f6d726170,0x6e69646469726732,0x7337315550475f67\n"
".quad 0x727241656c706d61,0x7463757274537961,0x616f6c6636506a50,0x65625f6666503274\n"
".quad 0x756d090a3b5d6174,0x2509203233662e6c,0x6625202c38343166,0x316625202c373431\n"
".quad 0x6c756d090a3b3634,0x662509203233662e,0x316625202c393431,0x34316625202c3834\n"
".quad 0x2e766f6d090a3b38,0x3166250920323366,0x65346630202c3035,0x203b643038626261\n"
".quad 0x2e31202f2f092020,0x302b653834303434,0x662e766f6d090a39,0x3531662509203233\n"
".quad 0x6164346630202c31,0x20203b3462663061,0x352e33202f2f0920,0x38302b6534343636\n"
".quad 0x33662e766f6d090a,0x3235316625092032,0x356162346630202c,0x2020203b62616331\n"
".quad 0x36312e32202f2f09,0x0a37302b65363134,0x3233662e766f6d09,0x2c33353166250920\n"
".quad 0x3962303934663020,0x092020203b326430,0x3636313735202f2f,0x662e766f6d090a31\n"
".quad 0x3531662509203233,0x3036346630202c34,0x20203b3462666331,0x303338202f2f0920\n"
".quad 0x6f6d090a33392e37,0x2509203233662e76,0x6630202c35353166,0x3231656436393234\n"
".quad 0x202f2f092020203b,0x0a373333342e3537,0x3233662e766f6d09,0x2c36353166250920\n"
".quad 0x3164656533663020,0x092020203b363538,0x3336342e30202f2f,0x766f6d090a363730\n"
".quad 0x662509203233662e,0x336630202c373531,0x3b35323563343062,0x30202f2f09202020\n"
".quad 0x393532303230302e,0x662e766f6d090a31,0x3531662509203233,0x6436336630202c38\n"
".quad 0x20203b3865626262,0x352e36202f2f0920,0x36302d6538353834,0x33662e766f6d090a\n"
".quad 0x3935316625092032,0x393832336630202c,0x2020203b38633161,0x30362e31202f2f09\n"
".quad 0x0a38302d65353232,0x3233662e766f6d09,0x2c30363166250920,0x3534306532663020\n"
".quad 0x092020203b356439,0x3930302e33202f2f,0x090a31312d653133,0x203233662e766f6d\n"
".quad 0x202c313631662509,0x3666333439326630,0x2f092020203b6339,0x323135332e34202f\n"
".quad 0x6d090a34312d6536,0x09203233662e766f,0x30202c3236316625,0x3561316435343266\n"
".quad 0x2f2f092020203b31,0x65343439372e3420,0x766f6d090a37312d,0x662509203233662e\n"
".quad 0x316630202c333631,0x3b36613963333366,0x33202f2f09202020,0x2d6535313730382e\n"
".quad 0x2e766f6d090a3032,0x3166250920323366,0x39316630202c3436,0x203b333439616263\n"
".quad 0x2e32202f2f092020,0x322d653138353031,0x722e616d66090a33,0x2509203233662e6e\n"
".quad 0x6625202c35363166,0x316625202c343631,0x36316625202c3934,0x2e616d66090a3b33\n"
".quad 0x09203233662e6e72,0x25202c3636316625,0x6625202c39343166,0x316625202c353631\n"
".quad 0x616d66090a3b3236,0x203233662e6e722e,0x202c373631662509,0x25202c3934316625\n"
".quad 0x6625202c36363166,0x6d66090a3b313631,0x3233662e6e722e61,0x2c38363166250920\n"
".quad 0x202c393431662520,0x25202c3736316625,0x66090a3b30363166,0x33662e6e722e616d\n"
".quad 0x3936316625092032,0x2c3934316625202c,0x202c383631662520,0x090a3b3935316625\n"
".quad 0x662e6e722e616d66,0x3731662509203233,0x3934316625202c30,0x2c3936316625202c\n"
".quad 0x0a3b383531662520,0x2e6e722e616d6609,0x3166250920323366,0x34316625202c3137\n"
".quad 0x3037316625202c39,0x3b3735316625202c,0x6e722e616d66090a,0x662509203233662e\n"
".quad 0x316625202c323731,0x37316625202c3934,0x3635316625202c31,0x722e616d66090a3b\n"
".quad 0x2509203233662e6e,0x6625202c33373166,0x316625202c393431,0x35316625202c3237\n"
".quad 0x2e616d66090a3b35,0x09203233662e6e72,0x25202c3437316625,0x6625202c39343166\n"
".quad 0x316625202c333731,0x616d66090a3b3435,0x203233662e6e722e,0x202c353731662509\n"
".quad 0x25202c3934316625,0x6625202c34373166,0x6d66090a3b333531,0x3233662e6e722e61\n"
".quad 0x2c36373166250920,0x202c393431662520,0x25202c3537316625,0x66090a3b32353166\n"
".quad 0x33662e6e722e616d,0x3737316625092032,0x2c3934316625202c,0x202c363731662520\n"
".quad 0x090a3b3135316625,0x662e6e722e616d66,0x3438662509203233,0x2c3934316625202c\n"
".quad 0x202c373731662520,0x090a3b3035316625,0x32093409636f6c2e,0x656e090a30093232\n"
".quad 0x2509203233662e67,0x386625202c353866,0x2e766f6d090a3b34,0x3166250920323366\n"
".quad 0x35636630202c3837,0x203b323837343034,0x332d202f2f092020,0x090a37342e363730\n"
".quad 0x203233662e646461,0x202c393731662509,0x25202c3934316625,0x6d090a3b38373166\n"
".quad 0x09203233662e766f,0x30202c3038316625,0x3963323435613466,0x2f2f092020203b64\n"
".quad 0x36323637342e3320,0x6d66090a36302b65,0x3233662e6e722e61,0x2c31383166250920\n"
".quad 0x202c393431662520,0x25202c3937316625,0x6d090a3b30383166,0x09203233662e766f\n"
".quad 0x30202c3238316625,0x3038626261656366,0x2f2f092020203b64,0x343034342e312d20\n"
".quad 0x66090a39302b6538,0x33662e6e722e616d,0x3338316625092032,0x2c3934316625202c\n"
".quad 0x202c313831662520,0x090a3b3238316625,0x727070612e766964,0x09203233662e786f\n"
".quad 0x25202c3438316625,0x316625202c353866,0x6c756d090a3b3338,0x662509203233662e\n"
".quad 0x316625202c353831,0x35336625202c3438,0x722e616d66090a3b,0x2509203233662e6e\n"
".quad 0x336625202c303266,0x3538316625202c33,0x0a3b30326625202c,0x093409636f6c2e09\n"
".quad 0x66090a3009333232,0x33662e6e722e616d,0x2c39316625092032,0x25202c3433662520\n"
".quad 0x6625202c35383166,0x6f6c2e090a3b3931,0x0934323209340963,0x662e747663090a30\n"
".quad 0x09203233662e3436,0x25202c3232646625,0x6f6d090a3b383166,0x2509203436662e76\n"
".quad 0x6430202c33326466,0x3030303030666633,0x3030303030303030,0x090a31202f2f093b\n"
".quad 0x203436662e646461,0x202c343264662509,0x25202c3232646625,0x63090a3b33326466\n"
".quad 0x33662e6e722e7476,0x2509203436662e32,0x646625202c383166,0x5f744c240a3b3432\n"
".quad 0x3a30393638315f33,0x6f6f6c3c2f2f200a,0x2074726150203e70,0x20706f6f6c20666f\n"
".quad 0x6e696c2079646f62,0x68202c3239312065,0x6562616c20646165,0x5f744c242064656c\n"
".quad 0x0a30313437315f33,0x2e6e722e74766309,0x203233732e323366,0x202c363831662509\n"
".quad 0x73090a3b35347225,0x09203233662e6275,0x25202c3738316625,0x316625202c393366\n"
".quad 0x616d66090a3b3638,0x203233662e6e722e,0x202c383831662509,0x25202c3738316625\n"
".quad 0x6625202c37383166,0x746573090a3b3634,0x3233662e746c2e70,0x202c353170250920\n"
".quad 0x25202c3838316625,0x2140090a3b383466,0x6172622035317025,0x5f335f744c240920\n"
".quad 0x200a3b3831303232,0x3e706f6f6c3c2f2f,0x666f207472615020,0x6f6220706f6f6c20\n"
".quad 0x20656e696c207964,0x616568202c323931,0x656c6562616c2064,0x5f335f744c242064\n"
".quad 0x2e090a3031343731,0x3232093409636f6c,0x766f6d090a300939,0x662509203436662e\n"
".quad 0x336430202c353264,0x3030303030306666,0x3b30303030303030,0x6c090a31202f2f09\n"
".quad 0x2e74736e6f632e64,0x3166250920323366,0x6f315f5b202c3938,0x666f747543726576\n"
".quad 0x090a3b5d635f3266,0x203233662e6c756d,0x202c303931662509,0x25202c3938316625\n"
".quad 0x63090a3b38383166,0x662e3436662e7476,0x3264662509203233,0x3039316625202c36\n"
".quad 0x662e627573090a3b,0x3264662509203436,0x3532646625202c37,0x3b3632646625202c\n"
".quad 0x6e722e747663090a,0x3436662e3233662e,0x2c31393166250920,0x0a3b373264662520\n"
".quad 0x093409636f6c2e09,0x7173090a30093239,0x33662e6e722e7472,0x3239316625092032\n"
".quad 0x3b3139316625202c,0x7261702e646c090a,0x09203233662e6d61,0x5b202c3339316625\n"
".quad 0x6170616475635f5f,0x6732315a5f5f6d72,0x5f676e6964646972,0x6d61733731555047\n"
".quad 0x7961727241656c70,0x6a50746375727453,0x3274616f6c663650,0x617465625f666650\n"
".quad 0x2e6c756d090a3b5d,0x3166250920323366,0x39316625202c3439,0x3239316625202c33\n"
".quad 0x662e6c756d090a3b,0x3931662509203233,0x3439316625202c35,0x3b3439316625202c\n"
".quad 0x33662e766f6d090a,0x3639316625092032,0x626165346630202c,0x2020203b64303862\n"
".quad 0x34342e31202f2f09,0x0a39302b65383430,0x3233662e766f6d09,0x2c37393166250920\n"
".quad 0x3061616434663020,0x092020203b346266,0x3636352e33202f2f,0x090a38302b653434\n"
".quad 0x203233662e766f6d,0x202c383931662509,0x6331356162346630,0x2f092020203b6261\n"
".quad 0x313436312e32202f,0x6d090a37302b6536,0x09203233662e766f,0x30202c3939316625\n"
".quad 0x6430396230393466,0x2f2f092020203b32,0x0a31363631373520,0x3233662e766f6d09\n"
".quad 0x2c30303266250920,0x6331303634663020,0x092020203b346266,0x2e37303338202f2f\n"
".quad 0x2e766f6d090a3339,0x3266250920323366,0x32346630202c3130,0x203b323165643639\n"
".quad 0x3537202f2f092020,0x6d090a373333342e,0x09203233662e766f,0x30202c3230326625\n"
".quad 0x3538316465653366,0x2f2f092020203b36,0x37303336342e3020,0x662e766f6d090a36\n"
".quad 0x3032662509203233,0x3062336630202c33,0x20203b3532356334,0x302e30202f2f0920\n"
".quad 0x0a31393532303230,0x3233662e766f6d09,0x2c34303266250920,0x6262643633663020\n"
".quad 0x092020203b386562,0x3834352e36202f2f,0x090a36302d653835,0x203233662e766f6d\n"
".quad 0x202c353032662509,0x3161393832336630,0x2f092020203b3863,0x323230362e31202f\n"
".quad 0x6d090a38302d6535,0x09203233662e766f,0x30202c3630326625,0x6439353430653266\n"
".quad 0x2f2f092020203b35,0x31333930302e3320,0x6f6d090a31312d65,0x2509203233662e76\n"
".quad 0x6630202c37303266,0x6339366633343932,0x202f2f092020203b,0x6536323135332e34\n"
".quad 0x766f6d090a34312d,0x662509203233662e,0x326630202c383032,0x3b31356131643534\n"
".quad 0x34202f2f09202020,0x312d65343439372e,0x662e766f6d090a37,0x3032662509203233\n"
".quad 0x3366316630202c39,0x20203b3661396333,0x382e33202f2f0920,0x30322d6535313730\n"
".quad 0x33662e766f6d090a,0x3031326625092032,0x626339316630202c,0x2020203b33343961\n"
".quad 0x30312e32202f2f09,0x0a33322d65313835,0x2e6e722e616d6609,0x3266250920323366\n"
".quad 0x31326625202c3131,0x3539316625202c30,0x3b3930326625202c,0x6e722e616d66090a\n"
".quad 0x662509203233662e,0x316625202c323132,0x31326625202c3539,0x3830326625202c31\n"
".quad 0x722e616d66090a3b,0x2509203233662e6e,0x6625202c33313266,0x326625202c353931\n"
".quad 0x30326625202c3231,0x2e616d66090a3b37,0x09203233662e6e72,0x25202c3431326625\n"
".quad 0x6625202c35393166,0x326625202c333132,0x616d66090a3b3630,0x203233662e6e722e\n"
".quad 0x202c353132662509,0x25202c3539316625,0x6625202c34313266,0x6d66090a3b353032\n"
".quad 0x3233662e6e722e61,0x2c36313266250920,0x202c353931662520,0x25202c3531326625\n"
".quad 0x66090a3b34303266,0x33662e6e722e616d,0x3731326625092032,0x2c3539316625202c\n"
".quad 0x202c363132662520,0x090a3b3330326625,0x662e6e722e616d66,0x3132662509203233\n"
".quad 0x3539316625202c38,0x2c3731326625202c,0x0a3b323032662520,0x2e6e722e616d6609\n"
".quad 0x3266250920323366,0x39316625202c3931,0x3831326625202c35,0x3b3130326625202c\n"
".quad 0x6e722e616d66090a,0x662509203233662e,0x316625202c303232,0x31326625202c3539\n"
".quad 0x3030326625202c39,0x722e616d66090a3b,0x2509203233662e6e,0x6625202c31323266\n"
".quad 0x326625202c353931,0x39316625202c3032,0x2e616d66090a3b39,0x09203233662e6e72\n"
".quad 0x25202c3232326625,0x6625202c35393166,0x316625202c313232,0x616d66090a3b3839\n"
".quad 0x203233662e6e722e,0x202c333232662509,0x25202c3539316625,0x6625202c32323266\n"
".quad 0x6d66090a3b373931,0x3233662e6e722e61,0x202c343866250920,0x25202c3539316625\n"
".quad 0x6625202c33323266,0x6c2e090a3b363931,0x303332093409636f,0x2e67656e090a3009\n"
".quad 0x3866250920323366,0x3b34386625202c35,0x33662e766f6d090a,0x3432326625092032\n"
".quad 0x303435636630202c,0x2020203b32383734,0x3730332d202f2f09,0x6461090a37342e36\n"
".quad 0x2509203233662e64,0x6625202c35323266,0x326625202c353931,0x766f6d090a3b3432\n"
".quad 0x662509203233662e,0x346630202c363232,0x3b64396332343561,0x33202f2f09202020\n"
".quad 0x2b6536323637342e,0x2e616d66090a3630,0x09203233662e6e72,0x25202c3732326625\n"
".quad 0x6625202c35393166,0x326625202c353232,0x766f6d090a3b3632,0x662509203233662e\n"
".quad 0x636630202c383232,0x3b64303862626165,0x2d202f2f09202020,0x6538343034342e31\n"
".quad 0x616d66090a39302b,0x203233662e6e722e,0x202c393232662509,0x25202c3539316625\n"
".quad 0x6625202c37323266,0x6964090a3b383232,0x786f727070612e76,0x662509203233662e\n"
".quad 0x386625202c303332,0x3932326625202c35,0x662e6c756d090a3b,0x3332662509203233\n"
".quad 0x3033326625202c31,0x0a3b35336625202c,0x2e6e722e616d6609,0x3166250920323366\n"
".quad 0x2c33336625202c37,0x202c313332662520,0x2e090a3b37316625,0x3332093409636f6c\n"
".quad 0x616d66090a300931,0x203233662e6e722e,0x25202c3631662509,0x326625202c343366\n"
".quad 0x36316625202c3133,0x09636f6c2e090a3b,0x0a30093233320934,0x3436662e74766309\n"
".quad 0x662509203233662e,0x316625202c383264,0x2e766f6d090a3b35,0x6466250920343666\n"
".quad 0x66336430202c3932,0x3030303030303066,0x093b303030303030,0x6461090a31202f2f\n"
".quad 0x2509203436662e64,0x6625202c30336466,0x646625202c383264,0x747663090a3b3932\n"
".quad 0x2e3233662e6e722e,0x3166250920343666,0x3033646625202c35,0x5f335f744c240a3b\n"
".quad 0x240a3a3831303232,0x373731315f335f4c,0x6c3c2f2f200a3a38,0x726150203e706f6f\n"
".quad 0x6f6f6c20666f2074,0x6c2079646f622070,0x2c32393120656e69,0x616c206461656820\n"
".quad 0x4c242064656c6562,0x313437315f335f74,0x09636f6c2e090a30,0x0a30093732320934\n"
".quad 0x3233732e64646109,0x202c383772250920,0x3b31202c38377225,0x36752e646461090a\n"
".quad 0x3733647225092034,0x2c3733647225202c,0x746573090a3b3420,0x3233732e656e2e70\n"
".quad 0x202c363170250920,0x7225202c38377225,0x702540090a3b3337,0x0920617262203631\n"
".quad 0x37315f335f744c24,0x744c240a3b303134,0x38393836315f335f,0x6f6c3c2f2f200a3a\n"
".quad 0x74726150203e706f,0x706f6f6c20666f20,0x696c2079646f6220,0x202c35373120656e\n"
".quad 0x62616c2064616568,0x744c242064656c65,0x30333136315f335f,0x33732e646461090a\n"
".quad 0x2c38367225092032,0x31202c3836722520,0x2e70746573090a3b,0x09203233752e656e\n"
".quad 0x7225202c37317025,0x35367225202c3836,0x3731702540090a3b,0x4c24092061726220\n"
".quad 0x333136315f335f74,0x335f744c240a3b30,0x0a3a38313635315f,0x706f6f6c3c2f2f20\n"
".quad 0x6f2074726150203e,0x6220706f6f6c2066,0x656e696c2079646f,0x6568202c32373120\n"
".quad 0x6c6562616c206461,0x335f744c24206465,0x090a32363335315f,0x203233732e646461\n"
".quad 0x25202c3435722509,0x0a3b31202c343572,0x3436732e64646109,0x2c37316472250920\n"
".quad 0x202c373164722520,0x090a3b3331647225,0x203436752e646461,0x202c313264722509\n"
".quad 0x25202c3331647225,0x73090a3b31326472,0x732e656e2e707465,0x3831702509203233\n"
".quad 0x202c37357225202c,0x40090a3b34357225,0x6172622038317025,0x5f335f744c240920\n"
".quad 0x240a3b3236333531,0x3834315f335f744c,0x3c2f2f200a3a3035,0x6150203e706f6f6c\n"
".quad 0x6f6c20666f207472,0x2079646f6220706f,0x31373120656e696c,0x6c2064616568202c\n"
".quad 0x242064656c656261,0x3534315f335f744c,0x2e646461090a3439,0x3272250920323373\n"
".quad 0x2c39327225202c39,0x746573090a3b3120,0x3233732e656e2e70,0x202c393170250920\n"
".quad 0x7225202c38347225,0x702540090a3b3932,0x0920617262203931,0x34315f335f744c24\n"
".quad 0x7262090a3b343935,0x240920696e752e61,0x3034315f335f744c,0x5f744c240a3b3238\n"
".quad 0x3a38333730325f33,0x6e6f632e646c090a,0x09203233732e7473,0x735b202c32357225\n"
".quad 0x635f79785f657a69,0x2e766f6d090a3b5d,0x3166250920323366,0x3030306630202c35\n"
".quad 0x20203b3030303030,0x0a30202f2f092020,0x3233662e766f6d09,0x202c363166250920\n"
".quad 0x3030303030306630,0x09202020203b3030,0x6f6d090a30202f2f,0x2509203233662e76\n"
".quad 0x306630202c373166,0x3b30303030303030,0x202f2f0920202020,0x662e766f6d090a30\n"
".quad 0x3831662509203233,0x303030306630202c,0x2020203b30303030,0x090a30202f2f0920\n"
".quad 0x203233662e766f6d,0x30202c3931662509,0x3030303030303066,0x2f09202020203b30\n"
".quad 0x766f6d090a30202f,0x662509203233662e,0x30306630202c3032,0x203b303030303030\n"
".quad 0x30202f2f09202020,0x33662e766f6d090a,0x2c31326625092032,0x3030303030663020\n"
".quad 0x202020203b303030,0x6d090a30202f2f09,0x09203233662e766f,0x6630202c32326625\n"
".quad 0x3030303030303030,0x2f2f09202020203b,0x2e766f6d090a3020,0x3266250920323366\n"
".quad 0x3030306630202c33,0x20203b3030303030,0x0a30202f2f092020,0x3233662e766f6d09\n"
".quad 0x202c343266250920,0x3030303030306630,0x09202020203b3030,0x6f6d090a30202f2f\n"
".quad 0x2509203233662e76,0x306630202c353266,0x3b30303030303030,0x202f2f0920202020\n"
".quad 0x662e766f6d090a30,0x3632662509203233,0x303030306630202c,0x2020203b30303030\n"
".quad 0x240a30202f2f0920,0x3034315f335f744c,0x6f6c2e090a3a3238,0x0930343209340963\n"
".quad 0x6c2e6c756d090a30,0x2509203233732e6f,0x367225202c303872,0x0a3b33337225202c\n"
".quad 0x2e6f6c2e6c756d09,0x3872250920323373,0x2c32357225202c31,0x090a3b3533722520\n"
".quad 0x203233732e646461,0x25202c3238722509,0x387225202c303872,0x2e646461090a3b31\n"
".quad 0x3872250920323373,0x2c32387225202c33,0x090a3b3133722520,0x2e3436732e747663\n"
".quad 0x6472250920323373,0x33387225202c3434,0x61702e646c090a3b,0x203436752e6d6172\n"
".quad 0x202c353464722509,0x70616475635f5f5b,0x32315a5f5f6d7261,0x676e696464697267\n"
".quad 0x617337315550475f,0x61727241656c706d,0x5074637572745379,0x74616f6c6636506a\n"
".quad 0x6972675f66665032,0x5d675f6174614464,0x772e6c756d090a3b,0x203233732e656469\n"
".quad 0x202c363464722509,0x3b38202c33387225,0x36752e646461090a,0x3734647225092034\n"
".quad 0x2c3534647225202c,0x0a3b363464722520,0x626f6c672e747309,0x33662e32762e6c61\n"
".quad 0x346472255b092032,0x257b202c5d302b37,0x353266252c363266,0x636f6c2e090a3b7d\n"
".quad 0x3009313432093409,0x7261702e646c090a,0x09203436752e6d61,0x5b202c3834647225\n"
".quad 0x6170616475635f5f,0x6732315a5f5f6d72,0x5f676e6964646972,0x6d61733731555047\n"
".quad 0x7961727241656c70,0x6a50746375727453,0x3274616f6c663650,0x706d61735f666650\n"
".quad 0x7469736e6544656c,0x6d090a3b5d675f79,0x2e656469772e6c75,0x6472250920323373\n"
".quad 0x33387225202c3934,0x6461090a3b34202c,0x2509203436752e64,0x7225202c30356472\n"
".quad 0x647225202c383464,0x2e7473090a3b3934,0x662e6c61626f6c67,0x6472255b09203233\n"
".quad 0x25202c5d302b3035,0x6c2e090a3b343266,0x333432093409636f,0x2e6c756d090a3009\n"
".quad 0x09203233752e6f6c,0x7225202c34387225,0x38317225202c3235,0x752e646461090a3b\n"
".quad 0x3538722509203233,0x202c33387225202c,0x63090a3b34387225,0x732e3436732e7476\n"
".quad 0x3564722509203233,0x3b35387225202c31,0x69772e6c756d090a,0x09203233732e6564\n"
".quad 0x25202c3235647225,0x0a3b38202c353872,0x3436752e64646109,0x2c33356472250920\n"
".quad 0x202c353464722520,0x090a3b3235647225,0x61626f6c672e7473,0x3233662e32762e6c\n"
".quad 0x33356472255b0920,0x66257b202c5d302b,0x7d323266252c3332,0x09636f6c2e090a3b\n"
".quad 0x0a30093434320934,0x6469772e6c756d09,0x2509203233732e65,0x7225202c34356472\n"
".quad 0x090a3b34202c3538,0x203436752e646461,0x202c353564722509,0x25202c3834647225\n"
".quad 0x73090a3b34356472,0x6c61626f6c672e74,0x255b09203233662e,0x2c5d302b35356472\n"
".quad 0x090a3b3132662520,0x32093409636f6c2e,0x6461090a30093634,0x2509203233752e64\n"
".quad 0x387225202c363872,0x3b35387225202c34,0x36732e747663090a,0x2509203233732e34\n"
".quad 0x7225202c36356472,0x6c756d090a3b3638,0x33732e656469772e,0x3735647225092032\n"
".quad 0x202c36387225202c,0x2e646461090a3b38,0x6472250920343675,0x34647225202c3835\n"
".quad 0x3735647225202c35,0x6c672e7473090a3b,0x2e32762e6c61626f,0x72255b0920323366\n"
".quad 0x202c5d302b383564,0x66252c303266257b,0x6c2e090a3b7d3931,0x373432093409636f\n"
".quad 0x2e6c756d090a3009,0x3233732e65646977,0x2c39356472250920,0x34202c3638722520\n"
".quad 0x752e646461090a3b,0x3664722509203436,0x3834647225202c30,0x3b3935647225202c\n"
".quad 0x6f6c672e7473090a,0x203233662e6c6162,0x2b30366472255b09,0x38316625202c5d30\n"
".quad 0x09636f6c2e090a3b,0x0a30093934320934,0x3233752e64646109,0x202c373872250920\n"
".quad 0x7225202c34387225,0x747663090a3b3638,0x3233732e3436732e,0x2c31366472250920\n"
".quad 0x090a3b3738722520,0x656469772e6c756d,0x722509203233732e,0x387225202c323664\n"
".quad 0x61090a3b38202c37,0x09203436752e6464,0x25202c3336647225,0x7225202c35346472\n"
".quad 0x7473090a3b323664,0x2e6c61626f6c672e,0x09203233662e3276,0x302b33366472255b\n"
".quad 0x373166257b202c5d,0x0a3b7d363166252c,0x093409636f6c2e09,0x6d090a3009303532\n"
".quad 0x2e656469772e6c75,0x6472250920323373,0x37387225202c3436,0x6461090a3b34202c\n"
".quad 0x2509203436752e64,0x7225202c35366472,0x647225202c383464,0x2e7473090a3b3436\n"
".quad 0x662e6c61626f6c67,0x6472255b09203233,0x25202c5d302b3536,0x6c2e090a3b353166\n"
".quad 0x313532093409636f,0x74697865090a3009,0x6e6557444c240a3b,0x726732315a5f5f64\n"
".quad 0x475f676e69646469,0x706d617337315550,0x537961727241656c,0x506a507463757274\n"
".quad 0x503274616f6c6636,0x2f207d090a3a6666,0x726732315a5f202f,0x475f676e69646469\n"
".quad 0x706d617337315550,0x537961727241656c,0x506a507463757274,0x503274616f6c6636\n"
".quad 0x000000000a0a6666\n"
".text");

extern "C" {

extern const unsigned long long __deviceText_$compute_20$[4013];

}

static __cudaFatPtxEntry __ptxEntries [] = {{(char*)"compute_20",(char*)__deviceText_$compute_20$},{0,0}};
static __cudaFatCubinEntry __cubinEntries[] = {{0,0}};
static __cudaFatDebugEntry __debugEntries0 = {0, 0, 0, 0} ;
static __cudaFatElfEntry __elfEntries0 = {0, 0, 0, 0} ;
static __cudaFatElfEntry __elfEntries1 = {(char*)"sm_20", (char*)__deviceText_$sm_20$, &__elfEntries0, (unsigned int)sizeof(__deviceText_$sm_20$)};



static __cudaFatCudaBinary __fatDeviceText __attribute__ ((section (".nvFatBinSegment")))= {0x1ee55a01,0x00000004,0x2e00b786,(char*)"9f0b21bf5f7367f8",(char*)"CUDA_interface.cu",(char*)" ",__ptxEntries,__cubinEntries,&__debugEntries0,0,0,0,0,0,0x47f24676,&__elfEntries1};
# 3 "/tmp/tmpxft_0000636c_00000000-1_CUDA_interface.cudafe1.stub.c" 2
struct __T20 {unsigned __par0;ReconstructionSample *__par1;unsigned *__par2;unsigned *__par3;unsigned *__par4;unsigned __par5;unsigned __par6;int __dummy_field;};
struct __T21 {int __par0;unsigned *__par1;ReconstructionSample *__par2;sampleArrayStruct __par3;int __dummy_field;};
struct __T22 {sampleArrayStruct __par0;unsigned *__par1;float2 *__par2;float *__par3;float __par4;int __dummy_field;};
extern void __device_stub__Z14binning_kerneljP20ReconstructionSamplePjS1_S1_jj(unsigned, ReconstructionSample *, unsigned *, unsigned *, unsigned *, unsigned, unsigned);
extern void __device_stub__Z14reorder_kerneliPjP20ReconstructionSample17sampleArrayStruct(int, unsigned *, ReconstructionSample *, sampleArrayStruct&);
extern void __device_stub__Z12gridding_GPU17sampleArrayStructPjP6float2Pff(sampleArrayStruct&, unsigned *, float2 *, float *, float);
static void __sti____cudaRegisterAll_50_tmpxft_0000636c_00000000_10_CUDA_interface_cpp1_ii_cc7a22b8(void) __attribute__((__constructor__));
void __device_stub__Z14binning_kerneljP20ReconstructionSamplePjS1_S1_jj(unsigned __par0, ReconstructionSample *__par1, unsigned *__par2, unsigned *__par3, unsigned *__par4, unsigned __par5, unsigned __par6){ struct __T20 *__T23 = 0;
if (cudaSetupArgument((void*)(char*)&__par0, sizeof(__par0), (size_t)&__T23->__par0) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par1, sizeof(__par1), (size_t)&__T23->__par1) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par2, sizeof(__par2), (size_t)&__T23->__par2) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par3, sizeof(__par3), (size_t)&__T23->__par3) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par4, sizeof(__par4), (size_t)&__T23->__par4) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par5, sizeof(__par5), (size_t)&__T23->__par5) != cudaSuccess) return;if (cudaSetupArgument((void*)(char*)&__par6, sizeof(__par6), (size_t)&__T23->__par6) != cudaSuccess) return;{ volatile static char *__f; __f = ((char *)((void ( *)(unsigned, ReconstructionSample *, unsigned *, unsigned *, unsigned *, unsigned, unsigned))binning_kernel)); (void)cudaLaunch(((char *)((void ( *)(unsigned, ReconstructionSample *, unsigned *, unsigned *, unsigned *, unsigned, unsigned))binning_kernel))); };}
void binning_kernel( unsigned __cuda_0,ReconstructionSample *__cuda_1,unsigned *__cuda_2,unsigned *__cuda_3,unsigned *__cuda_4,unsigned __cuda_5,unsigned __cuda_6)
# 33 "GPU_kernels.cu"
{__device_stub__Z14binning_kerneljP20ReconstructionSamplePjS1_S1_jj( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 59 "GPU_kernels.cu"
}
# 1 "/tmp/tmpxft_0000636c_00000000-1_CUDA_interface.cudafe1.stub.c"
void __device_stub__Z14reorder_kerneliPjP20ReconstructionSample17sampleArrayStruct( int __par0, unsigned *__par1, ReconstructionSample *__par2, sampleArrayStruct&__par3) { struct __T21 *__T24 = 0;
if (cudaSetupArgument((void*)(char*)&__par0, sizeof(__par0), (size_t)&__T24->__par0) != cudaSuccess) return; if (cudaSetupArgument((void*)(char*)&__par1, sizeof(__par1), (size_t)&__T24->__par1) != cudaSuccess) return; if (cudaSetupArgument((void*)(char*)&__par2, sizeof(__par2), (size_t)&__T24->__par2) != cudaSuccess) return; if (cudaSetupArgument((void*)(char*)&__par3, sizeof(__par3), (size_t)&__T24->__par3) != cudaSuccess) return; { volatile static char *__f; __f = ((char *)((void ( *)(int, unsigned *, ReconstructionSample *, sampleArrayStruct))reorder_kernel)); (void)cudaLaunch(((char *)((void ( *)(int, unsigned *, ReconstructionSample *, sampleArrayStruct))reorder_kernel))); }; }
void reorder_kernel( int __cuda_0,unsigned *__cuda_1,ReconstructionSample *__cuda_2,sampleArrayStruct __cuda_3)
# 61 "GPU_kernels.cu"
{__device_stub__Z14reorder_kerneliPjP20ReconstructionSample17sampleArrayStruct( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 83 "GPU_kernels.cu"
}
# 1 "/tmp/tmpxft_0000636c_00000000-1_CUDA_interface.cudafe1.stub.c"
void __device_stub__Z12gridding_GPU17sampleArrayStructPjP6float2Pff( sampleArrayStruct&__par0, unsigned *__par1, float2 *__par2, float *__par3, float __par4) { struct __T22 *__T25 = 0;
if (cudaSetupArgument((void*)(char*)&__par0, sizeof(__par0), (size_t)&__T25->__par0) != cudaSuccess) return; if (cudaSetupArgument((void*)(char*)&__par1, sizeof(__par1), (size_t)&__T25->__par1) != cudaSuccess) return; if (cudaSetupArgument((void*)(char*)&__par2, sizeof(__par2), (size_t)&__T25->__par2) != cudaSuccess) return; if (cudaSetupArgument((void*)(char*)&__par3, sizeof(__par3), (size_t)&__T25->__par3) != cudaSuccess) return; if (cudaSetupArgument((void*)(char*)&__par4, sizeof(__par4), (size_t)&__T25->__par4) != cudaSuccess) return; { volatile static char *__f; __f = ((char *)((void ( *)(sampleArrayStruct, unsigned *, float2 *, float *, float))gridding_GPU)); (void)cudaLaunch(((char *)((void ( *)(sampleArrayStruct, unsigned *, float2 *, float *, float))gridding_GPU))); }; }
void gridding_GPU( sampleArrayStruct __cuda_0,unsigned *__cuda_1,float2 *__cuda_2,float *__cuda_3,float __cuda_4)
# 109 "GPU_kernels.cu"
{__device_stub__Z12gridding_GPU17sampleArrayStructPjP6float2Pff( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 251 "GPU_kernels.cu"
}
# 1 "/tmp/tmpxft_0000636c_00000000-1_CUDA_interface.cudafe1.stub.c"
static void __sti____cudaRegisterAll_50_tmpxft_0000636c_00000000_10_CUDA_interface_cpp1_ii_cc7a22b8(void) { __cudaFatCubinHandle = __cudaRegisterFatBinary((void*)&__fatDeviceText); atexit(__cudaUnregisterBinaryUtil); __cudaRegisterFunction(__cudaFatCubinHandle, (const char*)((void ( *)(sampleArrayStruct, unsigned *, float2 *, float *, float))gridding_GPU), (char*)"_Z12gridding_GPU17sampleArrayStructPjP6float2Pff", "_Z12gridding_GPU17sampleArrayStructPjP6float2Pff", -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0); __cudaRegisterFunction(__cudaFatCubinHandle, (const char*)((void ( *)(int, unsigned *, ReconstructionSample *, sampleArrayStruct))reorder_kernel), (char*)"_Z14reorder_kerneliPjP20ReconstructionSample17sampleArrayStruct", "_Z14reorder_kerneliPjP20ReconstructionSample17sampleArrayStruct", -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0); __cudaRegisterFunction(__cudaFatCubinHandle, (const char*)((void ( *)(unsigned, ReconstructionSample *, unsigned *, unsigned *, unsigned *, unsigned, unsigned))binning_kernel), (char*)"_Z14binning_kerneljP20ReconstructionSamplePjS1_S1_jj", "_Z14binning_kerneljP20ReconstructionSamplePjS1_S1_jj", -1, (uint3*)0, (uint3*)0, (dim3*)0, (dim3*)0, (int*)0); __cudaRegisterVar(__cudaFatCubinHandle, (char*)&cutoff2_c, (char*)"cutoff2_c", "cutoff2_c", 0, 4, 1, 0); __cudaRegisterVar(__cudaFatCubinHandle, (char*)&cutoff_c, (char*)"cutoff_c", "cutoff_c", 0, 4, 1, 0); __cudaRegisterVar(__cudaFatCubinHandle, (char*)&gridSize_c, (char*)"gridSize_c", "gridSize_c", 0, 12, 1, 0); __cudaRegisterVar(__cudaFatCubinHandle, (char*)&size_xy_c, (char*)"size_xy_c", "size_xy_c", 0, 4, 1, 0); __cudaRegisterVar(__cudaFatCubinHandle, (char*)&_1overCutoff2_c, (char*)"_1overCutoff2_c", "_1overCutoff2_c", 0, 4, 1, 0); }
# 1 "/tmp/tmpxft_0000636c_00000000-1_CUDA_interface.cudafe1.stub.c" 2