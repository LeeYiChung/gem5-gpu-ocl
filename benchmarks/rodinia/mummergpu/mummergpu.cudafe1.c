# 1 "mummergpu.cu"
# 1017 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h"
struct CUstream_st;
# 206 "/usr/include/libio.h" 3
enum __codecvt_result {

__codecvt_ok,
__codecvt_partial,
__codecvt_error,
__codecvt_noconv};
# 271 "/usr/include/libio.h" 3
struct _IO_FILE;
# 75 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
struct timeval;
# 203 "/usr/include/math.h" 3
enum _ZUt_ {
FP_NAN,

FP_INFINITE,

FP_ZERO,

FP_SUBNORMAL,

FP_NORMAL};
# 296 "/usr/include/math.h" 3
enum _LIB_VERSION_TYPE {
_IEEE_ = (-1),
_SVID_,
_XOPEN_,
_POSIX_,
_ISOC_};
# 194 "/usr/include/x86_64-linux-gnu/bits/fcntl.h" 3
enum __pid_type {

F_OWNER_TID,
F_OWNER_PID,
F_OWNER_PGRP,
F_OWNER_GID = 2};
# 27 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
enum _ZUt0_ {
_PC_LINK_MAX,

_PC_MAX_CANON,

_PC_MAX_INPUT,

_PC_NAME_MAX,

_PC_PATH_MAX,

_PC_PIPE_BUF,

_PC_CHOWN_RESTRICTED,

_PC_NO_TRUNC,

_PC_VDISABLE,

_PC_SYNC_IO,

_PC_ASYNC_IO,

_PC_PRIO_IO,

_PC_SOCK_MAXBUF,

_PC_FILESIZEBITS,

_PC_REC_INCR_XFER_SIZE,

_PC_REC_MAX_XFER_SIZE,

_PC_REC_MIN_XFER_SIZE,

_PC_REC_XFER_ALIGN,

_PC_ALLOC_SIZE_MIN,

_PC_SYMLINK_MAX,

_PC_2_SYMLINKS};
# 74 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
enum _ZUt1_ {
_SC_ARG_MAX,

_SC_CHILD_MAX,

_SC_CLK_TCK,

_SC_NGROUPS_MAX,

_SC_OPEN_MAX,

_SC_STREAM_MAX,

_SC_TZNAME_MAX,

_SC_JOB_CONTROL,

_SC_SAVED_IDS,

_SC_REALTIME_SIGNALS,

_SC_PRIORITY_SCHEDULING,

_SC_TIMERS,

_SC_ASYNCHRONOUS_IO,

_SC_PRIORITIZED_IO,

_SC_SYNCHRONIZED_IO,

_SC_FSYNC,

_SC_MAPPED_FILES,

_SC_MEMLOCK,

_SC_MEMLOCK_RANGE,

_SC_MEMORY_PROTECTION,

_SC_MESSAGE_PASSING,

_SC_SEMAPHORES,

_SC_SHARED_MEMORY_OBJECTS,

_SC_AIO_LISTIO_MAX,

_SC_AIO_MAX,

_SC_AIO_PRIO_DELTA_MAX,

_SC_DELAYTIMER_MAX,

_SC_MQ_OPEN_MAX,

_SC_MQ_PRIO_MAX,

_SC_VERSION,

_SC_PAGESIZE,


_SC_RTSIG_MAX,

_SC_SEM_NSEMS_MAX,

_SC_SEM_VALUE_MAX,

_SC_SIGQUEUE_MAX,

_SC_TIMER_MAX,




_SC_BC_BASE_MAX,

_SC_BC_DIM_MAX,

_SC_BC_SCALE_MAX,

_SC_BC_STRING_MAX,

_SC_COLL_WEIGHTS_MAX,

_SC_EQUIV_CLASS_MAX,

_SC_EXPR_NEST_MAX,

_SC_LINE_MAX,

_SC_RE_DUP_MAX,

_SC_CHARCLASS_NAME_MAX,


_SC_2_VERSION,

_SC_2_C_BIND,

_SC_2_C_DEV,

_SC_2_FORT_DEV,

_SC_2_FORT_RUN,

_SC_2_SW_DEV,

_SC_2_LOCALEDEF,


_SC_PII,

_SC_PII_XTI,

_SC_PII_SOCKET,

_SC_PII_INTERNET,

_SC_PII_OSI,

_SC_POLL,

_SC_SELECT,

_SC_UIO_MAXIOV,

_SC_IOV_MAX = 60,

_SC_PII_INTERNET_STREAM,

_SC_PII_INTERNET_DGRAM,

_SC_PII_OSI_COTS,

_SC_PII_OSI_CLTS,

_SC_PII_OSI_M,

_SC_T_IOV_MAX,



_SC_THREADS,

_SC_THREAD_SAFE_FUNCTIONS,

_SC_GETGR_R_SIZE_MAX,

_SC_GETPW_R_SIZE_MAX,

_SC_LOGIN_NAME_MAX,

_SC_TTY_NAME_MAX,

_SC_THREAD_DESTRUCTOR_ITERATIONS,

_SC_THREAD_KEYS_MAX,

_SC_THREAD_STACK_MIN,

_SC_THREAD_THREADS_MAX,

_SC_THREAD_ATTR_STACKADDR,

_SC_THREAD_ATTR_STACKSIZE,

_SC_THREAD_PRIORITY_SCHEDULING,

_SC_THREAD_PRIO_INHERIT,

_SC_THREAD_PRIO_PROTECT,

_SC_THREAD_PROCESS_SHARED,


_SC_NPROCESSORS_CONF,

_SC_NPROCESSORS_ONLN,

_SC_PHYS_PAGES,

_SC_AVPHYS_PAGES,

_SC_ATEXIT_MAX,

_SC_PASS_MAX,


_SC_XOPEN_VERSION,

_SC_XOPEN_XCU_VERSION,

_SC_XOPEN_UNIX,

_SC_XOPEN_CRYPT,

_SC_XOPEN_ENH_I18N,

_SC_XOPEN_SHM,


_SC_2_CHAR_TERM,

_SC_2_C_VERSION,

_SC_2_UPE,


_SC_XOPEN_XPG2,

_SC_XOPEN_XPG3,

_SC_XOPEN_XPG4,


_SC_CHAR_BIT,

_SC_CHAR_MAX,

_SC_CHAR_MIN,

_SC_INT_MAX,

_SC_INT_MIN,

_SC_LONG_BIT,

_SC_WORD_BIT,

_SC_MB_LEN_MAX,

_SC_NZERO,

_SC_SSIZE_MAX,

_SC_SCHAR_MAX,

_SC_SCHAR_MIN,

_SC_SHRT_MAX,

_SC_SHRT_MIN,

_SC_UCHAR_MAX,

_SC_UINT_MAX,

_SC_ULONG_MAX,

_SC_USHRT_MAX,


_SC_NL_ARGMAX,

_SC_NL_LANGMAX,

_SC_NL_MSGMAX,

_SC_NL_NMAX,

_SC_NL_SETMAX,

_SC_NL_TEXTMAX,


_SC_XBS5_ILP32_OFF32,

_SC_XBS5_ILP32_OFFBIG,

_SC_XBS5_LP64_OFF64,

_SC_XBS5_LPBIG_OFFBIG,


_SC_XOPEN_LEGACY,

_SC_XOPEN_REALTIME,

_SC_XOPEN_REALTIME_THREADS,


_SC_ADVISORY_INFO,

_SC_BARRIERS,

_SC_BASE,

_SC_C_LANG_SUPPORT,

_SC_C_LANG_SUPPORT_R,

_SC_CLOCK_SELECTION,

_SC_CPUTIME,

_SC_THREAD_CPUTIME,

_SC_DEVICE_IO,

_SC_DEVICE_SPECIFIC,

_SC_DEVICE_SPECIFIC_R,

_SC_FD_MGMT,

_SC_FIFO,

_SC_PIPE,

_SC_FILE_ATTRIBUTES,

_SC_FILE_LOCKING,

_SC_FILE_SYSTEM,

_SC_MONOTONIC_CLOCK,

_SC_MULTI_PROCESS,

_SC_SINGLE_PROCESS,

_SC_NETWORKING,

_SC_READER_WRITER_LOCKS,

_SC_SPIN_LOCKS,

_SC_REGEXP,

_SC_REGEX_VERSION,

_SC_SHELL,

_SC_SIGNALS,

_SC_SPAWN,

_SC_SPORADIC_SERVER,

_SC_THREAD_SPORADIC_SERVER,

_SC_SYSTEM_DATABASE,

_SC_SYSTEM_DATABASE_R,

_SC_TIMEOUTS,

_SC_TYPED_MEMORY_OBJECTS,

_SC_USER_GROUPS,

_SC_USER_GROUPS_R,

_SC_2_PBS,

_SC_2_PBS_ACCOUNTING,

_SC_2_PBS_LOCATE,

_SC_2_PBS_MESSAGE,

_SC_2_PBS_TRACK,

_SC_SYMLOOP_MAX,

_SC_STREAMS,

_SC_2_PBS_CHECKPOINT,


_SC_V6_ILP32_OFF32,

_SC_V6_ILP32_OFFBIG,

_SC_V6_LP64_OFF64,

_SC_V6_LPBIG_OFFBIG,


_SC_HOST_NAME_MAX,

_SC_TRACE,

_SC_TRACE_EVENT_FILTER,

_SC_TRACE_INHERIT,

_SC_TRACE_LOG,


_SC_LEVEL1_ICACHE_SIZE,

_SC_LEVEL1_ICACHE_ASSOC,

_SC_LEVEL1_ICACHE_LINESIZE,

_SC_LEVEL1_DCACHE_SIZE,

_SC_LEVEL1_DCACHE_ASSOC,

_SC_LEVEL1_DCACHE_LINESIZE,

_SC_LEVEL2_CACHE_SIZE,

_SC_LEVEL2_CACHE_ASSOC,

_SC_LEVEL2_CACHE_LINESIZE,

_SC_LEVEL3_CACHE_SIZE,

_SC_LEVEL3_CACHE_ASSOC,

_SC_LEVEL3_CACHE_LINESIZE,

_SC_LEVEL4_CACHE_SIZE,

_SC_LEVEL4_CACHE_ASSOC,

_SC_LEVEL4_CACHE_LINESIZE,



_SC_IPV6 = 235,

_SC_RAW_SOCKETS,


_SC_V7_ILP32_OFF32,

_SC_V7_ILP32_OFFBIG,

_SC_V7_LP64_OFF64,

_SC_V7_LPBIG_OFFBIG,


_SC_SS_REPL_MAX,


_SC_TRACE_EVENT_NAME_MAX,

_SC_TRACE_NAME_MAX,

_SC_TRACE_SYS_MAX,

_SC_TRACE_USER_EVENT_MAX,


_SC_XOPEN_STREAMS,


_SC_THREAD_ROBUST_PRIO_INHERIT,

_SC_THREAD_ROBUST_PRIO_PROTECT};
# 536 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
enum _ZUt2_ {
_CS_PATH,


_CS_V6_WIDTH_RESTRICTED_ENVS,



_CS_GNU_LIBC_VERSION,

_CS_GNU_LIBPTHREAD_VERSION,


_CS_V5_WIDTH_RESTRICTED_ENVS,



_CS_V7_WIDTH_RESTRICTED_ENVS,



_CS_LFS_CFLAGS = 1000,

_CS_LFS_LDFLAGS,

_CS_LFS_LIBS,

_CS_LFS_LINTFLAGS,

_CS_LFS64_CFLAGS,

_CS_LFS64_LDFLAGS,

_CS_LFS64_LIBS,

_CS_LFS64_LINTFLAGS,


_CS_XBS5_ILP32_OFF32_CFLAGS = 1100,

_CS_XBS5_ILP32_OFF32_LDFLAGS,

_CS_XBS5_ILP32_OFF32_LIBS,

_CS_XBS5_ILP32_OFF32_LINTFLAGS,

_CS_XBS5_ILP32_OFFBIG_CFLAGS,

_CS_XBS5_ILP32_OFFBIG_LDFLAGS,

_CS_XBS5_ILP32_OFFBIG_LIBS,

_CS_XBS5_ILP32_OFFBIG_LINTFLAGS,

_CS_XBS5_LP64_OFF64_CFLAGS,

_CS_XBS5_LP64_OFF64_LDFLAGS,

_CS_XBS5_LP64_OFF64_LIBS,

_CS_XBS5_LP64_OFF64_LINTFLAGS,

_CS_XBS5_LPBIG_OFFBIG_CFLAGS,

_CS_XBS5_LPBIG_OFFBIG_LDFLAGS,

_CS_XBS5_LPBIG_OFFBIG_LIBS,

_CS_XBS5_LPBIG_OFFBIG_LINTFLAGS,


_CS_POSIX_V6_ILP32_OFF32_CFLAGS,

_CS_POSIX_V6_ILP32_OFF32_LDFLAGS,

_CS_POSIX_V6_ILP32_OFF32_LIBS,

_CS_POSIX_V6_ILP32_OFF32_LINTFLAGS,

_CS_POSIX_V6_ILP32_OFFBIG_CFLAGS,

_CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS,

_CS_POSIX_V6_ILP32_OFFBIG_LIBS,

_CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS,

_CS_POSIX_V6_LP64_OFF64_CFLAGS,

_CS_POSIX_V6_LP64_OFF64_LDFLAGS,

_CS_POSIX_V6_LP64_OFF64_LIBS,

_CS_POSIX_V6_LP64_OFF64_LINTFLAGS,

_CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS,

_CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS,

_CS_POSIX_V6_LPBIG_OFFBIG_LIBS,

_CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS,


_CS_POSIX_V7_ILP32_OFF32_CFLAGS,

_CS_POSIX_V7_ILP32_OFF32_LDFLAGS,

_CS_POSIX_V7_ILP32_OFF32_LIBS,

_CS_POSIX_V7_ILP32_OFF32_LINTFLAGS,

_CS_POSIX_V7_ILP32_OFFBIG_CFLAGS,

_CS_POSIX_V7_ILP32_OFFBIG_LDFLAGS,

_CS_POSIX_V7_ILP32_OFFBIG_LIBS,

_CS_POSIX_V7_ILP32_OFFBIG_LINTFLAGS,

_CS_POSIX_V7_LP64_OFF64_CFLAGS,

_CS_POSIX_V7_LP64_OFF64_LDFLAGS,

_CS_POSIX_V7_LP64_OFF64_LIBS,

_CS_POSIX_V7_LP64_OFF64_LINTFLAGS,

_CS_POSIX_V7_LPBIG_OFFBIG_CFLAGS,

_CS_POSIX_V7_LPBIG_OFFBIG_LDFLAGS,

_CS_POSIX_V7_LPBIG_OFFBIG_LIBS,

_CS_POSIX_V7_LPBIG_OFFBIG_LINTFLAGS,


_CS_V6_ENV,

_CS_V7_ENV};
# 57 "/usr/include/x86_64-linux-gnu/sys/time.h" 3
struct timezone;
# 93 "/usr/include/x86_64-linux-gnu/sys/time.h" 3
enum __itimer_which {


ITIMER_REAL,


ITIMER_VIRTUAL,



ITIMER_PROF};
# 195 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUipcMem_flags_enum {
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1};
# 204 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUctx_flags_enum {
CU_CTX_SCHED_AUTO,
CU_CTX_SCHED_SPIN,
CU_CTX_SCHED_YIELD,
CU_CTX_SCHED_BLOCKING_SYNC = 4,
CU_CTX_BLOCKING_SYNC = 4,


CU_CTX_SCHED_MASK = 7,
CU_CTX_MAP_HOST,
CU_CTX_LMEM_RESIZE_TO_MAX = 16,
CU_CTX_FLAGS_MASK = 31};
# 221 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUevent_flags_enum {
CU_EVENT_DEFAULT,
CU_EVENT_BLOCKING_SYNC,
CU_EVENT_DISABLE_TIMING,
CU_EVENT_INTERPROCESS = 4};
# 231 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUarray_format_enum {
CU_AD_FORMAT_UNSIGNED_INT8 = 1,
CU_AD_FORMAT_UNSIGNED_INT16,
CU_AD_FORMAT_UNSIGNED_INT32,
CU_AD_FORMAT_SIGNED_INT8 = 8,
CU_AD_FORMAT_SIGNED_INT16,
CU_AD_FORMAT_SIGNED_INT32,
CU_AD_FORMAT_HALF = 16,
CU_AD_FORMAT_FLOAT = 32};
# 245 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUaddress_mode_enum {
CU_TR_ADDRESS_MODE_WRAP,
CU_TR_ADDRESS_MODE_CLAMP,
CU_TR_ADDRESS_MODE_MIRROR,
CU_TR_ADDRESS_MODE_BORDER};
# 255 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUfilter_mode_enum {
CU_TR_FILTER_MODE_POINT,
CU_TR_FILTER_MODE_LINEAR};
# 263 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUdevice_attribute_enum {
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
CU_DEVICE_ATTRIBUTE_WARP_SIZE,
CU_DEVICE_ATTRIBUTE_MAX_PITCH,
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
CU_DEVICE_ATTRIBUTE_INTEGRATED,
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES,
CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE,
CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH};
# 362 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUpointer_attribute_enum {
CU_POINTER_ATTRIBUTE_CONTEXT = 1,
CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
CU_POINTER_ATTRIBUTE_HOST_POINTER};
# 372 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUfunction_attribute_enum {
# 378 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
# 385 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
# 391 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,




CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,




CU_FUNC_ATTRIBUTE_NUM_REGS,
# 410 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_PTX_VERSION,
# 419 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_BINARY_VERSION,

CU_FUNC_ATTRIBUTE_MAX};
# 427 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUfunc_cache_enum {
CU_FUNC_CACHE_PREFER_NONE,
CU_FUNC_CACHE_PREFER_SHARED,
CU_FUNC_CACHE_PREFER_L1,
CU_FUNC_CACHE_PREFER_EQUAL};
# 437 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUsharedconfig_enum {
CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE,
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE,
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE};
# 446 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUmemorytype_enum {
CU_MEMORYTYPE_HOST = 1,
CU_MEMORYTYPE_DEVICE,
CU_MEMORYTYPE_ARRAY,
CU_MEMORYTYPE_UNIFIED};
# 456 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUcomputemode_enum {
CU_COMPUTEMODE_DEFAULT,
CU_COMPUTEMODE_EXCLUSIVE,
CU_COMPUTEMODE_PROHIBITED,
CU_COMPUTEMODE_EXCLUSIVE_PROCESS};
# 466 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUjit_option_enum {
# 472 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_JIT_MAX_REGISTERS,
# 485 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_JIT_THREADS_PER_BLOCK,
# 492 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_JIT_WALL_TIME,
# 500 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_JIT_INFO_LOG_BUFFER,
# 508 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
# 516 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_JIT_ERROR_LOG_BUFFER,
# 524 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
# 531 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_JIT_OPTIMIZATION_LEVEL,
# 538 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_JIT_TARGET_FROM_CUCONTEXT,
# 544 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_JIT_TARGET,
# 551 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_JIT_FALLBACK_STRATEGY};
# 558 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUjit_target_enum {

CU_TARGET_COMPUTE_10,
CU_TARGET_COMPUTE_11,
CU_TARGET_COMPUTE_12,
CU_TARGET_COMPUTE_13,
CU_TARGET_COMPUTE_20,
CU_TARGET_COMPUTE_21,
CU_TARGET_COMPUTE_30};
# 572 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUjit_fallback_enum {

CU_PREFER_PTX,

CU_PREFER_BINARY};
# 583 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUgraphicsRegisterFlags_enum {
CU_GRAPHICS_REGISTER_FLAGS_NONE,
CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY,
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4,
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8};
# 594 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUgraphicsMapResourceFlags_enum {
CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE,
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY,
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD};
# 603 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUarray_cubemap_face_enum {
CU_CUBEMAP_FACE_POSITIVE_X,
CU_CUBEMAP_FACE_NEGATIVE_X,
CU_CUBEMAP_FACE_POSITIVE_Y,
CU_CUBEMAP_FACE_NEGATIVE_Y,
CU_CUBEMAP_FACE_POSITIVE_Z,
CU_CUBEMAP_FACE_NEGATIVE_Z};
# 615 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUlimit_enum {
CU_LIMIT_STACK_SIZE,
CU_LIMIT_PRINTF_FIFO_SIZE,
CU_LIMIT_MALLOC_HEAP_SIZE};
# 624 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum cudaError_enum {
# 630 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_SUCCESS,
# 636 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_INVALID_VALUE,
# 642 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_OUT_OF_MEMORY,
# 648 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_NOT_INITIALIZED,




CUDA_ERROR_DEINITIALIZED,
# 659 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_PROFILER_DISABLED,




CUDA_ERROR_PROFILER_NOT_INITIALIZED,




CUDA_ERROR_PROFILER_ALREADY_STARTED,




CUDA_ERROR_PROFILER_ALREADY_STOPPED,




CUDA_ERROR_NO_DEVICE = 100,
# 685 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_INVALID_DEVICE,
# 692 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_INVALID_IMAGE = 200,
# 702 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_INVALID_CONTEXT,
# 711 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_CONTEXT_ALREADY_CURRENT,




CUDA_ERROR_MAP_FAILED = 205,




CUDA_ERROR_UNMAP_FAILED,
# 727 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_ARRAY_IS_MAPPED,




CUDA_ERROR_ALREADY_MAPPED,
# 740 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_NO_BINARY_FOR_GPU,




CUDA_ERROR_ALREADY_ACQUIRED,




CUDA_ERROR_NOT_MAPPED,
# 756 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
# 762 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_NOT_MAPPED_AS_POINTER,
# 768 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_ECC_UNCORRECTABLE,
# 774 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_UNSUPPORTED_LIMIT,
# 781 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_CONTEXT_ALREADY_IN_USE,




CUDA_ERROR_INVALID_SOURCE = 300,




CUDA_ERROR_FILE_NOT_FOUND,




CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,




CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,




CUDA_ERROR_OPERATING_SYSTEM,
# 813 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_INVALID_HANDLE = 400,
# 820 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_NOT_FOUND = 500,
# 829 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_NOT_READY = 600,
# 840 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_LAUNCH_FAILED = 700,
# 851 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
# 862 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_LAUNCH_TIMEOUT,
# 868 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
# 875 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
# 882 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
# 888 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
# 895 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_CONTEXT_IS_DESTROYED,
# 903 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_ASSERT,
# 910 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_TOO_MANY_PEERS,
# 916 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
# 922 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,




CUDA_ERROR_UNKNOWN = 999};
# 118 "./common.cuh"
struct _ZN14TextureAddress4__C1Ut_E;
# 115 "./common.cuh"
union __C1;
# 114 "./common.cuh"
struct TextureAddress;
# 130 "./common.cuh"
struct PixelOfNode;
# 150 "./common.cuh"
struct PixelOfChildren;
# 179 "./common.cuh"
struct MatchInfo;
# 189 "./common.cuh"
struct Alignment;
# 6 "./mummergpu.h"
struct QuerySet;
# 26 "./mummergpu.h"
struct AuxiliaryNodeData;
# 33 "./mummergpu.h"
struct Reference;
# 84 "./mummergpu.h"
struct _ZN10MatchCoord4__C5Ut_E;
# 81 "./mummergpu.h"
union __C5;
# 79 "./mummergpu.h"
struct MatchCoord;
# 91 "./mummergpu.h"
struct MatchResults;
# 112 "./mummergpu.h"
struct Statistics;
# 139 "./mummergpu.h"
struct MatchContext;
# 164 "./mummergpu.h"
struct ReferencePage;
# 205 "./mummergpu.h"
struct Timer_t;
# 153 "./mummergpu_kernel.cuh"
struct _ZN11_MatchCoord4__C6Ut_E;
# 150 "./mummergpu_kernel.cuh"
union __C6;
# 147 "./mummergpu_kernel.cuh"
struct _MatchCoord;
# 170 "./mummergpu_kernel.cuh"
struct _ZN16_PixelOfChildren4__C74__C8Ut_E;
# 181 "./mummergpu_kernel.cuh"
struct _ZN16_PixelOfChildren4__C74__C8Ut0_E;
# 168 "./mummergpu_kernel.cuh"
union __C8;
# 164 "./mummergpu_kernel.cuh"
union __C7;
# 161 "./mummergpu_kernel.cuh"
struct _PixelOfChildren;
# 200 "./mummergpu_kernel.cuh"
struct _ZN22_PixelOfChildrenNoData4__C95__C10Ut_E;
# 211 "./mummergpu_kernel.cuh"
struct _ZN22_PixelOfChildrenNoData4__C95__C10Ut0_E;
# 198 "./mummergpu_kernel.cuh"
union __C10;
# 194 "./mummergpu_kernel.cuh"
union __C9;
# 191 "./mummergpu_kernel.cuh"
struct _PixelOfChildrenNoData;
# 221 "./mummergpu_kernel.cuh"
struct _PixelOfChildrenNoDataBasesOnly;
# 239 "./mummergpu_kernel.cuh"
struct _ZN12_PixelOfNode5__C11Ut_E;
# 236 "./mummergpu_kernel.cuh"
union __C11;
# 233 "./mummergpu_kernel.cuh"
struct _PixelOfNode;
# 253 "./mummergpu_kernel.cuh"
struct _PixelOfNodeNoData;
# 124 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_voidIvEUt_E { _ZNSt9__is_voidIvE7__valueE = 1};
# 144 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIbEUt_E { _ZNSt12__is_integerIbE7__valueE = 1};
# 151 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIcEUt_E { _ZNSt12__is_integerIcE7__valueE = 1};
# 158 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIaEUt_E { _ZNSt12__is_integerIaE7__valueE = 1};
# 165 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIhEUt_E { _ZNSt12__is_integerIhE7__valueE = 1};
# 173 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIwEUt_E { _ZNSt12__is_integerIwE7__valueE = 1};
# 197 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIsEUt_E { _ZNSt12__is_integerIsE7__valueE = 1};
# 204 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerItEUt_E { _ZNSt12__is_integerItE7__valueE = 1};
# 211 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIiEUt_E { _ZNSt12__is_integerIiE7__valueE = 1};
# 218 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIjEUt_E { _ZNSt12__is_integerIjE7__valueE = 1};
# 225 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIlEUt_E { _ZNSt12__is_integerIlE7__valueE = 1};
# 232 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerImEUt_E { _ZNSt12__is_integerImE7__valueE = 1};
# 239 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIxEUt_E { _ZNSt12__is_integerIxE7__valueE = 1};
# 246 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIyEUt_E { _ZNSt12__is_integerIyE7__valueE = 1};
# 264 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIfEUt_E { _ZNSt13__is_floatingIfE7__valueE = 1};
# 271 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIdEUt_E { _ZNSt13__is_floatingIdE7__valueE = 1};
# 278 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIeEUt_E { _ZNSt13__is_floatingIeE7__valueE = 1};
# 354 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIcEUt_E { _ZNSt9__is_charIcE7__valueE = 1};
# 362 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIwEUt_E { _ZNSt9__is_charIwE7__valueE = 1};
# 377 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIcEUt_E { _ZNSt9__is_byteIcE7__valueE = 1};
# 384 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIaEUt_E { _ZNSt9__is_byteIaE7__valueE = 1};
# 391 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIhEUt_E { _ZNSt9__is_byteIhE7__valueE = 1};
# 134 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIfEUt_E { _ZNSt12__is_integerIfE7__valueE}; enum _ZNSt12__is_integerIdEUt_E { _ZNSt12__is_integerIdE7__valueE};
# 64 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
union _ZZ10__signbitlEUt_;
# 211 "/usr/lib/gcc/x86_64-linux-gnu/4.4.7/include/stddef.h" 3
typedef unsigned long size_t;
#include "crt/host_runtime.h"
# 141 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
typedef long __off_t;
# 149 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
typedef long __time_t;

typedef long __suseconds_t;
# 49 "/usr/include/stdio.h" 3
typedef struct _IO_FILE FILE;
# 75 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
struct timeval {

__time_t tv_sec;
__suseconds_t tv_usec;};
# 63 "/usr/include/x86_64-linux-gnu/sys/time.h" 3
typedef struct timezone *__restrict__ __timezone_ptr_t;
# 118 "./common.cuh"
struct _ZN14TextureAddress4__C1Ut_E {




unsigned x;};
# 115 "./common.cuh"
union __C1 {
unsigned data;

struct  {




unsigned x;};};
# 114 "./common.cuh"
struct TextureAddress {
union  {
unsigned data;

struct  {




unsigned x;};};};
# 179 "./common.cuh"
struct MatchInfo {

unsigned resultsoffset;
unsigned queryid;
struct TextureAddress matchnode;
unsigned numLeaves;
unsigned short edgematch;
unsigned short qrystartpos;};
typedef struct MatchInfo MatchInfo;

struct Alignment {

int left_in_ref;
unsigned short matchlen;char __nv_no_debug_dummy_end_padding_0[2];};
typedef struct Alignment Alignment;
# 6 "./mummergpu.h"
struct QuerySet {
int qfile;

char *h_tex_array;
char *d_tex_array;
int *d_addrs_tex_array;
int *h_addrs_tex_array;
int *h_lengths_array;
int *d_lengths_array;

char **h_names;

unsigned count;
size_t texlen;


size_t bytes_on_board;};
# 33 "./mummergpu.h"
struct Reference {

char *str;
size_t len;
float t_load_from_disk;

unsigned pitch;
void *d_ref_array;
char *h_ref_array;


void *d_node_tex_array;
void *h_node_tex_array;

void *d_children_tex_array;
void *h_children_tex_array;

void *d_parent_tex_array;
void *h_parent_tex_array;
# 61 "./mummergpu.h"
unsigned tex_node_height;
unsigned tex_children_height;
unsigned tex_width;


size_t bytes_on_board;

struct AuxiliaryNodeData *aux_data;
int num_nodes;char __nv_no_debug_dummy_end_padding_0[4];};
# 84 "./mummergpu.h"
struct _ZN10MatchCoord4__C5Ut_E {
struct TextureAddress node;
int edge_match_length;};
# 81 "./mummergpu.h"
union __C5 {
int2 data;

struct  {
struct TextureAddress node;
int edge_match_length;};};
# 79 "./mummergpu.h"
struct MatchCoord {

union  {
int2 data;

struct  {
struct TextureAddress node;
int edge_match_length;};};};




struct MatchResults {


struct MatchCoord *d_match_coords;
struct MatchCoord *h_match_coords;

unsigned numCoords;
# 105 "./mummergpu.h"
int *h_coord_tex_array;


size_t bytes_on_board;};



struct Statistics {
float t_end_to_end;
float t_match_kernel;
float t_print_kernel;
float t_results_to_disk;
float t_queries_to_board;
float t_match_coords_to_board;
float t_match_coords_from_board;
float t_tree_to_board;
float t_ref_str_to_board;
float t_queries_from_disk;
float t_ref_from_disk;
float t_tree_construction;
float t_tree_reorder;
float t_tree_flatten;
float t_reorder_ref_str;
float t_build_coord_offsets;
float t_coords_to_buffers;
float bp_avg_query_length;};
# 139 "./mummergpu.h"
struct MatchContext {
char *full_ref;
size_t full_ref_len;

struct Reference *ref;
struct QuerySet *queries;
struct MatchResults results;

char on_cpu;

int min_match_length;

char reverse;
char forwardreverse;
char forwardcoordinates;
char show_query_length;
char maxmatch;

char *stats_file;
char *dotfilename;
char *texfilename;
struct Statistics statistics;};



struct ReferencePage {
int begin;
int end;
int shadow_left;
int shadow_right;
struct MatchResults results;
unsigned id;
struct Reference ref;};
# 205 "./mummergpu.h"
struct Timer_t {

struct timeval start_m;
struct timeval end_m;};
# 153 "./mummergpu_kernel.cuh"
struct _ZN11_MatchCoord4__C6Ut_E {
int node;
int edge_match_length;};
# 150 "./mummergpu_kernel.cuh"
union __C6 {
int2 data;

struct  {
int node;
int edge_match_length;};};
# 147 "./mummergpu_kernel.cuh"
struct _MatchCoord {


union  {
int2 data;

struct  {
int node;
int edge_match_length;};};} __attribute__((__aligned__(8)));
# 170 "./mummergpu_kernel.cuh"
struct _ZN16_PixelOfChildren4__C74__C8Ut_E {
uchar3 a;
uchar3 c;
uchar3 g;
uchar3 t;
uchar3 d;

char leafchar;};



struct _ZN16_PixelOfChildren4__C74__C8Ut0_E {
uchar3 leafid;
unsigned char pad[12];
char leafchar0;};
# 168 "./mummergpu_kernel.cuh"
union __C8 {

struct  {
uchar3 a;
uchar3 c;
uchar3 g;
uchar3 t;
uchar3 d;

char leafchar;};



struct  {
uchar3 leafid;
unsigned char pad[12];
char leafchar0;};};
# 164 "./mummergpu_kernel.cuh"
union __C7 {
uint4 data;


union  {

struct  {
uchar3 a;
uchar3 c;
uchar3 g;
uchar3 t;
uchar3 d;

char leafchar;};



struct  {
uchar3 leafid;
unsigned char pad[12];
char leafchar0;};};};
# 161 "./mummergpu_kernel.cuh"
struct _PixelOfChildren {


union  {
uint4 data;


union  {

struct  {
uchar3 a;
uchar3 c;
uchar3 g;
uchar3 t;
uchar3 d;

char leafchar;};



struct  {
uchar3 leafid;
unsigned char pad[12];
char leafchar0;};};};};
# 200 "./mummergpu_kernel.cuh"
struct _ZN22_PixelOfChildrenNoData4__C95__C10Ut_E {
uchar3 a;
uchar3 c;
uchar3 g;
uchar3 t;
uchar3 d;

char leafchar;};



struct _ZN22_PixelOfChildrenNoData4__C95__C10Ut0_E {
uchar3 leafid;
unsigned char pad[12];
char leafchar0;};
# 198 "./mummergpu_kernel.cuh"
union __C10 {

struct  {
uchar3 a;
uchar3 c;
uchar3 g;
uchar3 t;
uchar3 d;

char leafchar;};



struct  {
uchar3 leafid;
unsigned char pad[12];
char leafchar0;};};
# 194 "./mummergpu_kernel.cuh"
union __C9 {
uint4 data;


union  {

struct  {
uchar3 a;
uchar3 c;
uchar3 g;
uchar3 t;
uchar3 d;

char leafchar;};



struct  {
uchar3 leafid;
unsigned char pad[12];
char leafchar0;};};};
# 191 "./mummergpu_kernel.cuh"
struct _PixelOfChildrenNoData {


union  {
uint4 data;


union  {

struct  {
uchar3 a;
uchar3 c;
uchar3 g;
uchar3 t;
uchar3 d;

char leafchar;};



struct  {
uchar3 leafid;
unsigned char pad[12];
char leafchar0;};};};};
# 221 "./mummergpu_kernel.cuh"
struct _PixelOfChildrenNoDataBasesOnly {

uchar3 a;
uchar3 c;
uchar3 g;
uchar3 t;
uchar3 d;

char leafchar;};
# 239 "./mummergpu_kernel.cuh"
struct _ZN12_PixelOfNode5__C11Ut_E {
uchar3 parent;
uchar3 suffix;

uchar3 start;
uchar3 end;
uchar3 depth;

unsigned char pad;};
# 236 "./mummergpu_kernel.cuh"
union __C11 {
uint4 data;

struct  {
uchar3 parent;
uchar3 suffix;

uchar3 start;
uchar3 end;
uchar3 depth;

unsigned char pad;};};
# 233 "./mummergpu_kernel.cuh"
struct _PixelOfNode {


union  {
uint4 data;

struct  {
uchar3 parent;
uchar3 suffix;

uchar3 start;
uchar3 end;
uchar3 depth;

unsigned char pad;};};};
# 253 "./mummergpu_kernel.cuh"
struct _PixelOfNodeNoData {

uchar3 parent;
uchar3 suffix;

uchar3 start;
uchar3 end;
uchar3 depth;

unsigned char pad;};
# 56 "/usr/include/stdint.h" 3
typedef unsigned long uint64_t;
# 64 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
union _ZZ10__signbitlEUt_ { long double __l; int __i[3];};
void *memcpy(void*, const void*, size_t); void *memset(void*, int, size_t);
# 711 "/home/bachelor/deicide218/cuda-4.2/include/cuda_runtime_api.h"
extern cudaError_t cudaThreadExit(void);
# 735 "/home/bachelor/deicide218/cuda-4.2/include/cuda_runtime_api.h"
extern cudaError_t cudaThreadSynchronize(void);
# 958 "/home/bachelor/deicide218/cuda-4.2/include/cuda_runtime_api.h"
extern cudaError_t cudaGetLastError(void);
# 1013 "/home/bachelor/deicide218/cuda-4.2/include/cuda_runtime_api.h"
extern const char *cudaGetErrorString(cudaError_t);
# 1786 "/home/bachelor/deicide218/cuda-4.2/include/cuda_runtime_api.h"
extern cudaError_t cudaConfigureCall(dim3, dim3, size_t, cudaStream_t);
# 2055 "/home/bachelor/deicide218/cuda-4.2/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc(void **, size_t);
# 2189 "/home/bachelor/deicide218/cuda-4.2/include/cuda_runtime_api.h"
extern cudaError_t cudaFree(void *);
# 2870 "/home/bachelor/deicide218/cuda-4.2/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy(void *, const void *, size_t, enum cudaMemcpyKind);
# 3653 "/home/bachelor/deicide218/cuda-4.2/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset(void *, int, size_t);
# 4396 "/home/bachelor/deicide218/cuda-4.2/include/cuda_runtime_api.h"
extern struct cudaChannelFormatDesc cudaCreateChannelDesc(int, int, int, int, enum cudaChannelFormatKind);
# 70 "/home/bachelor/deicide218/cuda-4.2/include/common_functions.h"
extern void *memset(void *, int, size_t);
# 128 "/usr/include/string.h" 3
extern char *strcpy(char *__restrict__, const char *__restrict__);


extern char *strncpy(char *__restrict__, const char *__restrict__, size_t);
# 234 "/usr/include/stdio.h" 3
extern int fclose(FILE *);
# 269 "/usr/include/stdio.h" 3
extern FILE *fopen(const char *__restrict__, const char *__restrict__);
# 102 "/home/bachelor/deicide218/cuda-4.2/include/common_functions.h"
extern int fprintf(FILE *__restrict__, const char *__restrict__, ...);
# 101 "/home/bachelor/deicide218/cuda-4.2/include/common_functions.h"
extern int printf(const char *__restrict__, ...);
# 361 "/usr/include/stdio.h" 3
extern int sprintf(char *__restrict__, const char *__restrict__, ...);
# 103 "/home/bachelor/deicide218/cuda-4.2/include/common_functions.h"
extern __attribute__((__malloc__)) void *malloc(size_t);
# 473 "/usr/include/stdlib.h" 3
extern __attribute__((__malloc__)) void *calloc(size_t, size_t);
# 104 "/home/bachelor/deicide218/cuda-4.2/include/common_functions.h"
extern void free(void *);
# 544 "/usr/include/stdlib.h" 3
extern __attribute__((__noreturn__)) void exit(int);
# 117 "/home/bachelor/deicide218/cuda-4.2/include/common_functions.h"
extern __attribute__((__noreturn__)) void __assert_fail(const char *, const char *, unsigned, const char *);
# 189 "/home/bachelor/deicide218/cuda-4.2/include/math_functions.h"
extern int min(int, int);
# 225 "/home/bachelor/deicide218/cuda-4.2/include/math_functions.h"
extern int max(int, int);
# 1048 "/home/bachelor/deicide218/cuda-4.2/include/math_functions.h"
extern __attribute__((__const__)) double ceil(double);
# 38 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
extern  __attribute__((__weak__)) /* COMDAT group: __signbitf */ __inline__ __attribute__((__const__)) int __signbitf(float);
# 50 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
extern  __attribute__((__weak__)) /* COMDAT group: __signbit */ __inline__ __attribute__((__const__)) int __signbit(double);
# 62 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
extern  __attribute__((__weak__)) /* COMDAT group: __signbitl */ __inline__ __attribute__((__const__)) int __signbitl(long double);
# 119 "/usr/include/fcntl.h" 3
extern int open(const char *, int, ...);
# 331 "/usr/include/unistd.h" 3
extern __off_t lseek(int, __off_t, int);
# 350 "/usr/include/unistd.h" 3
extern int close(int);
# 47 "/usr/include/x86_64-linux-gnu/bits/errno.h" 3
extern __attribute__((__const__)) int *__errno_location(void);
# 73 "/usr/include/x86_64-linux-gnu/sys/time.h" 3
extern int gettimeofday(struct timeval *__restrict__, struct timezone *__restrict__);
# 27 "mummergpu.cu"
extern void m5_work_begin(uint64_t, uint64_t);
extern void m5_work_end(uint64_t, uint64_t);
# 39 "mummergpu.cu"
extern void _Z8trap_dbgv(void);
# 130 "mummergpu.cu"
extern void getReferenceString(const char *, char **, size_t *);


extern void createTreeTexture(const char *, struct PixelOfNode **, struct PixelOfChildren **, unsigned *, unsigned *, unsigned *, struct AuxiliaryNodeData **, int *, int, struct Statistics *, const char *, const char *);
# 147 "mummergpu.cu"
extern void getQueriesTexture(int, char **, size_t *, int **, char ***, int **, unsigned *, unsigned *, unsigned, int, char);
# 160 "mummergpu.cu"
extern int lookupNumLeaves(struct ReferencePage *, struct TextureAddress);

extern void _Z15printAlignmentsP13ReferencePageP9AlignmentPci14TextureAddressiiibb(struct ReferencePage *, Alignment *, char *, int, struct TextureAddress, int, int, int, char, char);
# 183 "mummergpu.cu"
extern char *createTimer(void);
# 190 "mummergpu.cu"
extern void startTimer(char *);




extern void stopTimer(char *);




extern float getTimerValue(char *);
# 216 "mummergpu.cu"
extern void deleteTimer(char *);





extern int createReference(const char *, struct Reference *);
# 240 "mummergpu.cu"
extern int destroyReference(struct Reference *);
# 261 "mummergpu.cu"
extern int createQuerySet(const char *, struct QuerySet *);
# 279 "mummergpu.cu"
extern int destroyQuerySet(struct QuerySet *);
# 289 "mummergpu.cu"
extern void printStringForError(int);





extern int createMatchContext(struct Reference *, struct QuerySet *, struct MatchResults *, char, int, char *, char, char, char, char, char *, char *, struct MatchContext *);
# 328 "mummergpu.cu"
extern int destroyMatchContext(struct MatchContext *);
# 336 "mummergpu.cu"
extern void _Z21buildReferenceTextureP9ReferencePcmmiS1_S1_P10Statistics(struct Reference *, char *, size_t, size_t, int, char *, char *, struct Statistics *);
# 469 "mummergpu.cu"
extern void _Z11boardMemoryPjS_(unsigned *, unsigned *);
# 482 "mummergpu.cu"
extern void _Z20loadReferenceTextureP12MatchContext(struct MatchContext *);
# 576 "mummergpu.cu"
extern void _Z21unloadReferenceStringP9Reference(struct Reference *);
# 591 "mummergpu.cu"
extern void _Z19unloadReferenceTreeP12MatchContext(struct MatchContext *);
# 650 "mummergpu.cu"
extern void _Z13loadReferenceP12MatchContext(struct MatchContext *);
# 906 "mummergpu.cu"
extern void _Z18dumpQueryBlockInfoP8QuerySet(struct QuerySet *);
# 913 "mummergpu.cu"
extern void _Z11loadQueriesP12MatchContext(struct MatchContext *);
# 983 "mummergpu.cu"
extern void _Z13unloadQueriesP12MatchContext(struct MatchContext *);
# 1009 "mummergpu.cu"
extern void _Z21buildCoordOffsetArrayP12MatchContextPPiPj(struct MatchContext *, int **, unsigned *);
# 1061 "mummergpu.cu"
extern void _Z16loadResultBufferP12MatchContext(struct MatchContext *);
# 1133 "mummergpu.cu"
extern void _Z18unloadResultBufferP12MatchContext(struct MatchContext *);
# 1143 "mummergpu.cu"
extern void _Z25transferResultsFromDeviceP12MatchContext(struct MatchContext *);
# 1206 "mummergpu.cu"
extern int _Z11flushOutputv(void);
extern int _Z11addToBufferPc(char *);



extern struct MatchCoord *_Z17coordForQueryCharP12MatchContextjj(struct MatchContext *, unsigned, unsigned);
# 1224 "mummergpu.cu"
extern void _Z20coordsToPrintBuffersP12MatchContextP13ReferencePagePP9MatchInfoPP9AlignmentjPjS9_S9_S9_S9_(struct MatchContext *, struct ReferencePage *, MatchInfo **, Alignment **, unsigned, unsigned *, unsigned *, unsigned *, unsigned *, unsigned *);
# 1350 "mummergpu.cu"
extern void _Z14runPrintKernelP12MatchContextP13ReferencePageP9MatchInfojP9Alignmentj(struct MatchContext *, struct ReferencePage *, MatchInfo *, unsigned, Alignment *, unsigned);
# 1468 "mummergpu.cu"
extern void _Z13runPrintOnCPUP12MatchContextP13ReferencePageP9MatchInfojP9Alignmentj(struct MatchContext *, struct ReferencePage *, MatchInfo *, unsigned, Alignment *, unsigned);
# 1516 "mummergpu.cu"
extern void _Z18getExactAlignmentsP12MatchContextP13ReferencePageb(struct MatchContext *, struct ReferencePage *, char);
# 1684 "mummergpu.cu"
extern int _Z13getQueryBlockP12MatchContextm(struct MatchContext *, size_t);
# 1730 "mummergpu.cu"
extern void _Z17destroyQueryBlockP8QuerySet(struct QuerySet *);
# 1750 "mummergpu.cu"
extern void _Z10resetStatsP10Statistics(struct Statistics *);
# 1788 "mummergpu.cu"
extern void _Z19writeStatisticsFileP10StatisticsPcS1_S1_(struct Statistics *, char *, char *, char *);
# 1914 "mummergpu.cu"
extern void _Z10matchOnCPUP12MatchContextb(struct MatchContext *, char);
# 1946 "mummergpu.cu"
extern void _Z10matchOnGPUP12MatchContextb(struct MatchContext *, char);
# 2009 "mummergpu.cu"
extern void _Z15getMatchResultsP12MatchContextj(struct MatchContext *, unsigned);





extern void _Z30matchQueryBlockToReferencePageP12MatchContextP13ReferencePageb(struct MatchContext *, struct ReferencePage *, char);
# 2050 "mummergpu.cu"
extern int _Z11matchSubsetP12MatchContextP13ReferencePage(struct MatchContext *, struct ReferencePage *);
# 2084 "mummergpu.cu"
extern int _Z19getFreeDeviceMemoryb(char);
# 2107 "mummergpu.cu"
extern int _Z27matchQueriesToReferencePageP12MatchContextP13ReferencePage(struct MatchContext *, struct ReferencePage *);
# 2138 "mummergpu.cu"
extern void _Z18initReferencePagesP12MatchContextPiPP13ReferencePage(struct MatchContext *, int *, struct ReferencePage **);
# 2179 "mummergpu.cu"
extern int _Z29streamReferenceAgainstQueriesP12MatchContext(struct MatchContext *);
# 2241 "mummergpu.cu"
extern int matchQueries(struct MatchContext *);
extern int __cudaSetupArgSimple();
extern int __cudaLaunch();
extern int __cudaRegisterBinary();
extern int __cudaRegisterEntry();
static void __sti___17_mummergpu_cpp1_ii_a6baf3c4(void) __attribute__((__constructor__));
# 166 "/usr/include/stdio.h" 3
extern struct _IO_FILE *stdout;
extern struct _IO_FILE *stderr;
# 32 "mummergpu.cu"
extern int USE_PRINT_KERNEL;





extern unsigned cuda_calls;
# 87 "mummergpu.cu"
extern unsigned num_bind_tex_calls;
# 1209 "mummergpu.cu"
char numbuffer[32];
extern  __attribute__((__weak__)) /* COMDAT group: _ZTSSt9exception */ const char _ZTSSt9exception[13] __attribute__((visibility("default")));
extern  __attribute__((__weak__)) /* COMDAT group: _ZTSSt9bad_alloc */ const char _ZTSSt9bad_alloc[13] __attribute__((visibility("default")));
# 32 "mummergpu.cu"
int USE_PRINT_KERNEL = 1;





unsigned cuda_calls = 0U;
# 87 "mummergpu.cu"
unsigned num_bind_tex_calls = 0U;
 __attribute__((__weak__)) /* COMDAT group: _ZTSSt9exception */ const char _ZTSSt9exception[13] __attribute__((visibility("default"))) = "St9exception";
 __attribute__((__weak__)) /* COMDAT group: _ZTSSt9bad_alloc */ const char _ZTSSt9bad_alloc[13] __attribute__((visibility("default"))) = "St9bad_alloc";
# 38 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
 __attribute__((__weak__)) /* COMDAT group: __signbitf */ __inline__ __attribute__((__const__)) int __signbitf( float __x)
{




 int __cuda_local_var_6664_7_non_const___m;
__asm("pmovmskb %1, %0" : "=r" (__cuda_local_var_6664_7_non_const___m) : "x" (__x));
return __cuda_local_var_6664_7_non_const___m & 8;

}

 __attribute__((__weak__)) /* COMDAT group: __signbit */ __inline__ __attribute__((__const__)) int __signbit( double __x)
{




 int __cuda_local_var_6676_7_non_const___m;
__asm("pmovmskb %1, %0" : "=r" (__cuda_local_var_6676_7_non_const___m) : "x" (__x));
return __cuda_local_var_6676_7_non_const___m & 128;

}

 __attribute__((__weak__)) /* COMDAT group: __signbitl */ __inline__ __attribute__((__const__)) int __signbitl( long double __x)
{
 union _ZZ10__signbitlEUt_ __cuda_local_var_6684_56_non_const___u;
# 64 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
(__cuda_local_var_6684_56_non_const___u.__l) = __x;
return (int)(((((__cuda_local_var_6684_56_non_const___u.__i))[2]) & 32768) != 0);
}
# 39 "mummergpu.cu"
void _Z8trap_dbgv(void)
{
fprintf(stderr, ((const char *)"Trapped\n")); 
}
# 183 "mummergpu.cu"
char *createTimer(void)
{
 unsigned *__cuda_local_var_20612_18_non_const_ptr;
# 185 "mummergpu.cu"
__cuda_local_var_20612_18_non_const_ptr = ((unsigned *)(malloc(32UL)));
memset(((void *)__cuda_local_var_20612_18_non_const_ptr), 0, 32UL);
return (char *)__cuda_local_var_20612_18_non_const_ptr;
}

void startTimer( char *ptr)
{
gettimeofday((&(((struct Timer_t *)ptr)->start_m)), ((struct timezone *)0LL)); 
}

void stopTimer( char *ptr)
{
gettimeofday((&(((struct Timer_t *)ptr)->end_m)), ((struct timezone *)0LL)); 
}

float getTimerValue( char *ptr)
{
 struct Timer_t *__cuda_local_var_20629_13_non_const_timer;
# 202 "mummergpu.cu"
__cuda_local_var_20629_13_non_const_timer = ((struct Timer_t *)ptr);

if (__cuda_local_var_20629_13_non_const_timer == ((struct Timer_t *)0LL))
{
fprintf(stderr, ((const char *)"Uninitialized timer!!!\n"));
return (0.0F);
}

if (((__cuda_local_var_20629_13_non_const_timer->end_m).tv_sec) == 0L) { stopTimer(ptr); }

return (float)(((1000.0) * ((double)(((__cuda_local_var_20629_13_non_const_timer->end_m).tv_sec) - ((__cuda_local_var_20629_13_non_const_timer->start_m).tv_sec)))) + ((0.001000000000000000021) * ((double)(((__cuda_local_var_20629_13_non_const_timer->end_m).tv_usec) - ((__cuda_local_var_20629_13_non_const_timer->start_m).tv_usec)))));

}

void deleteTimer( char *ptr)
{
free(((void *)((struct Timer_t *)ptr))); 
}


int createReference( const char *fromFile,  struct Reference *ref)
{



 char *__cuda_local_var_20654_11_non_const_loadreftimer;
# 224 "mummergpu.cu"
if ((!(fromFile)) || (!(ref))) {
return (-1); }

__cuda_local_var_20654_11_non_const_loadreftimer = (createTimer());
startTimer(__cuda_local_var_20654_11_non_const_loadreftimer);

getReferenceString(fromFile, (&(ref->str)), (&(ref->len)));

stopTimer(__cuda_local_var_20654_11_non_const_loadreftimer);
(ref->t_load_from_disk) += (getTimerValue(__cuda_local_var_20654_11_non_const_loadreftimer));
deleteTimer(__cuda_local_var_20654_11_non_const_loadreftimer);

return 0;
}


int destroyReference( struct Reference *ref)
{
free((ref->h_node_tex_array));
free((ref->h_children_tex_array));
free(((void *)(ref->str)));




free(((void *)(ref->aux_data)));




(ref->str) = ((char *)0LL);
(ref->len) = 0UL;

return 0;
}


int createQuerySet( const char *fromFile,  struct QuerySet *queries)
{


 int __cuda_local_var_20692_8_non_const_qfile;
# 264 "mummergpu.cu"
fprintf(stderr, ((const char *)"Opening %s...\n"), fromFile);
__cuda_local_var_20692_8_non_const_qfile = (open(fromFile, 0));

if (__cuda_local_var_20692_8_non_const_qfile == (-1))
{
fprintf(stderr, ((const char *)"Can\'t open %s: %d\n"), fromFile, (*(__errno_location())));
exit(1);
}

(queries->qfile) = __cuda_local_var_20692_8_non_const_qfile;

return 0;
}


int destroyQuerySet( struct QuerySet *queries)
{

if (queries->qfile) {
close((queries->qfile)); }

return 0;
}


void printStringForError( int err)
{  

}


int createMatchContext( struct Reference *ref, 
struct QuerySet *queries, 
struct MatchResults *matches, 
char on_cpu, 
int min_match_length, 
char *stats_file, 
char reverse, 
char forwardreverse, 
char forwardcoordinates, 
char showQueryLength, 
char *dotfilename, 
char *texfilename, 
struct MatchContext *ctx) {

(ctx->queries) = queries;
(ctx->ref) = ref;
(ctx->full_ref) = (ref->str);
(ctx->full_ref_len) = (ref->len);

(ctx->on_cpu) = on_cpu;
(ctx->min_match_length) = min_match_length;
(ctx->stats_file) = stats_file;
(ctx->reverse) = reverse;
(ctx->forwardreverse) = forwardreverse;
(ctx->forwardcoordinates) = forwardcoordinates;
(ctx->show_query_length) = showQueryLength;
(ctx->dotfilename) = dotfilename;
(ctx->texfilename) = texfilename;
return 0;
}



int destroyMatchContext( struct MatchContext *ctx)
{
free(((void *)(ctx->full_ref)));

destroyQuerySet((ctx->queries));
return 0;
}

void _Z21buildReferenceTextureP9ReferencePcmmiS1_S1_P10Statistics( struct Reference *ref, 
char *full_ref, 
size_t begin, 
size_t end, 
int min_match_len, 
char *dotfilename, 
char *texfilename, 
struct Statistics *statistics)
{


 struct PixelOfNode *__cuda_local_var_20774_18_non_const_nodeTexture;
 struct PixelOfChildren *__cuda_local_var_20775_23_non_const_childrenTexture;

 unsigned __cuda_local_var_20777_18_non_const_width;
 unsigned __cuda_local_var_20778_18_non_const_node_height;
 unsigned __cuda_local_var_20779_18_non_const_children_height;

 struct AuxiliaryNodeData *__cuda_local_var_20781_24_non_const_aux_data;
 int __cuda_local_var_20782_9_non_const_num_nodes;

 char *__cuda_local_var_20784_9_non_const_loadreftimer;
# 345 "mummergpu.cu"
fprintf(stderr, ((const char *)"Building reference texture...\n"));

__cuda_local_var_20774_18_non_const_nodeTexture = ((struct PixelOfNode *)0LL);
__cuda_local_var_20775_23_non_const_childrenTexture = ((struct PixelOfChildren *)0LL);

__cuda_local_var_20777_18_non_const_width = 0U;
__cuda_local_var_20778_18_non_const_node_height = 0U;
__cuda_local_var_20779_18_non_const_children_height = 0U;

__cuda_local_var_20781_24_non_const_aux_data = ((struct AuxiliaryNodeData *)0LL);


__cuda_local_var_20784_9_non_const_loadreftimer = (createTimer());
startTimer(__cuda_local_var_20784_9_non_const_loadreftimer);

(ref->len) = ((end - begin) + 3UL);
(ref->str) = ((char *)(malloc((ref->len))));
((ref->str)[0]) = ((char)115);
strncpy(((ref->str) + 1), ((const char *)(full_ref + begin)), ((ref->len) - 3UL));
strcpy((((ref->str) + (ref->len)) - 2), ((const char *)"$"));

stopTimer(__cuda_local_var_20784_9_non_const_loadreftimer);
(statistics->t_ref_from_disk) += ((getTimerValue(__cuda_local_var_20784_9_non_const_loadreftimer)) + (ref->t_load_from_disk));
deleteTimer(__cuda_local_var_20784_9_non_const_loadreftimer);

createTreeTexture(((const char *)(ref->str)), (&__cuda_local_var_20774_18_non_const_nodeTexture), (&__cuda_local_var_20775_23_non_const_childrenTexture), (&__cuda_local_var_20777_18_non_const_width), (&__cuda_local_var_20778_18_non_const_node_height), (&__cuda_local_var_20779_18_non_const_children_height), (&__cuda_local_var_20781_24_non_const_aux_data), (&__cuda_local_var_20782_9_non_const_num_nodes), min_match_len, statistics, ((const char *)dotfilename), ((const char *)texfilename));
# 383 "mummergpu.cu"
(ref->h_node_tex_array) = ((void *)__cuda_local_var_20774_18_non_const_nodeTexture);
(ref->h_children_tex_array) = ((void *)__cuda_local_var_20775_23_non_const_childrenTexture);
(ref->tex_width) = __cuda_local_var_20777_18_non_const_width;
(ref->tex_node_height) = __cuda_local_var_20778_18_non_const_node_height;
(ref->tex_children_height) = __cuda_local_var_20779_18_non_const_children_height;
# 394 "mummergpu.cu"
(ref->aux_data) = __cuda_local_var_20781_24_non_const_aux_data;
(ref->num_nodes) = __cuda_local_var_20782_9_non_const_num_nodes;

(ref->bytes_on_board) = ((((unsigned long)(__cuda_local_var_20777_18_non_const_width * __cuda_local_var_20778_18_non_const_node_height)) * 16UL) + (((unsigned long)(__cuda_local_var_20777_18_non_const_width * __cuda_local_var_20779_18_non_const_children_height)) * 16UL));

fprintf(stderr, ((const char *)"This tree will need %d bytes on the board\n"), (ref->bytes_on_board));
# 462 "mummergpu.cu"
fprintf(stderr, ((const char *)"The refstr requires %d bytes\n"), (ref->len));
(ref->bytes_on_board) += (ref->len); 



}

void _Z11boardMemoryPjS_( unsigned *free_mem,  unsigned *total_mem)
{



(*free_mem) = 536870912U;
(*total_mem) = 805306368U; 



}


void _Z20loadReferenceTextureP12MatchContext( struct MatchContext *ctx)
{  float __T22;
 struct Reference *__cuda_local_var_20850_16_non_const_ref;
 int __cuda_local_var_20851_9_non_const_numrows;
 int __cuda_local_var_20852_9_non_const_blocksize;


 struct cudaChannelFormatDesc __cuda_local_var_20855_27_non_const_refTextureDesc;
# 484 "mummergpu.cu"
__cuda_local_var_20850_16_non_const_ref = (ctx->ref);
__cuda_local_var_20851_9_non_const_numrows = ((int)((__T22 = (((float)(__cuda_local_var_20850_16_non_const_ref->len)) / ((float)(__cuda_local_var_20850_16_non_const_ref->pitch)))) , (__builtin_ceilf(__T22))));
__cuda_local_var_20852_9_non_const_blocksize = 4;
__cuda_local_var_20851_9_non_const_numrows += __cuda_local_var_20852_9_non_const_blocksize;

__cuda_local_var_20855_27_non_const_refTextureDesc = (cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSigned));


if (!(ctx->on_cpu)) {
 char *__cuda_local_var_20859_16_non_const_toboardtimer;
# 493 "mummergpu.cu"
__cuda_local_var_20859_16_non_const_toboardtimer = (createTimer());
startTimer(__cuda_local_var_20859_16_non_const_toboardtimer);
# 557 "mummergpu.cu"
do { cudaMalloc(((void **)(&(__cuda_local_var_20850_16_non_const_ref->d_ref_array))), (__cuda_local_var_20850_16_non_const_ref->len)); ++num_bind_tex_calls; } while (0);
do {  enum cudaError __cuda_local_var_20863_32_non_const_err;
# 558 "mummergpu.cu"
cuda_calls++; __cuda_local_var_20863_32_non_const_err = (cudaMemcpy(((void *)(__cuda_local_var_20850_16_non_const_ref->d_ref_array)), ((const void *)(__cuda_local_var_20850_16_non_const_ref->str)), (__cuda_local_var_20850_16_non_const_ref->len), cudaMemcpyHostToDevice)); if (0 != ((int)__cuda_local_var_20863_32_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 561, ((int)__cuda_local_var_20863_32_non_const_err), (cudaGetErrorString(__cuda_local_var_20863_32_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);




((ctx->ref)->bytes_on_board) += (__cuda_local_var_20850_16_non_const_ref->len);


stopTimer(__cuda_local_var_20859_16_non_const_toboardtimer);
((ctx->statistics).t_ref_str_to_board) += (getTimerValue(__cuda_local_var_20859_16_non_const_toboardtimer));
deleteTimer(__cuda_local_var_20859_16_non_const_toboardtimer);
}
else  {
(__cuda_local_var_20850_16_non_const_ref->d_ref_array) = ((void *)0LL);
} 
}


void _Z21unloadReferenceStringP9Reference( struct Reference *ref)
{
# 585 "mummergpu.cu"
do {  enum cudaError __cuda_local_var_20890_31_non_const_err;
# 585 "mummergpu.cu"
cuda_calls++; __cuda_local_var_20890_31_non_const_err = (cudaFree((ref->d_ref_array))); if (0 != ((int)__cuda_local_var_20890_31_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 585, ((int)__cuda_local_var_20890_31_non_const_err), (cudaGetErrorString(__cuda_local_var_20890_31_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);


(ref->d_ref_array) = ((void *)0LL); 
}

void _Z19unloadReferenceTreeP12MatchContext( struct MatchContext *ctx)
{
 struct Reference *__cuda_local_var_20898_15_non_const_ref;
# 593 "mummergpu.cu"
__cuda_local_var_20898_15_non_const_ref = (ctx->ref);
# 622 "mummergpu.cu"
do {  enum cudaError __cuda_local_var_20900_32_non_const_err;
# 622 "mummergpu.cu"
cuda_calls++; __cuda_local_var_20900_32_non_const_err = (cudaFree((__cuda_local_var_20898_15_non_const_ref->d_node_tex_array))); if (0 != ((int)__cuda_local_var_20900_32_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 622, ((int)__cuda_local_var_20900_32_non_const_err), (cudaGetErrorString(__cuda_local_var_20900_32_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);

(__cuda_local_var_20898_15_non_const_ref->d_node_tex_array) = ((void *)0LL);


if (__cuda_local_var_20898_15_non_const_ref->d_children_tex_array)
{




do {  enum cudaError __cuda_local_var_20911_31_non_const_err;
# 633 "mummergpu.cu"
cuda_calls++; __cuda_local_var_20911_31_non_const_err = (cudaFree((__cuda_local_var_20898_15_non_const_ref->d_children_tex_array))); if (0 != ((int)__cuda_local_var_20911_31_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 633, ((int)__cuda_local_var_20911_31_non_const_err), (cudaGetErrorString(__cuda_local_var_20911_31_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
(__cuda_local_var_20898_15_non_const_ref->d_children_tex_array) = ((void *)0LL);
} 
# 647 "mummergpu.cu"
}


void _Z13loadReferenceP12MatchContext( struct MatchContext *ctx) {

 struct Reference *__cuda_local_var_20920_16_non_const_ref;
# 652 "mummergpu.cu"
__cuda_local_var_20920_16_non_const_ref = (ctx->ref);

(__cuda_local_var_20920_16_non_const_ref->bytes_on_board) = 0UL;

_Z20loadReferenceTextureP12MatchContext(ctx);

if (!(ctx->on_cpu)) {
 char *__cuda_local_var_20927_16_non_const_toboardtimer;
# 659 "mummergpu.cu"
__cuda_local_var_20927_16_non_const_toboardtimer = (createTimer());
startTimer(__cuda_local_var_20927_16_non_const_toboardtimer);


(__cuda_local_var_20920_16_non_const_ref->bytes_on_board) += (((unsigned long)((__cuda_local_var_20920_16_non_const_ref->tex_width) * (__cuda_local_var_20920_16_non_const_ref->tex_node_height))) * 16UL);


(__cuda_local_var_20920_16_non_const_ref->bytes_on_board) += (((unsigned long)((__cuda_local_var_20920_16_non_const_ref->tex_width) * (__cuda_local_var_20920_16_non_const_ref->tex_children_height))) * 16UL);
# 777 "mummergpu.cu"
do { cudaMalloc(((void **)(&(__cuda_local_var_20920_16_non_const_ref->d_node_tex_array))), (((unsigned long)(__cuda_local_var_20920_16_non_const_ref->tex_node_height)) * 16UL)); ++num_bind_tex_calls; } while (0);


do {  enum cudaError __cuda_local_var_20939_38_non_const_err;
# 780 "mummergpu.cu"
cuda_calls++; __cuda_local_var_20939_38_non_const_err = (cudaMemcpy((__cuda_local_var_20920_16_non_const_ref->d_node_tex_array), ((const void *)(__cuda_local_var_20920_16_non_const_ref->h_node_tex_array)), (((unsigned long)(__cuda_local_var_20920_16_non_const_ref->tex_node_height)) * 16UL), cudaMemcpyHostToDevice)); if (0 != ((int)__cuda_local_var_20939_38_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 783, ((int)__cuda_local_var_20939_38_non_const_err), (cudaGetErrorString(__cuda_local_var_20939_38_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
# 794 "mummergpu.cu"
if (__cuda_local_var_20920_16_non_const_ref->tex_children_height)
{

do { cudaMalloc(((void **)(&(__cuda_local_var_20920_16_non_const_ref->d_children_tex_array))), (((unsigned long)(__cuda_local_var_20920_16_non_const_ref->tex_children_height)) * 16UL)); ++num_bind_tex_calls; } while (0);


do {  enum cudaError __cuda_local_var_20947_39_non_const_err;
# 800 "mummergpu.cu"
cuda_calls++; __cuda_local_var_20947_39_non_const_err = (cudaMemcpy((__cuda_local_var_20920_16_non_const_ref->d_children_tex_array), ((const void *)(__cuda_local_var_20920_16_non_const_ref->h_children_tex_array)), (((unsigned long)(__cuda_local_var_20920_16_non_const_ref->tex_children_height)) * 16UL), cudaMemcpyHostToDevice)); if (0 != ((int)__cuda_local_var_20947_39_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 803, ((int)__cuda_local_var_20947_39_non_const_err), (cudaGetErrorString(__cuda_local_var_20947_39_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
# 814 "mummergpu.cu"
}
# 892 "mummergpu.cu"
stopTimer(__cuda_local_var_20927_16_non_const_toboardtimer);
((ctx->statistics).t_tree_to_board) += (getTimerValue(__cuda_local_var_20927_16_non_const_toboardtimer));
deleteTimer(__cuda_local_var_20927_16_non_const_toboardtimer);

fprintf(stderr, ((const char *)"done\n"));
}
else  {
(__cuda_local_var_20920_16_non_const_ref->d_node_tex_array) = ((void *)0LL);
(__cuda_local_var_20920_16_non_const_ref->d_children_tex_array) = ((void *)0LL);
} 
}



void _Z18dumpQueryBlockInfoP8QuerySet( struct QuerySet *queries)
{
fprintf(stderr, ((const char *)"\tProcessing queries %s to %s\n"), ((queries->h_names)[0]), ((queries->h_names)[((queries->count) - 1U)])); 


}

void _Z11loadQueriesP12MatchContext( struct MatchContext *ctx)
{
 struct QuerySet *__cuda_local_var_20974_15_non_const_queries;


 unsigned __cuda_local_var_20977_18_non_const_numQueries;
# 915 "mummergpu.cu"
__cuda_local_var_20974_15_non_const_queries = (ctx->queries);
(__cuda_local_var_20974_15_non_const_queries->bytes_on_board) = 0UL;

__cuda_local_var_20977_18_non_const_numQueries = (__cuda_local_var_20974_15_non_const_queries->count);

if (!(ctx->on_cpu)) {


 char *__cuda_local_var_20982_12_non_const_toboardtimer;
# 921 "mummergpu.cu"
fprintf(stderr, ((const char *)"Allocating device memory for queries... "));

__cuda_local_var_20982_12_non_const_toboardtimer = (createTimer());
startTimer(__cuda_local_var_20982_12_non_const_toboardtimer);

_Z18dumpQueryBlockInfoP8QuerySet(__cuda_local_var_20974_15_non_const_queries);
do { cudaMalloc(((void **)(&(__cuda_local_var_20974_15_non_const_queries->d_tex_array))), (__cuda_local_var_20974_15_non_const_queries->texlen)); ++num_bind_tex_calls; } while (0);


(__cuda_local_var_20974_15_non_const_queries->bytes_on_board) += (__cuda_local_var_20974_15_non_const_queries->texlen);

do {  enum cudaError __cuda_local_var_20991_38_non_const_err;
# 932 "mummergpu.cu"
cuda_calls++; __cuda_local_var_20991_38_non_const_err = (cudaMemcpy(((void *)(__cuda_local_var_20974_15_non_const_queries->d_tex_array)), ((const void *)((__cuda_local_var_20974_15_non_const_queries->h_tex_array) + ((__cuda_local_var_20974_15_non_const_queries->h_addrs_tex_array)[0]))), (__cuda_local_var_20974_15_non_const_queries->texlen), cudaMemcpyHostToDevice)); if (0 != ((int)__cuda_local_var_20991_38_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 935, ((int)__cuda_local_var_20991_38_non_const_err), (cudaGetErrorString(__cuda_local_var_20991_38_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
# 946 "mummergpu.cu"
do { cudaMalloc(((void **)(&(__cuda_local_var_20974_15_non_const_queries->d_addrs_tex_array))), (((unsigned long)__cuda_local_var_20977_18_non_const_numQueries) * 4UL)); ++num_bind_tex_calls; } while (0);


(__cuda_local_var_20974_15_non_const_queries->bytes_on_board) += (((unsigned long)__cuda_local_var_20977_18_non_const_numQueries) * 4UL);

do {  enum cudaError __cuda_local_var_20998_38_non_const_err;
# 951 "mummergpu.cu"
cuda_calls++; __cuda_local_var_20998_38_non_const_err = (cudaMemcpy(((void *)(__cuda_local_var_20974_15_non_const_queries->d_addrs_tex_array)), ((const void *)(__cuda_local_var_20974_15_non_const_queries->h_addrs_tex_array)), (((unsigned long)__cuda_local_var_20977_18_non_const_numQueries) * 4UL), cudaMemcpyHostToDevice)); if (0 != ((int)__cuda_local_var_20998_38_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 954, ((int)__cuda_local_var_20998_38_non_const_err), (cudaGetErrorString(__cuda_local_var_20998_38_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);




do { cudaMalloc(((void **)(&(__cuda_local_var_20974_15_non_const_queries->d_lengths_array))), (((unsigned long)__cuda_local_var_20977_18_non_const_numQueries) * 4UL)); ++num_bind_tex_calls; } while (0);


(__cuda_local_var_20974_15_non_const_queries->bytes_on_board) += (((unsigned long)__cuda_local_var_20977_18_non_const_numQueries) * 4UL);

do {  enum cudaError __cuda_local_var_21008_38_non_const_err;
# 961 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21008_38_non_const_err = (cudaMemcpy(((void *)(__cuda_local_var_20974_15_non_const_queries->d_lengths_array)), ((const void *)(__cuda_local_var_20974_15_non_const_queries->h_lengths_array)), (((unsigned long)__cuda_local_var_20977_18_non_const_numQueries) * 4UL), cudaMemcpyHostToDevice)); if (0 != ((int)__cuda_local_var_21008_38_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 964, ((int)__cuda_local_var_21008_38_non_const_err), (cudaGetErrorString(__cuda_local_var_21008_38_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);



stopTimer(__cuda_local_var_20982_12_non_const_toboardtimer);
((ctx->statistics).t_queries_to_board) += (getTimerValue(__cuda_local_var_20982_12_non_const_toboardtimer));
deleteTimer(__cuda_local_var_20982_12_non_const_toboardtimer);

fprintf(stderr, ((const char *)"\tallocated %ld bytes\n"), (__cuda_local_var_20974_15_non_const_queries->bytes_on_board));

}
else  {
(__cuda_local_var_20974_15_non_const_queries->d_addrs_tex_array) = ((int *)0LL);
(__cuda_local_var_20974_15_non_const_queries->d_tex_array) = ((char *)0LL);
(__cuda_local_var_20974_15_non_const_queries->d_lengths_array) = ((int *)0LL);
fprintf(stderr, ((const char *)" allocated %ld bytes\n"), ((((unsigned long)(2U * __cuda_local_var_20977_18_non_const_numQueries)) * 4UL) + (__cuda_local_var_20974_15_non_const_queries->texlen)));
} 


}


void _Z13unloadQueriesP12MatchContext( struct MatchContext *ctx)
{
 struct QuerySet *__cuda_local_var_21032_14_non_const_queries;
# 985 "mummergpu.cu"
__cuda_local_var_21032_14_non_const_queries = (ctx->queries);

do {  enum cudaError __cuda_local_var_21034_33_non_const_err;
# 987 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21034_33_non_const_err = (cudaFree(((void *)(__cuda_local_var_21032_14_non_const_queries->d_tex_array)))); if (0 != ((int)__cuda_local_var_21034_33_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 987, ((int)__cuda_local_var_21034_33_non_const_err), (cudaGetErrorString(__cuda_local_var_21034_33_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
(__cuda_local_var_21032_14_non_const_queries->d_tex_array) = ((char *)0LL);

do {  enum cudaError __cuda_local_var_21037_33_non_const_err;
# 990 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21037_33_non_const_err = (cudaFree(((void *)(__cuda_local_var_21032_14_non_const_queries->d_addrs_tex_array)))); if (0 != ((int)__cuda_local_var_21037_33_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 990, ((int)__cuda_local_var_21037_33_non_const_err), (cudaGetErrorString(__cuda_local_var_21037_33_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
(__cuda_local_var_21032_14_non_const_queries->d_addrs_tex_array) = ((int *)0LL);

do {  enum cudaError __cuda_local_var_21040_33_non_const_err;
# 993 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21040_33_non_const_err = (cudaFree(((void *)(__cuda_local_var_21032_14_non_const_queries->d_lengths_array)))); if (0 != ((int)__cuda_local_var_21040_33_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 993, ((int)__cuda_local_var_21040_33_non_const_err), (cudaGetErrorString(__cuda_local_var_21040_33_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
(__cuda_local_var_21032_14_non_const_queries->d_lengths_array) = ((int *)0LL);

(__cuda_local_var_21032_14_non_const_queries->bytes_on_board) = 0UL; 
}
# 1009 "mummergpu.cu"
void _Z21buildCoordOffsetArrayP12MatchContextPPiPj( struct MatchContext *ctx, 
int **h_coord_offset_array, 
unsigned *num_coords)
{
 int __cuda_local_var_21060_9_non_const_numCoords;
 int __cuda_local_var_21061_9_non_const_match_length;
 int __cuda_local_var_21062_6_non_const_numQueries;
 int *__cuda_local_var_21063_7_non_const_lengths;

 int *__cuda_local_var_21065_10_non_const_coord_offsets;
# 1013 "mummergpu.cu"
__cuda_local_var_21060_9_non_const_numCoords = 0;
__cuda_local_var_21061_9_non_const_match_length = (ctx->min_match_length);
__cuda_local_var_21062_6_non_const_numQueries = ((int)((ctx->queries)->count));
__cuda_local_var_21063_7_non_const_lengths = ((ctx->queries)->h_lengths_array);

__cuda_local_var_21065_10_non_const_coord_offsets = ((int *)(calloc(((size_t)__cuda_local_var_21062_6_non_const_numQueries), 4UL))); {
# 1042 "mummergpu.cu"
 unsigned i;
# 1042 "mummergpu.cu"
i = 0U; for (; (i < ((unsigned)__cuda_local_var_21062_6_non_const_numQueries)); ++i)
{
 int __cuda_local_var_21069_7_non_const_qryoffset;
# 1044 "mummergpu.cu"
__cuda_local_var_21069_7_non_const_qryoffset = (((ctx->queries)->h_addrs_tex_array)[i]);
(__cuda_local_var_21065_10_non_const_coord_offsets[i]) = (__cuda_local_var_21069_7_non_const_qryoffset - (((int)i) * (__cuda_local_var_21061_9_non_const_match_length + 1)));
} }
if (__cuda_local_var_21062_6_non_const_numQueries > 0)
{
 unsigned __cuda_local_var_21074_16_non_const_last_qry;
 unsigned __cuda_local_var_21075_16_non_const_last_qry_len;
# 1049 "mummergpu.cu"
__cuda_local_var_21074_16_non_const_last_qry = ((unsigned)(__cuda_local_var_21062_6_non_const_numQueries - 1));
__cuda_local_var_21075_16_non_const_last_qry_len = ((unsigned)(((__cuda_local_var_21063_7_non_const_lengths[__cuda_local_var_21074_16_non_const_last_qry]) - __cuda_local_var_21061_9_non_const_match_length) + 1));
__cuda_local_var_21060_9_non_const_numCoords = ((int)(((unsigned)(__cuda_local_var_21065_10_non_const_coord_offsets[__cuda_local_var_21074_16_non_const_last_qry])) + __cuda_local_var_21075_16_non_const_last_qry_len));
fprintf(stderr, ((const char *)"Need %d match coords for this result array\n"), __cuda_local_var_21060_9_non_const_numCoords);

}

(*num_coords) = ((unsigned)__cuda_local_var_21060_9_non_const_numCoords);
(*h_coord_offset_array) = __cuda_local_var_21065_10_non_const_coord_offsets; 
}


void _Z16loadResultBufferP12MatchContext( struct MatchContext *ctx)
{ static const char __T23[38] = "void loadResultBuffer(MatchContext *)";
 unsigned __cuda_local_var_21088_18_non_const_numQueries;



 char *__cuda_local_var_21092_11_non_const_offsettimer;
# 1078 "mummergpu.cu"
 unsigned __cuda_local_var_21103_15_non_const_numCoords;



 unsigned __cuda_local_var_21107_18_non_const_boardFreeMemory;
 unsigned __cuda_local_var_21108_18_non_const_total_mem;
# 1063 "mummergpu.cu"
__cuda_local_var_21088_18_non_const_numQueries = ((ctx->queries)->count);

(__cuda_local_var_21088_18_non_const_numQueries) ? ((void)0) : (__assert_fail(((const char *)"numQueries"), ((const char *)"mummergpu.cu"), 1065U, __T23));

__cuda_local_var_21092_11_non_const_offsettimer = (createTimer());
startTimer(__cuda_local_var_21092_11_non_const_offsettimer);

_Z21buildCoordOffsetArrayP12MatchContextPPiPj(ctx, (&((ctx->results).h_coord_tex_array)), (&((ctx->results).numCoords)));



stopTimer(__cuda_local_var_21092_11_non_const_offsettimer);
((ctx->statistics).t_build_coord_offsets) += (getTimerValue(__cuda_local_var_21092_11_non_const_offsettimer));
deleteTimer(__cuda_local_var_21092_11_non_const_offsettimer);

__cuda_local_var_21103_15_non_const_numCoords = ((ctx->results).numCoords);
fprintf(stderr, ((const char *)"Allocating result array for %d queries (%d bytes) ..."), __cuda_local_var_21088_18_non_const_numQueries, (((unsigned long)__cuda_local_var_21103_15_non_const_numCoords) * 8UL));


__cuda_local_var_21107_18_non_const_boardFreeMemory = 0U;
__cuda_local_var_21108_18_non_const_total_mem = 0U;

_Z11boardMemoryPjS_((&__cuda_local_var_21107_18_non_const_boardFreeMemory), (&__cuda_local_var_21108_18_non_const_total_mem));

fprintf(stderr, ((const char *)"board free memory: %u total memory: %u\n"), __cuda_local_var_21107_18_non_const_boardFreeMemory, __cuda_local_var_21108_18_non_const_total_mem);


((ctx->results).h_match_coords) = ((struct MatchCoord *)(calloc(((size_t)__cuda_local_var_21103_15_non_const_numCoords), 8UL)));
if (((ctx->results).h_match_coords) == ((struct MatchCoord *)0LL))
{
_Z8trap_dbgv();
exit(1);
}

if (!(ctx->on_cpu)) {
 char *__cuda_local_var_21123_15_non_const_toboardtimer;
# 1098 "mummergpu.cu"
__cuda_local_var_21123_15_non_const_toboardtimer = (createTimer());
startTimer(__cuda_local_var_21123_15_non_const_toboardtimer);

((ctx->results).bytes_on_board) = 0UL;

do { cudaMalloc(((void **)(&((ctx->results).d_match_coords))), (((unsigned long)__cuda_local_var_21103_15_non_const_numCoords) * 8UL)); ++num_bind_tex_calls; } while (0);

((ctx->results).bytes_on_board) += (((unsigned long)__cuda_local_var_21103_15_non_const_numCoords) * 8UL);

do {  enum cudaError __cuda_local_var_21132_38_non_const_err;
# 1107 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21132_38_non_const_err = (cudaMemset(((void *)((ctx->results).d_match_coords)), 0, (((unsigned long)__cuda_local_var_21103_15_non_const_numCoords) * 8UL))); if (0 != ((int)__cuda_local_var_21132_38_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 1108, ((int)__cuda_local_var_21132_38_non_const_err), (cudaGetErrorString(__cuda_local_var_21132_38_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
# 1121 "mummergpu.cu"
stopTimer(__cuda_local_var_21123_15_non_const_toboardtimer);
((ctx->statistics).t_match_coords_to_board) += (getTimerValue(__cuda_local_var_21123_15_non_const_toboardtimer));
deleteTimer(__cuda_local_var_21123_15_non_const_toboardtimer);
}
else  {
((ctx->results).d_match_coords) = ((struct MatchCoord *)0LL);
}

fprintf(stderr, ((const char *)"done\n")); 
}


void _Z18unloadResultBufferP12MatchContext( struct MatchContext *ctx) {
do {  enum cudaError __cuda_local_var_21147_34_non_const_err;
# 1134 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21147_34_non_const_err = (cudaFree(((void *)((ctx->results).d_match_coords)))); if (0 != ((int)__cuda_local_var_21147_34_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 1134, ((int)__cuda_local_var_21147_34_non_const_err), (cudaGetErrorString(__cuda_local_var_21147_34_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
((ctx->results).d_match_coords) = ((struct MatchCoord *)0LL);
((ctx->results).bytes_on_board) = 0UL; 




}

void _Z25transferResultsFromDeviceP12MatchContext( struct MatchContext *ctx)
{
if (!(ctx->on_cpu))
{
 char *__cuda_local_var_21160_13_non_const_fromboardtimer;
# 1147 "mummergpu.cu"
__cuda_local_var_21160_13_non_const_fromboardtimer = (createTimer());
startTimer(__cuda_local_var_21160_13_non_const_fromboardtimer);

do {  enum cudaError __cuda_local_var_21163_33_non_const_err;
# 1150 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21163_33_non_const_err = (cudaMemcpy(((void *)((ctx->results).h_match_coords)), ((const void *)((ctx->results).d_match_coords)), (((unsigned long)((ctx->results).numCoords)) * 8UL), cudaMemcpyDeviceToHost)); if (0 != ((int)__cuda_local_var_21163_33_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 1153, ((int)__cuda_local_var_21163_33_non_const_err), (cudaGetErrorString(__cuda_local_var_21163_33_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
# 1198 "mummergpu.cu"
stopTimer(__cuda_local_var_21160_13_non_const_fromboardtimer);
((ctx->statistics).t_match_coords_from_board) += (getTimerValue(__cuda_local_var_21160_13_non_const_fromboardtimer));
deleteTimer(__cuda_local_var_21160_13_non_const_fromboardtimer);
} 

}
# 1211 "mummergpu.cu"
struct MatchCoord *_Z17coordForQueryCharP12MatchContextjj( struct MatchContext *ctx, 
unsigned qryid, 
unsigned qrychar)
{
 struct MatchResults *__cuda_local_var_21182_16_non_const_results;
 struct MatchCoord *__cuda_local_var_21183_17_non_const_coords;
# 1215 "mummergpu.cu"
__cuda_local_var_21182_16_non_const_results = (&(ctx->results));
__cuda_local_var_21183_17_non_const_coords = (__cuda_local_var_21182_16_non_const_results->h_match_coords);



return (__cuda_local_var_21183_17_non_const_coords + ((__cuda_local_var_21182_16_non_const_results->h_coord_tex_array)[qryid])) + qrychar;

}

void _Z20coordsToPrintBuffersP12MatchContextP13ReferencePagePP9MatchInfoPP9AlignmentjPjS9_S9_S9_S9_( struct MatchContext *ctx, 
struct ReferencePage *page, 
MatchInfo **matches, 
Alignment **alignments, 
unsigned mem_avail, 
unsigned *coord_idx, 
unsigned *match_idx, 
unsigned *align_idx, 
unsigned *nextqry, 
unsigned *nextqrychar)
{
 unsigned __cuda_local_var_21202_18_non_const_numQueries;
 int __cuda_local_var_21203_9_non_const_match_length;
 unsigned __cuda_local_var_21204_18_non_const_cidx;
 unsigned __cuda_local_var_21205_18_non_const_midx;

 unsigned __cuda_local_var_21207_15_non_const_numCoords;

 unsigned __cuda_local_var_21209_15_non_const_numMatches;
 unsigned __cuda_local_var_21210_15_non_const_numAlignments;

 int __cuda_local_var_21212_9_non_const_DEBUG;
# 1293 "mummergpu.cu"
 MatchInfo *__cuda_local_var_21260_16_non_const_M;
 unsigned __cuda_local_var_21261_18_non_const_alignmentOffset;

 int __cuda_local_var_21263_6_non_const_qry;
 int __cuda_local_var_21264_6_non_const_qrychar;
 char __cuda_local_var_21265_7_non_const_set_full;
# 1235 "mummergpu.cu"
__cuda_local_var_21202_18_non_const_numQueries = ((ctx->queries)->count);
__cuda_local_var_21203_9_non_const_match_length = (ctx->min_match_length);
__cuda_local_var_21204_18_non_const_cidx = (*coord_idx);
__cuda_local_var_21205_18_non_const_midx = 0U;

__cuda_local_var_21207_15_non_const_numCoords = ((ctx->results).numCoords);

__cuda_local_var_21209_15_non_const_numMatches = 0U;
__cuda_local_var_21210_15_non_const_numAlignments = 0U;

__cuda_local_var_21212_9_non_const_DEBUG = 0;
if ((__cuda_local_var_21212_9_non_const_DEBUG) && (__cuda_local_var_21204_18_non_const_cidx == 0U))
{ {
 int j;
# 1248 "mummergpu.cu"
j = 0; for (; (((unsigned)j) < __cuda_local_var_21207_15_non_const_numCoords); ++j)
{
 struct MatchCoord *__cuda_local_var_21217_22_non_const_coord;
# 1250 "mummergpu.cu"
__cuda_local_var_21217_22_non_const_coord = (((ctx->results).h_match_coords) + j);
if ((((__cuda_local_var_21217_22_non_const_coord->node).data) > 0U) && (!((__cuda_local_var_21217_22_non_const_coord->edge_match_length) & 32768)))
{


fprintf(stdout, ((const char *)"node: %d leaves:%d\n"), ((__cuda_local_var_21217_22_non_const_coord->node).data), (lookupNumLeaves(page, (__cuda_local_var_21217_22_non_const_coord->node))));

}
} }
exit(0);
} {



 int j;
# 1264 "mummergpu.cu"
j = ((int)__cuda_local_var_21204_18_non_const_cidx); for (; (((unsigned)j) < __cuda_local_var_21207_15_non_const_numCoords); ++j)
{
 struct MatchCoord *__cuda_local_var_21233_18_non_const_coord;

 int __cuda_local_var_21235_7_non_const_queryAlignments;
 int __cuda_local_var_21236_7_non_const_queryMatches;
# 1277 "mummergpu.cu"
 int __cuda_local_var_21244_7_non_const_allMatches;
 int __cuda_local_var_21245_7_non_const_allAlignments;

 int __cuda_local_var_21247_7_non_const_neededSize;
# 1266 "mummergpu.cu"
__cuda_local_var_21233_18_non_const_coord = (((ctx->results).h_match_coords) + j);

__cuda_local_var_21235_7_non_const_queryAlignments = 0;
__cuda_local_var_21236_7_non_const_queryMatches = 0;

if ((((__cuda_local_var_21233_18_non_const_coord->node).data) > 0U) && (!((__cuda_local_var_21233_18_non_const_coord->edge_match_length) & 32768)))
{
 int __cuda_local_var_21240_14_non_const_numLeaves;
# 1273 "mummergpu.cu"
__cuda_local_var_21240_14_non_const_numLeaves = (lookupNumLeaves(page, (__cuda_local_var_21233_18_non_const_coord->node)));
__cuda_local_var_21235_7_non_const_queryAlignments += __cuda_local_var_21240_14_non_const_numLeaves;
__cuda_local_var_21236_7_non_const_queryMatches++;
}
__cuda_local_var_21244_7_non_const_allMatches = ((int)(__cuda_local_var_21209_15_non_const_numMatches + ((unsigned)__cuda_local_var_21236_7_non_const_queryMatches)));
__cuda_local_var_21245_7_non_const_allAlignments = ((int)(__cuda_local_var_21210_15_non_const_numAlignments + ((unsigned)__cuda_local_var_21235_7_non_const_queryAlignments)));

__cuda_local_var_21247_7_non_const_neededSize = ((int)((((unsigned long)__cuda_local_var_21244_7_non_const_allMatches) * 20UL) + (((unsigned long)__cuda_local_var_21245_7_non_const_allAlignments) * 8UL)));

if ((((unsigned)__cuda_local_var_21247_7_non_const_neededSize) > mem_avail) || ((__cuda_local_var_21244_7_non_const_allMatches / 256) >= 65535))
{

goto __T24;
}

++__cuda_local_var_21204_18_non_const_cidx;
__cuda_local_var_21209_15_non_const_numMatches = ((unsigned)__cuda_local_var_21244_7_non_const_allMatches);
__cuda_local_var_21210_15_non_const_numAlignments = ((unsigned)__cuda_local_var_21245_7_non_const_allAlignments);
} } __T24:;

__cuda_local_var_21260_16_non_const_M = ((MatchInfo *)(calloc(((size_t)__cuda_local_var_21209_15_non_const_numMatches), 20UL)));
__cuda_local_var_21261_18_non_const_alignmentOffset = 0U;

__cuda_local_var_21263_6_non_const_qry = ((int)(*nextqry));
__cuda_local_var_21264_6_non_const_qrychar = ((int)(*nextqrychar));
__cuda_local_var_21265_7_non_const_set_full = ((char)0);
while (((unsigned)__cuda_local_var_21263_6_non_const_qry) < __cuda_local_var_21202_18_non_const_numQueries)
{

 int __cuda_local_var_21269_13_non_const_qlen;
# 1302 "mummergpu.cu"
__cuda_local_var_21269_13_non_const_qlen = (((((ctx->queries)->h_lengths_array)[__cuda_local_var_21263_6_non_const_qry]) + 1) - __cuda_local_var_21203_9_non_const_match_length);

while (__cuda_local_var_21264_6_non_const_qrychar < __cuda_local_var_21269_13_non_const_qlen)
{
# 1312 "mummergpu.cu"
 struct MatchCoord *__cuda_local_var_21279_25_non_const_coord;
# 1306 "mummergpu.cu"
if (__cuda_local_var_21205_18_non_const_midx >= __cuda_local_var_21209_15_non_const_numMatches)
{
__cuda_local_var_21265_7_non_const_set_full = ((char)1);
goto __T25;
}

__cuda_local_var_21279_25_non_const_coord = (_Z17coordForQueryCharP12MatchContextjj(ctx, ((unsigned)__cuda_local_var_21263_6_non_const_qry), ((unsigned)__cuda_local_var_21264_6_non_const_qrychar)));

if ((((__cuda_local_var_21279_25_non_const_coord->node).data) > 0U) && (!((__cuda_local_var_21279_25_non_const_coord->edge_match_length) & 32768)))
{
 MatchInfo __cuda_local_var_21283_27_non_const_m;
(__cuda_local_var_21283_27_non_const_m.resultsoffset) = __cuda_local_var_21261_18_non_const_alignmentOffset;
(__cuda_local_var_21283_27_non_const_m.qrystartpos) = ((unsigned short)__cuda_local_var_21264_6_non_const_qrychar);
(__cuda_local_var_21283_27_non_const_m.matchnode) = (__cuda_local_var_21279_25_non_const_coord->node);
(__cuda_local_var_21283_27_non_const_m.edgematch) = ((unsigned short)(__cuda_local_var_21279_25_non_const_coord->edge_match_length));
(__cuda_local_var_21283_27_non_const_m.numLeaves) = ((unsigned)(lookupNumLeaves(page, (__cuda_local_var_21283_27_non_const_m.matchnode))));
(__cuda_local_var_21283_27_non_const_m.queryid) = ((unsigned)__cuda_local_var_21263_6_non_const_qry);

__cuda_local_var_21261_18_non_const_alignmentOffset += (__cuda_local_var_21283_27_non_const_m.numLeaves);
(__cuda_local_var_21260_16_non_const_M[(__cuda_local_var_21205_18_non_const_midx++)]) = __cuda_local_var_21283_27_non_const_m;
}

++__cuda_local_var_21264_6_non_const_qrychar;
} __T25:;

if (__cuda_local_var_21265_7_non_const_set_full) {
goto __T26; }

++__cuda_local_var_21263_6_non_const_qry;
__cuda_local_var_21264_6_non_const_qrychar = 0;
} __T26:;

(*coord_idx) = __cuda_local_var_21204_18_non_const_cidx;
(*match_idx) = __cuda_local_var_21205_18_non_const_midx;
(*align_idx) = __cuda_local_var_21261_18_non_const_alignmentOffset;
(*matches) = __cuda_local_var_21260_16_non_const_M;
(*nextqry) = ((unsigned)__cuda_local_var_21263_6_non_const_qry);
(*nextqrychar) = ((unsigned)__cuda_local_var_21264_6_non_const_qrychar);
fprintf(stderr, ((const char *)"Allocing %d bytes of host memory for %d alignments\n"), (((unsigned long)__cuda_local_var_21261_18_non_const_alignmentOffset) * 8UL), __cuda_local_var_21210_15_non_const_numAlignments);
(*alignments) = ((struct Alignment *)(calloc(((size_t)__cuda_local_var_21261_18_non_const_alignmentOffset), 8UL))); 

}


void _Z14runPrintKernelP12MatchContextP13ReferencePageP9MatchInfojP9Alignmentj( struct MatchContext *ctx, 
struct ReferencePage *page, 
MatchInfo *h_matches, 
unsigned numMatches, 
Alignment *alignments, 
unsigned numAlignments)
{  unsigned __T27;
 float __T28;
 unsigned __T29;
# 1358 "mummergpu.cu"
 MatchInfo *__cuda_local_var_21325_16_non_const_d_matches;
 size_t __cuda_local_var_21326_12_non_const_matchesSize;


 struct Alignment *__cuda_local_var_21329_24_non_const_d_alignments;
 size_t __cuda_local_var_21330_12_non_const_alignmentSize;



 char *__cuda_local_var_21334_9_non_const_atimer;





 int __cuda_local_var_21340_9_non_const_DEBUG;
# 1393 "mummergpu.cu"
 float __cuda_local_var_21360_8_non_const_mtime;


 int __cuda_local_var_21363_9_non_const_blocksize;

 dim3 __cuda_local_var_21365_10_non_const_dimBlock;
 dim3 __cuda_local_var_21366_10_non_const_dimGrid;
# 1442 "mummergpu.cu"
 cudaError_t __cuda_local_var_21401_17_non_const_err;
# 1459 "mummergpu.cu"
 float __cuda_local_var_21418_8_non_const_atime;
# 1359 "mummergpu.cu"
__cuda_local_var_21326_12_non_const_matchesSize = (((unsigned long)numMatches) * 20UL);
do { cudaMalloc(((void **)(&__cuda_local_var_21325_16_non_const_d_matches)), __cuda_local_var_21326_12_non_const_matchesSize); ++num_bind_tex_calls; } while (0);


__cuda_local_var_21330_12_non_const_alignmentSize = (((unsigned long)numAlignments) * 8UL);
do { cudaMalloc(((void **)(&__cuda_local_var_21329_24_non_const_d_alignments)), __cuda_local_var_21330_12_non_const_alignmentSize); ++num_bind_tex_calls; } while (0);
do {  enum cudaError __cuda_local_var_21332_34_non_const_err;
# 1365 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21332_34_non_const_err = (cudaMemset(((void *)__cuda_local_var_21329_24_non_const_d_alignments), 0, __cuda_local_var_21330_12_non_const_alignmentSize)); if (0 != ((int)__cuda_local_var_21332_34_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 1365, ((int)__cuda_local_var_21332_34_non_const_err), (cudaGetErrorString(__cuda_local_var_21332_34_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);

__cuda_local_var_21334_9_non_const_atimer = (createTimer());
startTimer(__cuda_local_var_21334_9_non_const_atimer);

fprintf(stderr, ((const char *)"prepared %d matches %d alignments\n"), numMatches, numAlignments);
fprintf(stderr, ((const char *)"Copying %d bytes to host memory for %d alignments\n"), (((unsigned long)numAlignments) * 8UL), numAlignments);

__cuda_local_var_21340_9_non_const_DEBUG = 0;
if (__cuda_local_var_21340_9_non_const_DEBUG)
{ {
 int i;
# 1376 "mummergpu.cu"
i = 0; for (; (((unsigned)i) < numMatches); i++)
{
printf(((const char *)"m[%d]:\t%d\t%d\t%d\t%d\t%d\t%d\n"), i, ((h_matches[i]).resultsoffset), ((h_matches[i]).queryid), (((h_matches[i]).matchnode).data), ((h_matches[i]).numLeaves), ((int)((h_matches[i]).edgematch)), ((int)((h_matches[i]).qrystartpos)));
# 1386 "mummergpu.cu"
} }

exit(0);
}

do {  enum cudaError __cuda_local_var_21358_34_non_const_err;
# 1391 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21358_34_non_const_err = (cudaMemcpy(((void *)__cuda_local_var_21325_16_non_const_d_matches), ((const void *)h_matches), __cuda_local_var_21326_12_non_const_matchesSize, cudaMemcpyHostToDevice)); if (0 != ((int)__cuda_local_var_21358_34_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 1391, ((int)__cuda_local_var_21358_34_non_const_err), (cudaGetErrorString(__cuda_local_var_21358_34_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
stopTimer(__cuda_local_var_21334_9_non_const_atimer);
__cuda_local_var_21360_8_non_const_mtime = (getTimerValue(__cuda_local_var_21334_9_non_const_atimer));


__cuda_local_var_21363_9_non_const_blocksize = ((int)((numMatches > 256U) ? 256U : numMatches));

{ __T27 = ((unsigned)__cuda_local_var_21363_9_non_const_blocksize); { (__cuda_local_var_21365_10_non_const_dimBlock.x) = __T27; (__cuda_local_var_21365_10_non_const_dimBlock.y) = 1U; (__cuda_local_var_21365_10_non_const_dimBlock.z) = 1U; } }
{ __T29 = ((unsigned)((__T28 = (((float)numMatches) / (256.0F))) , (__builtin_ceilf(__T28)))); { (__cuda_local_var_21366_10_non_const_dimGrid.x) = __T29; (__cuda_local_var_21366_10_non_const_dimGrid.y) = 1U; (__cuda_local_var_21366_10_non_const_dimGrid.z) = 1U; } }

fprintf(stderr, ((const char *)"  Calling print kernel... "));

(cudaConfigureCall(__cuda_local_var_21366_10_non_const_dimGrid, __cuda_local_var_21365_10_non_const_dimBlock, 0UL, ((cudaStream_t)0LL))) ? ((void)0) : (__device_stub__Z11printKernelP9MatchInfoiP9AlignmentPcP12_PixelOfNodeP16_PixelOfChildrenPKiS9_iiiii(__cuda_local_var_21325_16_non_const_d_matches, ((int)numMatches), __cuda_local_var_21329_24_non_const_d_alignments, ((ctx->queries)->d_tex_array), ((struct _PixelOfNode *)((ctx->ref)->d_node_tex_array)), ((struct _PixelOfChildren *)((ctx->ref)->d_children_tex_array)), ((const int *)((ctx->queries)->d_addrs_tex_array)), ((const int *)((ctx->queries)->d_lengths_array)), (page->begin), (page->end), (page->shadow_left), (page->shadow_right), (ctx->min_match_length)));
# 1438 "mummergpu.cu"
cudaThreadSynchronize();



__cuda_local_var_21401_17_non_const_err = (cudaGetLastError());
if (0 != ((int)__cuda_local_var_21401_17_non_const_err))
{
fprintf(stderr, ((const char *)"Kernel execution failed: %s.\n"), (cudaGetErrorString(__cuda_local_var_21401_17_non_const_err)));

exit(1);
}

startTimer(__cuda_local_var_21334_9_non_const_atimer);

do {  enum cudaError __cuda_local_var_21411_34_non_const_err;
# 1452 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21411_34_non_const_err = (cudaMemcpy(((void *)alignments), ((const void *)((void *)__cuda_local_var_21329_24_non_const_d_alignments)), __cuda_local_var_21330_12_non_const_alignmentSize, cudaMemcpyDeviceToHost)); if (0 != ((int)__cuda_local_var_21411_34_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 1455, ((int)__cuda_local_var_21411_34_non_const_err), (cudaGetErrorString(__cuda_local_var_21411_34_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);



cudaThreadSynchronize();
stopTimer(__cuda_local_var_21334_9_non_const_atimer);

__cuda_local_var_21418_8_non_const_atime = (getTimerValue(__cuda_local_var_21334_9_non_const_atimer));
fprintf(stderr, ((const char *)"memcpy time= %f\n"), ((double)(__cuda_local_var_21418_8_non_const_atime + __cuda_local_var_21360_8_non_const_mtime)));
deleteTimer(__cuda_local_var_21334_9_non_const_atimer);

do {  enum cudaError __cuda_local_var_21422_34_non_const_err;
# 1463 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21422_34_non_const_err = (cudaFree(((void *)__cuda_local_var_21329_24_non_const_d_alignments))); if (0 != ((int)__cuda_local_var_21422_34_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 1463, ((int)__cuda_local_var_21422_34_non_const_err), (cudaGetErrorString(__cuda_local_var_21422_34_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
do {  enum cudaError __cuda_local_var_21423_34_non_const_err;
# 1464 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21423_34_non_const_err = (cudaFree(((void *)__cuda_local_var_21325_16_non_const_d_matches))); if (0 != ((int)__cuda_local_var_21423_34_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 1464, ((int)__cuda_local_var_21423_34_non_const_err), (cudaGetErrorString(__cuda_local_var_21423_34_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0); 
}


void _Z13runPrintOnCPUP12MatchContextP13ReferencePageP9MatchInfojP9Alignmentj( struct MatchContext *ctx,  struct ReferencePage *page, 
MatchInfo *h_matches, 
unsigned numMatches, 
Alignment *alignments, 
unsigned numAlignments)
{
 unsigned __cuda_local_var_21433_15_non_const_min_match_length;

 int *__cuda_local_var_21435_7_non_const_addrs;
 int *__cuda_local_var_21436_7_non_const_lengths;
 char *__cuda_local_var_21437_8_non_const_qrychars;




 int __cuda_local_var_21442_6_non_const_qry;
 unsigned __cuda_local_var_21443_15_non_const_qrylen;
# 1474 "mummergpu.cu"
__cuda_local_var_21433_15_non_const_min_match_length = ((unsigned)(ctx->min_match_length));

__cuda_local_var_21435_7_non_const_addrs = ((ctx->queries)->h_addrs_tex_array);
__cuda_local_var_21436_7_non_const_lengths = ((ctx->queries)->h_lengths_array);
__cuda_local_var_21437_8_non_const_qrychars = ((ctx->queries)->h_tex_array);

if (!(numMatches)) {
return; }

__cuda_local_var_21442_6_non_const_qry = (-1); {


 int i;
# 1486 "mummergpu.cu"
i = 0; for (; (((unsigned)i) < numMatches); ++i)
{
 MatchInfo *__cuda_local_var_21447_20_non_const_match;
# 1488 "mummergpu.cu"
__cuda_local_var_21447_20_non_const_match = (h_matches + i);
if ((__cuda_local_var_21447_20_non_const_match->queryid) != ((unsigned)__cuda_local_var_21442_6_non_const_qry))
{
__cuda_local_var_21442_6_non_const_qry = ((int)(__cuda_local_var_21447_20_non_const_match->queryid));
__cuda_local_var_21443_15_non_const_qrylen = ((unsigned)(__cuda_local_var_21436_7_non_const_lengths[__cuda_local_var_21442_6_non_const_qry]));
}
if (!(((int)(__cuda_local_var_21447_20_non_const_match->edgematch)) & 32768))
{
_Z15printAlignmentsP13ReferencePageP9AlignmentPci14TextureAddressiiibb(page, (alignments + (__cuda_local_var_21447_20_non_const_match->resultsoffset)), (__cuda_local_var_21437_8_non_const_qrychars + (__cuda_local_var_21435_7_non_const_addrs[__cuda_local_var_21442_6_non_const_qry])), ((int)__cuda_local_var_21443_15_non_const_qrylen), (__cuda_local_var_21447_20_non_const_match->matchnode), ((int)(__cuda_local_var_21447_20_non_const_match->qrystartpos)), ((int)(__cuda_local_var_21447_20_non_const_match->edgematch)), ((int)__cuda_local_var_21433_15_non_const_min_match_length), ((char)0), (ctx->forwardcoordinates));
# 1510 "mummergpu.cu"
}
} } 
}



void _Z18getExactAlignmentsP12MatchContextP13ReferencePageb( struct MatchContext *ctx,  struct ReferencePage *page,  char on_cpu)
{ static const char __T210[68] = "void getExactAlignments(MatchContext *, ReferencePage *, __nv_bool)";


 unsigned __cuda_local_var_21479_18_non_const_boardFreeMemory;
 unsigned __cuda_local_var_21480_18_non_const_total_mem;
# 1542 "mummergpu.cu"
 int __cuda_local_var_21501_9_non_const_rTotalMatches;
 int __cuda_local_var_21502_9_non_const_rTotalAlignments;
 int __cuda_local_var_21503_9_non_const_totalRounds;
 unsigned __cuda_local_var_21504_15_non_const_last_coord;
 unsigned __cuda_local_var_21505_15_non_const_next_coord;
 unsigned __cuda_local_var_21506_15_non_const_nextqry;
 unsigned __cuda_local_var_21507_15_non_const_nextqrychar;
 int __cuda_local_var_21508_6_non_const_lastqry;
# 1518 "mummergpu.cu"
((!(ctx->reverse)) && (!(ctx->forwardreverse))) ? ((void)0) : (__assert_fail(((const char *)"!ctx->reverse && !ctx->forwardreverse"), ((const char *)"mummergpu.cu"), 1518U, __T210));




if (!(on_cpu))
{
_Z11boardMemoryPjS_((&__cuda_local_var_21479_18_non_const_boardFreeMemory), (&__cuda_local_var_21480_18_non_const_total_mem));
fprintf(stderr, ((const char *)"board free memory: %u total memory: %u\n"), __cuda_local_var_21479_18_non_const_boardFreeMemory, __cuda_local_var_21480_18_non_const_total_mem);

}

else  {
__cuda_local_var_21479_18_non_const_boardFreeMemory = 268435456U;
__cuda_local_var_21480_18_non_const_total_mem = __cuda_local_var_21479_18_non_const_boardFreeMemory;
}
# 1539 "mummergpu.cu"
__cuda_local_var_21479_18_non_const_boardFreeMemory -= 16777216U;
fprintf(stderr, ((const char *)"board free memory: %u\n"), __cuda_local_var_21479_18_non_const_boardFreeMemory);

__cuda_local_var_21501_9_non_const_rTotalMatches = 0;
__cuda_local_var_21502_9_non_const_rTotalAlignments = 0;
__cuda_local_var_21503_9_non_const_totalRounds = 0;
__cuda_local_var_21504_15_non_const_last_coord = ((ctx->results).numCoords);
__cuda_local_var_21505_15_non_const_next_coord = 0U;
__cuda_local_var_21506_15_non_const_nextqry = 0U;
__cuda_local_var_21507_15_non_const_nextqrychar = 0U;
__cuda_local_var_21508_6_non_const_lastqry = (-1);
while (__cuda_local_var_21505_15_non_const_next_coord < __cuda_local_var_21504_15_non_const_last_coord) {
{



 unsigned __cuda_local_var_21514_22_non_const_numMatches;
 unsigned __cuda_local_var_21515_22_non_const_numAlignments;
 MatchInfo *__cuda_local_var_21516_14_non_const_h_matches;
 Alignment *__cuda_local_var_21517_14_non_const_h_alignments;
 int __cuda_local_var_21518_7_non_const_coord_left;
 char *__cuda_local_var_21519_9_non_const_btimer;





 float __cuda_local_var_21525_9_non_const_btime;
# 1577 "mummergpu.cu"
 char __cuda_local_var_21536_14_non_const_buf[256];
# 1591 "mummergpu.cu"
 char *__cuda_local_var_21550_15_non_const_ktimer;
# 1605 "mummergpu.cu"
 float __cuda_local_var_21564_12_non_const_ktime;
# 1628 "mummergpu.cu"
 char *__cuda_local_var_21569_9_non_const_otimer;
# 1553 "mummergpu.cu"
__cuda_local_var_21503_9_non_const_totalRounds++;

__cuda_local_var_21514_22_non_const_numMatches = 0U;
__cuda_local_var_21515_22_non_const_numAlignments = 0U;
__cuda_local_var_21516_14_non_const_h_matches = ((MatchInfo *)0LL);
__cuda_local_var_21517_14_non_const_h_alignments = ((Alignment *)0LL);
__cuda_local_var_21518_7_non_const_coord_left = ((int)__cuda_local_var_21505_15_non_const_next_coord);
__cuda_local_var_21519_9_non_const_btimer = (createTimer());
startTimer(__cuda_local_var_21519_9_non_const_btimer);
_Z20coordsToPrintBuffersP12MatchContextP13ReferencePagePP9MatchInfoPP9AlignmentjPjS9_S9_S9_S9_(ctx, page, (&__cuda_local_var_21516_14_non_const_h_matches), (&__cuda_local_var_21517_14_non_const_h_alignments), __cuda_local_var_21479_18_non_const_boardFreeMemory, (&__cuda_local_var_21505_15_non_const_next_coord), (&__cuda_local_var_21514_22_non_const_numMatches), (&__cuda_local_var_21515_22_non_const_numAlignments), (&__cuda_local_var_21506_15_non_const_nextqry), (&__cuda_local_var_21507_15_non_const_nextqrychar));

stopTimer(__cuda_local_var_21519_9_non_const_btimer);

__cuda_local_var_21525_9_non_const_btime = (getTimerValue(__cuda_local_var_21519_9_non_const_btimer));
((ctx->statistics).t_coords_to_buffers) += __cuda_local_var_21525_9_non_const_btime;
fprintf(stderr, ((const char *)"buffer prep time= %f\n"), ((double)__cuda_local_var_21525_9_non_const_btime));
deleteTimer(__cuda_local_var_21519_9_non_const_btimer);

fprintf(stderr, ((const char *)"Round %d: Printing results for match coords [%d-%d) of %d using %d matches and %d alignments\n"), __cuda_local_var_21503_9_non_const_totalRounds, __cuda_local_var_21518_7_non_const_coord_left, __cuda_local_var_21505_15_non_const_next_coord, __cuda_local_var_21504_15_non_const_last_coord, __cuda_local_var_21514_22_non_const_numMatches, __cuda_local_var_21515_22_non_const_numAlignments);


if (__cuda_local_var_21514_22_non_const_numMatches == 0U) {
goto __T211; }




__cuda_local_var_21502_9_non_const_rTotalAlignments += __cuda_local_var_21515_22_non_const_numAlignments;
__cuda_local_var_21501_9_non_const_rTotalMatches += __cuda_local_var_21514_22_non_const_numMatches;

if (num_bind_tex_calls > 100U)
{
cudaThreadExit();
num_bind_tex_calls = 0U;
_Z13loadReferenceP12MatchContext(ctx);
_Z11loadQueriesP12MatchContext(ctx);
}

__cuda_local_var_21550_15_non_const_ktimer = (createTimer());
startTimer(__cuda_local_var_21550_15_non_const_ktimer);
if (on_cpu)
{
_Z13runPrintOnCPUP12MatchContextP13ReferencePageP9MatchInfojP9Alignmentj(ctx, page, __cuda_local_var_21516_14_non_const_h_matches, __cuda_local_var_21514_22_non_const_numMatches, __cuda_local_var_21517_14_non_const_h_alignments, __cuda_local_var_21515_22_non_const_numAlignments);

}

else  {
_Z14runPrintKernelP12MatchContextP13ReferencePageP9MatchInfojP9Alignmentj(ctx, page, __cuda_local_var_21516_14_non_const_h_matches, __cuda_local_var_21514_22_non_const_numMatches, __cuda_local_var_21517_14_non_const_h_alignments, __cuda_local_var_21515_22_non_const_numAlignments);

}
stopTimer(__cuda_local_var_21550_15_non_const_ktimer);

__cuda_local_var_21564_12_non_const_ktime = (getTimerValue(__cuda_local_var_21550_15_non_const_ktimer));
((ctx->statistics).t_print_kernel) += __cuda_local_var_21564_12_non_const_ktime;
fprintf(stderr, ((const char *)"print kernel time= %f\n"), ((double)__cuda_local_var_21564_12_non_const_ktime));
deleteTimer(__cuda_local_var_21550_15_non_const_ktimer);
# 1628 "mummergpu.cu"
__cuda_local_var_21569_9_non_const_otimer = (createTimer());
startTimer(__cuda_local_var_21569_9_non_const_otimer); {

 int m;
# 1631 "mummergpu.cu"
m = 0; for (; (((unsigned)m) < __cuda_local_var_21514_22_non_const_numMatches); m++)
{
 int __cuda_local_var_21574_17_non_const_base;
# 1633 "mummergpu.cu"
__cuda_local_var_21574_17_non_const_base = ((int)((__cuda_local_var_21516_14_non_const_h_matches[m]).resultsoffset)); {
 int i;
# 1634 "mummergpu.cu"
i = 0; for (; (((unsigned)i) < ((__cuda_local_var_21516_14_non_const_h_matches[m]).numLeaves)); i++)
{

if (((__cuda_local_var_21517_14_non_const_h_alignments[(__cuda_local_var_21574_17_non_const_base + i)]).left_in_ref) == 0)
{
goto __T212;
}

if (((__cuda_local_var_21516_14_non_const_h_matches[m]).queryid) != ((unsigned)__cuda_local_var_21508_6_non_const_lastqry))
{
__cuda_local_var_21508_6_non_const_lastqry = ((int)((__cuda_local_var_21516_14_non_const_h_matches[m]).queryid));
_Z11addToBufferPc("> ");
_Z11addToBufferPc((*(((ctx->queries)->h_names) + __cuda_local_var_21508_6_non_const_lastqry)));
_Z11addToBufferPc("\n");
}

sprintf((__cuda_local_var_21536_14_non_const_buf), ((const char *)"%d\t%d\t%d\n"), ((__cuda_local_var_21517_14_non_const_h_alignments[(__cuda_local_var_21574_17_non_const_base + i)]).left_in_ref), (((int)((__cuda_local_var_21516_14_non_const_h_matches[m]).qrystartpos)) + 1), ((int)((__cuda_local_var_21517_14_non_const_h_alignments[(__cuda_local_var_21574_17_non_const_base + i)]).matchlen)));



_Z11addToBufferPc((__cuda_local_var_21536_14_non_const_buf));
# 1660 "mummergpu.cu"
} } __T212:;
} }


_Z11flushOutputv();

stopTimer(__cuda_local_var_21569_9_non_const_otimer);
((ctx->statistics).t_results_to_disk) += (getTimerValue(__cuda_local_var_21569_9_non_const_otimer));
deleteTimer(__cuda_local_var_21569_9_non_const_otimer);

free(((void *)__cuda_local_var_21516_14_non_const_h_matches));
free(((void *)__cuda_local_var_21517_14_non_const_h_alignments));


} __T211:; }
free(((void *)((ctx->results).h_coord_tex_array)));
free(((void *)((ctx->results).h_match_coords)));
((ctx->results).h_coord_tex_array) = ((int *)0LL);
((ctx->results).h_match_coords) = ((struct MatchCoord *)0LL);

fprintf(stderr, ((const char *)"Finished processing %d matches and %d potential alignments in %d rounds\n"), __cuda_local_var_21501_9_non_const_rTotalMatches, __cuda_local_var_21502_9_non_const_rTotalAlignments, __cuda_local_var_21503_9_non_const_totalRounds); 

}

int _Z13getQueryBlockP12MatchContextm( struct MatchContext *ctx,  size_t device_mem_avail)
{
 struct QuerySet *__cuda_local_var_21627_15_non_const_queries;
 char *__cuda_local_var_21628_12_non_const_queryTex;
 int *__cuda_local_var_21629_10_non_const_queryAddrs;
 int *__cuda_local_var_21630_10_non_const_queryLengths;
 unsigned __cuda_local_var_21631_18_non_const_numQueries;
 unsigned __cuda_local_var_21632_15_non_const_num_match_coords;
 size_t __cuda_local_var_21633_12_non_const_queryLen;
 char **__cuda_local_var_21634_12_non_const_names;



 char *__cuda_local_var_21638_11_non_const_queryreadtimer;
# 1686 "mummergpu.cu"
__cuda_local_var_21627_15_non_const_queries = (ctx->queries);
__cuda_local_var_21628_12_non_const_queryTex = ((char *)0LL);
__cuda_local_var_21629_10_non_const_queryAddrs = ((int *)0LL);
__cuda_local_var_21630_10_non_const_queryLengths = ((int *)0LL);
# 1695 "mummergpu.cu"
fprintf(stderr, ((const char *)"Loading query block... "));

__cuda_local_var_21638_11_non_const_queryreadtimer = (createTimer());
startTimer(__cuda_local_var_21638_11_non_const_queryreadtimer);

getQueriesTexture((__cuda_local_var_21627_15_non_const_queries->qfile), (&__cuda_local_var_21628_12_non_const_queryTex), (&__cuda_local_var_21633_12_non_const_queryLen), (&__cuda_local_var_21629_10_non_const_queryAddrs), (&__cuda_local_var_21634_12_non_const_names), (&__cuda_local_var_21630_10_non_const_queryLengths), (&__cuda_local_var_21631_18_non_const_numQueries), (&__cuda_local_var_21632_15_non_const_num_match_coords), ((unsigned)device_mem_avail), (ctx->min_match_length), ((char)((ctx->reverse) || (ctx->forwardreverse))));
# 1712 "mummergpu.cu"
stopTimer(__cuda_local_var_21638_11_non_const_queryreadtimer);
((ctx->statistics).t_queries_from_disk) += (getTimerValue(__cuda_local_var_21638_11_non_const_queryreadtimer));
deleteTimer(__cuda_local_var_21638_11_non_const_queryreadtimer);

(__cuda_local_var_21627_15_non_const_queries->h_tex_array) = __cuda_local_var_21628_12_non_const_queryTex;
(__cuda_local_var_21627_15_non_const_queries->count) = __cuda_local_var_21631_18_non_const_numQueries;
(__cuda_local_var_21627_15_non_const_queries->h_addrs_tex_array) = __cuda_local_var_21629_10_non_const_queryAddrs;
(__cuda_local_var_21627_15_non_const_queries->texlen) = __cuda_local_var_21633_12_non_const_queryLen;
(__cuda_local_var_21627_15_non_const_queries->h_names) = __cuda_local_var_21634_12_non_const_names;
(__cuda_local_var_21627_15_non_const_queries->h_lengths_array) = __cuda_local_var_21630_10_non_const_queryLengths;

((ctx->results).numCoords) = __cuda_local_var_21632_15_non_const_num_match_coords;

fprintf(stderr, ((const char *)"done.\n"));

return (int)__cuda_local_var_21631_18_non_const_numQueries;
}

void _Z17destroyQueryBlockP8QuerySet( struct QuerySet *queries)
{
free(((void *)(queries->h_tex_array)));
(queries->h_tex_array) = ((char *)0LL); {

 int i;
# 1735 "mummergpu.cu"
i = 0; for (; (((unsigned)i) < (queries->count)); ++i) {
free(((void *)((queries->h_names)[i]))); } }

free(((void *)(queries->h_names)));

(queries->count) = 0U;
(queries->texlen) = 0UL;

free(((void *)(queries->h_addrs_tex_array)));
(queries->h_addrs_tex_array) = ((int *)0LL);

free(((void *)(queries->h_lengths_array)));
(queries->h_lengths_array) = ((int *)0LL); 
}

void _Z10resetStatsP10Statistics( struct Statistics *stats)
{
(stats->t_end_to_end) = (0.0F);
(stats->t_match_kernel) = (0.0F);
(stats->t_print_kernel) = (0.0F);
(stats->t_queries_to_board) = (0.0F);
(stats->t_match_coords_to_board) = (0.0F);
(stats->t_match_coords_from_board) = (0.0F);
(stats->t_tree_to_board) = (0.0F);
(stats->t_ref_str_to_board) = (0.0F);
(stats->t_queries_from_disk) = (0.0F);
(stats->t_ref_from_disk) = (0.0F);
(stats->t_results_to_disk) = (0.0F);
(stats->t_tree_construction) = (0.0F);
(stats->t_tree_reorder) = (0.0F);
(stats->t_tree_flatten) = (0.0F);
(stats->t_reorder_ref_str) = (0.0F);
(stats->t_build_coord_offsets) = (0.0F);
(stats->t_coords_to_buffers) = (0.0F);
(stats->bp_avg_query_length) = (0.0F); 
# 1786 "mummergpu.cu"
}

void _Z19writeStatisticsFileP10StatisticsPcS1_S1_( struct Statistics *stats, 
char *stats_filename, 
char *node_hist_filename, 
char *child_hist_filename)
{
if (stats_filename)
{
 FILE *__cuda_local_var_21721_12_non_const_f;
# 1795 "mummergpu.cu"
__cuda_local_var_21721_12_non_const_f = (fopen(((const char *)stats_filename), ((const char *)"w")));

if (!(__cuda_local_var_21721_12_non_const_f))
{
fprintf(stderr, ((const char *)"WARNING: could not open %s for writing\n"), stats_filename);
}

else  {
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)"Q"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",R"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",T"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",m"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",r"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",t"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",n"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Total"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Match kernel"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Print Kernel"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Queries to board"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Match coords to board"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Match coords from board"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Tree to board"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Ref str to board"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Queries from disk"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Ref from disk"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Output to disk"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Tree construction"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Tree reorder"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Tree flatten"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Ref reorder"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Build coord table"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Coords to buffers"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",Avg qry length"));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)"\n"));

fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)"%d"), 0);
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%d"), 0);
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%d"), 0);
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%d"), 0);
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%d"), 0);
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%d"), 0);
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%d"), 0);
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_end_to_end)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_match_kernel)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_print_kernel)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_queries_to_board)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_match_coords_to_board)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_match_coords_from_board)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_tree_to_board)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_ref_str_to_board)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_queries_from_disk)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_ref_from_disk)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_results_to_disk)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_tree_construction)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_tree_reorder)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_tree_flatten)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_reorder_ref_str)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_build_coord_offsets)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->t_coords_to_buffers)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)",%f"), ((double)(stats->bp_avg_query_length)));
fprintf(__cuda_local_var_21721_12_non_const_f, ((const char *)"\n"));

fclose(__cuda_local_var_21721_12_non_const_f);
}
} 
# 1912 "mummergpu.cu"
}

void _Z10matchOnCPUP12MatchContextb( struct MatchContext *ctx,  char doRC)
{

fprintf(stderr, ((const char *)"WE DON\'T SUPPORT CHECKING ON CPU IN THE SIMULATOR!!!\n"));
exit(1); 
# 1944 "mummergpu.cu"
}

void _Z10matchOnGPUP12MatchContextb( struct MatchContext *ctx,  char doRC)
{  unsigned __T213;
 float __T214;
 unsigned __T215;
# 1948 "mummergpu.cu"
 int __cuda_local_var_21799_6_non_const_numQueries;
 int __cuda_local_var_21800_6_non_const_blocksize;

 dim3 __cuda_local_var_21802_7_non_const_dimBlock;

 dim3 __cuda_local_var_21804_7_non_const_dimGrid;
# 2001 "mummergpu.cu"
 cudaError_t __cuda_local_var_21845_14_non_const_err;
# 1948 "mummergpu.cu"
__cuda_local_var_21799_6_non_const_numQueries = ((int)((ctx->queries)->count));
__cuda_local_var_21800_6_non_const_blocksize = ((__cuda_local_var_21799_6_non_const_numQueries > 256) ? 256 : __cuda_local_var_21799_6_non_const_numQueries);

{ __T213 = ((unsigned)__cuda_local_var_21800_6_non_const_blocksize); { (__cuda_local_var_21802_7_non_const_dimBlock.x) = __T213; (__cuda_local_var_21802_7_non_const_dimBlock.y) = 1U; (__cuda_local_var_21802_7_non_const_dimBlock.z) = 1U; } }

{ __T215 = ((unsigned)((__T214 = (((float)__cuda_local_var_21799_6_non_const_numQueries) / (256.0F))) , (__builtin_ceilf(__T214)))); { (__cuda_local_var_21804_7_non_const_dimGrid.x) = __T215; (__cuda_local_var_21804_7_non_const_dimGrid.y) = 1U; (__cuda_local_var_21804_7_non_const_dimGrid.z) = 1U; } }


if (doRC) {

(cudaConfigureCall(__cuda_local_var_21804_7_non_const_dimGrid, __cuda_local_var_21802_7_non_const_dimBlock, 0UL, ((cudaStream_t)0LL))) ? ((void)0) : (__device_stub__Z17mummergpuRCKernelP10MatchCoordPcPKiS3_ii(((ctx->results).d_match_coords), ((ctx->queries)->d_tex_array), ((const int *)((ctx->queries)->d_addrs_tex_array)), ((const int *)((ctx->queries)->d_lengths_array)), __cuda_local_var_21799_6_non_const_numQueries, (ctx->min_match_length)));
# 1964 "mummergpu.cu"
}
else  {
(cudaConfigureCall(__cuda_local_var_21804_7_non_const_dimGrid, __cuda_local_var_21802_7_non_const_dimBlock, 0UL, ((cudaStream_t)0LL))) ? ((void)0) : (__device_stub__Z15mummergpuKernelPvPcP12_PixelOfNodeP16_PixelOfChildrenS0_PKiS6_ii(((void *)((ctx->results).d_match_coords)), ((ctx->queries)->d_tex_array), ((struct _PixelOfNode *)((ctx->ref)->d_node_tex_array)), ((struct _PixelOfChildren *)((ctx->ref)->d_children_tex_array)), ((char *)((ctx->ref)->d_ref_array)), ((const int *)((ctx->queries)->d_addrs_tex_array)), ((const int *)((ctx->queries)->d_lengths_array)), __cuda_local_var_21799_6_non_const_numQueries, (ctx->min_match_length)));
# 1998 "mummergpu.cu"
}


__cuda_local_var_21845_14_non_const_err = (cudaGetLastError());
if (0 != ((int)__cuda_local_var_21845_14_non_const_err)) {
fprintf(stderr, ((const char *)"Kernel execution failed: %s.\n"), (cudaGetErrorString(__cuda_local_var_21845_14_non_const_err)));

exit(1);
} 
}

void _Z15getMatchResultsP12MatchContextj( struct MatchContext *ctx, 
unsigned page_num)
{
_Z25transferResultsFromDeviceP12MatchContext(ctx); 
}

void _Z30matchQueryBlockToReferencePageP12MatchContextP13ReferencePageb( struct MatchContext *ctx, 
struct ReferencePage *page, 
char reverse_complement)
{
 char *__cuda_local_var_21863_8_non_const_ktimer;
# 2040 "mummergpu.cu"
 float __cuda_local_var_21884_8_non_const_ktime;
# 2019 "mummergpu.cu"
__cuda_local_var_21863_8_non_const_ktimer = (createTimer());

fprintf(stderr, ((const char *)"Memory footprint is:\n\tqueries: %d\n\tref: %d\n\tresults: %d\n"), ((ctx->queries)->bytes_on_board), ((ctx->ref)->bytes_on_board), ((ctx->results).bytes_on_board));




startTimer(__cuda_local_var_21863_8_non_const_ktimer);
if (ctx->on_cpu)
{
_Z10matchOnCPUP12MatchContextb(ctx, reverse_complement);
}

else  {

_Z10matchOnGPUP12MatchContextb(ctx, reverse_complement);
cudaThreadSynchronize();

}
stopTimer(__cuda_local_var_21863_8_non_const_ktimer);

__cuda_local_var_21884_8_non_const_ktime = (getTimerValue(__cuda_local_var_21863_8_non_const_ktimer));
((ctx->statistics).t_match_kernel) += __cuda_local_var_21884_8_non_const_ktime;
fprintf(stderr, ((const char *)"match kernel time= %f\n"), ((double)__cuda_local_var_21884_8_non_const_ktime));
deleteTimer(__cuda_local_var_21863_8_non_const_ktimer);

_Z15getMatchResultsP12MatchContextj(ctx, (page->id));
_Z18unloadResultBufferP12MatchContext(ctx); 
}


int _Z11matchSubsetP12MatchContextP13ReferencePage( struct MatchContext *ctx, 
struct ReferencePage *page)
{

_Z11loadQueriesP12MatchContext(ctx);

fprintf(stderr, ((const char *)"Matching queries %s - %s against ref coords %d - %d\n"), (((ctx->queries)->h_names)[0]), (((ctx->queries)->h_names)[(((ctx->queries)->count) - 1U)]), (page->begin), (page->end));
# 2063 "mummergpu.cu"
_Z16loadResultBufferP12MatchContext(ctx);



_Z30matchQueryBlockToReferencePageP12MatchContextP13ReferencePageb(ctx, page, ((char)0));

if ((USE_PRINT_KERNEL) && (!(ctx->on_cpu)))
{
_Z18getExactAlignmentsP12MatchContextP13ReferencePageb(ctx, page, ((char)0));
}


else  {
_Z18getExactAlignmentsP12MatchContextP13ReferencePageb(ctx, page, ((char)1));
}

_Z11flushOutputv();
_Z13unloadQueriesP12MatchContext(ctx);
return 0;
}

int _Z19getFreeDeviceMemoryb( char on_cpu)
{
 unsigned __cuda_local_var_21930_15_non_const_free_mem;
 unsigned __cuda_local_var_21931_15_non_const_total_mem;



 int *__cuda_local_var_21935_8_non_const_p;
# 2086 "mummergpu.cu"
__cuda_local_var_21930_15_non_const_free_mem = 0U;
__cuda_local_var_21931_15_non_const_total_mem = 0U;



__cuda_local_var_21935_8_non_const_p = ((int *)0LL);
do {  enum cudaError __cuda_local_var_21936_31_non_const_err;
# 2092 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21936_31_non_const_err = (cudaMalloc(((void **)(&__cuda_local_var_21935_8_non_const_p)), 4UL)); if (0 != ((int)__cuda_local_var_21936_31_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 2092, ((int)__cuda_local_var_21936_31_non_const_err), (cudaGetErrorString(__cuda_local_var_21936_31_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
do {  enum cudaError __cuda_local_var_21937_31_non_const_err;
# 2093 "mummergpu.cu"
cuda_calls++; __cuda_local_var_21937_31_non_const_err = (cudaFree(((void *)__cuda_local_var_21935_8_non_const_p))); if (0 != ((int)__cuda_local_var_21937_31_non_const_err)) { fprintf(stderr, ((const char *)"Cuda error in file \'%s\' in line %i : %d (%s).\n"), ((const char *)("mummergpu.cu")), 2093, ((int)__cuda_local_var_21937_31_non_const_err), (cudaGetErrorString(__cuda_local_var_21937_31_non_const_err))); _Z8trap_dbgv(); exit(1); } } while (0);
if (!(on_cpu)) {

_Z11boardMemoryPjS_((&__cuda_local_var_21930_15_non_const_free_mem), (&__cuda_local_var_21931_15_non_const_total_mem));
fprintf(stderr, ((const char *)"board free memory: %u total memory: %u\n"), __cuda_local_var_21930_15_non_const_free_mem, __cuda_local_var_21931_15_non_const_total_mem);

}
else  {
__cuda_local_var_21931_15_non_const_total_mem = (__cuda_local_var_21930_15_non_const_free_mem = 804585472U);
}

return (int)__cuda_local_var_21930_15_non_const_free_mem;
}

int _Z27matchQueriesToReferencePageP12MatchContextP13ReferencePage( struct MatchContext *ctx,  struct ReferencePage *page)
{


 int __cuda_local_var_21955_6_non_const_free_mem;

 int __cuda_local_var_21957_6_non_const_available_mem;
# 2109 "mummergpu.cu"
fprintf(stderr, ((const char *)"Beginning reference page %p\n"), page);

__cuda_local_var_21955_6_non_const_free_mem = (_Z19getFreeDeviceMemoryb((ctx->on_cpu)));

__cuda_local_var_21957_6_non_const_available_mem = ((int)((((unsigned long)__cuda_local_var_21955_6_non_const_free_mem) - ((page->ref).bytes_on_board)) - 16777216UL));
(ctx->ref) = (&(page->ref));
_Z13loadReferenceP12MatchContext(ctx);

while (_Z13getQueryBlockP12MatchContextm(ctx, ((size_t)__cuda_local_var_21957_6_non_const_available_mem))) {
_Z11matchSubsetP12MatchContextP13ReferencePage(ctx, page);
((ctx->statistics).bp_avg_query_length) = ((((float)((ctx->queries)->texlen)) / ((float)((ctx->queries)->count))) - (2.0F));

_Z17destroyQueryBlockP8QuerySet((ctx->queries));
if (num_bind_tex_calls > 100U)
{
cudaThreadExit();
num_bind_tex_calls = 0U;
_Z13loadReferenceP12MatchContext(ctx);
}
}

_Z21unloadReferenceStringP9Reference((ctx->ref));
_Z19unloadReferenceTreeP12MatchContext(ctx);
lseek(((ctx->queries)->qfile), 0L, 0);
return 0;
}



void _Z18initReferencePagesP12MatchContextPiPP13ReferencePage( struct MatchContext *ctx,  int *num_pages,  struct ReferencePage **pages_out) {
 unsigned __cuda_local_var_21983_18_non_const_bases_in_ref;
 unsigned __cuda_local_var_21984_18_non_const_page_size;

 unsigned __cuda_local_var_21986_18_non_const_num_reference_pages;



 unsigned __cuda_local_var_21990_18_non_const_page_overlap;
 struct ReferencePage *__cuda_local_var_21991_20_non_const_pages;
# 2139 "mummergpu.cu"
__cuda_local_var_21983_18_non_const_bases_in_ref = ((unsigned)((ctx->full_ref_len) - 3UL));
__cuda_local_var_21984_18_non_const_page_size = ((8388608U < __cuda_local_var_21983_18_non_const_bases_in_ref) ? 8388608U : __cuda_local_var_21983_18_non_const_bases_in_ref);

__cuda_local_var_21986_18_non_const_num_reference_pages = ((unsigned)(ceil(((((double)__cuda_local_var_21983_18_non_const_bases_in_ref) + (0.0)) / ((double)__cuda_local_var_21984_18_non_const_page_size)))));
fprintf(stderr, ((const char *)"Stream will use %d pages for %d bases, page size = %d\n"), __cuda_local_var_21986_18_non_const_num_reference_pages, __cuda_local_var_21983_18_non_const_bases_in_ref, __cuda_local_var_21984_18_non_const_page_size);


__cuda_local_var_21990_18_non_const_page_overlap = 8193U;
__cuda_local_var_21991_20_non_const_pages = ((struct ReferencePage *)(calloc(((size_t)__cuda_local_var_21986_18_non_const_num_reference_pages), 192UL)));


((__cuda_local_var_21991_20_non_const_pages[0]).begin) = 1;
((__cuda_local_var_21991_20_non_const_pages[0]).end) = ((int)((((double)(((unsigned)((__cuda_local_var_21991_20_non_const_pages[0]).begin)) + __cuda_local_var_21984_18_non_const_page_size)) + (ceil((((double)__cuda_local_var_21990_18_non_const_page_overlap) / (2.0))))) + (1.0)));


((__cuda_local_var_21991_20_non_const_pages[0]).shadow_left) = (-1);
((__cuda_local_var_21991_20_non_const_pages[0]).id) = 0U; {

 int i;
# 2157 "mummergpu.cu"
i = 1; for (; (((unsigned)i) < (__cuda_local_var_21986_18_non_const_num_reference_pages - 1U)); ++i) {
((__cuda_local_var_21991_20_non_const_pages[i]).begin) = ((int)(((unsigned)((__cuda_local_var_21991_20_non_const_pages[(i - 1)]).end)) - __cuda_local_var_21990_18_non_const_page_overlap));
((__cuda_local_var_21991_20_non_const_pages[i]).end) = ((int)((((unsigned)((__cuda_local_var_21991_20_non_const_pages[i]).begin)) + __cuda_local_var_21984_18_non_const_page_size) + __cuda_local_var_21990_18_non_const_page_overlap));
((__cuda_local_var_21991_20_non_const_pages[(i - 1)]).shadow_right) = ((__cuda_local_var_21991_20_non_const_pages[i]).begin);
((__cuda_local_var_21991_20_non_const_pages[i]).shadow_left) = ((__cuda_local_var_21991_20_non_const_pages[(i - 1)]).end);
((__cuda_local_var_21991_20_non_const_pages[i]).id) = ((unsigned)i);
} }

if (__cuda_local_var_21986_18_non_const_num_reference_pages > 1U) {
 int __cuda_local_var_22010_13_non_const_last_page;
# 2166 "mummergpu.cu"
__cuda_local_var_22010_13_non_const_last_page = ((int)(__cuda_local_var_21986_18_non_const_num_reference_pages - 1U));
((__cuda_local_var_21991_20_non_const_pages[__cuda_local_var_22010_13_non_const_last_page]).begin) = ((int)(((unsigned)((__cuda_local_var_21991_20_non_const_pages[(__cuda_local_var_22010_13_non_const_last_page - 1)]).end)) - __cuda_local_var_21990_18_non_const_page_overlap));
((__cuda_local_var_21991_20_non_const_pages[__cuda_local_var_22010_13_non_const_last_page]).end) = ((int)((ctx->full_ref_len) - 1UL));
((__cuda_local_var_21991_20_non_const_pages[(__cuda_local_var_22010_13_non_const_last_page - 1)]).shadow_right) = ((__cuda_local_var_21991_20_non_const_pages[__cuda_local_var_22010_13_non_const_last_page]).begin);
((__cuda_local_var_21991_20_non_const_pages[__cuda_local_var_22010_13_non_const_last_page]).shadow_right) = (-1);
((__cuda_local_var_21991_20_non_const_pages[__cuda_local_var_22010_13_non_const_last_page]).shadow_left) = ((__cuda_local_var_21991_20_non_const_pages[(__cuda_local_var_22010_13_non_const_last_page - 1)]).end);
((__cuda_local_var_21991_20_non_const_pages[__cuda_local_var_22010_13_non_const_last_page]).id) = ((unsigned)__cuda_local_var_22010_13_non_const_last_page);
}

(*pages_out) = __cuda_local_var_21991_20_non_const_pages;
(*num_pages) = ((int)__cuda_local_var_21986_18_non_const_num_reference_pages); 
}

int _Z29streamReferenceAgainstQueriesP12MatchContext( struct MatchContext *ctx) {
 int __cuda_local_var_22024_9_non_const_num_reference_pages;
 struct ReferencePage *__cuda_local_var_22025_20_non_const_pages;
# 2180 "mummergpu.cu"
__cuda_local_var_22024_9_non_const_num_reference_pages = 0;
__cuda_local_var_22025_20_non_const_pages = ((struct ReferencePage *)0LL);
_Z18initReferencePagesP12MatchContextPiPP13ReferencePage(ctx, (&__cuda_local_var_22024_9_non_const_num_reference_pages), (&__cuda_local_var_22025_20_non_const_pages));



m5_work_begin(0UL, 0UL);


_Z21buildReferenceTextureP9ReferencePcmmiS1_S1_P10Statistics((&((__cuda_local_var_22025_20_non_const_pages[0]).ref)), (ctx->full_ref), ((size_t)((__cuda_local_var_22025_20_non_const_pages[0]).begin)), ((size_t)((__cuda_local_var_22025_20_non_const_pages[0]).end)), (ctx->min_match_length), (ctx->dotfilename), (ctx->texfilename), (&(ctx->statistics)));
# 2199 "mummergpu.cu"
_Z27matchQueriesToReferencePageP12MatchContextP13ReferencePage(ctx, (__cuda_local_var_22025_20_non_const_pages + 0));
destroyReference((&((__cuda_local_var_22025_20_non_const_pages[0]).ref))); {

 int i;
# 2202 "mummergpu.cu"
i = 1; for (; (i < (__cuda_local_var_22024_9_non_const_num_reference_pages - 1)); ++i) {

_Z21buildReferenceTextureP9ReferencePcmmiS1_S1_P10Statistics((&((__cuda_local_var_22025_20_non_const_pages[i]).ref)), (ctx->full_ref), ((size_t)((__cuda_local_var_22025_20_non_const_pages[i]).begin)), ((size_t)((__cuda_local_var_22025_20_non_const_pages[i]).end)), (ctx->min_match_length), ((char *)0LL), ((char *)0LL), (&(ctx->statistics)));
# 2213 "mummergpu.cu"
_Z27matchQueriesToReferencePageP12MatchContextP13ReferencePage(ctx, (__cuda_local_var_22025_20_non_const_pages + i));
destroyReference((&((__cuda_local_var_22025_20_non_const_pages[i]).ref)));
} }

if (__cuda_local_var_22024_9_non_const_num_reference_pages > 1) {
 int __cuda_local_var_22062_13_non_const_last_page;
# 2218 "mummergpu.cu"
__cuda_local_var_22062_13_non_const_last_page = (__cuda_local_var_22024_9_non_const_num_reference_pages - 1);
_Z21buildReferenceTextureP9ReferencePcmmiS1_S1_P10Statistics((&((__cuda_local_var_22025_20_non_const_pages[__cuda_local_var_22062_13_non_const_last_page]).ref)), (ctx->full_ref), ((size_t)((__cuda_local_var_22025_20_non_const_pages[__cuda_local_var_22062_13_non_const_last_page]).begin)), ((size_t)((__cuda_local_var_22025_20_non_const_pages[__cuda_local_var_22062_13_non_const_last_page]).end)), (ctx->min_match_length), ((char *)0LL), ((char *)0LL), (&(ctx->statistics)));
# 2228 "mummergpu.cu"
_Z27matchQueriesToReferencePageP12MatchContextP13ReferencePage(ctx, (__cuda_local_var_22025_20_non_const_pages + __cuda_local_var_22062_13_non_const_last_page));
destroyReference((&((__cuda_local_var_22025_20_non_const_pages[__cuda_local_var_22062_13_non_const_last_page]).ref)));
}


m5_work_end(0UL, 0UL);

free(((void *)__cuda_local_var_22025_20_non_const_pages));
return 0;
}



int matchQueries( struct MatchContext *ctx) {
# 2252 "mummergpu.cu"
 char *__cuda_local_var_22096_11_non_const_ttimer;


 int __cuda_local_var_22099_9_non_const_ret;
# 2242 "mummergpu.cu"
(void)0;
(void)0;
# 2250 "mummergpu.cu"
_Z10resetStatsP10Statistics((&(ctx->statistics)));

__cuda_local_var_22096_11_non_const_ttimer = (createTimer());
startTimer(__cuda_local_var_22096_11_non_const_ttimer);



fprintf(stderr, ((const char *)"Streaming reference pages against all queries\n"));
__cuda_local_var_22099_9_non_const_ret = (_Z29streamReferenceAgainstQueriesP12MatchContext(ctx));

stopTimer(__cuda_local_var_22096_11_non_const_ttimer);
((ctx->statistics).t_end_to_end) += (getTimerValue(__cuda_local_var_22096_11_non_const_ttimer));
deleteTimer(__cuda_local_var_22096_11_non_const_ttimer);

_Z19writeStatisticsFileP10StatisticsPcS1_S1_((&(ctx->statistics)), (ctx->stats_file), "node_hist.out", "child_hist.out");

return __cuda_local_var_22099_9_non_const_ret;
}
static void __sti___17_mummergpu_cpp1_ii_a6baf3c4(void) {   }

#include "mummergpu.cudafe1.stub.c"