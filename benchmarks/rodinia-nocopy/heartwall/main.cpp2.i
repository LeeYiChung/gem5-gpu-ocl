# 1 "main.cudafe1.gpu"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "main.cudafe1.gpu"
typedef char __nv_bool;
# 1017 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h"
struct CUstream_st;
# 206 "/usr/include/libio.h" 3
enum __codecvt_result {
# 208 "/usr/include/libio.h" 3
__codecvt_ok,
# 209 "/usr/include/libio.h" 3
__codecvt_partial,
# 210 "/usr/include/libio.h" 3
__codecvt_error,
# 211 "/usr/include/libio.h" 3
__codecvt_noconv};
# 271 "/usr/include/libio.h" 3
struct _IO_FILE;
# 203 "/usr/include/math.h" 3
enum _ZUt_ {
# 204 "/usr/include/math.h" 3
FP_NAN,
# 206 "/usr/include/math.h" 3
FP_INFINITE,
# 208 "/usr/include/math.h" 3
FP_ZERO,
# 210 "/usr/include/math.h" 3
FP_SUBNORMAL,
# 212 "/usr/include/math.h" 3
FP_NORMAL};
# 296 "/usr/include/math.h" 3
enum _LIB_VERSION_TYPE {
# 297 "/usr/include/math.h" 3
_IEEE_ = (-1),
# 298 "/usr/include/math.h" 3
_SVID_,
# 299 "/usr/include/math.h" 3
_XOPEN_,
# 300 "/usr/include/math.h" 3
_POSIX_,
# 301 "/usr/include/math.h" 3
_ISOC_};
# 194 "/usr/include/x86_64-linux-gnu/bits/fcntl.h" 3
enum __pid_type {
# 196 "/usr/include/x86_64-linux-gnu/bits/fcntl.h" 3
F_OWNER_TID,
# 197 "/usr/include/x86_64-linux-gnu/bits/fcntl.h" 3
F_OWNER_PID,
# 198 "/usr/include/x86_64-linux-gnu/bits/fcntl.h" 3
F_OWNER_PGRP,
# 199 "/usr/include/x86_64-linux-gnu/bits/fcntl.h" 3
F_OWNER_GID = 2};
# 27 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
enum _ZUt0_ {
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
_PC_2_SYMLINKS};
# 74 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
enum _ZUt1_ {
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
_SC_THREAD_ROBUST_PRIO_PROTECT};
# 536 "/usr/include/x86_64-linux-gnu/bits/confname.h" 3
enum _ZUt2_ {
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
_CS_V7_ENV};
# 88 "./avilib.h"
struct avi_t;
# 195 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUipcMem_flags_enum {
# 196 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1};
# 204 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUctx_flags_enum {
# 205 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CTX_SCHED_AUTO,
# 206 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CTX_SCHED_SPIN,
# 207 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CTX_SCHED_YIELD,
# 208 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CTX_SCHED_BLOCKING_SYNC = 4,
# 209 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CTX_BLOCKING_SYNC = 4,
# 212 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CTX_SCHED_MASK = 7,
# 213 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CTX_MAP_HOST,
# 214 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CTX_LMEM_RESIZE_TO_MAX = 16,
# 215 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CTX_FLAGS_MASK = 31};
# 221 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUevent_flags_enum {
# 222 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_EVENT_DEFAULT,
# 223 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_EVENT_BLOCKING_SYNC,
# 224 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_EVENT_DISABLE_TIMING,
# 225 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_EVENT_INTERPROCESS = 4};
# 231 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUarray_format_enum {
# 232 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_AD_FORMAT_UNSIGNED_INT8 = 1,
# 233 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_AD_FORMAT_UNSIGNED_INT16,
# 234 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_AD_FORMAT_UNSIGNED_INT32,
# 235 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_AD_FORMAT_SIGNED_INT8 = 8,
# 236 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_AD_FORMAT_SIGNED_INT16,
# 237 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_AD_FORMAT_SIGNED_INT32,
# 238 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_AD_FORMAT_HALF = 16,
# 239 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_AD_FORMAT_FLOAT = 32};
# 245 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUaddress_mode_enum {
# 246 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TR_ADDRESS_MODE_WRAP,
# 247 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TR_ADDRESS_MODE_CLAMP,
# 248 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TR_ADDRESS_MODE_MIRROR,
# 249 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TR_ADDRESS_MODE_BORDER};
# 255 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUfilter_mode_enum {
# 256 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TR_FILTER_MODE_POINT,
# 257 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TR_FILTER_MODE_LINEAR};
# 263 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUdevice_attribute_enum {
# 264 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
# 265 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
# 266 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
# 267 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
# 268 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
# 269 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
# 270 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
# 271 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
# 272 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
# 273 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
# 274 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_WARP_SIZE,
# 275 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_PITCH,
# 276 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
# 277 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
# 278 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
# 279 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
# 280 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
# 281 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
# 282 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
# 283 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_INTEGRATED,
# 284 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
# 285 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
# 286 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
# 287 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
# 288 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
# 289 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
# 290 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
# 291 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
# 292 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
# 293 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
# 294 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
# 295 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
# 296 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT,
# 297 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES,
# 298 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
# 299 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
# 300 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
# 301 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
# 302 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
# 303 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
# 304 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
# 305 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
# 306 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
# 307 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
# 308 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
# 309 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
# 310 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
# 311 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
# 312 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER,
# 313 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH,
# 314 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT,
# 315 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE,
# 316 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE,
# 317 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE,
# 318 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
# 319 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
# 320 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH,
# 321 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH,
# 322 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS,
# 323 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH,
# 324 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH,
# 325 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT,
# 326 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH,
# 327 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT,
# 328 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH,
# 329 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH,
# 330 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS,
# 331 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH,
# 332 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT,
# 333 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS,
# 334 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH,
# 335 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH,
# 336 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS,
# 337 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH,
# 338 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH,
# 339 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT,
# 340 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH};
# 362 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUpointer_attribute_enum {
# 363 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_POINTER_ATTRIBUTE_CONTEXT = 1,
# 364 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
# 365 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
# 366 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_POINTER_ATTRIBUTE_HOST_POINTER};
# 372 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUfunction_attribute_enum {
# 378 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
# 385 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
# 391 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
# 396 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
# 401 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_NUM_REGS,
# 410 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_PTX_VERSION,
# 419 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_BINARY_VERSION,
# 421 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_ATTRIBUTE_MAX};
# 427 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUfunc_cache_enum {
# 428 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_CACHE_PREFER_NONE,
# 429 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_CACHE_PREFER_SHARED,
# 430 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_CACHE_PREFER_L1,
# 431 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_FUNC_CACHE_PREFER_EQUAL};
# 437 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUsharedconfig_enum {
# 438 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE,
# 439 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE,
# 440 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE};
# 446 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUmemorytype_enum {
# 447 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_MEMORYTYPE_HOST = 1,
# 448 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_MEMORYTYPE_DEVICE,
# 449 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_MEMORYTYPE_ARRAY,
# 450 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_MEMORYTYPE_UNIFIED};
# 456 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUcomputemode_enum {
# 457 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_COMPUTEMODE_DEFAULT,
# 458 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_COMPUTEMODE_EXCLUSIVE,
# 459 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_COMPUTEMODE_PROHIBITED,
# 460 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
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
# 560 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TARGET_COMPUTE_10,
# 561 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TARGET_COMPUTE_11,
# 562 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TARGET_COMPUTE_12,
# 563 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TARGET_COMPUTE_13,
# 564 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TARGET_COMPUTE_20,
# 565 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TARGET_COMPUTE_21,
# 566 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_TARGET_COMPUTE_30};
# 572 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUjit_fallback_enum {
# 574 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_PREFER_PTX,
# 576 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_PREFER_BINARY};
# 583 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUgraphicsRegisterFlags_enum {
# 584 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_NONE,
# 585 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY,
# 586 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
# 587 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4,
# 588 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8};
# 594 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUgraphicsMapResourceFlags_enum {
# 595 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE,
# 596 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY,
# 597 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD};
# 603 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUarray_cubemap_face_enum {
# 604 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CUBEMAP_FACE_POSITIVE_X,
# 605 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CUBEMAP_FACE_NEGATIVE_X,
# 606 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CUBEMAP_FACE_POSITIVE_Y,
# 607 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CUBEMAP_FACE_NEGATIVE_Y,
# 608 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CUBEMAP_FACE_POSITIVE_Z,
# 609 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_CUBEMAP_FACE_NEGATIVE_Z};
# 615 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
enum CUlimit_enum {
# 616 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_LIMIT_STACK_SIZE,
# 617 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CU_LIMIT_PRINTF_FIFO_SIZE,
# 618 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
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
# 653 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_DEINITIALIZED,
# 659 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_PROFILER_DISABLED,
# 664 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_PROFILER_NOT_INITIALIZED,
# 669 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_PROFILER_ALREADY_STARTED,
# 674 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_PROFILER_ALREADY_STOPPED,
# 679 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_NO_DEVICE = 100,
# 685 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_INVALID_DEVICE,
# 692 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_INVALID_IMAGE = 200,
# 702 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_INVALID_CONTEXT,
# 711 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
# 716 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_MAP_FAILED = 205,
# 721 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_UNMAP_FAILED,
# 727 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_ARRAY_IS_MAPPED,
# 732 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_ALREADY_MAPPED,
# 740 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_NO_BINARY_FOR_GPU,
# 745 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_ALREADY_ACQUIRED,
# 750 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
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
# 786 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_INVALID_SOURCE = 300,
# 791 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_FILE_NOT_FOUND,
# 796 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
# 801 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
# 806 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
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
# 927 "/home/bachelor/deicide218/cuda-4.2/include/cuda.h"
CUDA_ERROR_UNKNOWN = 999};
# 21 "define.c"
struct params_common_change;
# 38 "define.c"
struct params_common;
# 270 "define.c"
struct params_unique;
# 124 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_voidIvEUt_E {
# 124 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt9__is_voidIvE7__valueE = 1};
# 144 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIbEUt_E {
# 144 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIbE7__valueE = 1};
# 151 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIcEUt_E {
# 151 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIcE7__valueE = 1};
# 158 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIaEUt_E {
# 158 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIaE7__valueE = 1};
# 165 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIhEUt_E {
# 165 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIhE7__valueE = 1};
# 173 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIwEUt_E {
# 173 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIwE7__valueE = 1};
# 197 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIsEUt_E {
# 197 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIsE7__valueE = 1};
# 204 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerItEUt_E {
# 204 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerItE7__valueE = 1};
# 211 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIiEUt_E {
# 211 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIiE7__valueE = 1};
# 218 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIjEUt_E {
# 218 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIjE7__valueE = 1};
# 225 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIlEUt_E {
# 225 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIlE7__valueE = 1};
# 232 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerImEUt_E {
# 232 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerImE7__valueE = 1};
# 239 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIxEUt_E {
# 239 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIxE7__valueE = 1};
# 246 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIyEUt_E {
# 246 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIyE7__valueE = 1};
# 264 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIfEUt_E {
# 264 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt13__is_floatingIfE7__valueE = 1};
# 271 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIdEUt_E {
# 271 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt13__is_floatingIdE7__valueE = 1};
# 278 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIeEUt_E {
# 278 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt13__is_floatingIeE7__valueE = 1};
# 354 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIcEUt_E {
# 354 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt9__is_charIcE7__valueE = 1};
# 362 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIwEUt_E {
# 362 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt9__is_charIwE7__valueE = 1};
# 377 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIcEUt_E {
# 377 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt9__is_byteIcE7__valueE = 1};
# 384 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIaEUt_E {
# 384 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt9__is_byteIaE7__valueE = 1};
# 391 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIhEUt_E {
# 391 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt9__is_byteIhE7__valueE = 1};
# 134 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIfEUt_E {
# 134 "/usr/include/c++/4.4/bits/cpp_type_traits.h" 3
_ZNSt12__is_integerIfE7__valueE};
# 64 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
union _ZZ10__signbitlEUt_;
# 211 "/usr/lib/gcc/x86_64-linux-gnu/4.4.7/include/stddef.h" 3
typedef unsigned long size_t;
# 1 "/home/bachelor/deicide218/cuda-4.2/include/crt/device_runtime.h" 1 3
# 38 "/home/bachelor/deicide218/cuda-4.2/include/crt/device_runtime.h" 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/host_defines.h" 1 3
# 39 "/home/bachelor/deicide218/cuda-4.2/include/crt/device_runtime.h" 2 3




typedef __attribute__((device_builtin_texture_type)) const void *__texture_type__;
typedef __attribute__((device_builtin_surface_type)) const void *__surface_type__;
# 129 "/home/bachelor/deicide218/cuda-4.2/include/crt/device_runtime.h" 3
extern __attribute__((device)) void* malloc(size_t);
extern __attribute__((device)) void free(void*);

extern __attribute__((device)) void __assertfail(
  const void *message,
  const void *file,
  unsigned int line,
  const void *function,
  size_t charsize);
# 154 "/home/bachelor/deicide218/cuda-4.2/include/crt/device_runtime.h" 3
static __attribute__((device)) void __assert_fail(
  const char *__assertion,
  const char *__file,
  unsigned int __line,
  const char *__function)
{
  __assertfail(
    (const void *)__assertion,
    (const void *)__file,
                  __line,
    (const void *)__function,
    sizeof(char));
}
# 184 "/home/bachelor/deicide218/cuda-4.2/include/crt/device_runtime.h" 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/builtin_types.h" 1 3
# 56 "/home/bachelor/deicide218/cuda-4.2/include/builtin_types.h" 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/device_types.h" 1 3
# 53 "/home/bachelor/deicide218/cuda-4.2/include/device_types.h" 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/host_defines.h" 1 3
# 54 "/home/bachelor/deicide218/cuda-4.2/include/device_types.h" 2 3







enum __attribute__((device_builtin)) cudaRoundMode
{
    cudaRoundNearest,
    cudaRoundZero,
    cudaRoundPosInf,
    cudaRoundMinInf
};
# 57 "/home/bachelor/deicide218/cuda-4.2/include/builtin_types.h" 2 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 1 3
# 126 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
enum __attribute__((device_builtin)) cudaError
{





    cudaSuccess = 0,





    cudaErrorMissingConfiguration = 1,





    cudaErrorMemoryAllocation = 2,





    cudaErrorInitializationError = 3,
# 161 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
    cudaErrorLaunchFailure = 4,
# 170 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
    cudaErrorPriorLaunchFailure = 5,
# 180 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
    cudaErrorLaunchTimeout = 6,
# 189 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
    cudaErrorLaunchOutOfResources = 7,





    cudaErrorInvalidDeviceFunction = 8,
# 204 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
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
# 285 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
    cudaErrorAddressOfConstant = 22,
# 294 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
    cudaErrorTextureFetchFailed = 23,
# 303 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
    cudaErrorTextureNotBound = 24,
# 312 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
    cudaErrorSynchronizationError = 25,





    cudaErrorInvalidFilterSetting = 26,





    cudaErrorInvalidNormSetting = 27,







    cudaErrorMixedDeviceExecution = 28,






    cudaErrorCudartUnloading = 29,




    cudaErrorUnknown = 30,







    cudaErrorNotYetImplemented = 31,
# 361 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
    cudaErrorMemoryValueTooLarge = 32,






    cudaErrorInvalidResourceHandle = 33,







    cudaErrorNotReady = 34,






    cudaErrorInsufficientDriver = 35,
# 396 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
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
# 458 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
    cudaErrorDevicesUnavailable = 46,




    cudaErrorInvalidKernelImage = 47,







    cudaErrorNoKernelImageForDevice = 48,
# 484 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
    cudaErrorIncompatibleDriverContext = 49,






    cudaErrorPeerAccessAlreadyEnabled = 50,






    cudaErrorPeerAccessNotEnabled = 51,





    cudaErrorDeviceAlreadyInUse = 54,







    cudaErrorProfilerDisabled = 55,






    cudaErrorProfilerNotInitialized = 56,






    cudaErrorProfilerAlreadyStarted = 57,





     cudaErrorProfilerAlreadyStopped = 58,







    cudaErrorAssert = 59,






    cudaErrorTooManyPeers = 60,





    cudaErrorHostMemoryAlreadyRegistered = 61,





    cudaErrorHostMemoryNotRegistered = 62,




    cudaErrorOperatingSystem = 63,




    cudaErrorStartupFailure = 0x7f,







    cudaErrorApiFailureBase = 10000
};




enum __attribute__((device_builtin)) cudaChannelFormatKind
{
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3
};




struct __attribute__((device_builtin)) cudaChannelFormatDesc
{
    int x;
    int y;
    int z;
    int w;
    enum cudaChannelFormatKind f;
};




struct cudaArray;




enum __attribute__((device_builtin)) cudaMemoryType
{
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2
};




enum __attribute__((device_builtin)) cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};





struct __attribute__((device_builtin)) cudaPitchedPtr
{
    void *ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
};





struct __attribute__((device_builtin)) cudaExtent
{
    size_t width;
    size_t height;
    size_t depth;
};





struct __attribute__((device_builtin)) cudaPos
{
    size_t x;
    size_t y;
    size_t z;
};




struct __attribute__((device_builtin)) cudaMemcpy3DParms
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




struct __attribute__((device_builtin)) cudaMemcpy3DPeerParms
{
    struct cudaArray *srcArray;
    struct cudaPos srcPos;
    struct cudaPitchedPtr srcPtr;
    int srcDevice;

    struct cudaArray *dstArray;
    struct cudaPos dstPos;
    struct cudaPitchedPtr dstPtr;
    int dstDevice;

    struct cudaExtent extent;
};




struct cudaGraphicsResource;




enum __attribute__((device_builtin)) cudaGraphicsRegisterFlags
{
    cudaGraphicsRegisterFlagsNone = 0,
    cudaGraphicsRegisterFlagsReadOnly = 1,
    cudaGraphicsRegisterFlagsWriteDiscard = 2,
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,
    cudaGraphicsRegisterFlagsTextureGather = 8
};




enum __attribute__((device_builtin)) cudaGraphicsMapFlags
{
    cudaGraphicsMapFlagsNone = 0,
    cudaGraphicsMapFlagsReadOnly = 1,
    cudaGraphicsMapFlagsWriteDiscard = 2
};




enum __attribute__((device_builtin)) cudaGraphicsCubeFace
{
    cudaGraphicsCubeFacePositiveX = 0x00,
    cudaGraphicsCubeFaceNegativeX = 0x01,
    cudaGraphicsCubeFacePositiveY = 0x02,
    cudaGraphicsCubeFaceNegativeY = 0x03,
    cudaGraphicsCubeFacePositiveZ = 0x04,
    cudaGraphicsCubeFaceNegativeZ = 0x05
};




struct __attribute__((device_builtin)) cudaPointerAttributes
{




    enum cudaMemoryType memoryType;
# 758 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
    int device;





    void *devicePointer;





    void *hostPointer;
};




struct __attribute__((device_builtin)) cudaFuncAttributes
{





   size_t sharedSizeBytes;





   size_t constSizeBytes;




   size_t localSizeBytes;






   int maxThreadsPerBlock;




   int numRegs;






   int ptxVersion;






   int binaryVersion;
};




enum __attribute__((device_builtin)) cudaFuncCache
{
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3
};





enum __attribute__((device_builtin)) cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault = 0,
    cudaSharedMemBankSizeFourByte = 1,
    cudaSharedMemBankSizeEightByte = 2
};




enum __attribute__((device_builtin)) cudaComputeMode
{
    cudaComputeModeDefault = 0,
    cudaComputeModeExclusive = 1,
    cudaComputeModeProhibited = 2,
    cudaComputeModeExclusiveProcess = 3
};




enum __attribute__((device_builtin)) cudaLimit
{
    cudaLimitStackSize = 0x00,
    cudaLimitPrintfFifoSize = 0x01,
    cudaLimitMallocHeapSize = 0x02
};




enum __attribute__((device_builtin)) cudaOutputMode
{
    cudaKeyValuePair = 0x00,
    cudaCSV = 0x01
};




struct __attribute__((device_builtin)) cudaDeviceProp
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
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
};
# 993 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
struct __attribute__((device_builtin)) cudaIpcEventHandle_st
{
    char reserved[64];
};

struct __attribute__((device_builtin)) cudaIpcMemHandle_st
{
    char reserved[64];
};
# 1012 "/home/bachelor/deicide218/cuda-4.2/include/driver_types.h" 3
typedef __attribute__((device_builtin)) enum cudaError cudaError_t;




typedef __attribute__((device_builtin)) struct CUstream_st *cudaStream_t;




typedef __attribute__((device_builtin)) struct CUevent_st *cudaEvent_t;




typedef __attribute__((device_builtin)) struct cudaGraphicsResource *cudaGraphicsResource_t;




typedef __attribute__((device_builtin)) struct CUuuid_st cudaUUID_t;




typedef __attribute__((device_builtin)) struct cudaIpcEventHandle_st cudaIpcEventHandle_t;
typedef __attribute__((device_builtin)) struct cudaIpcMemHandle_st cudaIpcMemHandle_t;




typedef __attribute__((device_builtin)) enum cudaOutputMode cudaOutputMode_t;
# 58 "/home/bachelor/deicide218/cuda-4.2/include/builtin_types.h" 2 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/surface_types.h" 1 3
# 84 "/home/bachelor/deicide218/cuda-4.2/include/surface_types.h" 3
enum __attribute__((device_builtin)) cudaSurfaceBoundaryMode
{
    cudaBoundaryModeZero = 0,
    cudaBoundaryModeClamp = 1,
    cudaBoundaryModeTrap = 2
};




enum __attribute__((device_builtin)) cudaSurfaceFormatMode
{
    cudaFormatModeForced = 0,
    cudaFormatModeAuto = 1
};




struct __attribute__((device_builtin)) surfaceReference
{



    struct cudaChannelFormatDesc channelDesc;
};
# 59 "/home/bachelor/deicide218/cuda-4.2/include/builtin_types.h" 2 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/texture_types.h" 1 3
# 84 "/home/bachelor/deicide218/cuda-4.2/include/texture_types.h" 3
enum __attribute__((device_builtin)) cudaTextureAddressMode
{
    cudaAddressModeWrap = 0,
    cudaAddressModeClamp = 1,
    cudaAddressModeMirror = 2,
    cudaAddressModeBorder = 3
};




enum __attribute__((device_builtin)) cudaTextureFilterMode
{
    cudaFilterModePoint = 0,
    cudaFilterModeLinear = 1
};




enum __attribute__((device_builtin)) cudaTextureReadMode
{
    cudaReadModeElementType = 0,
    cudaReadModeNormalizedFloat = 1
};




struct __attribute__((device_builtin)) textureReference
{



    int normalized;



    enum cudaTextureFilterMode filterMode;



    enum cudaTextureAddressMode addressMode[3];



    struct cudaChannelFormatDesc channelDesc;



    int sRGB;
    int __cudaReserved[15];
};
# 60 "/home/bachelor/deicide218/cuda-4.2/include/builtin_types.h" 2 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/vector_types.h" 1 3
# 59 "/home/bachelor/deicide218/cuda-4.2/include/vector_types.h" 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/builtin_types.h" 1 3
# 60 "/home/bachelor/deicide218/cuda-4.2/include/builtin_types.h" 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/vector_types.h" 1 3
# 60 "/home/bachelor/deicide218/cuda-4.2/include/builtin_types.h" 2 3
# 60 "/home/bachelor/deicide218/cuda-4.2/include/vector_types.h" 2 3
# 94 "/home/bachelor/deicide218/cuda-4.2/include/vector_types.h" 3
struct __attribute__((device_builtin)) char1
{
    signed char x;
};

struct __attribute__((device_builtin)) uchar1
{
    unsigned char x;
};


struct __attribute__((device_builtin)) __attribute__((aligned(2))) char2
{
    signed char x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(2))) uchar2
{
    unsigned char x, y;
};

struct __attribute__((device_builtin)) char3
{
    signed char x, y, z;
};

struct __attribute__((device_builtin)) uchar3
{
    unsigned char x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) char4
{
    signed char x, y, z, w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) uchar4
{
    unsigned char x, y, z, w;
};

struct __attribute__((device_builtin)) short1
{
    short x;
};

struct __attribute__((device_builtin)) ushort1
{
    unsigned short x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) short2
{
    short x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(4))) ushort2
{
    unsigned short x, y;
};

struct __attribute__((device_builtin)) short3
{
    short x, y, z;
};

struct __attribute__((device_builtin)) ushort3
{
    unsigned short x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(8))) short4 { short x; short y; short z; short w; };
struct __attribute__((device_builtin)) __attribute__((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; };

struct __attribute__((device_builtin)) int1
{
    int x;
};

struct __attribute__((device_builtin)) uint1
{
    unsigned int x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(8))) int2 { int x; int y; };
struct __attribute__((device_builtin)) __attribute__((aligned(8))) uint2 { unsigned int x; unsigned int y; };

struct __attribute__((device_builtin)) int3
{
    int x, y, z;
};

struct __attribute__((device_builtin)) uint3
{
    unsigned int x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) int4
{
    int x, y, z, w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) uint4
{
    unsigned int x, y, z, w;
};

struct __attribute__((device_builtin)) long1
{
    long int x;
};

struct __attribute__((device_builtin)) ulong1
{
    unsigned long x;
};






struct __attribute__((device_builtin)) __attribute__((aligned(2*sizeof(long int)))) long2
{
    long int x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(2*sizeof(unsigned long int)))) ulong2
{
    unsigned long int x, y;
};



struct __attribute__((device_builtin)) long3
{
    long int x, y, z;
};

struct __attribute__((device_builtin)) ulong3
{
    unsigned long int x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) long4
{
    long int x, y, z, w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) ulong4
{
    unsigned long int x, y, z, w;
};

struct __attribute__((device_builtin)) float1
{
    float x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(8))) float2 { float x; float y; };

struct __attribute__((device_builtin)) float3
{
    float x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) float4
{
    float x, y, z, w;
};

struct __attribute__((device_builtin)) longlong1
{
    long long int x;
};

struct __attribute__((device_builtin)) ulonglong1
{
    unsigned long long int x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) longlong2
{
    long long int x, y;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) ulonglong2
{
    unsigned long long int x, y;
};

struct __attribute__((device_builtin)) longlong3
{
    long long int x, y, z;
};

struct __attribute__((device_builtin)) ulonglong3
{
    unsigned long long int x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) longlong4
{
    long long int x, y, z ,w;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) ulonglong4
{
    unsigned long long int x, y, z, w;
};

struct __attribute__((device_builtin)) double1
{
    double x;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) double2
{
    double x, y;
};

struct __attribute__((device_builtin)) double3
{
    double x, y, z;
};

struct __attribute__((device_builtin)) __attribute__((aligned(16))) double4
{
    double x, y, z, w;
};
# 338 "/home/bachelor/deicide218/cuda-4.2/include/vector_types.h" 3
typedef __attribute__((device_builtin)) struct char1 char1;
typedef __attribute__((device_builtin)) struct uchar1 uchar1;
typedef __attribute__((device_builtin)) struct char2 char2;
typedef __attribute__((device_builtin)) struct uchar2 uchar2;
typedef __attribute__((device_builtin)) struct char3 char3;
typedef __attribute__((device_builtin)) struct uchar3 uchar3;
typedef __attribute__((device_builtin)) struct char4 char4;
typedef __attribute__((device_builtin)) struct uchar4 uchar4;
typedef __attribute__((device_builtin)) struct short1 short1;
typedef __attribute__((device_builtin)) struct ushort1 ushort1;
typedef __attribute__((device_builtin)) struct short2 short2;
typedef __attribute__((device_builtin)) struct ushort2 ushort2;
typedef __attribute__((device_builtin)) struct short3 short3;
typedef __attribute__((device_builtin)) struct ushort3 ushort3;
typedef __attribute__((device_builtin)) struct short4 short4;
typedef __attribute__((device_builtin)) struct ushort4 ushort4;
typedef __attribute__((device_builtin)) struct int1 int1;
typedef __attribute__((device_builtin)) struct uint1 uint1;
typedef __attribute__((device_builtin)) struct int2 int2;
typedef __attribute__((device_builtin)) struct uint2 uint2;
typedef __attribute__((device_builtin)) struct int3 int3;
typedef __attribute__((device_builtin)) struct uint3 uint3;
typedef __attribute__((device_builtin)) struct int4 int4;
typedef __attribute__((device_builtin)) struct uint4 uint4;
typedef __attribute__((device_builtin)) struct long1 long1;
typedef __attribute__((device_builtin)) struct ulong1 ulong1;
typedef __attribute__((device_builtin)) struct long2 long2;
typedef __attribute__((device_builtin)) struct ulong2 ulong2;
typedef __attribute__((device_builtin)) struct long3 long3;
typedef __attribute__((device_builtin)) struct ulong3 ulong3;
typedef __attribute__((device_builtin)) struct long4 long4;
typedef __attribute__((device_builtin)) struct ulong4 ulong4;
typedef __attribute__((device_builtin)) struct float1 float1;
typedef __attribute__((device_builtin)) struct float2 float2;
typedef __attribute__((device_builtin)) struct float3 float3;
typedef __attribute__((device_builtin)) struct float4 float4;
typedef __attribute__((device_builtin)) struct longlong1 longlong1;
typedef __attribute__((device_builtin)) struct ulonglong1 ulonglong1;
typedef __attribute__((device_builtin)) struct longlong2 longlong2;
typedef __attribute__((device_builtin)) struct ulonglong2 ulonglong2;
typedef __attribute__((device_builtin)) struct longlong3 longlong3;
typedef __attribute__((device_builtin)) struct ulonglong3 ulonglong3;
typedef __attribute__((device_builtin)) struct longlong4 longlong4;
typedef __attribute__((device_builtin)) struct ulonglong4 ulonglong4;
typedef __attribute__((device_builtin)) struct double1 double1;
typedef __attribute__((device_builtin)) struct double2 double2;
typedef __attribute__((device_builtin)) struct double3 double3;
typedef __attribute__((device_builtin)) struct double4 double4;







struct __attribute__((device_builtin)) dim3
{
    unsigned int x, y, z;





};

typedef __attribute__((device_builtin)) struct dim3 dim3;
# 60 "/home/bachelor/deicide218/cuda-4.2/include/builtin_types.h" 2 3
# 185 "/home/bachelor/deicide218/cuda-4.2/include/crt/device_runtime.h" 2 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/device_launch_parameters.h" 1 3
# 66 "/home/bachelor/deicide218/cuda-4.2/include/device_launch_parameters.h" 3
uint3 __attribute__((device_builtin)) extern const threadIdx;
uint3 __attribute__((device_builtin)) extern const blockIdx;
dim3 __attribute__((device_builtin)) extern const blockDim;
dim3 __attribute__((device_builtin)) extern const gridDim;
int __attribute__((device_builtin)) extern const warpSize;
# 186 "/home/bachelor/deicide218/cuda-4.2/include/crt/device_runtime.h" 2 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/crt/storage_class.h" 1 3
# 186 "/home/bachelor/deicide218/cuda-4.2/include/crt/device_runtime.h" 2 3
# 213 "/usr/lib/gcc/x86_64-linux-gnu/4.4.7/include/stddef.h" 2 3
# 49 "/usr/include/stdio.h" 3
typedef struct _IO_FILE FILE;
# 56 "/usr/include/stdint.h" 3
typedef unsigned long uint64_t;
# 126 "./avilib.h"
typedef struct avi_t avi_t;
# 21 "define.c"
struct params_common_change {
# 27 "define.c"
float *d_frame;
# 28 "define.c"
int frame_no;char __nv_no_debug_dummy_end_padding_0[4];};
# 30 "define.c"
typedef struct params_common_change params_common_change;
# 38 "define.c"
struct params_common {
# 48 "define.c"
int sSize;
# 49 "define.c"
int tSize;
# 50 "define.c"
int maxMove;
# 51 "define.c"
float alpha;
# 57 "define.c"
int no_frames;
# 58 "define.c"
int frame_rows;
# 59 "define.c"
int frame_cols;
# 60 "define.c"
int frame_elem;
# 61 "define.c"
int frame_mem;
# 67 "define.c"
int endoPoints;
# 68 "define.c"
int endo_mem;
# 70 "define.c"
int *endoRow;
# 71 "define.c"
int *endoCol;
# 72 "define.c"
int *tEndoRowLoc;
# 73 "define.c"
int *tEndoColLoc;
# 75 "define.c"
int *d_endoRow;
# 76 "define.c"
int *d_endoCol;
# 77 "define.c"
int *d_tEndoRowLoc;
# 78 "define.c"
int *d_tEndoColLoc;
# 80 "define.c"
float *d_endoT;
# 85 "define.c"
int epiPoints;
# 86 "define.c"
int epi_mem;
# 88 "define.c"
int *epiRow;
# 89 "define.c"
int *epiCol;
# 90 "define.c"
int *tEpiRowLoc;
# 91 "define.c"
int *tEpiColLoc;
# 93 "define.c"
int *d_epiRow;
# 94 "define.c"
int *d_epiCol;
# 95 "define.c"
int *d_tEpiRowLoc;
# 96 "define.c"
int *d_tEpiColLoc;
# 98 "define.c"
float *d_epiT;
# 104 "define.c"
int allPoints;
# 110 "define.c"
int in_rows;
# 111 "define.c"
int in_cols;
# 112 "define.c"
int in_elem;
# 113 "define.c"
int in_mem;
# 119 "define.c"
int in2_rows;
# 120 "define.c"
int in2_cols;
# 121 "define.c"
int in2_elem;
# 122 "define.c"
int in2_mem;
# 128 "define.c"
int conv_rows;
# 129 "define.c"
int conv_cols;
# 130 "define.c"
int conv_elem;
# 131 "define.c"
int conv_mem;
# 132 "define.c"
int ioffset;
# 133 "define.c"
int joffset;
# 143 "define.c"
int in2_pad_add_rows;
# 144 "define.c"
int in2_pad_add_cols;
# 145 "define.c"
int in2_pad_cumv_rows;
# 146 "define.c"
int in2_pad_cumv_cols;
# 147 "define.c"
int in2_pad_cumv_elem;
# 148 "define.c"
int in2_pad_cumv_mem;
# 154 "define.c"
int in2_pad_cumv_sel_rows;
# 155 "define.c"
int in2_pad_cumv_sel_cols;
# 156 "define.c"
int in2_pad_cumv_sel_elem;
# 157 "define.c"
int in2_pad_cumv_sel_mem;
# 158 "define.c"
int in2_pad_cumv_sel_rowlow;
# 159 "define.c"
int in2_pad_cumv_sel_rowhig;
# 160 "define.c"
int in2_pad_cumv_sel_collow;
# 161 "define.c"
int in2_pad_cumv_sel_colhig;
# 167 "define.c"
int in2_pad_cumv_sel2_rowlow;
# 168 "define.c"
int in2_pad_cumv_sel2_rowhig;
# 169 "define.c"
int in2_pad_cumv_sel2_collow;
# 170 "define.c"
int in2_pad_cumv_sel2_colhig;
# 171 "define.c"
int in2_sub_cumh_rows;
# 172 "define.c"
int in2_sub_cumh_cols;
# 173 "define.c"
int in2_sub_cumh_elem;
# 174 "define.c"
int in2_sub_cumh_mem;
# 180 "define.c"
int in2_sub_cumh_sel_rows;
# 181 "define.c"
int in2_sub_cumh_sel_cols;
# 182 "define.c"
int in2_sub_cumh_sel_elem;
# 183 "define.c"
int in2_sub_cumh_sel_mem;
# 184 "define.c"
int in2_sub_cumh_sel_rowlow;
# 185 "define.c"
int in2_sub_cumh_sel_rowhig;
# 186 "define.c"
int in2_sub_cumh_sel_collow;
# 187 "define.c"
int in2_sub_cumh_sel_colhig;
# 193 "define.c"
int in2_sub_cumh_sel2_rowlow;
# 194 "define.c"
int in2_sub_cumh_sel2_rowhig;
# 195 "define.c"
int in2_sub_cumh_sel2_collow;
# 196 "define.c"
int in2_sub_cumh_sel2_colhig;
# 197 "define.c"
int in2_sub2_rows;
# 198 "define.c"
int in2_sub2_cols;
# 199 "define.c"
int in2_sub2_elem;
# 200 "define.c"
int in2_sub2_mem;
# 210 "define.c"
int in2_sqr_rows;
# 211 "define.c"
int in2_sqr_cols;
# 212 "define.c"
int in2_sqr_elem;
# 213 "define.c"
int in2_sqr_mem;
# 219 "define.c"
int in2_sqr_sub2_rows;
# 220 "define.c"
int in2_sqr_sub2_cols;
# 221 "define.c"
int in2_sqr_sub2_elem;
# 222 "define.c"
int in2_sqr_sub2_mem;
# 228 "define.c"
int in_sqr_rows;
# 229 "define.c"
int in_sqr_cols;
# 230 "define.c"
int in_sqr_elem;
# 231 "define.c"
int in_sqr_mem;
# 237 "define.c"
int tMask_rows;
# 238 "define.c"
int tMask_cols;
# 239 "define.c"
int tMask_elem;
# 240 "define.c"
int tMask_mem;
# 246 "define.c"
int mask_rows;
# 247 "define.c"
int mask_cols;
# 248 "define.c"
int mask_elem;
# 249 "define.c"
int mask_mem;
# 255 "define.c"
int mask_conv_rows;
# 256 "define.c"
int mask_conv_cols;
# 257 "define.c"
int mask_conv_elem;
# 258 "define.c"
int mask_conv_mem;
# 259 "define.c"
int mask_conv_ioffset;
# 260 "define.c"
int mask_conv_joffset;char __nv_no_debug_dummy_end_padding_0[4];};
# 262 "define.c"
typedef struct params_common params_common;
# 270 "define.c"
struct params_unique {
# 276 "define.c"
int *d_Row;
# 277 "define.c"
int *d_Col;
# 278 "define.c"
int *d_tRowLoc;
# 279 "define.c"
int *d_tColLoc;
# 280 "define.c"
float *d_T;
# 286 "define.c"
int point_no;
# 292 "define.c"
int in_pointer;
# 298 "define.c"
float *d_in2;
# 304 "define.c"
float *d_conv;
# 305 "define.c"
float *d_in_mod;
# 315 "define.c"
float *d_in2_pad_cumv;
# 321 "define.c"
float *d_in2_pad_cumv_sel;
# 327 "define.c"
float *d_in2_sub_cumh;
# 333 "define.c"
float *d_in2_sub_cumh_sel;
# 339 "define.c"
float *d_in2_sub2;
# 349 "define.c"
float *d_in2_sqr;
# 355 "define.c"
float *d_in2_sqr_sub2;
# 361 "define.c"
float *d_in_sqr;
# 367 "define.c"
float *d_tMask;
# 373 "define.c"
float *d_mask;
# 379 "define.c"
float *d_mask_conv;};
# 381 "define.c"
typedef struct params_unique params_unique;
# 64 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
union _ZZ10__signbitlEUt_ {
# 64 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
long double __l;
# 64 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
int __i[3];};
# 101 "/home/bachelor/deicide218/cuda-4.2/include/common_functions.h"
 __attribute__((device_builtin)) extern __attribute__((device)) int printf(const char *__restrict__, ...);
# 103 "/home/bachelor/deicide218/cuda-4.2/include/common_functions.h"
extern __attribute__((device)) __attribute__((__malloc__)) void *malloc(size_t);
# 104 "/home/bachelor/deicide218/cuda-4.2/include/common_functions.h"
extern __attribute__((device)) void free(void *);
# 38 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
 __attribute__((device_builtin)) extern __attribute__((device)) __inline__ __attribute__((__const__)) int __signbitf(float);
# 50 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
 __attribute__((device_builtin)) extern __attribute__((device)) __inline__ __attribute__((__const__)) int __signbit(double);
# 62 "/usr/include/x86_64-linux-gnu/bits/mathinline.h" 3
 __attribute__((device_builtin)) extern __attribute__((device)) __inline__ __attribute__((__const__)) int __signbitl(long double);
# 131 "/home/bachelor/deicide218/cuda-4.2/include/device_functions.h"
 __attribute__((device_builtin)) extern __attribute__((device)) void __syncthreads(void);
# 7 "kernel.cu"
__attribute__((global)) extern void _Z6kernel20params_common_change13params_commonP13params_unique(params_common_change, params_common, params_unique *);
# 1 "/home/bachelor/deicide218/cuda-4.2/include/common_functions.h" 1
# 159 "/home/bachelor/deicide218/cuda-4.2/include/common_functions.h"
# 1 "/home/bachelor/deicide218/cuda-4.2/include/math_functions.h" 1 3
# 7730 "/home/bachelor/deicide218/cuda-4.2/include/math_functions.h" 3
# 1 "/home/bachelor/deicide218/cuda-4.2/include/math_functions_dbl_ptx3.h" 1 3
# 7731 "/home/bachelor/deicide218/cuda-4.2/include/math_functions.h" 2 3
# 160 "/home/bachelor/deicide218/cuda-4.2/include/common_functions.h" 2
# 9 "kernel.cu" 2
# 7 "kernel.cu"
__attribute__((global)) void _Z6kernel20params_common_change13params_commonP13params_unique(
# 7 "kernel.cu"
params_common_change d_common_change,
# 7 "kernel.cu"
params_common d_common,
# 7 "kernel.cu"
params_unique *d_unique){
# 7 "kernel.cu"
{ float __T21;
# 13 "kernel.cu"
 float *__cuda_local_var_20037_9_non_const_d_in;
# 14 "kernel.cu"
 int __cuda_local_var_20038_6_non_const_rot_row;
# 15 "kernel.cu"
 int __cuda_local_var_20039_6_non_const_rot_col;
# 16 "kernel.cu"
 int __cuda_local_var_20040_6_non_const_in2_rowlow;
# 17 "kernel.cu"
 int __cuda_local_var_20041_6_non_const_in2_collow;
# 18 "kernel.cu"
 int __cuda_local_var_20042_6_non_const_ic;
# 19 "kernel.cu"
 int __cuda_local_var_20043_6_non_const_jc;
# 20 "kernel.cu"
 int __cuda_local_var_20044_6_non_const_jp1;
# 21 "kernel.cu"
 int __cuda_local_var_20045_6_non_const_ja1;
# 21 "kernel.cu"
 int __cuda_local_var_20045_11_non_const_ja2;
# 22 "kernel.cu"
 int __cuda_local_var_20046_6_non_const_ip1;
# 23 "kernel.cu"
 int __cuda_local_var_20047_6_non_const_ia1;
# 23 "kernel.cu"
 int __cuda_local_var_20047_11_non_const_ia2;
# 24 "kernel.cu"
 int __cuda_local_var_20048_6_non_const_ja;
# 24 "kernel.cu"
 int __cuda_local_var_20048_10_non_const_jb;
# 25 "kernel.cu"
 int __cuda_local_var_20049_6_non_const_ia;
# 25 "kernel.cu"
 int __cuda_local_var_20049_10_non_const_ib;
# 26 "kernel.cu"
 float __cuda_local_var_20050_8_non_const_s;
# 27 "kernel.cu"
 int __cuda_local_var_20051_6_non_const_i;
# 28 "kernel.cu"
 int __cuda_local_var_20052_6_non_const_j;
# 29 "kernel.cu"
 int __cuda_local_var_20053_6_non_const_row;
# 30 "kernel.cu"
 int __cuda_local_var_20054_6_non_const_col;
# 31 "kernel.cu"
 int __cuda_local_var_20055_6_non_const_ori_row;
# 32 "kernel.cu"
 int __cuda_local_var_20056_6_non_const_ori_col;
# 33 "kernel.cu"
 int __cuda_local_var_20057_6_non_const_position;
# 34 "kernel.cu"
 float __cuda_local_var_20058_8_non_const_sum;
# 35 "kernel.cu"
 int __cuda_local_var_20059_6_non_const_pos_ori;
# 36 "kernel.cu"
 float __cuda_local_var_20060_8_non_const_temp;
# 37 "kernel.cu"
 float __cuda_local_var_20061_8_non_const_temp2;
# 38 "kernel.cu"
 int __cuda_local_var_20062_6_non_const_location;
# 39 "kernel.cu"
 int __cuda_local_var_20063_6_non_const_cent;
# 40 "kernel.cu"
 int __cuda_local_var_20064_6_non_const_tMask_row;
# 41 "kernel.cu"
 int __cuda_local_var_20065_6_non_const_tMask_col;
# 42 "kernel.cu"
 float __cuda_local_var_20066_8_non_const_largest_value_current;
# 43 "kernel.cu"
 float __cuda_local_var_20067_8_non_const_largest_value;
# 44 "kernel.cu"
 int __cuda_local_var_20068_6_non_const_largest_coordinate_current;
# 45 "kernel.cu"
 int __cuda_local_var_20069_6_non_const_largest_coordinate;
# 46 "kernel.cu"
 float __cuda_local_var_20070_8_non_const_fin_max_val;
# 47 "kernel.cu"
 int __cuda_local_var_20071_6_non_const_fin_max_coo;
# 48 "kernel.cu"
 int __cuda_local_var_20072_6_non_const_largest_row;
# 49 "kernel.cu"
 int __cuda_local_var_20073_6_non_const_largest_col;
# 50 "kernel.cu"
 int __cuda_local_var_20074_6_non_const_offset_row;
# 51 "kernel.cu"
 int __cuda_local_var_20075_6_non_const_offset_col;
# 52 "kernel.cu"
 __attribute__((shared)) float __cuda_local_var_20076_32_non_const_in_partial_sum[51];
# 53 "kernel.cu"
 __attribute__((shared)) float __cuda_local_var_20077_32_non_const_in_sqr_partial_sum[51];
# 54 "kernel.cu"
 __attribute__((shared)) float __cuda_local_var_20078_32_non_const_in_final_sum;
# 55 "kernel.cu"
 __attribute__((shared)) float __cuda_local_var_20079_32_non_const_in_sqr_final_sum;
# 56 "kernel.cu"
 float __cuda_local_var_20080_8_non_const_mean;
# 57 "kernel.cu"
 float __cuda_local_var_20081_8_non_const_mean_sqr;
# 58 "kernel.cu"
 float __cuda_local_var_20082_8_non_const_variance;
# 59 "kernel.cu"
 float __cuda_local_var_20083_8_non_const_deviation;
# 60 "kernel.cu"
 __attribute__((shared)) float __cuda_local_var_20084_32_non_const_denomT;
# 61 "kernel.cu"
 __attribute__((shared)) float __cuda_local_var_20085_32_non_const_par_max_val[131];
# 62 "kernel.cu"
 __attribute__((shared)) int __cuda_local_var_20086_30_non_const_par_max_coo[131];
# 63 "kernel.cu"
 int __cuda_local_var_20087_6_non_const_pointer;
# 64 "kernel.cu"
 __attribute__((shared)) float __cuda_local_var_20088_32_non_const_d_in_mod_temp[2601];
# 65 "kernel.cu"
 int __cuda_local_var_20089_6_non_const_ori_pointer;
# 66 "kernel.cu"
 int __cuda_local_var_20090_6_non_const_loc_pointer;
# 72 "kernel.cu"
 int __cuda_local_var_20096_6_non_const_bx;
# 73 "kernel.cu"
 int __cuda_local_var_20097_6_non_const_tx;
# 74 "kernel.cu"
 int __cuda_local_var_20098_6_non_const_ei_new;
# 42 "kernel.cu"
__cuda_local_var_20066_8_non_const_largest_value_current = (0.0F);
# 43 "kernel.cu"
__cuda_local_var_20067_8_non_const_largest_value = (0.0F);
# 44 "kernel.cu"
__cuda_local_var_20068_6_non_const_largest_coordinate_current = 0;
# 45 "kernel.cu"
__cuda_local_var_20069_6_non_const_largest_coordinate = 0;
# 46 "kernel.cu"
__cuda_local_var_20070_8_non_const_fin_max_val = (0.0F);
# 47 "kernel.cu"
__cuda_local_var_20071_6_non_const_fin_max_coo = 0;
# 72 "kernel.cu"
__cuda_local_var_20096_6_non_const_bx = ((int)(blockIdx.x));
# 73 "kernel.cu"
__cuda_local_var_20097_6_non_const_tx = ((int)(threadIdx.x));
# 83 "kernel.cu"
if ((d_common_change.frame_no) == 0)
# 83 "kernel.cu"
{
# 90 "kernel.cu"
__cuda_local_var_20037_9_non_const_d_in = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_T) + ((d_unique[__cuda_local_var_20096_6_non_const_bx]).in_pointer));
# 97 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 98 "kernel.cu"
if (__cuda_local_var_20098_6_non_const_ei_new == 0)
# 98 "kernel.cu"
{
# 101 "kernel.cu"
__cuda_local_var_20087_6_non_const_pointer = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no) * (d_common.no_frames)) + (d_common_change.frame_no));
# 102 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_tRowLoc)[__cuda_local_var_20087_6_non_const_pointer]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Row)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)]);
# 103 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_tColLoc)[__cuda_local_var_20087_6_non_const_pointer]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Col)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)]);
# 105 "kernel.cu"
}
# 112 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 113 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in_elem))
# 113 "kernel.cu"
{
# 116 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in_rows)) - 1);
# 117 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in_rows)) + 1) - 1);
# 118 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in_rows)) == 0)
# 118 "kernel.cu"
{
# 119 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in_rows) - 1);
# 120 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 121 "kernel.cu"
}
# 124 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = ((((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Row)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)]) - 25) + __cuda_local_var_20053_6_non_const_row) - 1);
# 125 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = ((((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Col)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)]) - 25) + __cuda_local_var_20054_6_non_const_col) - 1);
# 126 "kernel.cu"
__cuda_local_var_20089_6_non_const_ori_pointer = ((__cuda_local_var_20056_6_non_const_ori_col * (d_common.frame_rows)) + __cuda_local_var_20055_6_non_const_ori_row);
# 129 "kernel.cu"
(__cuda_local_var_20037_9_non_const_d_in[((__cuda_local_var_20054_6_non_const_col * (d_common.in_rows)) + __cuda_local_var_20053_6_non_const_row)]) = ((d_common_change.d_frame)[__cuda_local_var_20089_6_non_const_ori_pointer]);
# 132 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 134 "kernel.cu"
}
# 136 "kernel.cu"
}
# 145 "kernel.cu"
if ((d_common_change.frame_no) != 0)
# 145 "kernel.cu"
{
# 151 "kernel.cu"
__cuda_local_var_20040_6_non_const_in2_rowlow = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Row)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)]) - (d_common.sSize));
# 152 "kernel.cu"
__cuda_local_var_20041_6_non_const_in2_collow = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Col)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)]) - (d_common.sSize));
# 155 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 156 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_elem))
# 156 "kernel.cu"
{
# 159 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_rows)) - 1);
# 160 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in2_rows)) + 1) - 1);
# 161 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_rows)) == 0)
# 161 "kernel.cu"
{
# 162 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in2_rows) - 1);
# 163 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 164 "kernel.cu"
}
# 167 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = ((__cuda_local_var_20053_6_non_const_row + __cuda_local_var_20040_6_non_const_in2_rowlow) - 1);
# 168 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = ((__cuda_local_var_20054_6_non_const_col + __cuda_local_var_20041_6_non_const_in2_collow) - 1);
# 169 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2)[__cuda_local_var_20098_6_non_const_ei_new]) = ((d_common_change.d_frame)[((__cuda_local_var_20056_6_non_const_ori_col * (d_common.frame_rows)) + __cuda_local_var_20055_6_non_const_ori_row)]);
# 172 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 174 "kernel.cu"
}
# 180 "kernel.cu"
__syncthreads();
# 191 "kernel.cu"
__cuda_local_var_20037_9_non_const_d_in = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_T) + ((d_unique[__cuda_local_var_20096_6_non_const_bx]).in_pointer));
# 194 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 195 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in_elem))
# 195 "kernel.cu"
{
# 198 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in_rows)) - 1);
# 199 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in_rows)) + 1) - 1);
# 200 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in_rows)) == 0)
# 200 "kernel.cu"
{
# 201 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in_rows) - 1);
# 202 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 203 "kernel.cu"
}
# 206 "kernel.cu"
__cuda_local_var_20038_6_non_const_rot_row = (((d_common.in_rows) - 1) - __cuda_local_var_20053_6_non_const_row);
# 207 "kernel.cu"
__cuda_local_var_20039_6_non_const_rot_col = (((d_common.in_rows) - 1) - __cuda_local_var_20054_6_non_const_col);
# 208 "kernel.cu"
((__cuda_local_var_20088_32_non_const_d_in_mod_temp)[__cuda_local_var_20098_6_non_const_ei_new]) = (__cuda_local_var_20037_9_non_const_d_in[((__cuda_local_var_20039_6_non_const_rot_col * (d_common.in_rows)) + __cuda_local_var_20038_6_non_const_rot_row)]);
# 211 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 213 "kernel.cu"
}
# 219 "kernel.cu"
__syncthreads();
# 226 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 227 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.conv_elem))
# 227 "kernel.cu"
{
# 230 "kernel.cu"
__cuda_local_var_20042_6_non_const_ic = ((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.conv_rows));
# 231 "kernel.cu"
__cuda_local_var_20043_6_non_const_jc = (((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.conv_rows)) + 1);
# 232 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.conv_rows)) == 0)
# 232 "kernel.cu"
{
# 233 "kernel.cu"
__cuda_local_var_20042_6_non_const_ic = (d_common.conv_rows);
# 234 "kernel.cu"
__cuda_local_var_20043_6_non_const_jc = (__cuda_local_var_20043_6_non_const_jc - 1);
# 235 "kernel.cu"
}
# 238 "kernel.cu"
__cuda_local_var_20052_6_non_const_j = (__cuda_local_var_20043_6_non_const_jc + (d_common.joffset));
# 239 "kernel.cu"
__cuda_local_var_20044_6_non_const_jp1 = (__cuda_local_var_20052_6_non_const_j + 1);
# 240 "kernel.cu"
if ((d_common.in2_cols) < __cuda_local_var_20044_6_non_const_jp1)
# 240 "kernel.cu"
{
# 241 "kernel.cu"
__cuda_local_var_20045_6_non_const_ja1 = (__cuda_local_var_20044_6_non_const_jp1 - (d_common.in2_cols));
# 242 "kernel.cu"
}
# 243 "kernel.cu"
else
# 243 "kernel.cu"
{
# 244 "kernel.cu"
__cuda_local_var_20045_6_non_const_ja1 = 1;
# 245 "kernel.cu"
}
# 246 "kernel.cu"
if ((d_common.in_cols) < __cuda_local_var_20052_6_non_const_j)
# 246 "kernel.cu"
{
# 247 "kernel.cu"
__cuda_local_var_20045_11_non_const_ja2 = (d_common.in_cols);
# 248 "kernel.cu"
}
# 249 "kernel.cu"
else
# 249 "kernel.cu"
{
# 250 "kernel.cu"
__cuda_local_var_20045_11_non_const_ja2 = __cuda_local_var_20052_6_non_const_j;
# 251 "kernel.cu"
}
# 253 "kernel.cu"
__cuda_local_var_20051_6_non_const_i = (__cuda_local_var_20042_6_non_const_ic + (d_common.ioffset));
# 254 "kernel.cu"
__cuda_local_var_20046_6_non_const_ip1 = (__cuda_local_var_20051_6_non_const_i + 1);
# 256 "kernel.cu"
if ((d_common.in2_rows) < __cuda_local_var_20046_6_non_const_ip1)
# 256 "kernel.cu"
{
# 257 "kernel.cu"
__cuda_local_var_20047_6_non_const_ia1 = (__cuda_local_var_20046_6_non_const_ip1 - (d_common.in2_rows));
# 258 "kernel.cu"
}
# 259 "kernel.cu"
else
# 259 "kernel.cu"
{
# 260 "kernel.cu"
__cuda_local_var_20047_6_non_const_ia1 = 1;
# 261 "kernel.cu"
}
# 262 "kernel.cu"
if ((d_common.in_rows) < __cuda_local_var_20051_6_non_const_i)
# 262 "kernel.cu"
{
# 263 "kernel.cu"
__cuda_local_var_20047_11_non_const_ia2 = (d_common.in_rows);
# 264 "kernel.cu"
}
# 265 "kernel.cu"
else
# 265 "kernel.cu"
{
# 266 "kernel.cu"
__cuda_local_var_20047_11_non_const_ia2 = __cuda_local_var_20051_6_non_const_i;
# 267 "kernel.cu"
}
# 269 "kernel.cu"
__cuda_local_var_20050_8_non_const_s = (0.0F);
# 271 "kernel.cu"
for (__cuda_local_var_20048_6_non_const_ja = __cuda_local_var_20045_6_non_const_ja1; (__cuda_local_var_20048_6_non_const_ja <= __cuda_local_var_20045_11_non_const_ja2); __cuda_local_var_20048_6_non_const_ja++)
# 271 "kernel.cu"
{
# 272 "kernel.cu"
__cuda_local_var_20048_10_non_const_jb = (__cuda_local_var_20044_6_non_const_jp1 - __cuda_local_var_20048_6_non_const_ja);
# 273 "kernel.cu"
for (__cuda_local_var_20049_6_non_const_ia = __cuda_local_var_20047_6_non_const_ia1; (__cuda_local_var_20049_6_non_const_ia <= __cuda_local_var_20047_11_non_const_ia2); __cuda_local_var_20049_6_non_const_ia++)
# 273 "kernel.cu"
{
# 274 "kernel.cu"
__cuda_local_var_20049_10_non_const_ib = (__cuda_local_var_20046_6_non_const_ip1 - __cuda_local_var_20049_6_non_const_ia);
# 275 "kernel.cu"
__cuda_local_var_20050_8_non_const_s = (__cuda_local_var_20050_8_non_const_s + (((__cuda_local_var_20088_32_non_const_d_in_mod_temp)[((((d_common.in_rows) * (__cuda_local_var_20048_6_non_const_ja - 1)) + __cuda_local_var_20049_6_non_const_ia) - 1)]) * (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2)[((((d_common.in2_rows) * (__cuda_local_var_20048_10_non_const_jb - 1)) + __cuda_local_var_20049_10_non_const_ib) - 1)])));
# 276 "kernel.cu"
}
# 277 "kernel.cu"
}
# 280 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_conv)[__cuda_local_var_20098_6_non_const_ei_new]) = __cuda_local_var_20050_8_non_const_s;
# 283 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 285 "kernel.cu"
}
# 291 "kernel.cu"
__syncthreads();
# 306 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 307 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_pad_cumv_elem))
# 307 "kernel.cu"
{
# 310 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_pad_cumv_rows)) - 1);
# 311 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in2_pad_cumv_rows)) + 1) - 1);
# 312 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_pad_cumv_rows)) == 0)
# 312 "kernel.cu"
{
# 313 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in2_pad_cumv_rows) - 1);
# 314 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 315 "kernel.cu"
}
# 318 "kernel.cu"
if ((((__cuda_local_var_20053_6_non_const_row > ((d_common.in2_pad_add_rows) - 1)) && (__cuda_local_var_20053_6_non_const_row < ((d_common.in2_pad_add_rows) + (d_common.in2_rows)))) && (__cuda_local_var_20054_6_non_const_col > ((d_common.in2_pad_add_cols) - 1))) && (__cuda_local_var_20054_6_non_const_col < ((d_common.in2_pad_add_cols) + (d_common.in2_cols))))
# 321 "kernel.cu"
{
# 322 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = (__cuda_local_var_20053_6_non_const_row - (d_common.in2_pad_add_rows));
# 323 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = (__cuda_local_var_20054_6_non_const_col - (d_common.in2_pad_add_cols));
# 324 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[__cuda_local_var_20098_6_non_const_ei_new]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2)[((__cuda_local_var_20056_6_non_const_ori_col * (d_common.in2_rows)) + __cuda_local_var_20055_6_non_const_ori_row)]);
# 325 "kernel.cu"
}
# 326 "kernel.cu"
else
# 326 "kernel.cu"
{
# 327 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[__cuda_local_var_20098_6_non_const_ei_new]) = (0.0F);
# 328 "kernel.cu"
}
# 331 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 333 "kernel.cu"
}
# 339 "kernel.cu"
__syncthreads();
# 346 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 347 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_pad_cumv_cols))
# 347 "kernel.cu"
{
# 350 "kernel.cu"
__cuda_local_var_20059_6_non_const_pos_ori = (__cuda_local_var_20098_6_non_const_ei_new * (d_common.in2_pad_cumv_rows));
# 353 "kernel.cu"
__cuda_local_var_20058_8_non_const_sum = (0.0F);
# 356 "kernel.cu"
for (__cuda_local_var_20057_6_non_const_position = __cuda_local_var_20059_6_non_const_pos_ori; (__cuda_local_var_20057_6_non_const_position < (__cuda_local_var_20059_6_non_const_pos_ori + (d_common.in2_pad_cumv_rows))); __cuda_local_var_20057_6_non_const_position = (__cuda_local_var_20057_6_non_const_position + 1))
# 356 "kernel.cu"
{
# 357 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[__cuda_local_var_20057_6_non_const_position]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[__cuda_local_var_20057_6_non_const_position]) + __cuda_local_var_20058_8_non_const_sum);
# 358 "kernel.cu"
__cuda_local_var_20058_8_non_const_sum = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[__cuda_local_var_20057_6_non_const_position]);
# 359 "kernel.cu"
}
# 362 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 364 "kernel.cu"
}
# 370 "kernel.cu"
__syncthreads();
# 377 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 378 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_pad_cumv_sel_elem))
# 378 "kernel.cu"
{
# 381 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_pad_cumv_sel_rows)) - 1);
# 382 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in2_pad_cumv_sel_rows)) + 1) - 1);
# 383 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_pad_cumv_sel_rows)) == 0)
# 383 "kernel.cu"
{
# 384 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in2_pad_cumv_sel_rows) - 1);
# 385 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 386 "kernel.cu"
}
# 389 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = ((__cuda_local_var_20053_6_non_const_row + (d_common.in2_pad_cumv_sel_rowlow)) - 1);
# 390 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = ((__cuda_local_var_20054_6_non_const_col + (d_common.in2_pad_cumv_sel_collow)) - 1);
# 391 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv_sel)[__cuda_local_var_20098_6_non_const_ei_new]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[((__cuda_local_var_20056_6_non_const_ori_col * (d_common.in2_pad_cumv_rows)) + __cuda_local_var_20055_6_non_const_ori_row)]);
# 394 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 396 "kernel.cu"
}
# 402 "kernel.cu"
__syncthreads();
# 413 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 414 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub_cumh_elem))
# 414 "kernel.cu"
{
# 417 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_sub_cumh_rows)) - 1);
# 418 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in2_sub_cumh_rows)) + 1) - 1);
# 419 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_sub_cumh_rows)) == 0)
# 419 "kernel.cu"
{
# 420 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in2_sub_cumh_rows) - 1);
# 421 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 422 "kernel.cu"
}
# 425 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = ((__cuda_local_var_20053_6_non_const_row + (d_common.in2_pad_cumv_sel2_rowlow)) - 1);
# 426 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = ((__cuda_local_var_20054_6_non_const_col + (d_common.in2_pad_cumv_sel2_collow)) - 1);
# 427 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[__cuda_local_var_20098_6_non_const_ei_new]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[((__cuda_local_var_20056_6_non_const_ori_col * (d_common.in2_pad_cumv_rows)) + __cuda_local_var_20055_6_non_const_ori_row)]);
# 430 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 432 "kernel.cu"
}
# 438 "kernel.cu"
__syncthreads();
# 445 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 446 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub_cumh_elem))
# 446 "kernel.cu"
{
# 449 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[__cuda_local_var_20098_6_non_const_ei_new]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv_sel)[__cuda_local_var_20098_6_non_const_ei_new]) - (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[__cuda_local_var_20098_6_non_const_ei_new]));
# 452 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 454 "kernel.cu"
}
# 460 "kernel.cu"
__syncthreads();
# 467 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 468 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub_cumh_rows))
# 468 "kernel.cu"
{
# 471 "kernel.cu"
__cuda_local_var_20059_6_non_const_pos_ori = __cuda_local_var_20098_6_non_const_ei_new;
# 474 "kernel.cu"
__cuda_local_var_20058_8_non_const_sum = (0.0F);
# 477 "kernel.cu"
for (__cuda_local_var_20057_6_non_const_position = __cuda_local_var_20059_6_non_const_pos_ori; (__cuda_local_var_20057_6_non_const_position < (__cuda_local_var_20059_6_non_const_pos_ori + (d_common.in2_sub_cumh_elem))); __cuda_local_var_20057_6_non_const_position = (__cuda_local_var_20057_6_non_const_position + (d_common.in2_sub_cumh_rows)))
# 477 "kernel.cu"
{
# 478 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[__cuda_local_var_20057_6_non_const_position]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[__cuda_local_var_20057_6_non_const_position]) + __cuda_local_var_20058_8_non_const_sum);
# 479 "kernel.cu"
__cuda_local_var_20058_8_non_const_sum = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[__cuda_local_var_20057_6_non_const_position]);
# 480 "kernel.cu"
}
# 483 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 485 "kernel.cu"
}
# 491 "kernel.cu"
__syncthreads();
# 498 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 499 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub_cumh_sel_elem))
# 499 "kernel.cu"
{
# 502 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_sub_cumh_sel_rows)) - 1);
# 503 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in2_sub_cumh_sel_rows)) + 1) - 1);
# 504 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_sub_cumh_sel_rows)) == 0)
# 504 "kernel.cu"
{
# 505 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in2_sub_cumh_sel_rows) - 1);
# 506 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 507 "kernel.cu"
}
# 510 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = ((__cuda_local_var_20053_6_non_const_row + (d_common.in2_sub_cumh_sel_rowlow)) - 1);
# 511 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = ((__cuda_local_var_20054_6_non_const_col + (d_common.in2_sub_cumh_sel_collow)) - 1);
# 512 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh_sel)[__cuda_local_var_20098_6_non_const_ei_new]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[((__cuda_local_var_20056_6_non_const_ori_col * (d_common.in2_sub_cumh_rows)) + __cuda_local_var_20055_6_non_const_ori_row)]);
# 515 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 517 "kernel.cu"
}
# 523 "kernel.cu"
__syncthreads();
# 534 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 535 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub2_elem))
# 535 "kernel.cu"
{
# 538 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_sub2_rows)) - 1);
# 539 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in2_sub2_rows)) + 1) - 1);
# 540 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_sub2_rows)) == 0)
# 540 "kernel.cu"
{
# 541 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in2_sub2_rows) - 1);
# 542 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 543 "kernel.cu"
}
# 546 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = ((__cuda_local_var_20053_6_non_const_row + (d_common.in2_sub_cumh_sel2_rowlow)) - 1);
# 547 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = ((__cuda_local_var_20054_6_non_const_col + (d_common.in2_sub_cumh_sel2_collow)) - 1);
# 548 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub2)[__cuda_local_var_20098_6_non_const_ei_new]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[((__cuda_local_var_20056_6_non_const_ori_col * (d_common.in2_sub_cumh_rows)) + __cuda_local_var_20055_6_non_const_ori_row)]);
# 551 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 553 "kernel.cu"
}
# 559 "kernel.cu"
__syncthreads();
# 566 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 567 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub2_elem))
# 567 "kernel.cu"
{
# 570 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub2)[__cuda_local_var_20098_6_non_const_ei_new]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh_sel)[__cuda_local_var_20098_6_non_const_ei_new]) - (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub2)[__cuda_local_var_20098_6_non_const_ei_new]));
# 573 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 575 "kernel.cu"
}
# 581 "kernel.cu"
__syncthreads();
# 592 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 593 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sqr_elem))
# 593 "kernel.cu"
{
# 595 "kernel.cu"
__cuda_local_var_20060_8_non_const_temp = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2)[__cuda_local_var_20098_6_non_const_ei_new]);
# 596 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sqr)[__cuda_local_var_20098_6_non_const_ei_new]) = (__cuda_local_var_20060_8_non_const_temp * __cuda_local_var_20060_8_non_const_temp);
# 599 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 601 "kernel.cu"
}
# 607 "kernel.cu"
__syncthreads();
# 618 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 619 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_pad_cumv_elem))
# 619 "kernel.cu"
{
# 622 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_pad_cumv_rows)) - 1);
# 623 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in2_pad_cumv_rows)) + 1) - 1);
# 624 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_pad_cumv_rows)) == 0)
# 624 "kernel.cu"
{
# 625 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in2_pad_cumv_rows) - 1);
# 626 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 627 "kernel.cu"
}
# 630 "kernel.cu"
if ((((__cuda_local_var_20053_6_non_const_row > ((d_common.in2_pad_add_rows) - 1)) && (__cuda_local_var_20053_6_non_const_row < ((d_common.in2_pad_add_rows) + (d_common.in2_sqr_rows)))) && (__cuda_local_var_20054_6_non_const_col > ((d_common.in2_pad_add_cols) - 1))) && (__cuda_local_var_20054_6_non_const_col < ((d_common.in2_pad_add_cols) + (d_common.in2_sqr_cols))))
# 633 "kernel.cu"
{
# 634 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = (__cuda_local_var_20053_6_non_const_row - (d_common.in2_pad_add_rows));
# 635 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = (__cuda_local_var_20054_6_non_const_col - (d_common.in2_pad_add_cols));
# 636 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[__cuda_local_var_20098_6_non_const_ei_new]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sqr)[((__cuda_local_var_20056_6_non_const_ori_col * (d_common.in2_sqr_rows)) + __cuda_local_var_20055_6_non_const_ori_row)]);
# 637 "kernel.cu"
}
# 638 "kernel.cu"
else
# 638 "kernel.cu"
{
# 639 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[__cuda_local_var_20098_6_non_const_ei_new]) = (0.0F);
# 640 "kernel.cu"
}
# 643 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 645 "kernel.cu"
}
# 651 "kernel.cu"
__syncthreads();
# 658 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 659 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_pad_cumv_cols))
# 659 "kernel.cu"
{
# 662 "kernel.cu"
__cuda_local_var_20059_6_non_const_pos_ori = (__cuda_local_var_20098_6_non_const_ei_new * (d_common.in2_pad_cumv_rows));
# 665 "kernel.cu"
__cuda_local_var_20058_8_non_const_sum = (0.0F);
# 668 "kernel.cu"
for (__cuda_local_var_20057_6_non_const_position = __cuda_local_var_20059_6_non_const_pos_ori; (__cuda_local_var_20057_6_non_const_position < (__cuda_local_var_20059_6_non_const_pos_ori + (d_common.in2_pad_cumv_rows))); __cuda_local_var_20057_6_non_const_position = (__cuda_local_var_20057_6_non_const_position + 1))
# 668 "kernel.cu"
{
# 669 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[__cuda_local_var_20057_6_non_const_position]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[__cuda_local_var_20057_6_non_const_position]) + __cuda_local_var_20058_8_non_const_sum);
# 670 "kernel.cu"
__cuda_local_var_20058_8_non_const_sum = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[__cuda_local_var_20057_6_non_const_position]);
# 671 "kernel.cu"
}
# 674 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 676 "kernel.cu"
}
# 682 "kernel.cu"
__syncthreads();
# 689 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 690 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_pad_cumv_sel_elem))
# 690 "kernel.cu"
{
# 693 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_pad_cumv_sel_rows)) - 1);
# 694 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in2_pad_cumv_sel_rows)) + 1) - 1);
# 695 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_pad_cumv_sel_rows)) == 0)
# 695 "kernel.cu"
{
# 696 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in2_pad_cumv_sel_rows) - 1);
# 697 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 698 "kernel.cu"
}
# 701 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = ((__cuda_local_var_20053_6_non_const_row + (d_common.in2_pad_cumv_sel_rowlow)) - 1);
# 702 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = ((__cuda_local_var_20054_6_non_const_col + (d_common.in2_pad_cumv_sel_collow)) - 1);
# 703 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv_sel)[__cuda_local_var_20098_6_non_const_ei_new]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[((__cuda_local_var_20056_6_non_const_ori_col * (d_common.in2_pad_cumv_rows)) + __cuda_local_var_20055_6_non_const_ori_row)]);
# 706 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 708 "kernel.cu"
}
# 714 "kernel.cu"
__syncthreads();
# 725 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 726 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub_cumh_elem))
# 726 "kernel.cu"
{
# 729 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_sub_cumh_rows)) - 1);
# 730 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in2_sub_cumh_rows)) + 1) - 1);
# 731 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_sub_cumh_rows)) == 0)
# 731 "kernel.cu"
{
# 732 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in2_sub_cumh_rows) - 1);
# 733 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 734 "kernel.cu"
}
# 737 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = ((__cuda_local_var_20053_6_non_const_row + (d_common.in2_pad_cumv_sel2_rowlow)) - 1);
# 738 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = ((__cuda_local_var_20054_6_non_const_col + (d_common.in2_pad_cumv_sel2_collow)) - 1);
# 739 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[__cuda_local_var_20098_6_non_const_ei_new]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv)[((__cuda_local_var_20056_6_non_const_ori_col * (d_common.in2_pad_cumv_rows)) + __cuda_local_var_20055_6_non_const_ori_row)]);
# 742 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 744 "kernel.cu"
}
# 750 "kernel.cu"
__syncthreads();
# 757 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 758 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub_cumh_elem))
# 758 "kernel.cu"
{
# 761 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[__cuda_local_var_20098_6_non_const_ei_new]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_pad_cumv_sel)[__cuda_local_var_20098_6_non_const_ei_new]) - (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[__cuda_local_var_20098_6_non_const_ei_new]));
# 764 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 766 "kernel.cu"
}
# 773 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 774 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub_cumh_rows))
# 774 "kernel.cu"
{
# 777 "kernel.cu"
__cuda_local_var_20059_6_non_const_pos_ori = __cuda_local_var_20098_6_non_const_ei_new;
# 780 "kernel.cu"
__cuda_local_var_20058_8_non_const_sum = (0.0F);
# 783 "kernel.cu"
for (__cuda_local_var_20057_6_non_const_position = __cuda_local_var_20059_6_non_const_pos_ori; (__cuda_local_var_20057_6_non_const_position < (__cuda_local_var_20059_6_non_const_pos_ori + (d_common.in2_sub_cumh_elem))); __cuda_local_var_20057_6_non_const_position = (__cuda_local_var_20057_6_non_const_position + (d_common.in2_sub_cumh_rows)))
# 783 "kernel.cu"
{
# 784 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[__cuda_local_var_20057_6_non_const_position]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[__cuda_local_var_20057_6_non_const_position]) + __cuda_local_var_20058_8_non_const_sum);
# 785 "kernel.cu"
__cuda_local_var_20058_8_non_const_sum = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[__cuda_local_var_20057_6_non_const_position]);
# 786 "kernel.cu"
}
# 789 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 791 "kernel.cu"
}
# 797 "kernel.cu"
__syncthreads();
# 804 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 805 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub_cumh_sel_elem))
# 805 "kernel.cu"
{
# 808 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_sub_cumh_sel_rows)) - 1);
# 809 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in2_sub_cumh_sel_rows)) + 1) - 1);
# 810 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_sub_cumh_sel_rows)) == 0)
# 810 "kernel.cu"
{
# 811 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in2_sub_cumh_sel_rows) - 1);
# 812 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 813 "kernel.cu"
}
# 816 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = ((__cuda_local_var_20053_6_non_const_row + (d_common.in2_sub_cumh_sel_rowlow)) - 1);
# 817 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = ((__cuda_local_var_20054_6_non_const_col + (d_common.in2_sub_cumh_sel_collow)) - 1);
# 818 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh_sel)[__cuda_local_var_20098_6_non_const_ei_new]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[((__cuda_local_var_20056_6_non_const_ori_col * (d_common.in2_sub_cumh_rows)) + __cuda_local_var_20055_6_non_const_ori_row)]);
# 821 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 823 "kernel.cu"
}
# 829 "kernel.cu"
__syncthreads();
# 840 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 841 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub2_elem))
# 841 "kernel.cu"
{
# 844 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_sub2_rows)) - 1);
# 845 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in2_sub2_rows)) + 1) - 1);
# 846 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in2_sub2_rows)) == 0)
# 846 "kernel.cu"
{
# 847 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in2_sub2_rows) - 1);
# 848 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 849 "kernel.cu"
}
# 852 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = ((__cuda_local_var_20053_6_non_const_row + (d_common.in2_sub_cumh_sel2_rowlow)) - 1);
# 853 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = ((__cuda_local_var_20054_6_non_const_col + (d_common.in2_sub_cumh_sel2_collow)) - 1);
# 854 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sqr_sub2)[__cuda_local_var_20098_6_non_const_ei_new]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh)[((__cuda_local_var_20056_6_non_const_ori_col * (d_common.in2_sub_cumh_rows)) + __cuda_local_var_20055_6_non_const_ori_row)]);
# 857 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 859 "kernel.cu"
}
# 865 "kernel.cu"
__syncthreads();
# 872 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 873 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub2_elem))
# 873 "kernel.cu"
{
# 876 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sqr_sub2)[__cuda_local_var_20098_6_non_const_ei_new]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub_cumh_sel)[__cuda_local_var_20098_6_non_const_ei_new]) - (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sqr_sub2)[__cuda_local_var_20098_6_non_const_ei_new]));
# 879 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 881 "kernel.cu"
}
# 887 "kernel.cu"
__syncthreads();
# 898 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 899 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub2_elem))
# 899 "kernel.cu"
{
# 901 "kernel.cu"
__cuda_local_var_20060_8_non_const_temp = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub2)[__cuda_local_var_20098_6_non_const_ei_new]);
# 902 "kernel.cu"
__cuda_local_var_20061_8_non_const_temp2 = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sqr_sub2)[__cuda_local_var_20098_6_non_const_ei_new]) - ( fdividef((__cuda_local_var_20060_8_non_const_temp * __cuda_local_var_20060_8_non_const_temp) , ((float)(d_common.in_elem)))));
# 903 "kernel.cu"
if (__cuda_local_var_20061_8_non_const_temp2 < (0.0F))
# 903 "kernel.cu"
{
# 904 "kernel.cu"
__cuda_local_var_20061_8_non_const_temp2 = (0.0F);
# 905 "kernel.cu"
}
# 906 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sqr_sub2)[__cuda_local_var_20098_6_non_const_ei_new]) = (sqrtf(__cuda_local_var_20061_8_non_const_temp2));
# 910 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 912 "kernel.cu"
}
# 918 "kernel.cu"
__syncthreads();
# 925 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 926 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in_sqr_elem))
# 926 "kernel.cu"
{
# 928 "kernel.cu"
__cuda_local_var_20060_8_non_const_temp = (__cuda_local_var_20037_9_non_const_d_in[__cuda_local_var_20098_6_non_const_ei_new]);
# 929 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in_sqr)[__cuda_local_var_20098_6_non_const_ei_new]) = (__cuda_local_var_20060_8_non_const_temp * __cuda_local_var_20060_8_non_const_temp);
# 932 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 934 "kernel.cu"
}
# 940 "kernel.cu"
__syncthreads();
# 947 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 948 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in_cols))
# 948 "kernel.cu"
{
# 950 "kernel.cu"
__cuda_local_var_20058_8_non_const_sum = (0.0F);
# 951 "kernel.cu"
for (__cuda_local_var_20051_6_non_const_i = 0; (__cuda_local_var_20051_6_non_const_i < (d_common.in_rows)); __cuda_local_var_20051_6_non_const_i++)
# 951 "kernel.cu"
{
# 953 "kernel.cu"
__cuda_local_var_20058_8_non_const_sum = (__cuda_local_var_20058_8_non_const_sum + (__cuda_local_var_20037_9_non_const_d_in[((__cuda_local_var_20098_6_non_const_ei_new * (d_common.in_rows)) + __cuda_local_var_20051_6_non_const_i)]));
# 955 "kernel.cu"
}
# 956 "kernel.cu"
((__cuda_local_var_20076_32_non_const_in_partial_sum)[__cuda_local_var_20098_6_non_const_ei_new]) = __cuda_local_var_20058_8_non_const_sum;
# 959 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 961 "kernel.cu"
}
# 967 "kernel.cu"
__syncthreads();
# 973 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 974 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in_sqr_rows))
# 974 "kernel.cu"
{
# 976 "kernel.cu"
__cuda_local_var_20058_8_non_const_sum = (0.0F);
# 977 "kernel.cu"
for (__cuda_local_var_20051_6_non_const_i = 0; (__cuda_local_var_20051_6_non_const_i < (d_common.in_sqr_cols)); __cuda_local_var_20051_6_non_const_i++)
# 977 "kernel.cu"
{
# 979 "kernel.cu"
__cuda_local_var_20058_8_non_const_sum = (__cuda_local_var_20058_8_non_const_sum + (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in_sqr)[(__cuda_local_var_20098_6_non_const_ei_new + ((d_common.in_sqr_rows) * __cuda_local_var_20051_6_non_const_i))]));
# 981 "kernel.cu"
}
# 982 "kernel.cu"
((__cuda_local_var_20077_32_non_const_in_sqr_partial_sum)[__cuda_local_var_20098_6_non_const_ei_new]) = __cuda_local_var_20058_8_non_const_sum;
# 985 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 987 "kernel.cu"
}
# 993 "kernel.cu"
__syncthreads();
# 999 "kernel.cu"
if (__cuda_local_var_20097_6_non_const_tx == 0)
# 999 "kernel.cu"
{
# 1001 "kernel.cu"
__cuda_local_var_20078_32_non_const_in_final_sum = (0.0F);
# 1002 "kernel.cu"
for (__cuda_local_var_20051_6_non_const_i = 0; (__cuda_local_var_20051_6_non_const_i < (d_common.in_cols)); __cuda_local_var_20051_6_non_const_i++)
# 1002 "kernel.cu"
{
# 1003 "kernel.cu"
__cuda_local_var_20078_32_non_const_in_final_sum = (__cuda_local_var_20078_32_non_const_in_final_sum + ((__cuda_local_var_20076_32_non_const_in_partial_sum)[__cuda_local_var_20051_6_non_const_i]));
# 1004 "kernel.cu"
}
# 1006 "kernel.cu"
} else {
# 1006 "kernel.cu"
if (__cuda_local_var_20097_6_non_const_tx == 1)
# 1006 "kernel.cu"
{
# 1008 "kernel.cu"
__cuda_local_var_20079_32_non_const_in_sqr_final_sum = (0.0F);
# 1009 "kernel.cu"
for (__cuda_local_var_20051_6_non_const_i = 0; (__cuda_local_var_20051_6_non_const_i < (d_common.in_sqr_cols)); __cuda_local_var_20051_6_non_const_i++)
# 1009 "kernel.cu"
{
# 1010 "kernel.cu"
__cuda_local_var_20079_32_non_const_in_sqr_final_sum = (__cuda_local_var_20079_32_non_const_in_sqr_final_sum + ((__cuda_local_var_20077_32_non_const_in_sqr_partial_sum)[__cuda_local_var_20051_6_non_const_i]));
# 1011 "kernel.cu"
}
# 1013 "kernel.cu"
} }
# 1019 "kernel.cu"
__syncthreads();
# 1025 "kernel.cu"
if (__cuda_local_var_20097_6_non_const_tx == 0)
# 1025 "kernel.cu"
{ float __T22;
# 1027 "kernel.cu"
__cuda_local_var_20080_8_non_const_mean = ( fdividef(__cuda_local_var_20078_32_non_const_in_final_sum , ((float)(d_common.in_elem))));
# 1028 "kernel.cu"
__cuda_local_var_20081_8_non_const_mean_sqr = (__cuda_local_var_20080_8_non_const_mean * __cuda_local_var_20080_8_non_const_mean);
# 1029 "kernel.cu"
__cuda_local_var_20082_8_non_const_variance = (( fdividef(__cuda_local_var_20079_32_non_const_in_sqr_final_sum , ((float)(d_common.in_elem)))) - __cuda_local_var_20081_8_non_const_mean_sqr);
# 1030 "kernel.cu"
__cuda_local_var_20083_8_non_const_deviation = (sqrtf(__cuda_local_var_20082_8_non_const_variance));
# 1032 "kernel.cu"
__cuda_local_var_20084_32_non_const_denomT = (((__T22 = ((float)((d_common.in_elem) - 1))) , (sqrtf(__T22))) * __cuda_local_var_20083_8_non_const_deviation);
# 1034 "kernel.cu"
}
# 1040 "kernel.cu"
__syncthreads();
# 1047 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 1048 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub2_elem))
# 1048 "kernel.cu"
{
# 1050 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sqr_sub2)[__cuda_local_var_20098_6_non_const_ei_new]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sqr_sub2)[__cuda_local_var_20098_6_non_const_ei_new]) * __cuda_local_var_20084_32_non_const_denomT);
# 1053 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 1055 "kernel.cu"
}
# 1061 "kernel.cu"
__syncthreads();
# 1068 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 1069 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.conv_elem))
# 1069 "kernel.cu"
{
# 1071 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_conv)[__cuda_local_var_20098_6_non_const_ei_new]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_conv)[__cuda_local_var_20098_6_non_const_ei_new]) - ( fdividef(((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sub2)[__cuda_local_var_20098_6_non_const_ei_new]) * __cuda_local_var_20078_32_non_const_in_final_sum) , ((float)(d_common.in_elem)))));
# 1074 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 1076 "kernel.cu"
}
# 1082 "kernel.cu"
__syncthreads();
# 1089 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 1090 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in2_sub2_elem))
# 1090 "kernel.cu"
{
# 1092 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sqr_sub2)[__cuda_local_var_20098_6_non_const_ei_new]) = ( fdividef((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_conv)[__cuda_local_var_20098_6_non_const_ei_new]) , (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sqr_sub2)[__cuda_local_var_20098_6_non_const_ei_new])));
# 1095 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 1097 "kernel.cu"
}
# 1103 "kernel.cu"
__syncthreads();
# 1109 "kernel.cu"
__cuda_local_var_20063_6_non_const_cent = (((d_common.sSize) + (d_common.tSize)) + 1);
# 1110 "kernel.cu"
if ((d_common_change.frame_no) == 0)
# 1110 "kernel.cu"
{
# 1111 "kernel.cu"
__cuda_local_var_20064_6_non_const_tMask_row = (((__cuda_local_var_20063_6_non_const_cent + (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Row)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)])) - (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Row)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)])) - 1);
# 1112 "kernel.cu"
__cuda_local_var_20065_6_non_const_tMask_col = (((__cuda_local_var_20063_6_non_const_cent + (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Col)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)])) - (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Col)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)])) - 1);
# 1113 "kernel.cu"
}
# 1114 "kernel.cu"
else
# 1114 "kernel.cu"
{
# 1115 "kernel.cu"
__cuda_local_var_20087_6_non_const_pointer = (((d_common_change.frame_no) - 1) + (((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no) * (d_common.no_frames)));
# 1116 "kernel.cu"
__cuda_local_var_20064_6_non_const_tMask_row = (((__cuda_local_var_20063_6_non_const_cent + (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_tRowLoc)[__cuda_local_var_20087_6_non_const_pointer])) - (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Row)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)])) - 1);
# 1117 "kernel.cu"
__cuda_local_var_20065_6_non_const_tMask_col = (((__cuda_local_var_20063_6_non_const_cent + (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_tColLoc)[__cuda_local_var_20087_6_non_const_pointer])) - (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Col)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)])) - 1);
# 1118 "kernel.cu"
}
# 1122 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 1123 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.tMask_elem))
# 1123 "kernel.cu"
{
# 1125 "kernel.cu"
__cuda_local_var_20062_6_non_const_location = ((__cuda_local_var_20065_6_non_const_tMask_col * (d_common.tMask_rows)) + __cuda_local_var_20064_6_non_const_tMask_row);
# 1127 "kernel.cu"
if (__cuda_local_var_20098_6_non_const_ei_new == __cuda_local_var_20062_6_non_const_location)
# 1127 "kernel.cu"
{
# 1128 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_tMask)[__cuda_local_var_20098_6_non_const_ei_new]) = (1.0F);
# 1129 "kernel.cu"
}
# 1130 "kernel.cu"
else
# 1130 "kernel.cu"
{
# 1131 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_tMask)[__cuda_local_var_20098_6_non_const_ei_new]) = (0.0F);
# 1132 "kernel.cu"
}
# 1135 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 1137 "kernel.cu"
}
# 1143 "kernel.cu"
__syncthreads();
# 1150 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 1151 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.mask_conv_elem))
# 1151 "kernel.cu"
{
# 1154 "kernel.cu"
__cuda_local_var_20042_6_non_const_ic = ((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.mask_conv_rows));
# 1155 "kernel.cu"
__cuda_local_var_20043_6_non_const_jc = (((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.mask_conv_rows)) + 1);
# 1156 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.mask_conv_rows)) == 0)
# 1156 "kernel.cu"
{
# 1157 "kernel.cu"
__cuda_local_var_20042_6_non_const_ic = (d_common.mask_conv_rows);
# 1158 "kernel.cu"
__cuda_local_var_20043_6_non_const_jc = (__cuda_local_var_20043_6_non_const_jc - 1);
# 1159 "kernel.cu"
}
# 1162 "kernel.cu"
__cuda_local_var_20052_6_non_const_j = (__cuda_local_var_20043_6_non_const_jc + (d_common.mask_conv_joffset));
# 1163 "kernel.cu"
__cuda_local_var_20044_6_non_const_jp1 = (__cuda_local_var_20052_6_non_const_j + 1);
# 1164 "kernel.cu"
if ((d_common.mask_cols) < __cuda_local_var_20044_6_non_const_jp1)
# 1164 "kernel.cu"
{
# 1165 "kernel.cu"
__cuda_local_var_20045_6_non_const_ja1 = (__cuda_local_var_20044_6_non_const_jp1 - (d_common.mask_cols));
# 1166 "kernel.cu"
}
# 1167 "kernel.cu"
else
# 1167 "kernel.cu"
{
# 1168 "kernel.cu"
__cuda_local_var_20045_6_non_const_ja1 = 1;
# 1169 "kernel.cu"
}
# 1170 "kernel.cu"
if ((d_common.tMask_cols) < __cuda_local_var_20052_6_non_const_j)
# 1170 "kernel.cu"
{
# 1171 "kernel.cu"
__cuda_local_var_20045_11_non_const_ja2 = (d_common.tMask_cols);
# 1172 "kernel.cu"
}
# 1173 "kernel.cu"
else
# 1173 "kernel.cu"
{
# 1174 "kernel.cu"
__cuda_local_var_20045_11_non_const_ja2 = __cuda_local_var_20052_6_non_const_j;
# 1175 "kernel.cu"
}
# 1177 "kernel.cu"
__cuda_local_var_20051_6_non_const_i = (__cuda_local_var_20042_6_non_const_ic + (d_common.mask_conv_ioffset));
# 1178 "kernel.cu"
__cuda_local_var_20046_6_non_const_ip1 = (__cuda_local_var_20051_6_non_const_i + 1);
# 1180 "kernel.cu"
if ((d_common.mask_rows) < __cuda_local_var_20046_6_non_const_ip1)
# 1180 "kernel.cu"
{
# 1181 "kernel.cu"
__cuda_local_var_20047_6_non_const_ia1 = (__cuda_local_var_20046_6_non_const_ip1 - (d_common.mask_rows));
# 1182 "kernel.cu"
}
# 1183 "kernel.cu"
else
# 1183 "kernel.cu"
{
# 1184 "kernel.cu"
__cuda_local_var_20047_6_non_const_ia1 = 1;
# 1185 "kernel.cu"
}
# 1186 "kernel.cu"
if ((d_common.tMask_rows) < __cuda_local_var_20051_6_non_const_i)
# 1186 "kernel.cu"
{
# 1187 "kernel.cu"
__cuda_local_var_20047_11_non_const_ia2 = (d_common.tMask_rows);
# 1188 "kernel.cu"
}
# 1189 "kernel.cu"
else
# 1189 "kernel.cu"
{
# 1190 "kernel.cu"
__cuda_local_var_20047_11_non_const_ia2 = __cuda_local_var_20051_6_non_const_i;
# 1191 "kernel.cu"
}
# 1193 "kernel.cu"
__cuda_local_var_20050_8_non_const_s = (0.0F);
# 1195 "kernel.cu"
for (__cuda_local_var_20048_6_non_const_ja = __cuda_local_var_20045_6_non_const_ja1; (__cuda_local_var_20048_6_non_const_ja <= __cuda_local_var_20045_11_non_const_ja2); __cuda_local_var_20048_6_non_const_ja++)
# 1195 "kernel.cu"
{
# 1196 "kernel.cu"
__cuda_local_var_20048_10_non_const_jb = (__cuda_local_var_20044_6_non_const_jp1 - __cuda_local_var_20048_6_non_const_ja);
# 1197 "kernel.cu"
for (__cuda_local_var_20049_6_non_const_ia = __cuda_local_var_20047_6_non_const_ia1; (__cuda_local_var_20049_6_non_const_ia <= __cuda_local_var_20047_11_non_const_ia2); __cuda_local_var_20049_6_non_const_ia++)
# 1197 "kernel.cu"
{
# 1198 "kernel.cu"
__cuda_local_var_20049_10_non_const_ib = (__cuda_local_var_20046_6_non_const_ip1 - __cuda_local_var_20049_6_non_const_ia);
# 1199 "kernel.cu"
__cuda_local_var_20050_8_non_const_s = (__cuda_local_var_20050_8_non_const_s + ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_tMask)[((((d_common.tMask_rows) * (__cuda_local_var_20048_6_non_const_ja - 1)) + __cuda_local_var_20049_6_non_const_ia) - 1)]) * (1.0F)));
# 1200 "kernel.cu"
}
# 1201 "kernel.cu"
}
# 1204 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_mask_conv)[__cuda_local_var_20098_6_non_const_ei_new]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_in2_sqr_sub2)[__cuda_local_var_20098_6_non_const_ei_new]) * __cuda_local_var_20050_8_non_const_s);
# 1207 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 1209 "kernel.cu"
}
# 1215 "kernel.cu"
__syncthreads();
# 1225 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 1226 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.mask_conv_rows))
# 1226 "kernel.cu"
{
# 1228 "kernel.cu"
for (__cuda_local_var_20051_6_non_const_i = 0; (__cuda_local_var_20051_6_non_const_i < (d_common.mask_conv_cols)); __cuda_local_var_20051_6_non_const_i++)
# 1228 "kernel.cu"
{
# 1229 "kernel.cu"
__cuda_local_var_20068_6_non_const_largest_coordinate_current = ((__cuda_local_var_20098_6_non_const_ei_new * (d_common.mask_conv_rows)) + __cuda_local_var_20051_6_non_const_i);
# 1230 "kernel.cu"
__cuda_local_var_20066_8_non_const_largest_value_current = ((__T21 = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_mask_conv)[__cuda_local_var_20068_6_non_const_largest_coordinate_current])) , (fabsf(__T21)));
# 1231 "kernel.cu"
if (__cuda_local_var_20066_8_non_const_largest_value_current > __cuda_local_var_20067_8_non_const_largest_value)
# 1231 "kernel.cu"
{
# 1232 "kernel.cu"
__cuda_local_var_20069_6_non_const_largest_coordinate = __cuda_local_var_20068_6_non_const_largest_coordinate_current;
# 1233 "kernel.cu"
__cuda_local_var_20067_8_non_const_largest_value = __cuda_local_var_20066_8_non_const_largest_value_current;
# 1234 "kernel.cu"
}
# 1235 "kernel.cu"
}
# 1236 "kernel.cu"
((__cuda_local_var_20086_30_non_const_par_max_coo)[__cuda_local_var_20098_6_non_const_ei_new]) = __cuda_local_var_20069_6_non_const_largest_coordinate;
# 1237 "kernel.cu"
((__cuda_local_var_20085_32_non_const_par_max_val)[__cuda_local_var_20098_6_non_const_ei_new]) = __cuda_local_var_20067_8_non_const_largest_value;
# 1240 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 1242 "kernel.cu"
}
# 1248 "kernel.cu"
__syncthreads();
# 1254 "kernel.cu"
if (__cuda_local_var_20097_6_non_const_tx == 0)
# 1254 "kernel.cu"
{
# 1256 "kernel.cu"
for (__cuda_local_var_20051_6_non_const_i = 0; (__cuda_local_var_20051_6_non_const_i < (d_common.mask_conv_rows)); __cuda_local_var_20051_6_non_const_i++)
# 1256 "kernel.cu"
{
# 1257 "kernel.cu"
if (((__cuda_local_var_20085_32_non_const_par_max_val)[__cuda_local_var_20051_6_non_const_i]) > __cuda_local_var_20070_8_non_const_fin_max_val)
# 1257 "kernel.cu"
{
# 1258 "kernel.cu"
__cuda_local_var_20070_8_non_const_fin_max_val = ((__cuda_local_var_20085_32_non_const_par_max_val)[__cuda_local_var_20051_6_non_const_i]);
# 1259 "kernel.cu"
__cuda_local_var_20071_6_non_const_fin_max_coo = ((__cuda_local_var_20086_30_non_const_par_max_coo)[__cuda_local_var_20051_6_non_const_i]);
# 1260 "kernel.cu"
}
# 1261 "kernel.cu"
}
# 1264 "kernel.cu"
__cuda_local_var_20072_6_non_const_largest_row = (((__cuda_local_var_20071_6_non_const_fin_max_coo + 1) % (d_common.mask_conv_rows)) - 1);
# 1265 "kernel.cu"
__cuda_local_var_20073_6_non_const_largest_col = ((__cuda_local_var_20071_6_non_const_fin_max_coo + 1) / (d_common.mask_conv_rows));
# 1266 "kernel.cu"
if (((__cuda_local_var_20071_6_non_const_fin_max_coo + 1) % (d_common.mask_conv_rows)) == 0)
# 1266 "kernel.cu"
{
# 1267 "kernel.cu"
__cuda_local_var_20072_6_non_const_largest_row = ((d_common.mask_conv_rows) - 1);
# 1268 "kernel.cu"
__cuda_local_var_20073_6_non_const_largest_col = (__cuda_local_var_20073_6_non_const_largest_col - 1);
# 1269 "kernel.cu"
}
# 1272 "kernel.cu"
__cuda_local_var_20072_6_non_const_largest_row = (__cuda_local_var_20072_6_non_const_largest_row + 1);
# 1273 "kernel.cu"
__cuda_local_var_20073_6_non_const_largest_col = (__cuda_local_var_20073_6_non_const_largest_col + 1);
# 1274 "kernel.cu"
__cuda_local_var_20074_6_non_const_offset_row = ((__cuda_local_var_20072_6_non_const_largest_row - (d_common.in_rows)) - ((d_common.sSize) - (d_common.tSize)));
# 1275 "kernel.cu"
__cuda_local_var_20075_6_non_const_offset_col = ((__cuda_local_var_20073_6_non_const_largest_col - (d_common.in_cols)) - ((d_common.sSize) - (d_common.tSize)));
# 1276 "kernel.cu"
__cuda_local_var_20087_6_non_const_pointer = ((d_common_change.frame_no) + (((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no) * (d_common.no_frames)));
# 1277 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_tRowLoc)[__cuda_local_var_20087_6_non_const_pointer]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Row)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)]) + __cuda_local_var_20074_6_non_const_offset_row);
# 1278 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_tColLoc)[__cuda_local_var_20087_6_non_const_pointer]) = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Col)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)]) + __cuda_local_var_20075_6_non_const_offset_col);
# 1280 "kernel.cu"
}
# 1286 "kernel.cu"
__syncthreads();
# 1288 "kernel.cu"
}
# 1299 "kernel.cu"
if (((d_common_change.frame_no) != 0) && (((d_common_change.frame_no) % 10) == 0))
# 1299 "kernel.cu"
{
# 1302 "kernel.cu"
__cuda_local_var_20090_6_non_const_loc_pointer = ((((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no) * (d_common.no_frames)) + (d_common_change.frame_no));
# 1303 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Row)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_tRowLoc)[__cuda_local_var_20090_6_non_const_loc_pointer]);
# 1304 "kernel.cu"
(((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Col)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)]) = (((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_tColLoc)[__cuda_local_var_20090_6_non_const_loc_pointer]);
# 1307 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = __cuda_local_var_20097_6_non_const_tx;
# 1308 "kernel.cu"
while (__cuda_local_var_20098_6_non_const_ei_new < (d_common.in_elem))
# 1308 "kernel.cu"
{
# 1311 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in_rows)) - 1);
# 1312 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = ((((__cuda_local_var_20098_6_non_const_ei_new + 1) / (d_common.in_rows)) + 1) - 1);
# 1313 "kernel.cu"
if (((__cuda_local_var_20098_6_non_const_ei_new + 1) % (d_common.in_rows)) == 0)
# 1313 "kernel.cu"
{
# 1314 "kernel.cu"
__cuda_local_var_20053_6_non_const_row = ((d_common.in_rows) - 1);
# 1315 "kernel.cu"
__cuda_local_var_20054_6_non_const_col = (__cuda_local_var_20054_6_non_const_col - 1);
# 1316 "kernel.cu"
}
# 1319 "kernel.cu"
__cuda_local_var_20055_6_non_const_ori_row = ((((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Row)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)]) - 25) + __cuda_local_var_20053_6_non_const_row) - 1);
# 1320 "kernel.cu"
__cuda_local_var_20056_6_non_const_ori_col = ((((((d_unique[__cuda_local_var_20096_6_non_const_bx]).d_Col)[((d_unique[__cuda_local_var_20096_6_non_const_bx]).point_no)]) - 25) + __cuda_local_var_20054_6_non_const_col) - 1);
# 1321 "kernel.cu"
__cuda_local_var_20089_6_non_const_ori_pointer = ((__cuda_local_var_20056_6_non_const_ori_col * (d_common.frame_rows)) + __cuda_local_var_20055_6_non_const_ori_row);
# 1324 "kernel.cu"
(__cuda_local_var_20037_9_non_const_d_in[__cuda_local_var_20098_6_non_const_ei_new]) = ((float)(((double)((d_common.alpha) * (__cuda_local_var_20037_9_non_const_d_in[__cuda_local_var_20098_6_non_const_ei_new]))) + (((1.0) - ((double)(d_common.alpha))) * ((double)((d_common_change.d_frame)[__cuda_local_var_20089_6_non_const_ori_pointer])))));
# 1327 "kernel.cu"
__cuda_local_var_20098_6_non_const_ei_new = (__cuda_local_var_20098_6_non_const_ei_new + 512);
# 1329 "kernel.cu"
}
# 1331 "kernel.cu"
}
# 1333 "kernel.cu"
}}
