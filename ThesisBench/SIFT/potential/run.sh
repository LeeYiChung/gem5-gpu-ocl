#!/bin/bash
unset GEM5GPU_SIGNAL_BLOCK_READY
${GEM5GPU_OCL}/gem5/build/X86_VI_hammer_GPU/gem5.opt --outdir=./ ${GEM5GPU_OCL}/gem5-gpu/configs/se_fusion.py --clusters=1 --cores_per_cluster=1 --sc_l1_size=16kB --sc_l2_size=256kB --sc_l1_assoc=4 --sc_l2_assoc=16 --gpu-core-clock=1700MHz \
 --cpu-type=detailed --cpu-clock=2000MHz --pf-on --pf_assoc=16 --work-end-exit-count=1 --ports=32 --l1d_assoc=4 --l2_size=32MB --l2_assoc=16 --prefetcher \
 --num-dirs=2 --mem-type=LPDDR3_1600_x32 \
 -c ${GEM5GPU_OCL}/ThesisBench/SIFT/potential/gem5_fusion_SIFT \
 -o "/home/bachelor/b00902051/gem5-gpu-ocl/ThesisBench/SIFT/potential/data/input2_512.bmp" > out 2> err &

