CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**  
**Author:** Yi Liu

## Tested On
- **OS:** Windows 11 Home Version 24H2 (OS Build 26100.4061)  
- **CPU:** Intel(R) Core(TM) i9-14900K @ 3.20GHz, 24 cores / 32 threads  
- **RAM:** 64 GB  
- **GPU:** NVIDIA GeForce RTX 4090, 24 GB VRAM  
- **Environment:** Visual Studio 2022, CUDA 12.6, CMake 3.27
---

## 🧩 Overview

This project focuses on implementing and optimizing **stream compaction** using CUDA. Stream compaction is a fundamental parallel algorithm that removes unwanted elements (e.g., zeros) from an array while preserving the order of the remaining elements.

To achieve this, multiple versions of **prefix sum (scan)** were implemented and benchmarked, as scan is a key building block in efficient stream compaction.

The primary goals of this project were:
- Implement different versions of stream compaction on the GPU.
- Compare them against a CPU baseline.
- Investigate performance characteristics and bottlenecks.
- Explore scan implementations as subroutines.

---

## ✅ Features Implemented

### 1. **CPU Stream Compaction (Baseline)**
- Simple sequential compaction for correctness verification and timing comparison.
- Iterates through input array, writing out only non-zero elements.

### 2. **GPU Stream Compaction with Scan**
- Performs compaction in three steps:
  1. **Map to Boolean**: Marks non-zero entries as 1 and zeros as 0.
  2. **Exclusive Scan**: Computes destination indices for the non-zero values.
  3. **Scatter**: Writes non-zero values to compacted output using scanned indices.

### 3. **Scan Implementations (for Compaction Support)**

#### • Naive GPU Scan
- Textbook parallel scan with \( O(n \log n) \) operations.
- New kernel launched at every depth level.

#### • Work-Efficient GPU Scan (Blelloch)
- Upsweep and downsweep phases for \( O(n) \) total work.
- Multiple kernel launches for each pass.

#### • Shared Memory Optimized Scan
- Uses block-level shared memory to reduce global memory traffic.
- Suitable for small arrays or single-block inputs.

#### • Thrust Library Scan
- Calls `thrust::exclusive_scan()` as a baseline for optimized GPU performance.

### 4. **Benchmarking & Performance Comparison**
- All implementations were benchmarked for varying input sizes.
- GPU timings use CUDA events; CPU uses standard timers.
- Results written to CSV files for analysis.

