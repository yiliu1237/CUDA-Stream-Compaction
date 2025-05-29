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


## Test Output Summary

Below are the results from running the full test suite, including scan, stream compaction, and radix sort performance and correctness checks.

<details>
<summary><strong>Click to expand full test log</strong></summary>

```plaintext
****************
** SCAN TESTS **
****************
    [   9  48  21  27  35  11   8   3  34   0   0   0  38 ...   2   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 23.1201ms    (std::chrono Measured)
    [   0   9  57  78 105 140 151 159 162 196 196 196 196 ... 410887318 410887320 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 23.2731ms    (std::chrono Measured)
    [   0   9  57  78 105 140 151 159 162 196 196 196 196 ... 410887242 410887288 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 5.2679ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 4.88928ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 15.4344ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 15.4737ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 53.3576ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 24.6813ms    (CUDA Measured)
    passed
==== shared memory naive scan, power-of-two ====
   elapsed time: 0.334848ms    (CUDA Measured)
    passed
==== shared memory naive scan, non-power-of-two ====
   elapsed time: 0.013312ms    (CUDA Measured)
    passed
==== shared memory naive scan, small manual ====
   elapsed time: 0ms    (CUDA Measured)
    [   0   0   1   3   6  10  10  11  13  16  20  20  21 ...  60  60 ]
    passed
==== shared memory efficient scan, power-of-two ====
   elapsed time: 0.361472ms    (CUDA Measured)
    passed
==== shared memory efficient scan, small manual ====
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   0   3   3   1   3   0   1   0   0   0   2   2 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 42.3429ms    (std::chrono Measured)
    [   3   3   3   1   3   1   2   2   3   3   1   3   2 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 43.0611ms    (std::chrono Measured)
    [   3   3   3   1   3   1   2   2   3   3   1   3   2 ...   1   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 23.6391ms    (std::chrono Measured)
    [   3   3   3   1   3   1   2   2   3   3   1   3   2 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 4.49686ms    (CUDA Measured)
    [   3   3   3   1   3   1   2   2   3   3   1   3   2 ...   2   2 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 4.20435ms    (CUDA Measured)
    [   3   3   3   1   3   1   2   2   3   3   1   3   2 ...   1   2 ]
    passed

***********************
** RADIX SORT TESTS  **
***********************
==== radix sort - random ints ====
    [   0   1   2   3   4   5   6   7   8   9 ]
    passed
==== radix sort - already sorted ====
    [   0   1   2   3   4   5   6   7 ]
    passed
==== radix sort - reverse sorted ====
    [   0   1   2   3   4   5   6   7 ]
    passed
==== radix sort - identical elements ====
    [  42  42  42  42  42  42 ]
    passed
==== radix sort - contains duplicates ====
    [   0   1   1   2   3   3   5   7   9 ]
    passed
==== radix sort - large array (pow2) ====
   elapsed time: 0.43328ms    (CUDA Measured)
    passed
==== radix sort - large array (non-pow2) ====
   elapsed time: 0.444416ms    (CUDA Measured)
    passed
==== radix sort - nearly sorted with random swaps ====
   elapsed time: 0.873472ms    (CUDA Measured)
    passed
```
</details> 