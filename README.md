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

---

## Test Cases Summary

This project includes a comprehensive suite of test cases across **scan**, **stream compaction**, and **radix sort**, covering both correctness and performance. Each test is designed to validate behavior on different input types, sizes (including power-of-two and non-power-of-two), and algorithm variants.

### Scan Tests
Tested scan variants include CPU, naive GPU, work-efficient GPU, Thrust, and shared memory scans.

| Test Description                               | Input Size         | Notes                                                |
|------------------------------------------------|--------------------|------------------------------------------------------|
| CPU scan                                       | 2¹⁹ = 524,288      | Baseline for correctness and performance             |
| Naive GPU scan (power-of-two / non-power-of-two) | 2¹⁹ and 393,931  | Launches multiple kernels per level                 |
| Work-efficient GPU scan (power-of-two / NPOT)  | 2¹⁹ and 393,931    | Implements upsweep/downsweep                        |
| Thrust GPU scan (power-of-two / NPOT)          | 2¹⁹ and 393,931    | Uses `thrust::exclusive_scan`                       |
| Shared memory naive scan                       | 512, 500, 32       | Small sizes for validating shared mem behavior      |
| Shared memory efficient scan                   | 512, 32            | Efficient scan in shared memory with loop unrolling |


### Stream Compaction Tests
Tested both CPU and GPU compaction with and without scan, for both power-of-two and non-power-of-two sizes.

| Test Description                               | Input Size         | Notes                                                |
|------------------------------------------------|--------------------|------------------------------------------------------|
| CPU compaction without scan (POT / NPOT)       | 524,288 / 393,931  | Sequential traversal                                 |
| CPU compaction with scan                       | 524,288            | Uses scan + scatter                                  |
| Work-efficient GPU compaction (POT / NPOT)     | 524,288 / 393,931  | Uses map-to-boolean + scan + scatter                |

### Radix Sort Tests
Includes correctness tests for various distributions and sizes, with special cases and large arrays.

| Test Case Description                          | Input Size         | Notes                                                |
|------------------------------------------------|--------------------|------------------------------------------------------|
| Random values                                  | 10                 | Basic unsorted small input                          |
| Already sorted                                 | 8                  | Best-case scenario                                  |
| Reverse sorted                                 | 8                  | Worst-case scenario                                 |
| Identical values                               | 6                  | Edge case: no change after sort                     |
| Contains duplicates                            | 9                  | Tests stable ordering                               |
| Large array (power-of-two)                     | 65,536             | Stress test for GPU sort                            |
| Large array (non-power-of-two)                 | 65,519             | Non-POT performance and correctness                 |
| Nearly sorted array with random swaps          | 16,384             | Realistic scenario with small local disorder        |

---

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

---

## Scan Runtime Analysis

The figure below shows the elapsed time (in milliseconds) of four scan implementations measured across input sizes ranging from \(2^{13}\) to \(2^{27}\), using a fixed block size of 256. All tests were run in **Release mode** to ensure optimized performance, particularly for the Thrust-based implementation.

![Scan Elapsed Time Plot](visualization/scan_size_plot_13_27.png)

### Key Observations

#### Performance at Small Sizes (\(2^{13} → 2^{17}\))
- **Thrust Scan** performs exceptionally well, benefiting from Release-mode optimizations that eliminate overhead seen in debug builds.
- **Efficient GPU Scan** is also strong in this range, clearly outperforming the naive version.
- **CPU Scan** starts very fast but scales poorly beyond this point.

#### Mid-Range Sizes (\(2^{18} → 2^{24}\))
- **Thrust Scan** continues to lead, maintaining low latency while other methods begin to scale more steeply.
- **Naive GPU Scan** begins to show inefficiencies due to redundant memory access and higher algorithmic complexity.
- **Efficient GPU Scan** remains competitive but starts to lag behind Thrust.

#### Large Sizes (\(2^{25} → 2^{27}\))
- **Thrust Scan** remains the fastest and scales efficiently, highlighting its well-optimized internal operations.
- **Naive GPU Scan** slows down significantly due to its \(O(n \log n)\) complexity and less efficient memory use.
- **CPU Scan** becomes the slowest by far, with consistent linear growth.

Overall, Thrust offers the best performance across all input sizes when compiled in Release mode, while the Efficient GPU Scan provides a solid custom alternative with strong performance at small to mid-range sizes. The CPU scan, although fast for small inputs, follows a linear \(O(n)\) time complexity and becomes the slowest as input sizes grow.

---

## Compact Runtime Analysis

The plot below shows the runtime performance (in milliseconds) of three different stream compaction implementations as a function of input size \(N\), ranging from \(2^5\) to \(2^{27}\). All tests were conducted using a fixed CUDA block size of 256 and compiled in **Release mode** to ensure optimal performance.

![Compact Elapsed Time Plot](visualization/compact_size_plot.png)

### Key Observations

#### CPU vs CPU with Scan
- For small input sizes (\(N ≤ 2^{16}\)), both CPU variants show very similar runtimes, indicating that the scan step contributes little overhead in this range.
- As \(N\) increases, "CPU with Scan" becomes slightly slower than CPU-only compaction due to the added cost of prefix sum computation.
- Both exhibit consistent linear growth on the log-log plot, confirming the expected **\(O(n)\)** time complexity for serial execution.

#### Efficient GPU Scan
- The GPU implementation shows near-constant runtime across small and mid-sized inputs (up to \(2^{21}\)), demonstrating excellent scalability due to parallel execution and efficient memory usage.
- Beyond \(2^{21}\), the runtime begins to increase gradually. This likely reflects:
  - The need to process more data in global memory
  - Increased number of kernel launches
- Even at \(N = 2^{27}\), GPU runtime remains well under 50 ms — significantly outperforming the CPU implementations, which exceed 100 ms at that scale.

---

## Efficient GPU Scan Runtime Analysis (Global Memory Implementation)

The bar chart below displays the runtime (in milliseconds) of a work-efficient Blelloch-style GPU scan implementation that operates entirely in global memory. No shared memory or warp-level primitives are used. 

![Efficient Scan Bar Plot](visualization/efficient_scan_bar_plot.png)

### Key Observations

- For mid-sized inputs (\(2^6\) to \(2^9\)), the scan shows consistently low runtimes (~0.07 ms), indicating effective thread-level parallelism despite global memory latency.
- Noticeable spikes occur at sizes \(2^5\), \(2^{10}\), and \(2^{13}\), where runtime increases by 2–3× compared to neighboring sizes. These performance dips likely stem from:
  - **Uncoalesced memory access** due to thread divergence at these specific sizes
  - **Extra overhead** from partial warp utilization or thread underpopulation in early/late stages of the scan
  - **Depth-related kernel launches**: For \(N = 2^{13}\), the number of upsweep and downsweep steps increases, amplifying launch and global memory access costs
- The runtime drop at \(2^{11}\) suggests that thread/block configuration for that size aligns better with the kernel design, temporarily improving efficiency.

### Best Performance

The most optimal performance is observed between \(2^6\) and \(2^9\), where the runtime stabilizes around **0.07 ms**. These sizes likely strike a balance where:
- The number of elements fits well within available thread blocks,
- Memory access patterns remain more coalesced,
- And kernel launch overhead is minimal due to shallower recursion depths in the scan tree.

This range can be considered the **sweet spot** for this global memory-based scan implementation, offering the lowest latency and most consistent performance across all tested input sizes. 
