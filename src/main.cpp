/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include <stream_compaction/sharednaivemem.h>
#include <stream_compaction/sharedefficientmem.h>
#include "testing_helpers.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>  // C++17 required
#include <direct.h>    // For _getcwd()

const int SIZE = 1 << 24; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int* a = new int[SIZE];
int* b = new int[SIZE];
int* c = new int[SIZE];


void ensureFileExists(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.good()) {
        std::ofstream outfile(filename);  // Create the file
        if (outfile) {
            std::cout << "[INFO] File created: " << filename << std::endl;
        }
        else {
            std::cerr << "[ERROR] Failed to create file: " << filename << std::endl;
        }
    }
    else {
        std::cout << "[INFO] File already exists: " << filename << std::endl;
    }
}

void benchmarkScans() {
    const std::string filename = "C:\\Users\\thero\\OneDrive\\Documents\\GitHub\\Project2-Stream-Compaction\\visualization\\scan_timing_results.csv";
    ensureFileExists(filename);

    std::ofstream fout(filename);
    fout << "N,CPU(ms),Naive(ms),Efficient(ms),Thrust(ms)" << std::endl;

    char cwd[1024];
    _getcwd(cwd, sizeof(cwd));  // current working directory
    std::string fullPath = std::string(cwd) + "\\" + filename;
    std::cout << "[DEBUG] Attempting to write to: " << fullPath << std::endl;

    if (!fout.is_open()) {
        std::cerr << "[ERROR] Failed to open scan_timing_results.csv for writing." << std::endl;
        return;
    }

    for (int exp = 35; exp <= 45; exp++) {  // Increase range for more realistic timings
        int N = 1 << exp;

        int* input = new int[N];
        int* output = new int[N];

        genArray(N, input, 10000);  // Fill with random integers

        // CPU scan
        zeroArray(N, output);
        StreamCompaction::CPU::scan(N, output, input);
        cudaDeviceSynchronize();
        float cpuTime = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

        // Naive scan
        zeroArray(N, output);
        StreamCompaction::Naive::scan(N, output, input);
        cudaDeviceSynchronize();
        float naiveTime = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();

        // Efficient scan
        zeroArray(N, output);
        StreamCompaction::Efficient::scan(N, output, input);
        cudaDeviceSynchronize();
        float efficientTime = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();


        // Thrust scan. ERROR
        zeroArray(N, output);
        StreamCompaction::Thrust::scan(N, output, input);
        cudaDeviceSynchronize();
        float thrustTime = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();

        std::cout << N << "," << cpuTime << "," << naiveTime << "," << efficientTime << "," << thrustTime  << std::endl;

        fout << N << "," << cpuTime << "," << naiveTime << "," << efficientTime << "," << thrustTime  << std::endl;

        delete[] input;
        delete[] output;
    }

    fout.close();
    printf("Scan timing results written to scan_timing_results.csv\n");
}



int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    const int SAFE_SIZE = 512;     // Power-of-two and ≤ 1024
    const int SAFE_NPOT = 500;     // Non-power-of-two and ≤ 1024

    // ======= shared memory naive scan, power-of-two =======
    {
        int* a = new int[SAFE_SIZE];
        int* b = new int[SAFE_SIZE];
        int* c = new int[SAFE_SIZE];

        genArray(SAFE_SIZE, a, 50);    
        StreamCompaction::CPU::scan(SAFE_SIZE, b, a); 

        zeroArray(SAFE_SIZE, c);
        printDesc("shared memory naive scan, power-of-two");
        StreamCompaction::SharedNaiveMem::scanSharedNaive(SAFE_SIZE, c, a);
        printElapsedTime(StreamCompaction::SharedNaiveMem::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        printCmpResult(SAFE_SIZE, b, c);

        delete[] a;
        delete[] b;
        delete[] c;
    }

    // ======= shared memory naive scan, non-power-of-two =======
    {
        int* a = new int[SAFE_NPOT];
        int* b = new int[SAFE_NPOT];
        int* c = new int[SAFE_NPOT];

        genArray(SAFE_NPOT, a, 50);
        StreamCompaction::CPU::scan(SAFE_NPOT, b, a);

        zeroArray(SAFE_NPOT, c);
        printDesc("shared memory naive scan, non-power-of-two");
        StreamCompaction::SharedNaiveMem::scanSharedNaive(SAFE_NPOT, c, a);
        printElapsedTime(StreamCompaction::SharedNaiveMem::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        printCmpResult(SAFE_NPOT, b, c);

        delete[] a;
        delete[] b;
        delete[] c;
    }


    {
        const int N = 32;
        int input[N], output[N], expected[N];
        for (int i = 0; i < N; ++i) input[i] = i % 5;
        printDesc("shared memory naive scan, small manual");
        StreamCompaction::CPU::scan(N, expected, input);
        StreamCompaction::SharedNaiveMem::scanSharedNaive(N, output, input);
        printElapsedTime(StreamCompaction::SharedEfficientMem::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        printArray(N, output, true);
        printCmpResult(N, expected, output);
    }

    {
        int* a = new int[SAFE_SIZE];
        int* b = new int[SAFE_SIZE];
        int* c = new int[SAFE_SIZE];

        genArray(SAFE_SIZE, a, 50);
        StreamCompaction::CPU::scan(SAFE_SIZE, b, a);

        zeroArray(SAFE_SIZE, c);
        printDesc("shared memory efficient scan, power-of-two");
        StreamCompaction::SharedEfficientMem::scanSharedEfficient(SAFE_SIZE, c, a);
        printElapsedTime(StreamCompaction::SharedEfficientMem::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        printCmpResult(SAFE_SIZE, b, c);

        delete[] a;
        delete[] b;
        delete[] c;
    }


    {
        const int N = 32;
        int input[N], output[N], expected[N];
        for (int i = 0; i < N; ++i) input[i] = i % 4;
        StreamCompaction::CPU::scan(N, expected, input);
        StreamCompaction::SharedEfficientMem::scanSharedEfficient(N, output, input);
        printDesc("shared memory efficient scan, small manual");
        printCmpResult(N, expected, output);
    }


    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);


    printf("\n");
    printf("***********************\n");
    printf("** RADIX SORT TESTS  **\n");
    printf("***********************\n");

    // Test 1: Random integers
    {
        const int N = 10;
        int input[N] = { 3, 1, 4, 9, 2, 0, 5, 8, 7, 6 };
        int output[N], expected[N];
        memcpy(expected, input, sizeof(int) * N);
        std::sort(expected, expected + N);

        StreamCompaction::Radix::sort(N, output, input);
        printDesc("radix sort - random ints");
        printArray(N, output, true);
        printCmpResult(N, expected, output);
    }

    // Test 2: Already sorted
    {
        const int N = 8;
        int input[N] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        int output[N], expected[N];
        memcpy(expected, input, sizeof(int) * N);

        StreamCompaction::Radix::sort(N, output, input);
        printDesc("radix sort - already sorted");
        printArray(N, output, true);
        printCmpResult(N, expected, output);
    }

    // Test 3: Reverse sorted
    {
        const int N = 8;
        int input[N] = { 7, 6, 5, 4, 3, 2, 1, 0 };
        int output[N], expected[N] = { 0, 1, 2, 3, 4, 5, 6, 7 };

        StreamCompaction::Radix::sort(N, output, input);
        printDesc("radix sort - reverse sorted");
        printArray(N, output, true);
        printCmpResult(N, expected, output);
    }

    // Test 4: All same
    {
        const int N = 6;
        int input[N] = { 42, 42, 42, 42, 42, 42 };
        int output[N], expected[N] = { 42, 42, 42, 42, 42, 42 };

        StreamCompaction::Radix::sort(N, output, input);
        printDesc("radix sort - identical elements");
        printArray(N, output, true);
        printCmpResult(N, expected, output);
    }

    // Test 5: Contains duplicates
    {
        const int N = 9;
        int input[N] = { 5, 3, 3, 7, 1, 1, 2, 9, 0 };
        int output[N], expected[N];
        memcpy(expected, input, sizeof(int) * N);
        std::sort(expected, expected + N);

        StreamCompaction::Radix::sort(N, output, input);
        printDesc("radix sort - contains duplicates");
        printArray(N, output, true);
        printCmpResult(N, expected, output);
    }


    // Test 6: Large array (power-of-two size)
    {
        const int N = 1 << 16; // 65536
        int* input = new int[N];
        int* output = new int[N];
        int* expected = new int[N];

        genArray(N, input, 10000);  // Fill with random ints in range [0, 9999]
        memcpy(expected, input, sizeof(int) * N);
        std::sort(expected, expected + N);

        printDesc("radix sort - large array (pow2)");
        StreamCompaction::Radix::sort(N, output, input);
        printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        printCmpResult(N, expected, output);

        delete[] input;
        delete[] output;
        delete[] expected;
    }

    // Test 7: Large array (non-power-of-two size)
    {
        const int N = (1 << 16) - 17; // 65519
        int* input = new int[N];
        int* output = new int[N];
        int* expected = new int[N];

        genArray(N, input, 50000);  // Wider range of input
        memcpy(expected, input, sizeof(int) * N);
        std::sort(expected, expected + N);

        printDesc("radix sort - large array (non-pow2)");
        StreamCompaction::Radix::sort(N, output, input);
        printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        printCmpResult(N, expected, output);

        delete[] input;
        delete[] output;
        delete[] expected;
    }

    // Test 8: Nearly sorted array with minor disorder
    {
        const int N = 1 << 14; // 16384
        int* input = new int[N];
        int* output = new int[N];
        int* expected = new int[N];

        // Generate ascending array
        for (int i = 0; i < N; ++i) {
            input[i] = i;
        }

        // Add small random noise
        for (int i = 0; i < 100; ++i) {
            int x = rand() % N;
            int y = rand() % N;
            std::swap(input[x], input[y]);
        }

        memcpy(expected, input, sizeof(int) * N);
        std::sort(expected, expected + N);

        printDesc("radix sort - nearly sorted with random swaps");
        StreamCompaction::Radix::sort(N, output, input);
        printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        printCmpResult(N, expected, output);

        delete[] input;
        delete[] output;
        delete[] expected;
    }

    benchmarkScans();


    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}
