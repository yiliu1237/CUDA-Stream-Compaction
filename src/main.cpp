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
#include "testing_helpers.hpp"

const int SIZE = 1 << 8; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

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
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
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

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}
