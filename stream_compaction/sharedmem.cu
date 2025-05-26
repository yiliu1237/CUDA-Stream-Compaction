#include <cuda.h>
#include <cuda_runtime.h>
#include "sharedmem.h"

namespace StreamCompaction {
    namespace SharedMem {

        // ---------------- Efficient Shared Memory Scan ------------------
        __global__ void scanSharedEfficientKernel(int* odata, const int* idata, int n) {
            extern __shared__ int temp[];
            int tid = threadIdx.x;

            if (2 * tid < n)     temp[2 * tid] = idata[2 * tid];
            if (2 * tid + 1 < n) temp[2 * tid + 1] = idata[2 * tid + 1];

            __syncthreads();

            int offset = 1;
            for (int d = n >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (tid < d) {
                    int ai = offset * (2 * tid + 1) - 1;
                    int bi = offset * (2 * tid + 2) - 1;
                    if (bi < n) temp[bi] += temp[ai];
                }
                offset <<= 1;
            }

            if (tid == 0 && n > 0) temp[n - 1] = 0;

            for (int d = 1; d < n; d <<= 1) {
                offset >>= 1;
                __syncthreads();
                if (tid < d) {
                    int ai = offset * (2 * tid + 1) - 1;
                    int bi = offset * (2 * tid + 2) - 1;
                    if (bi < n) {
                        int t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                    }
                }
            }

            __syncthreads();

            if (2 * tid < n)     odata[2 * tid] = (2 * tid == 0) ? 0 : temp[2 * tid - 1];
            if (2 * tid + 1 < n) odata[2 * tid + 1] = temp[2 * tid];
        }

        void scanEfficient(int n, int* odata, const int* idata) {
            int* dev_in, * dev_out;
            cudaMalloc(&dev_in, n * sizeof(int));
            cudaMalloc(&dev_out, n * sizeof(int));
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int threads = (n + 1) / 2;
            scanSharedEfficientKernel << <1, threads, sizeof(int)* n >> > (dev_out, dev_in, n);
            cudaDeviceSynchronize();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_in);
            cudaFree(dev_out);
        }

        // ---------------- Naive Shared Memory Scan ------------------
        __global__ void scanSharedNaiveKernel(int* odata, const int* idata, int n) {
            extern __shared__ int temp[];
            int tid = threadIdx.x;

            if (tid < n) {
                temp[tid] = idata[tid];
            }
            __syncthreads();

            for (int offset = 1; offset < n; offset *= 2) {
                int val = 0;
                if (tid >= offset && tid < n) {
                    val = temp[tid - offset];
                }
                __syncthreads();
                if (tid >= offset && tid < n) {
                    temp[tid] += val;
                }
                __syncthreads();
            }

            if (tid < n) {
                odata[tid] = (tid == 0) ? 0 : temp[tid - 1];
            }
        }

        void scanNaive(int n, int* odata, const int* idata) {
            int* dev_in, * dev_out;
            cudaMalloc(&dev_in, n * sizeof(int));
            cudaMalloc(&dev_out, n * sizeof(int));
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            scanSharedNaiveKernel << <1, n, sizeof(int)* n >> > (dev_out, dev_in, n);
            cudaDeviceSynchronize();

            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_in);
            cudaFree(dev_out);
        }

    } // namespace SharedMem
} // namespace StreamCompaction