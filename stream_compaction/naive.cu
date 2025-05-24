#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void naiveScanStep(int n, int d, const int* input, int* output) {
            int k = threadIdx.x + blockIdx.x * blockDim.x;
            if (k >= n) return;

            if (k >= (1 << (d - 1))) {
                output[k] = input[k - (1 << (d - 1))] + input[k];
            }
            else {
                output[k] = input[k];
            }
        }

        int ilog2ceil(int x) {
            return static_cast<int>(std::ceil(std::log2f(static_cast<float>(x))));
        }

        void scan(int n, int* odata, const int* idata) {
            timer().startGpuTimer();

            int* dev_ping;
            int* dev_pong;

            cudaMalloc(&dev_ping, n * sizeof(int));
            cudaMalloc(&dev_pong, n * sizeof(int));

            cudaMemcpy(dev_ping, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int numBlocks = (n + blockSize - 1) / blockSize;

            int depth = ilog2ceil(n);
            for (int d = 1; d <= depth; ++d) {
                naiveScanStep << <numBlocks, blockSize >> > (n, d, dev_ping, dev_pong);
                cudaDeviceSynchronize();

                // Swap buffers
                std::swap(dev_ping, dev_pong);
            }

            // dev_ping now has the inclusive scan result
            // Convert to exclusive scan
            cudaMemcpy(odata + 1, dev_ping, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0;

            cudaFree(dev_ping);
            cudaFree(dev_pong);

            timer().endGpuTimer();
        }
    }
}
