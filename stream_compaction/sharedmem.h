#pragma once

namespace StreamCompaction {
	namespace SharedMem {

		// Work-efficient scan using shared memory (Blelloch scan)
		void scanEfficient(int n, int* odata, const int* idata);

		// Naive scan using shared memory (work-inefficient, O(n log n))
		void scanNaive(int n, int* odata, const int* idata);

	} 
} 