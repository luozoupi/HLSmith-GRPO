Convert the following plain C/C++ kernel code into Xilinx Vitis HLS-optimized code.

Requirements:
1. Use a top-level `workload()` function wrapped in `extern "C" { }`.
   If the input already contains a `workload()` wrapper, preserve and upgrade that wrapper instead of creating a second wrapper.
   If there is no wrapper yet, add one that calls the kernel.
2. Add HLS INTERFACE pragmas to the workload function:
   - `#pragma HLS INTERFACE m_axi port=<ptr> offset=slave bundle=gmem` for pointer arguments
   - `#pragma HLS INTERFACE s_axilite port=<arg> bundle=control` for all arguments
   - `#pragma HLS INTERFACE s_axilite port=return bundle=control`
3. Add performance pragmas to the kernel:
   - `#pragma HLS PIPELINE` on innermost loops where appropriate
   - `#pragma HLS UNROLL` where beneficial for parallelism
   - `#pragma HLS ARRAY_PARTITION` for arrays that need parallel access
4. Keep the original algorithm logic UNCHANGED
5. Include the original header file
6. Do NOT copy or re-declare structs, typedefs, constants, or function prototypes that already exist in the header; include the header once and reuse its declarations
7. The code must be synthesizable with Vitis HLS targeting the Alveo U280 (vivado flow)

Benchmark-specific guidance:
The top-level function must be `workload`.

Checklist before returning:
- Include the header exactly once.
- Reuse existing function names and signatures from the plain input when possible.
- Preserve the plain-input helper and wrapper structure unless a change is required for valid Vitis HLS pragmas.
- Prefer minimal edits to the plain input over creative rewrites.
- Do not redeclare header-owned structs/types like `bench_args_t`.
- Do not invent undeclared helper arrays or buffers like `l_*`; if a local buffer is needed, declare it and fill it explicitly.
- Keep every `#pragma HLS` inside a function body or loop body, never at global scope.

Here is the header:
```cpp
#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __SYNTHESIS__
#include "support/common/mc.h"
#endif

#define FLT_MAX 3.40282347e+38

#define NPOINTS (819200/2)
#define NFEATURES 34
#define NCLUSTERS 5
#define TILE_SIZE 4096

const int WIDTH_FACTOR = 16;

const int NUM_TILES = NPOINTS/TILE_SIZE;

// void workload(float  *feature, /* [npoints][nfeatures] */
// 			  float  *clusters, /* [n_clusters][n_features] */
// 			  int *membership);

struct bench_args_t {
	float FEATURE[NPOINTS*NFEATURES];
	float CLUSTER[NCLUSTERS*NFEATURES];
	int MEMBERSHIP[NPOINTS];
};

#endif

```

Here is the plain C kernel:
```cpp
#include "kmeans.h"


void workload(float  *feature, /* [npoints][nfeatures] */
			float  *clusters, /* [n_clusters][n_features] */
			int *membership)
{

	UPDATE_MEMBER: for (int i = 0; i < NPOINTS; i++) {
		float min_dist = FLT_MAX;
		int index = 0;

		/* find the cluster center id with min distance to pt */
		MIN: for (int j = 0; j < NCLUSTERS; j++) {
			float dist = 0.0;

			DIST: for (int k = 0; k < NFEATURES; k++) {
				float diff = feature[NFEATURES * i + k] - clusters[NFEATURES * j + k];
				dist += diff * diff;
			}
			if (dist < min_dist) {
				min_dist = dist;
				index = j;
			}
		}
		/* assign the membership to object i */
		membership[i] = index;
	}
}
```

Provide the complete HLS-optimized code in a ```cpp code fence.