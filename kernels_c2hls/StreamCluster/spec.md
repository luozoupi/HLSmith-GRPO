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
Preserve the existing helper-call structure from the plain input instead of rewriting the whole benchmark around a new buffer scheme.
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
#ifndef STREAMCLUSTER_H
#define STREAMCLUSTER_H

#include <string.h>

#define OFFSET 1  
//2973//((OFFSET) << 17)
//#define BATCH_SIZE 1024
#define DIM 200
#define BATCH_SIZE 1024
#define MAX_WORK_MEM_SIZE 128

struct bench_args_t {
    
};
#endif
```

Here is the plain C kernel:
```cpp
#include "streamcluster.h"


void workload(    
    float* coord,                      
    float* weight,                      
    float* cost, 
    float* target,
    int* assign,
    int* center_table,
    char* switch_membership,
    float* work_mem,
    int num,
    float* cost_of_opening_x,
    int numcenter            
)
{


    int i, j;
    for( i = 0; i < BATCH_SIZE; i++ ) {
        float sum = 0;
        for( j = 0; j < DIM; j++ ) {
            float a = coord[i * DIM + j] - target[j];
            sum += a * a;
        }
        float current_cost = sum * weight[i] - cost[i];
        int local_center_index = center_table[assign[i]];
        if (current_cost < 0) {

            // point i would save cost just by switching to x
            // (note that i cannot be a median, 
            // or else dist(p[i], p[x]) would be 0)
      
            switch_membership[i] = 1;
            cost_of_opening_x[0] += current_cost;

        } 
        else {

            // cost of assigning i to x is at least current assignment cost of i

            // consider the savings that i's **current** median would realize
            // if we reassigned that median and all its members to x;
            // note we've already accounted for the fact that the median
            // would save z by closing; now we have to subtract from the savings
            // the extra cost of reassigning that median and its members 
            work_mem[local_center_index] -= current_cost;
        }
    }
    
    return;
}
```

Provide the complete HLS-optimized code in a ```cpp code fence.