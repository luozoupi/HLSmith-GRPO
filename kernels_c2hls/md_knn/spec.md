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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define TYPE double
#define nAtoms        256
#define maxNeighbors  16
#define lj1           1.5
#define lj2           2.0

void md_kernel(TYPE force_x[nAtoms], TYPE force_y[nAtoms], TYPE force_z[nAtoms],
               TYPE position_x[nAtoms], TYPE position_y[nAtoms], TYPE position_z[nAtoms],
               int32_t NL[nAtoms * maxNeighbors]);

struct bench_args_t {
    TYPE force_x[nAtoms];
    TYPE force_y[nAtoms];
    TYPE force_z[nAtoms];
    TYPE position_x[nAtoms];
    TYPE position_y[nAtoms];
    TYPE position_z[nAtoms];
    int32_t NL[nAtoms * maxNeighbors];
};

```

Here is the plain C kernel:
```cpp
#include "md.h"

void md_kernel(TYPE force_x[nAtoms], TYPE force_y[nAtoms], TYPE force_z[nAtoms],
               TYPE position_x[nAtoms], TYPE position_y[nAtoms], TYPE position_z[nAtoms],
               int32_t NL[nAtoms * maxNeighbors]) {
    TYPE delx, dely, delz, r2inv;
    TYPE r6inv, potential, force, j_x, j_y, j_z;
    TYPE i_x, i_y, i_z, fx, fy, fz;
    int32_t i, j, jidx;

    loop_i: for (i = 0; i < nAtoms; i++) {
        i_x = position_x[i];
        i_y = position_y[i];
        i_z = position_z[i];
        fx = 0;
        fy = 0;
        fz = 0;
        loop_j: for (j = 0; j < maxNeighbors; j++) {
            jidx = NL[i * maxNeighbors + j];
            j_x = position_x[jidx];
            j_y = position_y[jidx];
            j_z = position_z[jidx];
            delx = i_x - j_x;
            dely = i_y - j_y;
            delz = i_z - j_z;
            r2inv = 1.0 / (delx * delx + dely * dely + delz * delz);
            r6inv = r2inv * r2inv * r2inv;
            potential = r6inv * (lj1 * r6inv - lj2);
            force = r2inv * potential;
            fx += delx * force;
            fy += dely * force;
            fz += delz * force;
        }
        force_x[i] = fx;
        force_y[i] = fy;
        force_z[i] = fz;
    }
}


void workload(TYPE* force_x, TYPE* force_y, TYPE* force_z,
              TYPE* position_x, TYPE* position_y, TYPE* position_z,
              int32_t* NL) {

    TYPE l_fx[nAtoms], l_fy[nAtoms], l_fz[nAtoms];
    TYPE l_px[nAtoms], l_py[nAtoms], l_pz[nAtoms];
    int32_t l_NL[nAtoms * maxNeighbors];
    int i;

    for (i = 0; i < nAtoms; i++) {
        l_px[i] = position_x[i];
        l_py[i] = position_y[i];
        l_pz[i] = position_z[i];
    }
    for (i = 0; i < nAtoms * maxNeighbors; i++) l_NL[i] = NL[i];

    md_kernel(l_fx, l_fy, l_fz, l_px, l_py, l_pz, l_NL);

    for (i = 0; i < nAtoms; i++) {
        force_x[i] = l_fx[i];
        force_y[i] = l_fy[i];
        force_z[i] = l_fz[i];
    }
}
```

Provide the complete HLS-optimized code in a ```cpp code fence.