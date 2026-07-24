Apply PIPELINE optimization to the following HLS code.

Pipeline means:
- Add `#pragma HLS PIPELINE II=1` to the innermost compute loops
- Add `#pragma HLS ARRAY_PARTITION` on local arrays that need parallel access within the pipeline
- Add `#pragma HLS DEPENDENCE variable=X inter false` where loop-carried dependencies are false
- Add `#pragma HLS LOOP_TRIPCOUNT min=N max=N` for variable-bound loops

Do NOT change the algorithmic structure. Only add pipeline/partition/dependence pragmas.

Current synthesis report:
  lat_worst: 5396
  ii_max: 5397
  lut: 123719
  ff: 205159
  dsp: 1840
  bram: 142
  clk_est_ns: 2.431

Header:
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

Current HLS code:
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

extern "C" {
void workload(TYPE* force_x, TYPE* force_y, TYPE* force_z,
              TYPE* position_x, TYPE* position_y, TYPE* position_z,
              int32_t* NL) {
#pragma HLS INTERFACE m_axi port=force_x offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=force_y offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=force_z offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=position_x offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=position_y offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=position_z offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=NL offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=force_x bundle=control
#pragma HLS INTERFACE s_axilite port=force_y bundle=control
#pragma HLS INTERFACE s_axilite port=force_z bundle=control
#pragma HLS INTERFACE s_axilite port=position_x bundle=control
#pragma HLS INTERFACE s_axilite port=position_y bundle=control
#pragma HLS INTERFACE s_axilite port=position_z bundle=control
#pragma HLS INTERFACE s_axilite port=NL bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

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
}

```

Provide the complete pipeline-optimized code in a ```cpp code fence.