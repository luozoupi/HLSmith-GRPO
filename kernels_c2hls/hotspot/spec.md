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
#ifndef HOTSPOT_H
#define HOTSPOT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define SIM_TIME 64

#define GRID_ROWS 512
#define GRID_COLS 512

#define TILE_ROWS 64

#define PARA_FACTOR 16
/* maximum power density possible (say 300W for a 10mm x 10mm chip) */

#define MAX_PD  (3.0e4)

/* required precision in degrees  */
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100

/* capacitance fitting factor */
#define FACTOR_CHIP 0.5
#define OPEN

/* chip parameters  */
#define T_CHIP 0.0005
#define CHIP_HEIGHT 0.016
#define CHIP_WIDTH 0.016

#define AMB_TEMP 80.0

#define TOP 0
#define BOTTOM (GRID_ROWS/TILE_ROWS - 1)

#define TYPE float


struct bench_args_t {
    float temp[GRID_ROWS * GRID_COLS];
    float power[GRID_ROWS * GRID_COLS];
};


#endif

```

Here is the plain C kernel:
```cpp
#include"hotspot.h"



void hotspot(float result[GRID_ROWS * GRID_COLS], float temp[GRID_ROWS * GRID_COLS], float power[GRID_ROWS * GRID_COLS], float Cap_1, float Rx_1, float Ry_1, float Rz_1) {
    float amb_temp = 80.0;
    float delta;

    for (int r = 0; r < GRID_ROWS; r++)
        for (int c = 0; c < GRID_COLS; c++) {
            if (r == 0 || c == 0 || r == GRID_ROWS - 1 || c == GRID_COLS - 1) {

                /* Corner 1 */
                if ((r == 0) && (c == 0)) {
                    delta = (Cap_1) * (power[0] +
                        (temp[1] - temp[0]) * Rx_1 +
                        (temp[GRID_COLS] - temp[0]) * Ry_1 +
                        (amb_temp - temp[0]) * Rz_1);
                }   
    
                /* Corner 2 */
                else if ((r == 0) && (c == GRID_COLS - 1)) {
                    delta = (Cap_1) * (power[c] +
                        (temp[c - 1] - temp[c]) * Rx_1 +
                        (temp[c + GRID_COLS] - temp[c]) * Ry_1 +
                        (amb_temp - temp[c]) * Rz_1);
                }   
    
                /* Corner 3 */
                else if ((r == GRID_ROWS - 1) && (c == GRID_COLS - 1)) {
                    delta = (Cap_1) * (power[r*GRID_COLS + c] +
                        (temp[r*GRID_COLS + c - 1] - temp[r*GRID_COLS + c]) * Rx_1 +
                        (temp[(r - 1)*GRID_COLS + c] - temp[r*GRID_COLS + c]) * Ry_1 +
                        (amb_temp - temp[r*GRID_COLS + c]) * Rz_1);
                }   
    
                /* Corner 4 */
                else if ((r == GRID_ROWS - 1) && (c == 0)) {
                    delta = (Cap_1) * (power[r*GRID_COLS] +
                        (temp[r*GRID_COLS + 1] - temp[r*GRID_COLS]) * Rx_1 +
                        (temp[(r - 1)*GRID_COLS] - temp[r*GRID_COLS]) * Ry_1 +
                        (amb_temp - temp[r*GRID_COLS]) * Rz_1);
                }   
    
                /* Edge 1 */
                else if (r == 0) {
                    delta = (Cap_1) * (power[c] +
                        (temp[c + 1] + temp[c - 1] - 2.0*temp[c]) * Rx_1 +
                        (temp[GRID_COLS + c] - temp[c]) * Ry_1 +
                        (amb_temp - temp[c]) * Rz_1);
                }   
    
                /* Edge 2 */
                else if (c == GRID_COLS - 1) {
                    delta = (Cap_1) * (power[r*GRID_COLS + c] +
                        (temp[(r + 1)*GRID_COLS + c] + temp[(r - 1)*GRID_COLS + c] - 2.0*temp[r*GRID_COLS + c]) * Ry_1 +
                        (temp[r*GRID_COLS + c - 1] - temp[r*GRID_COLS + c]) * Rx_1 +
                        (amb_temp - temp[r*GRID_COLS + c]) * Rz_1);
                }   
    
                /* Edge 3 */
                else if (r == GRID_ROWS - 1) {
                    delta = (Cap_1) * (power[r*GRID_COLS + c] +
                        (temp[r*GRID_COLS + c + 1] + temp[r*GRID_COLS + c - 1] - 2.0*temp[r*GRID_COLS + c]) * Rx_1 +
                        (temp[(r - 1)*GRID_COLS + c] - temp[r*GRID_COLS + c]) * Ry_1 +
                        (amb_temp - temp[r*GRID_COLS + c]) * Rz_1);
                }   
    
                /* Edge 4 */
                else if (c == 0) {
                    delta = (Cap_1) * (power[r*GRID_COLS] +
                        (temp[(r + 1)*GRID_COLS] + temp[(r - 1)*GRID_COLS] - 2.0*temp[r*GRID_COLS]) * Ry_1 +
                        (temp[r*GRID_COLS + 1] - temp[r*GRID_COLS]) * Rx_1 +
                        (amb_temp - temp[r*GRID_COLS]) * Rz_1);
                }

            }

            else {
                    delta = (Cap_1 * (power[r*GRID_COLS + c] +
                        (temp[(r + 1)*GRID_COLS + c] + temp[(r - 1)*GRID_COLS + c] - 2.f*temp[r*GRID_COLS + c]) * Ry_1 +
                        (temp[r*GRID_COLS + c + 1] + temp[r*GRID_COLS + c - 1] - 2.f*temp[r*GRID_COLS + c]) * Rx_1 +
                        (amb_temp - temp[r*GRID_COLS + c]) * Rz_1));
            }

            result[r*GRID_COLS + c] = temp[r*GRID_COLS + c] + delta;

        }

    return;
}




void workload(float result[GRID_ROWS * GRID_COLS], float temp[GRID_ROWS * GRID_COLS], float power[GRID_ROWS * GRID_COLS])
{

    
    
    
    float grid_height = CHIP_HEIGHT / GRID_ROWS;
    float grid_width = CHIP_WIDTH / GRID_COLS;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * T_CHIP * grid_height);
    float Ry = grid_height / (2.0 * K_SI * T_CHIP * grid_width);
    float Rz = T_CHIP / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI);
    float step = PRECISION / max_slope / 1000.0;

    float Rx_1=1.f / Rx;
    float Ry_1=1.f / Ry;
    float Rz_1=1.f / Rz;
    float Cap_1 = step / Cap;

    int i;
    for (i = 0; i < SIM_TIME/2; i++) {
       hotspot(result, temp, power, Cap_1, Rx_1, Ry_1, Rz_1);
       
       hotspot(temp, result, power, Cap_1, Rx_1, Ry_1, Rz_1);

    }

    return;
}

```

Provide the complete HLS-optimized code in a ```cpp code fence.