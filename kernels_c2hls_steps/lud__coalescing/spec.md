Apply MEMORY COALESCING optimization to the following HLS code.

Memory coalescing means:
- Change pointer arguments in the workload() function to use wide bus types: `ap_uint<512>*` (or `ap_uint<LARGE_BUS>*`)
- Use wide bus read/write helper functions (memcpy_wide_bus_read_float, memcpy_wide_bus_write_float, etc.)
- Include the wide bus header: `#include "../../../common/mc.h"` (defines LARGE_BUS=512, MARS_WIDE_BUS_TYPE, and provides helper functions)
- Update INTERFACE pragmas to use the wide bus pointer types
- Increase burst lengths where possible (max_read_burst_length=256, max_write_burst_length=256)
- Add cyclic array partitioning with appropriate factors for local buffers

Keep all existing double-buffering, pipeline, and unroll optimizations.

Current synthesis report:
  lat_worst: 239522305
  ii_max: 239522306
  lut: 2947
  ff: 2809
  dsp: 5
  bram: 2
  clk_est_ns: 2.431

Header:
```cpp
#ifndef LUD_H
#define LUD_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define GRID_ROWS 256
#define GRID_COLS 256
#define SIZE GRID_ROWS
#define TILE_ROWS 4
#define PARA_FACTOR 8
#define TOP 0
#define BOTTOM (GRID_ROWS / TILE_ROWS - 1)

#define TYPE float

struct bench_args_t {
    float result[GRID_ROWS * GRID_COLS];
};

// void workload(float result[GRID_ROWS * GRID_COLS]);
#endif

```

Current HLS code:
```cpp
#include"lud.h"
extern "C"{
void lud(float result[GRID_ROWS * GRID_COLS])
	{
		int i, j, k; 
		float sum;
	 
		for (i=0; i<SIZE; i++){
		     for (j=i; j<SIZE; j++){
		         sum=result[i*SIZE+j];
		         for (k=0; k<i; k++) sum -= result[i*SIZE+k]*result[k*SIZE+j];
		         result[i*SIZE+j]=sum;
		     }

		     for (j=i+1;j<SIZE; j++){
		         sum=result[j*SIZE+i];
		         for (k=0; k<i; k++) sum -= result[j*SIZE+k]*result[k*SIZE+i];
		         result[j*SIZE+i]=sum/result[i*SIZE+i];
		     }
		 }
		
		 return;
	}

	void workload(float result[GRID_ROWS * GRID_COLS])
	{

		#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem		
		#pragma HLS INTERFACE s_axilite port=result bundle=control
		#pragma HLS INTERFACE s_axilite port=return bundle=control

		lud(result);

		return;

	}
}

```

Provide the complete coalescing-optimized code in a ```cpp code fence.