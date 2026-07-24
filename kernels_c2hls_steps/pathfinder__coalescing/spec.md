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
  lat_worst: 2113553
  ii_max: 2113554
  lut: 3253
  ff: 2447
  dsp: 17
  bram: 7
  clk_est_ns: 2.545

Header:
```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//-----------------------------------------------
//Original
// #define ROWS 16384
// #define COLS 16384
//-----------------------------------------------
//-----------------------------------------------
//Alec-added
#define ROWS 1024
#define COLS 1024
//-----------------------------------------------
#define TYPE int32_t
#define MIN(a,b) ((a)<=(b) ? (a) : (b))

void pathfinder_kernel(int32_t J[ROWS*COLS], int32_t Jout[COLS]);

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  int32_t J[ROWS*COLS];
  int32_t Jout[COLS];
};

```

Current HLS code:
```cpp
#include "pathfinder.h"
#include "support/common/mc.h"

void pathfinder_kernel(int32_t J[ROWS*COLS], int32_t Jout[COLS]){
	int32_t temp;
	int32_t i, t, tt, n;
	int32_t min, before;

	KERNEL2: for(t = 0; t < ROWS-1 ;t++){
		for(n = 0; n < COLS; n++){
			min = Jout[n];
			
			if(n > 0){
				min = MIN(min,before);
			}
	
			if(n < COLS-1){
				min = MIN(min,Jout[n+1]);
			}
			before = Jout[n];
			
			Jout[n] = J[(t+1) * COLS + n]+min;
		}
	}
}

extern "C" {
void workload(int32_t J[ROWS * COLS], int32_t Jout[COLS]) {
  
 	#pragma HLS INTERFACE m_axi port=J offset=slave bundle=gmem
 	#pragma HLS INTERFACE m_axi port=Jout offset=slave bundle=gmem
  	#pragma HLS INTERFACE s_axilite port=J bundle=control
  	#pragma HLS INTERFACE s_axilite port=Jout bundle=control
  	#pragma HLS INTERFACE s_axilite port=return bundle=control
	
	int32_t dst[COLS], src[COLS];
	int32_t i, t, tt, n;
	int32_t min;

	memcpy(dst,J,sizeof(int32_t) * COLS);

	KERNEL_OUTER: for(t = 0; t < ROWS-1 ;t++){
		KERNEL_INNER: for(n = 0; n < COLS; n++){
			min = dst[n];
			
			if(n > 0){
				min = MIN(min,dst[n-1]);
			}
	
			if(n < COLS-1){
				min = MIN(min,dst[n+1]);
			}
			
			src[n] = J[(t+1) * COLS + n]+min;
		}
		memcpy(dst,src,sizeof(int32_t) * COLS);
	}  	
	memcpy(Jout,dst,sizeof(int32_t) * COLS);
	return;
							  
}
}
```

Provide the complete coalescing-optimized code in a ```cpp code fence.