Apply PIPELINE optimization to the following HLS code.

Pipeline means:
- Add `#pragma HLS PIPELINE II=1` to the innermost compute loops
- Add `#pragma HLS ARRAY_PARTITION` on local arrays that need parallel access within the pipeline
- Add `#pragma HLS DEPENDENCE variable=X inter false` where loop-carried dependencies are false
- Add `#pragma HLS LOOP_TRIPCOUNT min=N max=N` for variable-bound loops

Do NOT change the algorithmic structure. Only add pipeline/partition/dependence pragmas.

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

Provide the complete pipeline-optimized code in a ```cpp code fence.