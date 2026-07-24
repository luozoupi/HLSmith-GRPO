Apply TILING optimization to the following HLS code.

Tiling means:
- Buffer input data from global memory into local arrays (use memcpy or manual loops)
- Separate the code into load(), compute(), store() phases
- Process data in tiles/chunks of a reasonable size (e.g., 256 elements)
- The compute phase should operate on local buffers instead of directly on AXI memory

Keep all existing INTERFACE pragmas. Keep the extern "C" workload() wrapper.

Current synthesis report:
  lat_worst: 15070072
  ii_max: 15070073
  lut: 89206
  ff: 158522
  dsp: 1190
  bram: 5
  clk_est_ns: 2.798

Header:
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

Current HLS code:
```cpp
#include "kmeans.h"

extern"C"{
void load_local_cluster(float local_clusters[NCLUSTERS * NFEATURES], float clusters[NCLUSTERS * NFEATURES])
{
	for (int i(0); i<NCLUSTERS; ++i){
		for (int j(0); j<NFEATURES; ++j){
#pragma HLS PIPELINE II=1
			local_clusters[i*NFEATURES+j] = clusters[i*NFEATURES+j];
		}
	}
}

void load_local_feature(int flag, float local_feature[TILE_SIZE * NFEATURES],
		float feature[NPOINTS * NFEATURES], int tile_idx)
{
	if (flag){
	for (int i(0); i<TILE_SIZE; ++i){
		for (int j(0); j<NFEATURES; ++j){
#pragma HLS PIPELINE II=1
			local_feature[i*NFEATURES+j] = feature[(tile_idx*TILE_SIZE+i)*NFEATURES+j];
		}
	}
	}
}

void compute_local_membership(int flag, float local_feature[TILE_SIZE * NFEATURES],
		float local_clusters[NCLUSTERS * NFEATURES],int local_membership[TILE_SIZE])
{
	if (flag){
    for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS PIPELINE II=1
    	float min_dist = FLT_MAX;
        int index = 0;

        /* find the cluster center id with min distance to pt */
        MIN: for (int j = 0; j < NCLUSTERS; j++) {
#pragma HLS UNROLL
        	float dist = 0.0;

            DIST: for (int k = 0; k < NFEATURES; k++) {
#pragma HLS UNROLL
                float diff = local_feature[NFEATURES * i + k] - local_clusters[NFEATURES * j + k];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                index = j;
            }
        }

        /* assign the membership to object i */
        local_membership[i] = index;
    }
	}
}

void store_local_membership(int flag, int local_membership[TILE_SIZE], int membership[NPOINTS], int tile_idx)
{
	if (flag){
	for (int i(0); i<TILE_SIZE; ++i){
#pragma HLS PIPELINE II=1
		membership[tile_idx*TILE_SIZE+i] = local_membership[i];
	}
	}
}

void workload(float  *feature, /* [npoints][nfeatures] */
              float  *clusters, /* [n_clusters][n_features] */
			  int *membership)
{
#pragma HLS INTERFACE m_axi port=feature offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=membership offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=clusters offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=feature bundle=control
#pragma HLS INTERFACE s_axilite port=membership bundle=control
#pragma HLS INTERFACE s_axilite port=clusters bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int local_membership_0[TILE_SIZE];
    float local_feature_0[TILE_SIZE * NFEATURES];
#pragma HLS ARRAY_PARTITION variable=local_feature_0 cyclic factor=34 //NFEATURES

    int local_membership_1[TILE_SIZE];
    float local_feature_1[TILE_SIZE * NFEATURES];
#pragma HLS ARRAY_PARTITION variable=local_feature_1 cyclic factor=34 //NFEATURES

    int local_membership_2[TILE_SIZE];
    float local_feature_2[TILE_SIZE * NFEATURES];
#pragma HLS ARRAY_PARTITION variable=local_feature_2 cyclic factor=34 //NFEATURES

    float local_clusters[NCLUSTERS * NFEATURES];
#pragma HLS ARRAY_PARTITION variable=local_clusters complete

    load_local_cluster(local_clusters, clusters);

    for (int i=0; i<NUM_TILES+2; ++i)
    {
        int load_flag = (i >= 0) && (i < NUM_TILES);
        int compute_flag = (i >= 1) && (i < NUM_TILES+1);
        int store_flag = (i >= 2) && (i < NUM_TILES+2);

        if (i % 3 == 0){
	    	load_local_feature(load_flag, local_feature_0, feature, i);
	    	compute_local_membership(compute_flag, local_feature_2, local_clusters, local_membership_2);
	        store_local_membership(store_flag, local_membership_1, membership, i-2);
        }
        else if (i % 3 == 1){
	    	load_local_feature(load_flag, local_feature_1, feature, i);
	    	compute_local_membership(compute_flag, local_feature_0, local_clusters, local_membership_0);
	        store_local_membership(store_flag, local_membership_2, membership, i-2);
        }
        else{
	    	load_local_feature(load_flag, local_feature_2, feature, i);
	    	compute_local_membership(compute_flag, local_feature_1, local_clusters, local_membership_1);
	        store_local_membership(store_flag, local_membership_0, membership, i-2);
        }


    }
}

}
```

Provide the complete tiling-optimized code in a ```cpp code fence.