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
typedef uint8_t tok_t;
typedef TYPE prob_t;
typedef uint8_t state_t;
typedef int32_t step_t;

#define N_STATES  64
#define N_OBS     140
#define N_TOKENS  64

int viterbi(tok_t obs[N_OBS], prob_t init[N_STATES],
            prob_t transition[N_STATES * N_STATES],
            prob_t emission[N_STATES * N_TOKENS],
            state_t path[N_OBS]);

struct bench_args_t {
    tok_t obs[N_OBS];
    prob_t init[N_STATES];
    prob_t transition[N_STATES * N_STATES];
    prob_t emission[N_STATES * N_TOKENS];
    state_t path[N_OBS];
};

```

Here is the plain C kernel:
```cpp
#include "viterbi.h"

int viterbi(tok_t obs[N_OBS], prob_t init[N_STATES],
            prob_t transition[N_STATES * N_STATES],
            prob_t emission[N_STATES * N_TOKENS],
            state_t path[N_OBS]) {
    prob_t llike[N_OBS][N_STATES];
    step_t t;
    state_t prev, curr;
    prob_t min_p, p;
    state_t min_s, s;

    L_init: for (s = 0; s < N_STATES; s++) {
        llike[0][s] = init[s] + emission[s * N_TOKENS + obs[0]];
    }

    L_timestep: for (t = 1; t < N_OBS; t++) {
        L_curr_state: for (curr = 0; curr < N_STATES; curr++) {
            prev = 0;
            min_p = llike[t - 1][prev] +
                    transition[prev * N_STATES + curr] +
                    emission[curr * N_TOKENS + obs[t]];
            L_prev_state: for (prev = 1; prev < N_STATES; prev++) {
                p = llike[t - 1][prev] +
                    transition[prev * N_STATES + curr] +
                    emission[curr * N_TOKENS + obs[t]];
                if (p < min_p) {
                    min_p = p;
                }
            }
            llike[t][curr] = min_p;
        }
    }

    min_s = 0;
    min_p = llike[N_OBS - 1][min_s];
    L_end: for (s = 1; s < N_STATES; s++) {
        p = llike[N_OBS - 1][s];
        if (p < min_p) {
            min_p = p;
            min_s = s;
        }
    }
    path[N_OBS - 1] = min_s;

    L_backtrack: for (t = N_OBS - 2; t >= 0; t--) {
        min_s = 0;
        min_p = llike[t][min_s] + transition[min_s * N_STATES + path[t + 1]];
        L_state: for (s = 1; s < N_STATES; s++) {
            p = llike[t][s] + transition[s * N_STATES + path[t + 1]];
            if (p < min_p) {
                min_p = p;
                min_s = s;
            }
        }
        path[t] = min_s;
    }

    return 0;
}


void workload(tok_t* obs, prob_t* init, prob_t* transition,
              prob_t* emission, state_t* path) {

    tok_t l_obs[N_OBS];
    prob_t l_init[N_STATES];
    prob_t l_transition[N_STATES * N_STATES];
    prob_t l_emission[N_STATES * N_TOKENS];
    state_t l_path[N_OBS];
    int i;

    for (i = 0; i < N_OBS; i++) l_obs[i] = obs[i];
    for (i = 0; i < N_STATES; i++) l_init[i] = init[i];
    for (i = 0; i < N_STATES * N_STATES; i++) l_transition[i] = transition[i];
    for (i = 0; i < N_STATES * N_TOKENS; i++) l_emission[i] = emission[i];

    viterbi(l_obs, l_init, l_transition, l_emission, l_path);

    for (i = 0; i < N_OBS; i++) path[i] = l_path[i];
}
```

Provide the complete HLS-optimized code in a ```cpp code fence.