#pragma once
#include <cuda_runtime.h>

struct simParams {
    float tStart;
    float tEnd;
    int nSteps;
    int nNeurons;
    float pulseStart;
    float pulseEnd;
    float pulseAmp;
};

__global__ void hello(int n, float* d_v, float* d_v_next);
void runMemTest(simParams params, float* results);