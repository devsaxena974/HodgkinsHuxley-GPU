// hh_cuda.h
#pragma once
#include <cuda_runtime.h>

// struct HHParameters {
//     float Cm;
//     float gNa, gK, gL;
//     float ENa, EK, EL;
//     float dt;
//     int nSteps;
//     int nNeurons;
// };

// struct CurrentStimulus {
//     float amplitude;
//     float startTime;
//     float duration;
//     int stimulusType;  // 0: none, 1: step, 2: ramp, 3: sine
// };
struct simParams {
    float tStart;
    float tEnd;
    int nSteps;
    int nNeurons;
    float pulseStart;
    float pulseEnd;
    float pulseAmp;
};

// Host functions
void runHHSimulation(const simParams& params, float* h_results);
// void setupHHSimulation(HHParameters& params, CurrentStimulus& stim);
// void runHHSimulation(float* h_V, float* h_m, float* h_h, float* h_n,
//                     const HHParameters& params, const CurrentStimulus& stim);