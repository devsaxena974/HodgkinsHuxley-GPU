// hh_cuda.cu
#include "hh_cuda.h"
#include <math_constants.h>
#include <iostream>

// Device functions for channel kinetics
__device__ float alphaM(float v) {
    return (0.1f * (v + 40.0f)) / (1.0f - expf(-(v + 40.0f)/10.0f));
}

__device__ float betaM(float v) {
    return 4.0f * expf(-(v + 65.0f)/18.0f);
}

__device__ float alphaH(float v) {
    return 0.07f * expf(-(v + 65.0f) / 20.0f);
}

__device__ float betaH(float v) {
    return 1.0f / (1.0f + expf(-(v + 35.0f) / 10.0f));
}

__device__ float alphaN(float v) {
    return (0.01f * (v + 55.0f)) / (1.0f - expf(-(v + 55.0f) / 10.0f));
}

__device__ float betaN(float v) {
    return 0.125f * expf(-(v + 65.0f) / 80.0f);
}

__device__ void ydot(float t, float* w, float* z, const simParams params) {
    // HH Model params
    const float Cm = 1.0f;
    const float gNa = 120.0f;
    const float gK = 36.0f;
    const float gL = 0.3f;
    const float ENa = 50.0f;
    const float EK = -77.0f;
    const float EL = -54.4f;

    // Calculate the input current I (square pulse)
    float inter = (params.pulseStart + params.pulseEnd) / 2;
    float len = params.pulseEnd - params.pulseStart;
    float Iext = params.pulseAmp * (1.0f - signbit(abs(t - inter) - len / 2)) / 2.0f;

    // calculate current state variables
    float v = w[0];
    float m = w[1];
    float h = w[2];
    float n = w[3];

    // calc time constants and steady state values
    float am = alphaM(v);
    float bm = betaM(v);
    float ah = alphaH(v);
    float bh = betaH(v);
    float an = alphaN(v);
    float bn = betaN(v);

    float tau_m = 1.0f / (am + bm);
    float tau_h = 1.0f / (ah + bh);
    float tau_n = 1.0f / (an + bn);

    float m_inf = am/(am + bm);
    float h_inf = ah/(ah + bh);
    float n_inf = an/(an + bn);

    // calculate ionic currents
    float INa = gNa * powf(m, 3) * h * (v - ENa);
    float IK = gK * powf(n, 4) * (v - EK);
    float IL = gL * (v - EL);

    // voltage equation
    z[0] = (Iext - INa - IK - IL) / Cm;

    // gating equations
    z[1] = (m_inf - m) / tau_m;
    z[2] = (h_inf - h) / tau_h; 
    z[3] = (n_inf - n) / tau_n;
}

// Forward Euler kernel
__global__ void eulerKernel(float t, float h, float* w, float* w_next, const simParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= params.nNeurons) return;

    int offset = idx * 4;
    float state[4], derivatives[4];

    // get the current state
    for(int i = 0; i < 4; i++) {
        state[i] = w[offset + i];
    }

    ydot(t, state, derivatives, params);

    // perform the fwd euler update
    for(int i = 0; i < 4; i++) {
        w_next[offset + i] = state[i] + h * derivatives[i];
    }
}

// RK4 step kernel
__global__ void rk4stepKernel(float t, float h, float* w, float* w_next, const simParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.nNeurons) return;

    // Offset for this neuron's state variables
    int offset = idx * 4;  // 4 state variables per neuron
    float state[4], k1[4], k2[4], k3[4], k4[4], temp[4];

    // Load current state
    for (int i = 0; i < 4; i++) {
        state[i] = w[offset + i];
    }

    // k1 = h * f(t, w)
    ydot(t, state, k1, params);
    for (int i = 0; i < 4; i++) {
        k1[i] *= h;
        temp[i] = state[i] + k1[i]/2;
    }

    // k2 = h * f(t + h/2, w + k1/2)
    ydot(t + h/2, temp, k2, params);
    for (int i = 0; i < 4; i++) {
        k2[i] *= h;
        temp[i] = state[i] + k2[i]/2;
    }

    // k3 = h * f(t + h/2, w + k2/2)
    ydot(t + h/2, temp, k3, params);
    for (int i = 0; i < 4; i++) {
        k3[i] *= h;
        temp[i] = state[i] + k3[i];
    }

    // k4 = h * f(t + h, w + k3)
    ydot(t + h, temp, k4, params);
    for (int i = 0; i < 4; i++) {
        k4[i] *= h;
    }

    // Update state: w_next = w + (k1 + 2k2 + 2k3 + k4)/6
    for (int i = 0; i < 4; i++) {
        w_next[offset + i] = state[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6;
    }
}


// Host-side implementation
void runHHSimulation(const simParams& params, float* h_results) {
    float h = (params.tEnd - params.tStart) / params.nSteps;
    // we have 4 state variables per neuron
    size_t stateSize = params.nNeurons * 4 * sizeof(float);

    // Allocate device memory
    float *d_w, *d_w_next;
    cudaMalloc(&d_w, stateSize);
    cudaMalloc(&d_w_next, stateSize);

    // initialize device mem
    float* h_init = new float[params.nNeurons * 4];
    for(int i = 0; i < params.nNeurons; i++) {
        // v0
        h_init[i * 4 + 0] = -65.0f;
        // m0
        h_init[i * 4 + 1] = 0.0f;
        // h0
        h_init[i * 4 + 2] = 0.3f;
        // n0
        h_init[i * 4 + 3] = 0.6f;
    }

    // copy host init array to device equivalent
    cudaMemcpy(d_w, h_init, stateSize, cudaMemcpyHostToDevice);
    delete[] h_init;

    // launch kernel below
    int threadsPerBlock = 256;
    int blocks = (params.nNeurons + threadsPerBlock - 1) / threadsPerBlock;

    // main loop stepping through time
    float t = params.tStart;
    for(int step = 0; step < params.nSteps; step++) {
        //rk4stepKernel<<<blocks, threadsPerBlock>>>(t, h, d_w, d_w_next, params);
        eulerKernel<<<blocks, threadsPerBlock>>>(t, h, d_w, d_w_next, params);
        std::swap(d_w, d_w_next);
        t += h;
    }

    // copy results back to host
    cudaMemcpy(h_results, d_w, stateSize, cudaMemcpyDeviceToHost);

    //cleanup
    cudaFree(d_w);
    cudaFree(d_w_next);
}