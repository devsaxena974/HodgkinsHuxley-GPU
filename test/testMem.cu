#include "testMem.h"
#include <stdio.h>


__global__ void hello(int n, float* d_v, float* d_v_next) {

    int neuronIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if(neuronIdx < n) {
        // print the current grid indices
        printf("blockIdx.x: %i, threadIdx.x: %i, blockDim.y: %i, threadIdx.y: %i, neuronIdx: %i\n",
            blockIdx.x, threadIdx.x, blockDim.y, threadIdx.y, neuronIdx);

        int start = neuronIdx * 4;
        
        for(int i = 0; i < 4; i++) {
            d_v_next[start + i] = d_v[start + i];
        }

        printf("d_v_next values:\n");

        // print the device array values
        printf("v: %f, m: %f, h: %f, n: %f\n", 
            d_v_next[start], 
            d_v_next[start + 1],
            d_v_next[start + 2],
            d_v_next[start + 3]);
    }
} 

// function to get device information
void getDeviceInfo() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Max Threads Per Block: %i\n",
            prop.maxThreadsPerBlock);
        printf("  Max Blocks Per SM: %i\n",
            prop.maxBlocksPerMultiProcessor);
        printf("  Multiprocessor Count: %i\n",
            prop.multiProcessorCount);
        printf("  Shared Memory Per Block (bytes): %zu\n\n",
            prop.sharedMemPerBlock);
    }
}

// call the kernel
void runMemTest(simParams params, float* results) {
    // display device info
    getDeviceInfo();

    float dt = (params.tEnd - params.tStart) / params.nSteps;

    // figure out the state size per neuron = 4 floats
    size_t sizePerNeuron = params.nNeurons * 4 * sizeof(float); 

    float *d_w, *d_w_next;
    cudaMalloc(&d_w, sizePerNeuron * params.nNeurons);
    cudaMalloc(&d_w_next, sizePerNeuron * params.nNeurons);

    // initialize device memory
    float* h_init = new float[params.nNeurons * 4];

    for(int i = 0; i < params.nNeurons; i++) {
        h_init[i * 4 + 0] = -65.0f;
        h_init[i * 4 + 1] = 0.0f;
        h_init[i * 4 + 2] = 0.3f;
        h_init[i * 4 + 3] = 0.6f;
    }

    // copy host init array to the device equivalent
    cudaMemcpy(d_w, h_init, (sizePerNeuron * params.nNeurons), cudaMemcpyHostToDevice);

    // figure out kernel dimensions
    //int threadsPerBlock = 256;
    //dim3 threadsPerBlock(2, 1);
    //dim3 numBlocks((params.nNeurons + threadsPerBlock.x - 1) / threadsPerBlock.x, 
    //    (params.nNeurons+threadsPerBlock.y - 1) / threadsPerBlock.y);
    //dim3 numBlocks(1, 1);
    //int numBlocks = (params.nNeurons + threadsPerBlock - 1) / threadsPerBlock;
    dim3 threadsPerBlock(32, 1, 1);  // x = 32 threads, y = 1, z = 1
    dim3 numBlocks((params.nNeurons + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);


    // float* d_v;
    // size_t size = 4 * sizeof(float);
    // cudaMalloc(&d_v, size);
    // cudaMemcpy(d_v, results, size, cudaMemcpyHostToDevice);

    // float t = params.tStart;
    // for(int step = 0; step < params.nSteps; step++) {
    //     //rk4stepKernel<<<blocks, threadsPerBlock>>>(t, h, d_w, d_w_next, params);
    //     hello<<<blocks, threadsPerBlock>>>(t, h, d_w, d_w_next, params);
    //     std::swap(d_w, d_w_next);
    //     t += h;
    // }

    hello<<<numBlocks, threadsPerBlock>>>(params.nNeurons, d_w, d_w_next);

    cudaMemcpy(results, d_w_next, sizePerNeuron * params.nNeurons, cudaMemcpyDeviceToHost);

    cudaFree(d_w);
}