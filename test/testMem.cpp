#include <iostream>
#include "testMem.h"

int main() {
    simParams params = {
        0.0f, //tStart
        100.0f, //tEnd
        200, //nSteps
        4, //nNeurons
        50.0f, //pulseStart
        51.0f, //pulseEnd
        7.0f //pulseAmp
    };

    float* results = new float[params.nNeurons * 4];

    runMemTest(params, results);

    // printf("Printing results from CPU:\n");
    // for(int i = 0; i < 4; i++) {
    //     printf("%f\n", results[i]);
    // }

    cudaDeviceSynchronize();

    return 0;

}