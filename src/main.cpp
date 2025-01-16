#include "hh_cuda.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;

// function to write sim results
void writeResults(const simParams& params, const float* h_results) {
    fstream outFile;
    //outFile << std::setprecision(6) << std::fixed;
    outFile.open("hh_simulation_results.csv", ios::trunc | ios::out | ios::in);

    if (!outFile) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    while(outFile) {
        // write header
        outFile << "time,voltage,m,h,n\n";

        cout << "Number of neurons: " << params.nNeurons << endl;
        cout << "Number of steps: " << params.nSteps << endl;

        cout << "Should have printed the header by now" << endl;

        // calculate the time step
        float dt = (params.tEnd - params.tStart) / params.nSteps;

        // write data for each time step
        for(int step = 0; step < params.nSteps; step++) {
            float t = params.tStart + step * dt;

            // for first neuron
            int idx = step * params.nNeurons * 4;

            outFile << t << ","
                << h_results[idx + 0] << ","    // voltage
                << h_results[idx + 1] << ","    // m
                << h_results[idx + 2] << ","    // h
                << h_results[idx + 3] << "\n";  // n
        }
    }

    outFile.close();
}

int main() {
    // Setup simulation parameters
    simParams params = {
        0.0f, //tStart
        100.0f, //tEnd
        200, //nSteps
        1, //nNeurons
        50.0f, //pulseStart
        51.0f, //pulseEnd
        7.0f //pulseAmp
    };

    // allocate host mem to store results
    float* h_results = new float[params.nNeurons * 4];

    runHHSimulation(params, h_results);

    ofstream outFile("hh_results.csv");

    outFile << setprecision(6) << fixed;

    outFile << "time,voltage,m,h,n\n";

    cout << "First few values in h_results:" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << h_results[i] << endl;
    }

    cout << "Number of neurons: " << params.nNeurons << endl;
    cout << "Number of steps: " << params.nSteps << endl;

    cout << "Should have printed the header by now" << endl;

    // calculate the time step
    float dt = (params.tEnd - params.tStart) / params.nSteps;

    //write data for each time step
    for(int step = 0; step < params.nSteps; step++) {
        float t = params.tStart + (step * dt);

        // for first neuron
        int idx = step * params.nNeurons * 4;

        //cout << t << "," << h_results[idx+0] << endl;

        outFile << t << ","
            << h_results[idx + 0] << ","    // voltage
            << h_results[idx + 1] << ","    // m
            << h_results[idx + 2] << ","    // h
            << h_results[idx + 3] << "\n";  // n
    }

    outFile.close();

    // write results to csv file
    //writeResults(params, h_results);

    delete[] h_results;
    return 0;
}
