#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

#define MAX_THREADS_PER_BLOCK 1024

//**************************************************************************
__global__ void cuda_compute_C(float *A, float *B, float *C, int Bsize, int NBlocks, int N) {
    extern __shared__ float shared_C[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N) {
        int k = tid / Bsize;
        int istart = k * Bsize;
        int iend = istart + Bsize;

        shared_C[threadIdx.x] = 0.0;

        for (int i = istart; i < iend; i++) {
            float a = A[i] * tid;
            if (static_cast<int>(ceilf(a)) % 2 == 0)
                shared_C[threadIdx.x] += a + B[i];
            else
                shared_C[threadIdx.x] += a - B[i];
        }

        // Synchronize threads within the block
        __syncthreads();

        // Perform reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                shared_C[threadIdx.x] += shared_C[threadIdx.x + s];
            }
            // Synchronize threads within the block
            __syncthreads();
        }

        // Store the result in global memory
        if (threadIdx.x == 0) {
            C[tid] = shared_C[0];
        }

        tid += blockDim.x * gridDim.x;
    }
}

//**************************************************************************
int main(int argc, char *argv[]) {
    int Bsize, NBlocks;
    if (argc != 3) {
        cout << "Uso: transformacion Num_bloques Tam_bloque  " << endl;
        return 0;
    } else {
        NBlocks = atoi(argv[1]);
        Bsize = atoi(argv[2]);
    }

    const int N = Bsize * NBlocks;
    float *A, *B, *C, *D;

    A = new float[N];
    B = new float[N];
    C = new float[N];
    D = new float[NBlocks];
    float mx; // maximum of C

    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(1.5 * (1 + (5 * i) % 7) / (1 + i % 5));
        B[i] = static_cast<float>(2.0 * (2 + i % 5) / (1 + i % 7));
    }

    // Time measurement for sequential execution
    double t1 = clock();

    // Compute C[i], d[K] and mx
    for (int k = 0; k < NBlocks; k++) {
        int istart = k * Bsize;
        int iend = istart + Bsize;

        for (int i = istart; i < iend; i++) {
            C[i] = 0.0;

            for (int j = istart; j < iend; j++) {
                float a = A[j] * i;
                if (static_cast<int>(ceilf(a)) % 2 == 0)
                    C[i] += a + B[j];
                else
                    C[i] += a - B[j];
            }
        }
    }

    double t2 = (clock() - t1) / CLOCKS_PER_SEC;

    // Compute mx
    mx = C[0];
    for (int i = 1; i < N; i++) {
        mx = max(C[i], mx);
    }

    // Compute d[K]
    for (int k = 0; k < NBlocks; k++) {
        int istart = k * Bsize;
        int iend = istart + Bsize;
        D[k] = 0.0;

        for (int i = istart; i < iend; i++) {
            D[k] += C[i];
        }
    }

    cout << "....................................................." << endl;
    cout << "....................................................." << endl
         << "El valor máximo en C es:  " << mx << endl;

    cout << endl
         << "N=" << N << "= " << Bsize << "*" << NBlocks << "  ........  Tiempo algoritmo secuencial (solo CPU)= " << t2 << endl
         << endl;

    // Allocate memory on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Time measurement for GPU execution
    double t3 = clock();

    // Calculate grid and block sizes for CUDA kernel
    int threads_per_block = min(MAX_THREADS_PER_BLOCK, N);
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Launch CUDA kernel with shared memory
    cuda_compute_C<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(d_A, d_B, d_C, Bsize, NBlocks, N);
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    double t4 = (clock() - t3) / CLOCKS_PER_SEC;

    cout << "....................................................." << endl;
    cout << "....................................................." << endl
         << "El valor máximo en C es:  " << mx << endl;

    cout << endl
         << "N=" << N << "= " << Bsize << "*" << NBlocks << "  ........  Tiempo algoritmo paralelo (CPU + GPU)= " << t4 << endl
         << endl;
    cout << "Speedup TCPU/TGPU= " << t2/t4 << endl;
    cout<<"....................................................."<<endl<<endl;
    // Free memory on GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free memory on CPU
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;

    return 0;
}
