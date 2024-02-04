#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"

using namespace std;



//**************************************************************************
// Reducción en GPU
__global__ void reduction_kernel(int *input, int *output, const int nverts) {
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Cargar datos en memoria compartida
    if (index < nverts * nverts) {
        shared_data[tid] = input[index];
    } else {
        shared_data[tid] = 0;
    }

    __syncthreads();

    // Realizar reducción en memoria compartida
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // El hilo 0 de cada bloque escribe el resultado parcial en la salida global
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

//**************************************************************************
// Kernel de Floyd-Warshall
__global__ void floyd_kernel(int *M, const int nverts, const int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nverts && j < nverts) {
        int Mij = M[i * nverts + j];

        if (i != j && i != k && j != k) {
            int Mikj = M[i * nverts + k] + M[k * nverts + j];
            Mij = min(Mij, Mikj);
            M[i * nverts + j] = Mij;
        }
    }
}

//**************************************************************************
// ************  MAIN FUNCTION *********************************************
int main(int argc, char *argv[]) {

    double time, Tcpu, Tgpu;

    if (argc != 3) {
        cerr << "Sintaxis: " << argv[0] << " <archivo de grafo> <blocksize>" << endl;
        return (-1);
    }

    // Obtén el valor de blocksize desde el segundo argumento
    int blocksize = atoi(argv[2]);
    // Get GPU information
    int num_devices, devID;
    cudaDeviceProp props;
    cudaError_t err;

    err = cudaGetDeviceCount(&num_devices);
    if (err == cudaSuccess) {
        cout << endl << num_devices << " CUDA-enabled GPUs detected in this computer system" << endl << endl;
        cout << "....................................................." << endl << endl;
    } else {
        cerr << "ERROR detecting CUDA devices......" << endl;
        exit(-1);
    }

    for (int i = 0; i < num_devices; i++) {
        devID = i;
        err = cudaGetDeviceProperties(&props, devID);
        cout << "Device " << devID << ": " << props.name << " with Compute Capability: " << props.major << "." << props.minor << endl << endl;
        if (err != cudaSuccess) {
            cerr << "ERROR getting CUDA devices" << endl;
        }
    }

    devID = 0;
    cout << "Using Device " << devID << endl;
    cout << "....................................................." << endl << endl;

    err = cudaSetDevice(devID);
    if (err != cudaSuccess) {
        cerr << "ERROR setting CUDA device" << devID << endl;
    }

    // Declaration of the Graph object
    Graph G;

    // Read the Graph
    G.lee(argv[1]);

    const int nverts = G.vertices;
    const int nverts2 = nverts * nverts;
    int *c_Out_M = new int[nverts2];
    int *d_In_M = NULL;
    int size = nverts2 * sizeof(int);

    err = cudaMalloc((void **)&d_In_M, size);
    if (err != cudaSuccess) {
        cerr << "ERROR MALLOC" << endl;
    }

    int *A = G.Get_Matrix();

    time = clock();

    err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << "ERROR CUDA MEM. COPY" << endl;
    }

    dim3 threadsPerBlock(blocksize, blocksize);
    dim3 blocksPerGrid((nverts + threadsPerBlock.x - 1) / threadsPerBlock.x, (nverts + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Main Loop
    for (int k = 0; k < nverts; k++) {
        floyd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_In_M, nverts, k);
        err = cudaGetLastError();

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch kernel! ERROR= %d\n", err);
            exit(EXIT_FAILURE);
        }
    }

    err = cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        cout << "ERROR CUDA MEM. COPY" << endl;
    }

    cout << "niters= " << nverts << endl << endl;
    cout << "blocksize= " << blocksize << "x" << blocksize << "=" <<  blocksize*blocksize << endl << endl << endl << endl;
    Tgpu = (clock() - time) / CLOCKS_PER_SEC;
    cout << "Time spent on GPU= " << Tgpu << endl << endl;

    // Reducción en GPU
    int *d_partial_sums;
    int num_blocks_reduction = (nverts2 + blocksize - 1) / blocksize;
    size_t shared_memory_size = blocksize * sizeof(int);
    err = cudaMalloc((void **)&d_partial_sums, num_blocks_reduction * sizeof(int));
    if (err != cudaSuccess) {
        cerr << "ERROR MALLOC" << endl;
    }

    reduction_kernel<<<num_blocks_reduction, blocksize, shared_memory_size>>>(d_In_M, d_partial_sums, nverts);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Failed to launch reduction kernel! ERROR= " << err << endl;
        exit(EXIT_FAILURE);
    }

    int *h_partial_sums = new int[num_blocks_reduction];
    err = cudaMemcpy(h_partial_sums, d_partial_sums, num_blocks_reduction * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "ERROR CUDA MEM. COPY" << endl;
    }
//**************************************************************************
	// CPU phase
	//**************************************************************************

	time=clock();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for(int k = 0; k < nverts; k++) {
          kn = k * nverts;
	  for(int i=0;i<nverts;i++) {
			in = i * nverts;
			for(int j = 0; j < nverts; j++)
	       			if (i!=j && i!=k && j!=k){
			 	    inj = in + j;
			 	    A[inj] = min(A[in+k] + A[kn+j], A[inj]);
	       }
	   }
	}
  
  Tcpu=(clock()-time)/CLOCKS_PER_SEC;
  cout << "Time spent on CPU= " << Tcpu << endl << endl;
  cout<<"....................................................."<<endl<<endl;

  cout << "Speedup TCPU/TGPU= " << Tcpu / Tgpu << endl;
  cout<<"....................................................."<<endl<<endl;

  
  bool errors=false;
  // Error Checking (CPU vs. GPU)
  for(int i = 0; i < nverts; i++)
    for(int j = 0; j < nverts; j++)
       if (abs(c_Out_M[i*nverts+j] - G.arista(i,j)) > 0)
         {cout << "Error (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl;
		  errors=true;
		 }


  if (!errors){ 
    cout<<"....................................................."<<endl;
	cout<< "WELL DONE!!! No errors found ............................"<<endl;
	cout<<"....................................................."<<endl<<endl;

  }
    // Reducción final en el host
    long long int h_sum = 0;
    for (int i = 0; i < num_blocks_reduction; i++) {
        h_sum += h_partial_sums[i];
    }

    int total_elements = nverts * nverts;
    double average = static_cast<double>(h_sum) / static_cast<double>(total_elements);
    cout << "Average of elements in the matrix: " << average << endl;

    cudaFree(d_In_M);
    cudaFree(d_partial_sums);
    delete[] c_Out_M;
    delete[] h_partial_sums;

    return 0;
}
