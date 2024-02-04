#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"

using namespace std;

#define blocksize 256

//**************************************************************************
__global__ void floyd_kernel(int * M, const int nverts, const int k) {
    int ij = threadIdx.x + blockDim.x * blockIdx.x;
    int i= ij / nverts;
    int j= ij - i * nverts;
    if (i<nverts && j< nverts) {
    int Mij = M[ij];
    if (i != j && i != k && j != k) {
	int Mikj = M[i * nverts + k] + M[k * nverts + j];
    Mij = (Mij > Mikj) ? Mikj : Mij;
    M[ij] = Mij;}
  }
}
//**************************************************************************

//**************************************************************************
// ************  MAIN FUNCTION *********************************************
int main (int argc, char *argv[]) {

    double time, Tcpu, Tgpu;

    if (argc != 2) {
	    cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return(-1);
	}	

    //Get GPU information
    int num_devices,devID;
    cudaDeviceProp props;
    cudaError_t err;

	err=cudaGetDeviceCount(&num_devices);
	if (err == cudaSuccess) { 
	    cout <<endl<< num_devices <<" CUDA-enabled  GPUs detected in this computer system"<<endl<<endl;
		cout<<"....................................................."<<endl<<endl;}	
	else 
	    { cerr << "ERROR detecting CUDA devices......" << endl; exit(-1);}
	    
	for (int i = 0; i < num_devices; i++) {
	    devID=i;
	    err = cudaGetDeviceProperties(&props, devID);
        cout<<"Device "<<devID<<": "<< props.name <<" with Compute Capability: "<<props.major<<"."<<props.minor<<endl<<endl;
        if (err != cudaSuccess) {
		  cerr << "ERROR getting CUDA devices" << endl;
	    }


	}
	devID = 0;    
        cout<<"Using Device "<<devID<<endl;
        cout<<"....................................................."<<endl<<endl;

	err = cudaSetDevice(devID); 
    if(err != cudaSuccess) {
		cerr << "ERROR setting CUDA device" <<devID<< endl;
	}

	// Declaration of the Graph object
	Graph G;
	
	// Read the Graph
	G.lee(argv[1]);

	//cout << "The input Graph:"<<endl;
	//G.imprime();
	const int nverts = G.vertices;
	const int niters = nverts;
	const int nverts2 = nverts * nverts;

	int *c_Out_M = new int[nverts2];
	int size = nverts2*sizeof(int);
	int * d_In_M = NULL;

	err = cudaMalloc((void **) &d_In_M, size);
	if (err != cudaSuccess) {
		cerr << "ERROR MALLOC" << endl;
	}
    // Get the integer 2D array for the dense graph
	int *A = G.Get_Matrix();

    //**************************************************************************
	// GPU phase
	//**************************************************************************
	
    time=clock();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 

    // Main Loop
	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
	 	int threadsPerBlock = blocksize;
	 	int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;
        // Kernel Launch
	    floyd_kernel<<<blocksPerGrid,threadsPerBlock >>>(d_In_M, nverts, k);
	    err = cudaGetLastError();

	    if (err != cudaSuccess) {
	  	    fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
	  	    exit(EXIT_FAILURE);
		}
	}
	err =cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 
	cout << "niters= " << niters << endl << endl;
	cout << "blocksize= " << blocksize << endl << endl;
	Tgpu=(clock()-time)/CLOCKS_PER_SEC;
	
	cout << "Time spent on GPU= " << Tgpu << endl << endl;

    //**************************************************************************
	// CPU phase
	//**************************************************************************

	time=clock();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for(int k = 0; k < niters; k++) {
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
  cudaFree(d_In_M);
}

