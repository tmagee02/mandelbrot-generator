#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

extern void matToImage(char* filename, int* mat, int* dims);
extern void matToImageColor(char* filename, int* mat, int* dims);

int main(int argc, char **argv){
	int rank, numranks;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);
	MPI_Status stat;

	double timeStart, timeEnd;
	double nodeStart, nodeEnd, nodeTotal;
	double* nodeTimes;
	if(rank == 0) timeStart = omp_get_wtime();

	int numThreads;
	int* threadNumChunks;
	double* threadTime;

	int i, j, index;
	double x0, y0, x, y, iter, temp;

    	int nx=27000;
    	int ny=18000;
	int max = nx * ny;
    	int* matrix=(int*)malloc(nx*ny*sizeof(int));

    	int maxIter=255;
    	double xStart=-2;
    	double xEnd=1;
    	double yStart=-1;
    	double yEnd=1;

	int chunkSize = 50 * nx; // __ rows per rank
	//int chunkSize = nx * ny / numranks; //unbalanced nodes
	int startNode= 0 - chunkSize;
	int endNode = 0;

	int* matrixNode;
	int* nodeRange = (int*) malloc(2*sizeof(int));
	int nodeRangeDiff;
	int nodey;

	if(rank == 0){
		nodeStart = MPI_Wtime();
		int workersFinished = 0;
		while(workersFinished <= numranks-1){
			for(int i=1; i<numranks; i++){
				startNode += chunkSize;
				endNode += chunkSize;
				if(endNode > max) endNode = max;
				if(startNode > max){
					startNode = max + 1;
					endNode = max + 1;
					workersFinished++;
				}
				MPI_Send(&startNode, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&endNode, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
			if(workersFinished >= numranks-1) break;
			for(int i=1; i<numranks; i++){
				MPI_Recv(nodeRange, 2, MPI_INT, i, 0, MPI_COMM_WORLD, &stat);
				nodeRangeDiff = nodeRange[1]-nodeRange[0];
				matrixNode = (int*) malloc(nodeRangeDiff*sizeof(int));
				MPI_Recv(matrixNode, nodeRangeDiff, MPI_INT, i, 0, MPI_COMM_WORLD, &stat);
				for(int k=nodeRange[0]; k<nodeRange[1]; k++){
					matrix[k] = matrixNode[k-nodeRange[0]];	
				}
			}
		}
		nodeEnd = MPI_Wtime();
		nodeTotal = nodeEnd - nodeStart;
		nodeTimes = (double*) malloc(numranks*sizeof(double));
	}

	else{
		#pragma omp parallel private(i, j, index, x0, y0, x, y, iter, temp)
		while(startNode <= max){
		{
			#pragma omp master
			{
			MPI_Recv(&startNode, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&endNode, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
			nodeStart = MPI_Wtime();
			nodeRange[0] = startNode;
			nodeRange[1] = endNode;
			nodeRangeDiff = nodeRange[1] - nodeRange[0];
			matrixNode = (int*) calloc(nodeRangeDiff, sizeof(int));
			nodey = nodeRangeDiff / nx;
			}
			#pragma omp barrier

			#pragma omp for schedule(dynamic, 2)
    			for(i=0;i<nodey;i++){ //row
				//printf("Rank %d Thread %d: Indexes %d to %d\n", rank, omp_get_thread_num(), i*nx, ((i+1)*nx)-1);
				//fflush(stdout);
        			for(j=0;j<nx;j++){ //col
            				index=i*nx+j;
            				//C=x0+iy0
            				x0=xStart+(1.0*j/nx)*(xEnd-xStart);
            				y0=yStart+(1.0*(i+startNode/nx)/ny)*(yEnd-yStart);
            				//Z0=0
            				x=0;
            				y=0;
            				iter=0;

            				while(iter<maxIter){
                				temp=x*x-y*y+x0;
                				y=2*x*y+y0;
                				x=temp;
                				iter++;
                				if(x*x+y*y>4){
                    					break;
                				}
            				}//end of while
            				matrixNode[index]=iter;
        			}
    			}
			#pragma omp master
			{
			nodeEnd = MPI_Wtime();
			nodeTotal += nodeEnd - nodeStart;
			MPI_Send(nodeRange, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
			MPI_Send(matrixNode, nodeRangeDiff, MPI_INT, 0, 0, MPI_COMM_WORLD);
			free(matrixNode);
			}
		}
		}
	}

	printf("RANK %d at BARRIER wit nodeTotal = %f\n", rank, nodeTotal);
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Gather(&nodeTotal, 1, MPI_DOUBLE, nodeTimes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(rank == 0){
    		int dims[2];
    		dims[0]=ny;
    		dims[1]=nx;

		for(int k=0; k<numranks; k++){
			printf("Rank %d time: %f\n", k, nodeTimes[k]);
		}

		timeEnd = omp_get_wtime();
		printf("Total time: %f\n", timeEnd-timeStart);
		//printf("Random val: %d\n", matrix[20]);

/**
		fflush(stdout);
		printf("------------------\n-----------------\n---------------\n");
		fflush(stdout);
		for(int w=0; w<chunkSize; w++){
			printf("index %d: %d\n", w, matrix[w]);
		}
**/

    		matToImageColor("mandelbrot.jpg", matrix, dims);
	}

	MPI_Finalize();
    	return 0;
}

