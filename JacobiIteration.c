/*******************************************
*Samuel Mendheim - COMP233 - JacobiIteration
*
* This program will be using Jacobi Iterations
* to generate a ppm image. OpenMP was used to
* speed up the time on this program
*
* The original program was written by Argonne National Laboratory
* The original code for this project can be found here:
*	https://www.mcs.anl.gov/research/projects/mpi/tutorial
*   /mpiexmpl/src/jacobi/C/main.html
********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>


// max iterations to happen
#define MAXN 1000
// bounds for the array size
#define WIDTH 1000
#define HEIGHT 1000
// elements for the values of the plate
#define NORTH 100
#define EAST 0
#define SOUTH 100
#define WEST 0
// interior values for the plates
#define INTERIOR ((NORTH + EAST + SOUTH + WEST) / 4)


int main(argc, argv)
int argc;
char** argv;
{
   // start of the program
   printf("\n\nSamuel Mendheim ~ COMP 233 ~ Jacobi Iteration\n\n");

	// color values used to create the image
	int colorBlue, colorRed, colorValue;
	// clock times for calculating time
	double startTime, endTime, totalTime;
	// total calculated time, epsilonValue, diffNorm and gDiffNorm
	float epsilonValue, diffNorm, gDiffNorm;
	// variables used for loops, values, counting iterations and num threads
	int rank, value, size, errorCount, totalError, r, c, itCount, numThreads;
	// the first row, last row and max iterations
	int firstRow, lastRow, maxIter;

	// start timing the program
	startTime = omp_get_wtime();


	// allocate a chunck of memory for the xLocal array
	float** xLocal = (float**)malloc(WIDTH * sizeof(float*));

	for (r = 0; r < WIDTH; r++) {
		// allocate the inside memory
		xLocal[r] = (float*)malloc(HEIGHT * sizeof(float));
	}

	// allocate a chunck of memory for the xNew array
	float** xNew = (float**)malloc(WIDTH * sizeof(float*));

	for (r = 0; r < WIDTH; r++) {
		// allocate the inside memory
		xNew[r] = (float*)malloc(HEIGHT * sizeof(float));
	}


	// getting the first point to iterate through
	firstRow = 1;
	// getting the last point to iterate through
	lastRow = MAXN;

	// getting the command line arguments
	maxIter = atoi(argv[1]);
	epsilonValue = atof(argv[2]);
	numThreads = atoi(argv[3]);
 
 


	/* Fill the data as specified */
	for (r = 1; r < MAXN - 1; r++) {
		for (c = 1; c < MAXN - 1; c++) {
			// filling the array with the rank of the node
			xLocal[r][c] = INTERIOR;
		}
	}
	// for loop to full up the rows
	for (c = 0; c < MAXN; c++) {
		// setting the first row of every element to EAST
		xLocal[c][lastRow - 1] = EAST;
		// setting the last row of every element to WEST
		xLocal[c][firstRow - 1] = WEST;
		// setting the first row of every element in xNew to EAST
		xNew[c][lastRow - 1] = EAST;
		// setting the first row of every element in xNew to WEST
		xNew[c][firstRow - 1] = WEST;
	}
	// for loop to fill up the columns
	for (r = 0; r < MAXN; r++) {
		// setting the first columns of every element to NORTH
		xLocal[firstRow - 1][r] = NORTH;
		// setting the last columns of every element to SOUTH
		xLocal[lastRow - 1][r] = SOUTH;
		// setting the first columns of every element in xNew to NORTH
		xNew[firstRow - 1][r] = NORTH;
		// setting the last columns of every element in xNew to SOUTH
		xNew[lastRow - 1][r] = SOUTH;
	}
 
  // table formatting
	printf("\n-------------------------------------\n");
	printf("      itCount\t |\t diffNorm");
	printf("\n-------------------------------------\n");

	


	itCount = 0;
	do {
		/* Compute new values (but not on boundary) */
		itCount++;
		diffNorm = 0.0;

#pragma omp parallel for reduction(+:diffNorm) private(r,c) num_threads(numThreads)

		for (r = firstRow; r < lastRow - 1; r++) {
			for (c = 1; c < MAXN - 1; c++) {

				// calculating the values for all of the new values
				xNew[r][c] = (xLocal[r][c + 1] + xLocal[r][c - 1] +
					xLocal[r + 1][c] + xLocal[r - 1][c]) / 4.0;

				// calculating the diffNorm of the variables
				diffNorm += (xNew[r][c] - xLocal[r][c]) *
					(xNew[r][c] - xLocal[r][c]);
			}
		}

		/* Only transfer the interior points */
	// make a temporary block of memory
		float** temp = xLocal;
		// move the new values to the local array
		xLocal = xNew;
		// move the new values to the temporary block of memory
		xNew = temp;

		// calculating the gDiffNorm
		gDiffNorm = sqrt(diffNorm);

		if (itCount % 1000 == 0) {
			// every thousand iterations print out the ongoing results
			printf("\t%d\t | \t%e\t\n", itCount, gDiffNorm);
		}
	} while (gDiffNorm > epsilonValue && itCount < maxIter);

	// making a new ppm file to have our image
	FILE* jacobiImage = fopen("JacobiImage.ppm", "w+");
	// printing the inital requirements for the ppm image
	fprintf(jacobiImage, "P3\n1000 1000\n255\n");
  fprintf(jacobiImage, "# Samuel Mendheim ~ COMP 233 ~ Jacobi Iteration\n");
  fprintf(jacobiImage, "# This image took %d iterations to converge\n", itCount);

	for (r = 0; r < MAXN; r++) {
		for (c = 0; c < MAXN; c++) {
			// getting the value of the color at the point in the local array
			colorValue = xLocal[r][c];
			// finding the blue percentage of the photo
			colorBlue = 100 - colorValue;
			// finding the red percentage of the photo
			colorRed = 100 - colorBlue;
			// calculating the blue color
			colorBlue = (colorBlue * 0.01) * 255;
			// calculating the red color
			colorRed = (colorRed * 0.01) * 255;
			// printing the results to a file
			fprintf(jacobiImage, "%d 0 %d\n", colorRed, colorBlue);
		}
	}

	

	for (r = 0; r < MAXN; r++) {
    // freeing up the memory
		free(xLocal[r]);
		free(xNew[r]);
	}
  // freeing up the original pointer
	free(xLocal);
	free(xNew);

	// getting the end time of the calculation
	endTime = omp_get_wtime();

	// calculating the total time of the work horse do while loop
	totalTime = (endTime - startTime);

	printf("-------------------------------------\n");

	// printing the total time taken for the program
	printf("\nTotal time: %.2f seconds\n\n", totalTime);
 
  printf("-------------------------------------\n");

	printf("<--------Normal Termination-------->\n");

	return 0;
}