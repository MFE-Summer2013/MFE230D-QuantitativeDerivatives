#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include "mkl.h"
#include "mkl_lapacke.h"
#include "mkl_cblas.h"
#include "mkl_vsl.h"

using namespace std;

void displayMatrix(ofstream& myfile, const double* M, unsigned int nrows, unsigned int ncols);
double* relativeError(const double* est, const double* tgt, unsigned int nrows, unsigned int ncols);

int main()
{
	ofstream myfile;
	myfile.open ("test.txt");
	myfile << "THIS IS THE COMPUTATIONAL PROGRAM FOR ASSIGNMENT 1 \n\n";
	myfile << "INITIALIZATION OF MATRIX" << endl;
	unsigned int m = 3, n = 3;

	double CORR[] = {1.0, 0.8, 0.6,
					  0.8, 1.0, 0.7,
					  0.6, 0.7, 1.0};


	double SIGMA[] = {1.0, 0.8, 0.6,
					  0.8, 1.0, 0.7,
					  0.6, 0.7, 1.0};

	myfile << "The intialized Matrix R: " << endl;

	displayMatrix(myfile, SIGMA, m, n);

	myfile << "--Perform Cholesky Decomposition (LAPACK ROUTINE)" << endl;
	LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', m, SIGMA, n);
	
	displayMatrix(myfile, SIGMA, m, n);
	for(int i = 0; i < 3; ++i){
		for(int j = 2; j > i; --j)
			SIGMA[i*n + j] = 0.0;
	}

	displayMatrix(myfile, SIGMA, m, n);


	
	myfile << "--Initization of VSL Random Stream" << endl;
	VSLStreamStatePtr stream;
	int seed = 2;
	vslNewStream(&stream, VSL_BRNG_MT19937, seed);
	myfile << "Random Stream is Successfully Initiated \n\n" << endl;

	for(int times = 1; times <= 6; ++times){

		unsigned int NUMBER_OF_ENTRIES = pow(10,times);

		myfile << "===SIMULATIONS OF RANDOM VARIABLES=====" << endl;
		myfile << "   SAMPLE SIZE: " << NUMBER_OF_ENTRIES << "\n\n";

		myfile << "--Generate Gaussian Random Variable, Standard Normal" << endl;
		double *RANDOMGAUSSIAN = new double[NUMBER_OF_ENTRIES * n];
	
		time_t simulationStart, simulationEnd;
		time(&simulationStart);
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, NUMBER_OF_ENTRIES * n, RANDOMGAUSSIAN, 0.0,1.0);
		time(&simulationEnd);
		double seconds = difftime(simulationEnd, simulationStart);
		myfile << "--Simulation of " << NUMBER_OF_ENTRIES * n << " tuples of standard normal variables is completed" << endl;
		myfile << "  Takes " << seconds << " seconds" << endl;

		myfile << "--Recovering Correlated Random Variables (CBLAS_DGEMM)" << endl;
	
		double *RANDOM_CORRELATED = new double[NUMBER_OF_ENTRIES * n];
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, NUMBER_OF_ENTRIES, n, m, 1.0, RANDOMGAUSSIAN, n, SIGMA, n, 0, RANDOM_CORRELATED, n);
		delete [] RANDOMGAUSSIAN;
	
		myfile << "--Finished Simulation and Sampling \n" << endl;

		myfile << "--Calculating the Summary Statistics\n" << endl;
		VSLSSTaskPtr task;
		MKL_INT dim;
		MKL_INT nobs;
		MKL_INT x_storage;
		MKL_INT cov_storage;
		MKL_INT cor_storage;
	
		dim         = n;
		nobs        = NUMBER_OF_ENTRIES;
		x_storage   = VSL_SS_MATRIX_STORAGE_COLS;
		cov_storage = VSL_SS_MATRIX_STORAGE_FULL;
		cor_storage = VSL_SS_MATRIX_STORAGE_FULL;

		double *cov = new double[n*n];
		double *cor = new double[n*n];
		double *mean = new double[n];

		/***** Create Summary Statistics task *****/
		vsldSSNewTask( &task, &dim, &nobs, &x_storage, RANDOM_CORRELATED, 0, 0 );

		/***** Initialization of the task parameters using FULL_STORAGE
			for covariance/correlation matrices *****/
		vsldSSEditCovCor(task, mean, (double*)cov, &cov_storage, (double*)cor, &cor_storage );

		/***** Compute covariance/correlation matrices using FAST method  *****/
		vsldSSCompute(task, VSL_SS_COV|VSL_SS_COR, VSL_SS_METHOD_FAST);

		myfile << "-- Finish calculation of summary statistics,OUTPUT:" << endl;
		myfile << "Computed Mean" << endl;
		displayMatrix(myfile, mean, 1, 3);
		myfile << "Covariance Matrix" << endl;
		displayMatrix(myfile, cov, n,n);
		myfile << "Correlation Matrix" << endl;
		displayMatrix(myfile, cor, n,n);
	
		myfile << "ERROR OF ENTRIES:" << endl;
		double* rErr = relativeError(cor, CORR, n, n);
		displayMatrix(myfile, rErr, n, n);

		delete [] cov;
		delete [] cor;
		delete [] mean;
		delete [] rErr;
	
	}

	system("pause");
}

void displayMatrix(ofstream& myfile, const double* M, unsigned int nrows, unsigned int ncols)
{
	for(int i = 0; i < nrows; ++i){
		myfile << "\t";
		for(int j = 0; j < ncols; ++j)
			myfile << setprecision(5) << M[i*ncols+j] << "\t";
		myfile << endl;
	}
	myfile << endl;
}

double* relativeError(const double* est, const double* tgt, unsigned int nrows, unsigned int ncols)
{
	double *errorMatrix = new double[nrows*ncols];
	for(int i = 0; i < nrows * ncols; ++i){
		errorMatrix[i] = (est[i] - tgt[i]) / (tgt[i]);
	}
	return errorMatrix;
}