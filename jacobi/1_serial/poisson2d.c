/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>

#define NY 4096
#define NX 4096

double A[NX][NY];
double Anew[NX][NY];
double rhs[NX][NY];

int main(int argc, char** argv)
{
    int iter_max = 1000;
    const double tol = 1.0e-5;

	struct timeval start_time, stop_time, elapsed_time;

    /* ---------------------------------------------
        Initialize Arrays
    --------------------------------------------- */

	// Set rhs
    for (int iy = 1; iy < NY-1; iy++)
    {
        for( int ix = 1; ix < NX-1; ix++ )
        {
            const double x = -1.0 + (2.0*ix/(NX-1));
            const double y = -1.0 + (2.0*iy/(NY-1));
            rhs[iy][ix] = exp(-10.0*(x*x + y*y));
        }
    }

	// Set A
    for(int iy = 0; iy < NY; iy++)
    {
        for(int ix = 0; ix < NX; ix++)
        {
            A[iy][ix]    = 0.0;
        }
    }
    
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", NY, NX);

    /* ---------------------------------------------
        Serial Execution
    --------------------------------------------- */

	// Start timer
	gettimeofday(&start_time, NULL);

    int iter  = 0;
    double error = 1.0;
   
	// Main iteration loop 
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

        for (int iy = 1; iy < NY-1; iy++)
        {
            for( int ix = 1; ix < NX-1; ix++ )
            {
                Anew[iy][ix] = -0.25 * (rhs[iy][ix] - ( A[iy][ix+1] + A[iy][ix-1]
                                                       + A[iy-1][ix] + A[iy+1][ix] ));
                error = fmax( error, fabs(Anew[iy][ix]-A[iy][ix]));
            }
        }
        
        for (int iy = 1; iy < NY-1; iy++)
        {
            for( int ix = 1; ix < NX-1; ix++ )
            {
                A[iy][ix] = Anew[iy][ix];
            }
        }
        
        //Periodic boundary conditions
        for( int ix = 1; ix < NX-1; ix++ )
        {
                A[0][ix]      = A[(NY-2)][ix];
                A[(NY-1)][ix] = A[1][ix];
        }
        for (int iy = 1; iy < NY-1; iy++)
        {
                A[iy][0]      = A[iy][(NX-2)];
                A[iy][(NX-1)] = A[iy][1];
        }
        
        if((iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

	// Stop timer
	gettimeofday(&stop_time, NULL);
	timersub(&stop_time, &start_time, &elapsed_time);
	double runtime = elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0;

	printf("Elapsed Time (s): %8.4f\n", runtime);

    return 0;
}
