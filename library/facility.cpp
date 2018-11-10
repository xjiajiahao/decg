// #include <stdlib.h>
#include <stdio.h>
#include <string.h>
// #include <random>
#include "facility.h"
#define DEBUG 0

void stochastic_gradient_extension(double *x, long dim, double *ratings, long num_rows, long nnz, long sample_times, long *indices_in_ratings, double *stochastic_gradient, double *rand_vec, double (*julia_rand)())
{
#if DEBUG
    // printf("dim: %ld, num_rows: %ld, nnz: %ld\n", dim, num_rows, nnz);
#endif
    // Step 1. find the the indices in the ratings matrix
    if (num_rows != 2)
    {
        printf("[Error] num_rows must be 2!\n");
        return;
    }
    long i, j, k, tmp_idx, curr_idx;
    double tmp_f, curr_partial;
    memset(indices_in_ratings, 0, dim * sizeof(indices_in_ratings[0]));  // @NOTE indices_in_ratings is a Vector{Int64} of size dim-by-1
    for (j = 0; j < nnz; j++) {
        curr_idx = ratings[j<<1];
#if DEBUG
        // if (j==0)
        //    printf("ratings[0]: %ld\n", curr_idx);
#endif
        indices_in_ratings[curr_idx-1] = j+1;  // @NOTE the original array counts from 1, so we have to store j+1 instead of j
    }

    // Step 2. repeate computing stochastic gradient for multiple times and take average
    // std::random_device rd;  //Will be used to obtain a seed for the random number engine
    // std::mt19937 genSample(rd()); //Standard mersenne_twister_engine seeded with rd()
    // std::uniform_real_distribution<> uniformDist(0.0, 1.0);

    for (k = 0; k < sample_times; k++)
    {
        // Step 2.1 generate a random vector in [0.0, 1.0]
        for (j = 0; j < nnz; j++)
        {
            rand_vec[j] = julia_rand();  // julia's rand() function is faster than C++'s uniform_real_distribution
            // rand_vec[j] = uniformDist(genSample);
            // rand_vec[j] = j*1.0/nnz;
        }
#if DEBUG
        // printf("rand_vec[0]: %lf\n", rand_vec[0]);
#endif

        // Stpe 2.2 compute the partital derivative of each coordinate
        for (i = 0; i < dim; i++)
        {
            curr_idx = indices_in_ratings[i];
            // if the movie has not rated, just ignore that coordinate since f(X\curr_idx) = f(X + curr_idx),
            if (curr_idx == 0)
                continue;
            // compute f(X\curr_idx) @NOTE f(X+curr_idx) is simply max{f(X\curr_idx), f({curr_idx})}
            tmp_f = 0;
            for (j = 0; j < nnz; j++)
            {
                if (j+1 == curr_idx)
                    continue;
                tmp_idx = ratings[j<<1];  // @NOTE tmp_idx counts from 1, but in C, we count from 0
                if (rand_vec[j] <= x[tmp_idx - 1])
                {
                    tmp_f = ratings[(j << 1) + 1];
                    break;
                }
            }
            curr_partial = ratings[((curr_idx-1) << 1) + 1] - tmp_f;
            curr_partial = curr_partial > 0.0 ? curr_partial : 0.0;
#if DEBUG
            // if (k == 0 && i == 0)
            //     printf("curr_partial: %lf\n", curr_partial);
#endif
            stochastic_gradient[i] += curr_partial;
        }
    }
}
