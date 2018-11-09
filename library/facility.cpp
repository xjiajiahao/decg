#include <stdlib.h>
#include <string.h>
#include <random>
#include "facility.h"

void stochastic_gradient_extension(double *x, long dim, double *ratings, long num_rows, long nnz, long sample_times, long *indices_in_ratings, double *stochastic_gradient, double *rand_vec)
{
    // Step 1. find the the indices in the ratings matrix
    long i, j, k, tmp_idx, curr_idx;
    double tmp_f, curr_partial;
    memset(indices_in_ratings, 0, dim * sizeof(indices_in_ratings[0]));
    for (j = 0; j < nnz; j++) {
        curr_idx = ratings[j*num_rows];
        indices_in_ratings[curr_idx] = j;
    }

    // Step 2. repeate computing stochastic gradient for multiple times and take average
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 genSample(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> uniformDist(0.0, 1.0);

    for (k = 0; k < sample_times; k++)
    {
        // Step 2.1 generate a random vector in [0.0, 1.0]
        for (j = 0; j < nnz; j++)
            rand_vec[j] = uniformDist(genSample);

        // Stpe 2.2 compute the partital derivative of each coordinate
        for (i = 0; i < dim; i++)
        {
            curr_idx = indices_in_ratings[i];
            // if the movie has not rated, just ignore that coordinate
            if (curr_idx == 0)
                continue;
            // compute f(X\curr_idx) @NOTE f(X+curr_idx) is simply max{f(X\curr_idx), f({curr_idx})}
            tmp_f = 0;
            for (j = 0; j < nnz; j++)
            {
                if (j == curr_idx)
                    continue;
                tmp_idx = ratings[j*num_rows];
                if (rand_vec[j] < x[tmp_idx])
                {
                    tmp_f = ratings[j*num_rows + 1];
                    break;
                }
            }
            curr_partial = ratings[curr_idx * num_rows] - tmp_f;
            curr_partial = curr_partial > 0.0 ? curr_partial : 0.0;
            stochastic_gradient[i] += curr_partial;
        }
    }
}
