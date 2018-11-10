// #ifndef __FACILITY_H__
// #define __FACILITY_H__

#ifdef __cplusplus
extern "C" {
#endif

void stochastic_gradient_extension(double *x, long dim, double *ratings, long num_rows, long nnz, long sample_times, long *indices_in_ratings, double *stochastic_gradient, double *rand_vec, double (*julia_rand)());

#ifdef __cplusplus
}
#endif

// #endif
