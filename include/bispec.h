#ifndef _BISPEC_H_
#define _BISPEC_H_

#include <vector>
#include <fftw3.h>
#include "tpods.h"

double get_monopole_shotnoise(double P_1, double P_2, double P_3, double alpha, vec3<double> gal_bk_nbw,
                              vec3<double> ran_bk_nbw);

void get_bispectrum(std::vector<vec3<double>> k1, std::vector<vec3<double>> k2, double k3, double Delta_k,
                    fftw_complex *A_0, fftw_complex *A_2, vec3<int> N, vec3<double> L, 
                    std::vector<double> &B_0, std::vector<double> &B_2, std::vector<size_t> &N_tri, 
                    int bispecBin, double I33, double shotnoise);

int getNumBispecBins(double k_min, double k_max, double binWidth, std::vector<vec3<double>> &ks);

#endif
