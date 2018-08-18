#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include "../include/tpods.h"
#include "../include/bispec.h"
#include "../include/shells.h"
#include "../include/file_io.h"
#include "../include/transformers.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

vec4<size_t> get_index(vec3<double> k, vec3<double> k_f, vec3<int> N) {
    vec4<size_t> index = {0, 0, 0, 0};
    if (k.x < 0) {
        index.x = int(k.x/k_f.x) + N.x;
    } else {
        index.x = int(k.x/k_f.x);
    }
    
    if (k.y < 0) {
        index.y = int(k.y/k_f.y) + N.y;
    } else {
        index.y = int(k.y/k_f.y);
    }
    
    if (k.z < 0) {
        index.z = int(k.z/k_f.z) + N.z;
        vec3<double> k_neg = {-k.x, -k.y, -k.z};
        vec4<size_t> index_neg = get_index(k_neg, k_f, N);
        index.w = index_neg.w;
    } else {
        index.z = int(k.z/k_f.z);
        index.w = index.z + (N.z/2 + 1)*(index.y + N.y*index.x);
    }
    
    return index;
}

double real_part(vec4<size_t> k1, vec4<size_t> k2, vec4<size_t> k3, vec3<int> N, fftw_complex *A_0) {
    fftw_complex dk1, dk2, dk3;
    if (k1.z > N.z/2) {
        dk1[0] = A_0[k1.w][0];
        dk1[1] = -A_0[k1.w][1];
    } else {
        dk1[0] = A_0[k1.w][0];
        dk1[1] = A_0[k1.w][1];
    }
    if (k2.z > N.z/2) {
        dk2[0] = A_0[k2.w][0];
        dk2[1] = -A_0[k2.w][1];
    } else {
        dk2[0] = A_0[k2.w][0];
        dk2[1] = A_0[k2.w][1];
    }
    if (k3.z > N.z/2) {
        dk3[0] = A_0[k3.w][0];
        dk3[1] = -A_0[k3.w][1];
    } else {
        dk3[0] = A_0[k3.w][0];
        dk3[1] = A_0[k3.w][1];
    }
    return dk1[0]*dk2[0]*dk3[0] - dk1[0]*dk2[1]*dk3[1] - dk1[1]*dk2[0]*dk3[1] - dk1[1]*dk2[1]*dk3[0];
}

double real_part(vec4<size_t> k1, vec4<size_t> k2, vec4<size_t> k3, vec3<int> N, fftw_complex *A_0, 
                 fftw_complex *A_2) {
    fftw_complex dk1, dk2, dk3;
    if (k1.z > N.z/2) {
        dk1[0] = A_2[k1.w][0];
        dk1[1] = -A_2[k1.w][1];
    } else {
        dk1[0] = A_2[k1.w][0];
        dk1[1] = A_2[k1.w][1];
    }
    if (k2.z > N.z/2) {
        dk2[0] = A_0[k2.w][0];
        dk2[1] = -A_0[k2.w][1];
    } else {
        dk2[0] = A_0[k2.w][0];
        dk2[1] = A_0[k2.w][1];
    }
    if (k3.z > N.z/2) {
        dk3[0] = A_0[k3.w][0];
        dk3[1] = -A_0[k3.w][1];
    } else {
        dk3[0] = A_0[k3.w][0];
        dk3[1] = A_0[k3.w][1];
    }
    return dk1[0]*dk2[0]*dk3[0] - dk1[0]*dk2[1]*dk3[1] - dk1[1]*dk2[0]*dk3[1] - dk1[1]*dk2[1]*dk3[0];
}

double get_monopole_shotnoise(double P_1, double P_2, double P_3, double alpha, vec3<double> gal_bk_nbw,
                              vec3<double> ran_bk_nbw) {
    double shotnoise = (P_1 + P_2 + P_3)*(gal_bk_nbw.y/gal_bk_nbw.z);
    shotnoise += ((gal_bk_nbw.x/gal_bk_nbw.z) - alpha*alpha*(ran_bk_nbw.x/ran_bk_nbw.z));
    return shotnoise;
}

void get_bispectrum(std::vector<vec3<double>> k1, std::vector<vec3<double>> k2, double k3, double Delta_k,
                    fftw_complex *A_0, fftw_complex *A_2, vec3<int> N, vec3<double> L, 
                    std::vector<double> &B_0, std::vector<double> &B_2, std::vector<size_t> &N_tri, 
                    int bispecBin, double I33, double shotnoise) {
    std::vector<size_t> n_tri(omp_get_max_threads());
    std::vector<double> b_0(omp_get_max_threads());
    std::vector<double> b_2(omp_get_max_threads());
    
    vec3<double> k_f = {fundamental_frequency(N.x, L.x), fundamental_frequency(N.y, L.y),
                        fundamental_frequency(N.z, L.z)};
    
    #pragma omp parallel for
    for (size_t i = 0; i < k1.size(); ++i) {
        int tid = omp_get_thread_num();
        vec4<size_t> k1_index = get_index(k1[i], k_f, N);
        for (size_t j = 0; j < k2.size(); ++j) {
            vec4<size_t> k2_index = get_index(k2[j], k_f, N);
            vec3<double> k3_vec = {-k1[i].x - k2[j].x, -k1[i].y - k2[j].y, -k1[i].z - k2[j].z};
            double k3_mag = sqrt(k3_vec.x*k3_vec.x + k3_vec.y*k3_vec.y + k3_vec.z*k3_vec.z);
            if (k3_mag >= k3 - 0.5*Delta_k && k3_mag < k3 + 0.5*Delta_k) {
                vec4<size_t> k3_index = get_index(k3_vec, k_f, N);
                n_tri[tid]++;
                b_0[tid] += real_part(k1_index, k2_index, k3_index, N, A_0);
                b_2[tid] += real_part(k1_index, k2_index, k3_index, N, A_0, A_2);
            }
        }
    }
    
    for (int i = 1; i < omp_get_max_threads(); ++i) {
        n_tri[0] += n_tri[i];
        b_0[0] += b_0[i];
        b_2[0] += b_2[i];
    }
    
    N_tri[bispecBin] = n_tri[0];
    B_0[bispecBin] = b_0[0]/(I33*n_tri[0]) - shotnoise;
    B_2[bispecBin] = b_2[0]/(I33*n_tri[0]);
}

int getNumBispecBins(double k_min, double k_max, double binWidth, std::vector<vec3<double>> &ks) {
    int totBins = 0;
    int N = (k_max - k_min)/binWidth;
    
    for (int i = 0; i < N; ++i) {
        double k_1 = k_min + (i + 0.5)*binWidth;
        for (int j = i; j < N; ++j) {
            double k_2 = k_min + (j + 0.5)*binWidth;
            for (int k = j; k < N; ++k) {
                double k_3 = k_min + (k + 0.5)*binWidth;
                if (k_3 <= k_1 + k_2 && k_3 <= k_max) {
                    totBins++;
                    vec3<double> kt = {k_1, k_2, k_3};
                    ks.push_back(kt);
                }
            }
        }
    }
    
    return totBins;
}
