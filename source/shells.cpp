#include <vector>
#include <cmath>
#include "../include/tpods.h"
#include "../include/shells.h"
#include "../include/transformers.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

std::vector<vec3<double>> get_shell(const vec3<int> N, const vec3<double> L, double k_shell, double Delta_k) {
    std::vector<vec3<double>> shell;
    
    std::vector<double> kx = fft_freq(N.x, L.x);
    std::vector<double> ky = fft_freq(N.y, L.y);
    std::vector<double> kz = fft_freq(N.z, L.z);
    
    for (size_t i = 0; i < N.x; ++i) {
        for (size_t j = 0; j < N.y; ++j) {
            for (size_t k = 0; k < N.z; ++k) {
                double k_mag = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k]);
                
                if (k_mag >= k_shell - 0.5*Delta_k && k_mag < k_shell + 0.5*Delta_k) {
                    vec3<double> k_vec = {kx[i], ky[j], kz[k]};
                    shell.push_back(k_vec);
                }
            }
        }
    }
    
    return shell;
}
