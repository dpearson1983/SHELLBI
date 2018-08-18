#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <fftw3.h>
#include <omp.h>
#include "include/cic.h"
#include "include/cosmology.h"
#include "include/file_io.h"
#include "include/galaxy.h"
#include "include/harppi.h"
#include "include/line_of_sight.h"
#include "include/power.h"
#include "include/shells.h"
#include "include/tpods.h"
#include "include/transformers.h"
#include "include/bispec.h"

int main(int argc, char *argv[]) {
    parameters p(argv[1]);
    p.print();
    
    // Setup the cosmology class object with values needed to get comoving distances.
    // NOTE: The distances returned will be in h^-1 Mpc, so the value of H_0 is not actually used.
    cosmology cosmo(p.getd("H_0"), p.getd("Omega_M"), p.getd("Omega_L"));
    
    // Storage for values
    vec3<double> gal_pk_nbw = {0.0, 0.0, 0.0};
    vec3<double> gal_bk_nbw = {0.0, 0.0, 0.0};
    vec3<double> ran_pk_nbw = {0.0, 0.0, 0.0};
    vec3<double> ran_bk_nbw = {0.0, 0.0, 0.0};
    
    // Both r_min and L will be set automatically when reading in randoms
    vec3<double> r_min;
    vec3<double> L;
    
    vec3<int> N = {p.geti("Nx"), p.geti("Ny"), p.geti("Nz")};
    
    std::cout << "Setting file type variables..." << std::endl;
    FileType dataFileType, ranFileType;
    setFileType(p.gets("dataFileType"), dataFileType);
    setFileType(p.gets("ranFileType"), ranFileType);
    
    double alpha;
    std::vector<double> delta(N.x*N.y*N.z);
    
    std::cout << "Reading in data and randoms files..." << std::endl;
    // Since the N's can be large values, individual arrays for the FFTs will be quite large. Instead
    // of reusing a fixed number of arrays, by using braced enclosed sections, variables declared
    // within the braces will go out of scope, freeing the associated memory. Here, given how the
    // backend code works, there are two temporary arrays to store the galaxy field and the randoms
    // field.
    {
        std::vector<double> ran(N.x*N.y*N.z);
        std::vector<double> gal(N.x*N.y*N.z);
        
        std::cout << "   Getting randoms..." << std::endl;
        readFile(p.gets("randomsFile"), ran, N, L, r_min, cosmo, ran_pk_nbw, ran_bk_nbw, 
                 p.getd("z_min"), p.getd("z_max"), ranFileType);
        std::cout << "   Getting galaxies..." << std::endl;
        readFile(p.gets("dataFile"), gal, N, L, r_min, cosmo, gal_pk_nbw, gal_bk_nbw, p.getd("z_min"),
                 p.getd("z_max"), dataFileType);
        
        alpha = gal_pk_nbw.x/ran_pk_nbw.x;
        
        std::cout << "   Computing overdensity..." << std::endl;
        #pragma omp parallel for
        for (size_t i = 0; i < gal.size(); ++i) {
            delta[i] = gal[i] - alpha*ran[i];
        }
    }
    std::cout << "Done!" << std::endl;
    
    std::vector<double> kx = fft_freq(N.x, L.x);
    std::vector<double> ky = fft_freq(N.y, L.y);
    std::vector<double> kz = fft_freq(N.z, L.z);
    
    std::vector<double> A_0(N.x*N.y*2*(N.z/2 + 1));
    std::vector<double> A_2(N.x*N.y*2*(N.z/2 + 1));
    
    get_A0(delta, A_0, N);
    get_A2(delta, A_2, N, L, r_min);
    CICbinningCorrection((fftw_complex *)A_0.data(), N, L, kx, ky, kz);
    CICbinningCorrection((fftw_complex *)A_2.data(), N, L, kx, ky, kz);
    
    int numShells = (p.getd("k_max") - p.getd("k_min"))/p.getd("Delta_k");
    std::vector<double> k_P;
    std::vector<std::vector<vec3<double>>> shells;
    for (int i = 0; i < numShells; ++i) {
        double k_shell = p.getd("k_min") + (i + 0.5)*p.getd("Delta_k");
        std::vector<vec3<double>> shell = get_shell(N, L, k_shell, p.getd("Delta_k"));
        std::cout << k_shell << " " << shell.size() << std::endl;
        shells.push_back(shell);
        k_P.push_back(k_shell);
    }
    
    std::vector<double> P(numShells);
    std::vector<int> N_k(numShells);
    double SN = gal_pk_nbw.y + alpha*alpha*ran_pk_nbw.y;
    binFrequencies((fftw_complex *)A_0.data(), P, N_k, N, kx, ky, kz, p.getd("Delta_k"), p.getd("k_min"),
                   p.getd("k_max"), SN);
    normalizePower(P, N_k, gal_pk_nbw.z);
    writePowerSpectrumFile(p.gets("pkFile"), k_P, P);
    
    std::vector<vec3<double>> ks;
    int numBispecBins = getNumBispecBins(p.getd("k_min"), p.getd("k_max"), p.getd("Delta_k"), ks);
    
    std::vector<size_t> N_tri(numBispecBins);
    std::vector<double> B_0(numBispecBins);
    std::vector<double> B_2(numBispecBins);
    
    int bispecBin = 0;
    for (int i = 0; i < numShells; ++i) {
        double k1 = p.getd("k_min") + (i + 0.5)*p.getd("Delta_k");
        for (int j = i; j < numShells; ++j) {
            double k2 = p.getd("k_min") + (j + 0.5)*p.getd("Delta_k");
            for (int k = j; k < numShells; ++k) {
                double k3 = p.getd("k_min") + (k + 0.5)*p.getd("Delta_k");
                if (k3 <= k1 + k2) {
                    double shotnoise = get_monopole_shotnoise(P[i], P[j], P[k], alpha, gal_bk_nbw, ran_bk_nbw);
                    get_bispectrum(shells[i], shells[j], k3, p.getd("Delta_k"), (fftw_complex *)A_0.data(),
                                   (fftw_complex *)A_2.data(), N, L, B_0, B_2, N_tri, bispecBin, gal_bk_nbw.z, 
                                   shotnoise);
                    std::cout << k1 << " " << k2 << " " << k3 << " " << B_0[bispecBin] << " ";
                    std::cout << B_2[bispecBin] << " " << N_tri[bispecBin] << std::endl;
                    bispecBin++;
                }
            }
        }
    }
    
    writeBispectrumFile(p.gets("outFile"), ks, B_0, B_2, N_tri);
    
    return 0;
}
