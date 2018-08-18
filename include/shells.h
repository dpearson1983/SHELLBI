#ifndef _SHELLS_H_
#define _SHELLS_H_

#include <vector>
#include "tpods.h"

std::vector<vec3<double>> get_shell(const vec3<int> N, const vec3<double> L, double k_shell, double Delta_k);

#endif
