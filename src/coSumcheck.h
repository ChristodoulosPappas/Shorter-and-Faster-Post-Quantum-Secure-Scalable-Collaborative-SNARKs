#pragma once
#include "config_pc.hpp"
#include "utils.hpp"
#include "MPI_utils.hpp"
#include <math.h>
#include "Fiat_Shamir.h"


vector<pair<F,vector<F>>> quadratic_coSumcheck(vector<F> &p1, vector<F> &p2, vector<vector<F>> R, F K, F K_r, vector<int> degrees, int N, double &ps);
vector<pair<F,vector<F>>> cubic_coSumcheck(vector<F> &p1, vector<F> &p2, vector<F> &p3, vector<F> &r1, vector<F> &r2, vector<vector<F>> R, F K, F K_r, vector<int> degrees, int N, double &ps);