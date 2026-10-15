#include "config_pc.hpp"
#include "utils.hpp"
#include "MPI_utils.hpp"
#include "merkle_tree.h"
#include <math.h>


pair<F,vector<F>> prove_multi_folding(vector<vector<F>> &segments, vector<F> L, vector<F> y, vector<vector<u32>> queries, int n, double &ps);
pair<F,vector<F>> prove_extension(vector<vector<F>> &segments, vector<vector<F>> &extended_segments, vector<F> o, vector<int> queries,int rate,double &ps);
F delegate_beta_eval(vector<F> r, vector<vector<F>> L, F y, int offset, double &ps);

F delegate_lagrange_eval(vector<vector<F>> L1, vector<vector<F>> L2, F y, int offset, double &ps);
F delegate_lagrange_eval(vector<vector<F>> L1, vector<vector<F>> L2, F y, int offset, int queries, double &ps);