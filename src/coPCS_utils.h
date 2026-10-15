
#include "config_pc.hpp"
#include "constants.h"
#include <vector>
#include <math.h>
#include "utils.hpp"

#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <mutex>
#include "polynomial.h"
//#include "coPIOP.h"
//#include "coSumcheck.h"


void RS_fold(vector<F> &codeword, F a);
void setup(vector<vector<F>> &R_shares, int N, int M, int l, int k, int _k);
void encode_protocol_step2(vector<F> &row, vector<F> &rand, vector<F> &codeword);
void RS_multifold2(vector<F> &codeword, F a, int fold_degree);
void RS_multifold3(vector<F> &codeword, vector<F> a, int fold_degree);
void RS_multifold4(vector<F> &codeword, vector<F> a, int fold_degree);
void RS_multifold5(vector<F> &codeword, vector<F> a, int fold_degree);
F RS_multifold_segment(vector<F> segment, vector<F> L, int n, int idx);
vector<F> get_RS_multifold_segment(vector<F> segment, vector<F> L, int n, int idx);
vector<F> get_RS_multifold_reduced_matrix(vector<F> r, int n, int seg_n);
F get_RS_multifold_segment_tensor(vector<F> segment, vector<F> r, int n, int idx);
void RS_distributed_fold(vector<F> &codeword, vector<F> a, int fold_degree, int seg_size, int N, int id);