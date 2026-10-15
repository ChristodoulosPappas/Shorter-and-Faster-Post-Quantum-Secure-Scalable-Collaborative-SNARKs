#pragma once
#include "config_pc.hpp"
#include "utils.hpp"
#include "MPI_utils.hpp"
#include "delegate_protocol.h"
#include "merkle_tree.h"
#include <math.h>
#include "Whir.h"

vector<int> get_degrees(int rounds, int Row_Size);
vector<int> get_queries(int rounds, vector<vector<int>> rates);
vector<int> get_queries(int rounds, vector<vector<int>> rates);
void get_witness(vector<vector<F>> &Shares, vector<vector<F>> &Random, vector<vector<int>> rates,int rounds,int N, int M);
void new_commit(vector<vector<_hash>> &C, 
                vector<vector<vector<F>>> &Codeword, 
                vector<vector<vector<F>>> &Commit_data, 
                vector<vector<F>> &Shares, vector<vector<F>> &Random, 
                vector<vector<int>> rate);
F evaluate_vector(vector<F> r, vector<vector<vector<F>>> &Commit_data);
void open(vector<vector<_hash>> &C, vector<vector<vector<F>>> &Codeword, vector<vector<vector<F>>> &Commit_data, 
          vector<F> r, F _y, vector<vector<int>> rates);
F fast_eval_powers_y(F c_r, int point, int N);
F fast_eval_powers_x(vector<vector<F>> &L, int point, int rows, int rate);