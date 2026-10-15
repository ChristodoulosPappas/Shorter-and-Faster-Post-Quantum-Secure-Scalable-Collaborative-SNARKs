#pragma once
#include "config_pc.hpp"
#include "utils.hpp"
#include "MPI_utils.hpp"
#include "delegate_protocol.h"
#include "merkle_tree.h"
#include <math.h>
#include "Whir.h"
#include "Fiat_Shamir.h"



struct MT{
    vector<vector<_hash>> Base_MT;
    vector<vector<_hash>> Root_MT;
};


void PCS_commit(MT &C, 
                vector<vector<F>> &Codeword, 
                vector<vector<F>> &Commit_data, 
                vector<F> &Shares, vector<F> &Random_Shares, 
                vector<vector<int>> rate, int N);


void prepare_parameters(vector<vector<int>> rates,int rounds,int N, int M);
//void PCS_open(MT MT_C, vector<vector<F>> &Codeword, vector<vector<F>> &Commit_data, vector<F> r, F _y, vector<vector<int>> rates, int N);
void distributed_commit(MT &Com, int N);
void distributed_commit_batch(vector<MT> &Com, int N);
void PCS_open(MT &MT_C, vector<vector<F>> &Codeword, vector<vector<F>> &Commit_data, vector<vector<F>> input_vec, vector<vector<F>> _Lagrange_Evals_input, F K, vector<vector<int>> rates, int N);
void distributed_MT(vector<vector<F>> &Codeword, MT &Com, int N);
void PCS_open_new(MT &MT_C, vector<vector<F>> &Codeword, vector<vector<F>> &Commit_data, vector<vector<F>> input_vec, vector<vector<F>> _Lagrange_Evals_input, F K, vector<vector<int>> rates, int N, bool isPIOP );
void distributed_commit2(MT &Com, int N);
vector<F> tensor_eval_vector_plaintext(vector<u32> point, int cols, int rows, int rate);