#pragma once
#include "config_pc.hpp"
#include "utils.hpp"
#include "MPI_utils.hpp"
#include "merkle_tree.h"
#include <math.h>
extern int whir_k;


void Whir_Commit(vector<F> &arr, vector<F> &codeword, vector<vector<_hash>> &MT, int rate, int kappa = whir_k);
void Whir_Open(vector<F> &arr, vector<F> &codeword, vector<vector<_hash>> &MT, vector<pair<F,vector<F>>> claims, int rate);
//void Whir_Batch_Open(vector<vector<F>> &arrays, vector<vector<F>> &codewords, vector<vector<vector<_hash>>> &MTs, vector<vector<pair<F,vector<F>>>> claims, int rate);
vector<vector<F>> aggregate_claims(vector<vector<F>> &arrays, vector<F> &arr, vector<F> &b_arr, vector<vector<pair<F,vector<F>>>> claims, F &K );
void aggregate_commitments(vector<F> &codeword, vector<vector<_hash>> &MT, 
                            vector<vector<F>> &codewords_aggr, vector<vector<vector<_hash>>> &MT_aggr, vector<F> r, int rate);
pair<F,vector<F>> _Whir_Open(vector<F> &arr, vector<F> &codeword, vector<vector<_hash>> &MT, vector<F> &b, vector<vector<pair<F,vector<F>>>> claims, vector<F> challenges, F K, int rate, bool Verify = true);
void Whir_Commit_Codeword(vector<F> &codeword, vector<vector<_hash>> &MT, int rate);
void Whir_Batch_Open(vector<vector<F>> &arrays, vector<vector<F>> &codewords, vector<vector<vector<_hash>>> &MTs, vector<F> evals, vector<vector<F>> &betas, int rate, bool Verify);
void get_whir_rates(int rate);
void Whir_Commit_initial(vector<F> &arr, vector<F> &codeword, vector<vector<_hash>> &MT, int rate, int kappa);