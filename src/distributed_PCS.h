#pragma once
#include "config_pc.hpp"
#include "utils.hpp"
#include "MPI_utils.hpp"
#include "delegate_protocol.h"
#include "merkle_tree.h"
#include <math.h>
#include "Whir.h"
#include "Fiat_Shamir.h"
#include "coPCS.h"
#include "coPCS_utils.h"



void dOpen(vector<vector<F>> &commit_data, vector<vector<F>> &codewords, MT &C, vector<F> r, F y, int N);
void Batch_dCommit(vector<F> &arr, vector<F> &arr2, vector<vector<F>> &commit_data, vector<vector<F>> &Codeword, MT &Com, int rate, int N, int BATCH_SIZE);
void prepare_data(int N, int rate);
void dOpen_batch(vector<vector<vector<F>>> &commit_data, vector<vector<vector<F>>> &codewords, vector<MT> &C, vector<vector<F>> r, vector<F> y, int N);
void dCommit(vector<F> &arr, vector<F> &codeword, MT &C, int rate, int N, int batch_size = 1);
void PIOP_dOpen_batch(vector<vector<vector<F>>> &commit_data, vector<vector<vector<F>>> &codewords, vector<MT> &C, vector<vector<F>> r, vector<F> y, vector<bool> Verify, int N);
void PIOP_dOpen_batch_new(vector<vector<vector<F>>> &commit_data, vector<vector<vector<F>>> &codewords, vector<MT> &C, vector<vector<F>> r, vector<F> y, vector<bool> Verify, int N);