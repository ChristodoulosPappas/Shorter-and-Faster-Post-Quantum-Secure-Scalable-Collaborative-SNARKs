#pragma once
#include "coPIOP.h"
#include "Distributed_Sumcheck.h"
#include "Fiat_Shamir.h"
#include "MPI_utils.hpp"
#include "coPCS.h"
#include "coSumcheck.h"
#include "coSumcheck_MPI.h"
#include "distributed_PCS.h"
#include "sparse_eval.hpp"
#include "timer.hpp"
#include <cstdlib>
#include <cstring>
int com_rounds = 0;
// R1CS matrixes
vector<vector<pair<int, int>>> A, B, C;
// Transposed R1CS matrixes
vector<vector<pair<int, int>>> tA, tB, tC;
int logm, logn;
bool bit_method = false;
int index_rate = 4;
// int index_rate = 8;
bool data_parallel = false;
extern bool isLAN;
extern double ps_plain;
double cm = 0.0;
int gate_ctr = 0;
timer pt, pt_cp, temp_pc;
timer_cpu pt_cpu;
extern timer sch_com;
MT index_Com, witness_Com;
int min_seg_size;
extern vector<vector<F>> mul_tree_row_data, mul_tree_codeword;
extern MT MulTree_Com;
extern vector<pair<F, vector<F>>> MulTree_Claims;
// int witness_commit_rate = 8;
int witness_commit_rate = 4;
int witness_batch_size = 8;

MT R1_MT, R2_MT;
vector<F> R1_codeword, R2_codeword;

bool debug_mode = true;
timer vt;
int mask_l = 129;

vector<vector<F>> whir_polys, whir_c, whir_betas;
vector<vector<vector<_hash>>> whir_MTs;
vector<F> whir_claims;
vector<int> whir_sizes;
vector<int> whir_rates;
vector<int> sorted_indexes;
vector<int> sorted_whir_rates;
#include <unordered_map>
unordered_map<int, int> whir_kappas;

vector<F> sumcheck1_ip, sumcheck2_ip, sumcheck3_ip;
vector<vector<F>> _sumcheck1_ip, _sumcheck2_ip, _sumcheck3_ip;
extern vector<vector<F>> ip_vec;

void transpose_R1CS_matrixes() {
  tA.resize(1 << logn);
  tB.resize(1 << logn);
  tC.resize(1 << logn);

  for (int i = 0; i < A.size(); i++) {
    for (int j = 0; j < A[i].size(); j++) {
      if (tA.size() <= A[i][j].first) {
        printf("Error %d,%d\n", tA.size(), A[i][j].first);
        exit(-1);
      }
      tA[A[i][j].first].push_back(make_pair(A[i][j].first, A[i][j].second));
    }
  }
  for (int i = 0; i < B.size(); i++) {
    for (int j = 0; j < B[i].size(); j++)
      tB[B[i][j].first].push_back(make_pair(B[i][j].first, B[i][j].second));
  }

  for (int i = 0; i < C.size(); i++) {
    for (int j = 0; j < C[i].size(); j++)
      tC[C[i][j].first].push_back(make_pair(C[i][j].first, C[i][j].second));
  }

  /*
  for(int i = 0; i < tA.size(); i++){
      if(tA[i].size() > 1 || tB[i].size() > 1 || tC[i].size() > 1){
          printf("> Error\n");
          exit(-1);
      }
  }
  */
}

gate add_input() {
  gate gt;
  gt.id = gate_ctr++;
  gt.id_left.push_back(-1);
  gt.id_right = -1;
  gt.type = 0;
  return gt;
}

gate add_mul(int size, int add_ctr) {
  gate gt;
  vector<int> wires;
  wires.push_back((unsigned int)rand() % size);
  for (int i = 0; i < add_ctr - 1; i++) {
    while (true) {
      bool is_dublicate = false;
      unsigned int num = (unsigned int)rand() % size;
      for (int j = 0; j < wires.size(); j++) {
        if (num == wires[j]) {
          is_dublicate = true;
          break;
        }
      }
      if (!is_dublicate) {
        wires.push_back(num);
        break;
      }
    }
  }

  gt.id = gate_ctr++;
  gt.id_left = wires;
  gt.id_right = (unsigned int)rand() % size;
  gt.type = 2;
  return gt;
}

// Dummy computation represeting multiplication tree
// 0: Default, multree
// 1: Random circuit
void generate_R1CS_matrixes(size_t size, int type) {
  int n = size;
  int m = 0;
  int counter = 0;
  if (type == 0) {
    A.resize(size);
    B.resize(size);
    C.resize(size);
    for (int j = 0; j < (int)log2(size); j++) {
      for (int i = 0; i < size / (1 << (j + 1)); i++) {
        A[counter].resize(1);
        B[counter].resize(1);
        C[counter].resize(1);
        A[counter][0] = {2 * i + m, i + m / 2};
        B[counter][0] = {2 * i + 1 + m, i + m / 2};
        C[counter][0] = {n + i, i + m / 2};
        counter++;
      }
      m += size / (1 << (j));
      n += size / (1 << (j + 1));
    }
    logm = (int)log2(size);
    logn = (int)log2(size) + 1;
  } else {
    vector<gate> gates;
    for (int i = 0; i < size / 16; i++)
      gates.push_back(add_input());
    int add_ctr = 0;
    int additions = 0;
    srand(42);
    for (int i = size / 16; i < size; i++) {
      if (rand() % 2 == 0) {
        add_ctr++;
        additions++;
      } else {
        gates.push_back(add_mul(gates.size(), add_ctr));
        add_ctr = 0;
      }
    }
    if (add_ctr != 0)
      gates.push_back(add_mul(gates.size(), add_ctr));

    A.resize(gates.size());
    B.resize(gates.size());
    C.resize(gates.size());
    for (int i = size / 16; i < A.size(); i++) {
      A[i].resize(gates[i].id_left.size());
      B[i].resize(1);
      C[i].resize(1);
      for (int j = 0; j < A[i].size(); j++) {
        A[i][j] = {gates[i].id_left[j], gates[i].id - size / 16};
      }
      B[i][0] = {gates[i].id_right, gates[i].id - size / 16};
      C[i][0] = {gates[i].id, gates[i].id - size / 16};
    }
    logm = (int)log2(next_pow2(gates.size() - size / 16));
    logn = (int)log2(next_pow2(gates.size()));
  }
  transpose_R1CS_matrixes();
}

void reduce_R1CS_matrixes(size_t size, vector<F> r, vector<F> &RA,
                          vector<F> &RB, vector<F> &RC) {
  vector<F> beta;
  precompute_beta(r, beta);
  RA.resize(1 << logn, F(0));
  RB.resize(1 << logn, F(0));
  RC.resize(1 << logn, F(0));
  for (int i = 0; i < A.size(); i++) {
    for (int j = 0; j < A[i].size(); j++) {
      RA[A[i][j].first] += beta[A[i][j].second];
    }
    for (int j = 0; j < B[i].size(); j++) {
      RB[B[i][j].first] += beta[B[i][j].second];
    }
    for (int j = 0; j < C[i].size(); j++) {
      RC[C[i][j].first] += beta[C[i][j].second];
    }
  }
}

void commit_randomness(vector<F> commit_data_R1, vector<F> commit_data_R2,
                       int rate, int N) {
  pad_vector(commit_data_R1);
  pad_vector(commit_data_R2);

  R1_codeword = commit_data_R1;
  R2_codeword = commit_data_R2;
  R1_codeword.resize(rate * R1_codeword.size(), F(0));
  R2_codeword.resize(rate * R2_codeword.size(), F(0));

  fft(R1_codeword, (int)log2(R1_codeword.size()), false);
  fft(R2_codeword, (int)log2(R2_codeword.size()), false);
  merkle_tree::merkle_tree_prover::MT_commit_Blake(
      R1_codeword.data(), R1_MT.Base_MT, R1_codeword.size());
  merkle_tree::merkle_tree_prover::MT_commit_Blake(
      R2_codeword.data(), R2_MT.Base_MT, R2_codeword.size());

  distributed_commit2(R1_MT, N);
  distributed_commit2(R2_MT, N);
}

vector<vector<F>> R1_p(2), R2_p(2);
F R1_K, R2_K;

vector<pair<F, vector<F>>>
prove_phase1(vector<F> vL, vector<F> vR, vector<F> vO, vector<F> rL,
             vector<F> rR, vector<F> rO, vector<vector<F>> R1,
             vector<int> degrees, int N, int _k, int k, double &ps) {

  pt_cp.start();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  vector<F> _r1, _r2;

  for (int i = 0; i < (int)log2(vL.size()); i++)
    _r1.push_back(hash_to_field({0}));
  for (int i = 0; i < (int)log2(k); i++)
    _r2.push_back(hash_to_field({0}));

  vector<F> beta1, beta2;
  // precompute_beta(_r1,beta1);
  precompute_beta(_r2, beta2);
  F y = 0;
  for (int i = 0; i < rL.size(); i++) {
    y += _beta(i, _r1) * (rL[i] * rR[i] - rO[i]);
  }
  vector<u64> buff_u64(2);
  com_rounds++;
  pt_cp.end();

  if (rank == 0) {
    vector<F> Y(N);
    Y[0] = y;
    for (int i = 1; i < N; i++) {
      MPI_Recv(buff_u64.data(), 2, MPI_UINT64_T, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      Y[i].real = buff_u64[0];
      Y[i].img = buff_u64[1];
    }
    pt_cp.start();
    fft(Y, (int)log2(N), true);
    F mul = F(1);
    F omega = getRootOfUnity(1 + (int)log2(N)).inv();
    for (int i = 0; i < Y.size(); i++) {
      Y[i] = mul * Y[i];
      mul = omega * mul;
    }
    fft(Y, (int)log2(N), false);
    y = 0;
    for (int i = 0; i < k; i++) {
      y += beta2[i] * Y[i * N / _k];
    }
    buff_u64 = {y.real, y.img};
    pt_cp.end();

  } else {
    buff_u64 = {y.real, y.img};
    cm += 8 * buff_u64.size() / 1024.0;
    MPI_Send(buff_u64.data(), 2, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
  }
  if (rank == 0)
    cm += (N - 1) * 8 * buff_u64.size() / 1024.0;
  com_rounds++;
  MPI_Bcast(buff_u64.data(), 2, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  pt_cp.start();

  y.real = buff_u64[0];
  y.img = buff_u64[1];
  vector<F> r1 = _r1, r2 = _r2;
  r1.push_back(hash_to_field({0}));
  vL.resize(2 * vL.size(), F(0));
  vR.resize(2 * vR.size(), F(0));
  vO.resize(2 * vO.size(), F(0));
  for (int i = vL.size() / 2; i < vL.size() / 2 + rL.size(); i++) {
    vL[i] = rL[i - vL.size() / 2];
    vR[i] = rR[i - vL.size() / 2];
    vO[i] = rO[i - vL.size() / 2];
  }

  y = r1[r1.size() - 1] * y;

  pt_cp.end();

  F K_r = compute_Kr(R1, beta2, N);

  /*************************** */
  // To be replaced with sumcheck
  /*************************** */
  vector<F> r_vec = convert2vector(R1);
  vector<pair<F, vector<F>>> claims =
      cubic_coSumcheck(vL, vR, vO, r1, r2, R1, y, K_r, degrees, N, ps);

  sumcheck1_ip = convert2vector(ip_vec);
  vector<F> b1 = compute_lagrange_coeff(getRootOfUnity((int)log2(N / 4)),
                                        claims[0].second[0], N / 4);

  _sumcheck1_ip.push_back(sumcheck1_ip);
  _sumcheck1_ip.push_back(b1);

  b1.resize(2 * b1.size(), F(0));
  fft(b1, (int)log2(b1.size()), true);
  b1.resize(2 * N, F(0));
  fft(b1, (int)log2(b1.size()), false);

  for (int i = 0; i < sumcheck1_ip.size(); i++) {
    sumcheck1_ip[i] = b1[2 * rank + 1] * sumcheck1_ip[i];
  }
  return claims;
}

void compute_beta_shares_multivariate(vector<F> &shares, vector<F> r,
                                      vector<int> degrees, int k, int N,
                                      int _k) {
  shares.clear();
  vector<F> beta1 = compute_lagrange_coeff(
      getRootOfUnity((int)log2(degrees[0])), r[1], degrees[0]);
  for (int i = 1; i < degrees.size() - 1; i++) {
    vector<F> buff(degrees[i] * beta1.size());
    vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees[i])),
                                         r[i + 1], degrees[i]);
    for (int k = 0; k < L.size(); k++) {
      for (int j = 0; j < buff.size() / L.size(); j++) {
        buff[j + k * buff.size() / L.size()] = L[k] * beta1[j];
      }
    }
    beta1 = buff;
  }
  vector<F> beta2 =
      compute_lagrange_coeff(getRootOfUnity((int)log2(N / 4)), r[0], N / 4);
  vector<F> buff(_k, 0);
  shares = beta1;
  for (int i = 0; i < k; i++)
    buff[i] = beta2[i];
  fft(buff, (int)log2(buff.size()), true);
  buff.resize(2 * N, F(0));
  fft(buff, (int)log2(buff.size()), false);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  for (int i = 0; i < shares.size(); i++) {
    shares[i] = buff[2 * rank + 1] * shares[i];
  }
}

void compute_beta_shares(vector<F> &shares, vector<F> r, int k, int N, int _k) {
  shares.clear();
  vector<F> r1, r2, b1;
  for (int i = 0; i < (int)log2(k); i++)
    r1.push_back(r[i]);
  for (int i = (int)log2(k); i < r.size(); i++)
    r2.push_back(r[i]);

  precompute_beta(r1, b1);
  precompute_beta(r2, shares);
  vector<F> buff(_k, 0);

  for (int i = 0; i < k; i++)
    buff[i] = b1[i];
  fft(buff, (int)log2(buff.size()), true);
  buff.resize(2 * N, F(0));
  fft(buff, (int)log2(buff.size()), false);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  for (int i = 0; i < shares.size(); i++) {
    shares[i] = buff[2 * rank + 1] * shares[i];
  }
}

vector<pair<F, vector<F>>>
prove_phase2(vector<F> &w, vector<F> rL, vector<F> rR, vector<F> rO,
             vector<F> &RA, vector<F> &RB, vector<F> &RC, vector<vector<F>> &R1,
             vector<F> r, vector<int> degrees1, vector<int> degrees2, F yL,
             F yR, F yO, F K_r, int N, int size, int _k, int k, F &a, F &b,
             F &c, double &ps) {

  RA.clear();
  RB.clear();
  RC.clear();

  vector<F> _r = r;
  _r.pop_back();

  vector<F> rL_old = rL, rR_old = rR, rO_old = rO;

  pt_cp.start();
  /*************** */
  // TODO: Possible need to change to aggregate lagrange functions and not beta
  // ones
  //_reduce_R1CS_matrixes(size, _r, RA, RB, RC, N);
  _reduce_R1CS_matrixes_multivariate(size, _r, RA, RB, RC, degrees1, N);
  /*************** */

  a = hash_to_field({0});
  b = hash_to_field({0});
  c = hash_to_field({0});

  // Initialize main inputs
  // w.insert(w.end(),r_w.begin(),r_w.end());
  // w.resize(next_pow2(w.size()),F(0));
  rL.resize(w.size(), F(0));

  for (int i = 0; i < rR.size(); i++)
    rL[i] = a * rL[i] + b * rR[i] + c * rO[i];

  // Initialize secondary inputs, R_aggr the reduced R1CS matrixes
  vector<F> R_aggr(RA.size(), F(0));

  vector<F> L_last =
      compute_lagrange_coeff(getRootOfUnity(1), r[r.size() - 1], 2);

  for (int i = 0; i < RA.size(); i++) {
    R_aggr[i] = L_last[0] * (a * RA[i] + b * RB[i] + c * RC[i]);
  }

  pt_cp.end();
  secret_share_vector(R_aggr, _k, k, N);
  pt_cp.start();
  // R_aggr.resize(2*R_aggr.size(),F(0));

  /*************************/
  // TODO: Change beta_shares to lagrange ones
  vector<F> beta_shares;
  F i = L_last[0].inv();
  compute_beta_shares_multivariate(beta_shares, _r, degrees1, k, N, _k);
  // compute_beta_shares(beta_shares, _r, k, N, _k);
  for (int i = 0; i < beta_shares.size(); i++) {
    beta_shares[i] = L_last[1] * beta_shares[i];
  }
  beta_shares.resize(4 * beta_shares.size(), F(0));
  /*************************/

  // DO NOT MODIFY THESE
  int segment_size = 2 * R_aggr.size() / 32;
  vector<vector<F>> new_R_aggr(32);
  int ctr = 0, rnd_ctr = 0;
  for (int i = 0; i < 32; i++) {
    new_R_aggr[i].resize(segment_size, F(0));
  }
  for (int i = 0; i < 32; i++) {
    new_R_aggr[i].resize(segment_size, F(0));

    for (int j = 0; j < segment_size - mask_l; j++) {
      if (ctr == R_aggr.size())
        break;
      new_R_aggr[i][j] = R_aggr[ctr++];
    }
    if (ctr == R_aggr.size())
      break;
  }
  R_aggr = convert2vector(new_R_aggr);

  R_aggr.insert(R_aggr.end(), beta_shares.begin(), beta_shares.end());
  w.insert(w.end(), rL.begin(), rL.end());
  vector<F> r_vec = convert2vector(R1);
  pt_cp.end();

  // exit(-1);
  vector<pair<F, vector<F>>> claims = quadratic_coSumcheck(
      w, R_aggr, R1, a * yL + b * yR + c * yO, K_r, degrees2, N, ps);

  /*******************/
  // TO BE REPLACED WITH SUMCHECK
  /*
  F K = F(0);
  for(int i = 0; i < w.size(); i++) K += rL[i]*beta_shares[i];
  for(int i = 0; i < w.size(); i++) K += w[i]*R_aggr[i];

  K =  F_sum(K, N);
  if(K != a*yL + b*yR + c*yO){
      printf("ERROR\n");
  }
  */
  /*
   vector<F> final_r;
   for(int i = 0; i < degrees2.size()+1; i++){
       final_r.push_back(hash_to_field({0}));
   }
   R_aggr = convert2vector(new_R_aggr);

   vector<F> data = R_aggr;
   vector<vector<F>> v = {R_aggr,w};
   vector<pair<F,vector<F>>> _claims =
   compute_evaluation_claims(v,claims[0].second,degrees2,N);
  */
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  /*
  data = send_data(data, N);
  v.clear(); v.push_back(convert2vector(R1));
  int len = v.size();
  for(int i = 0; i < len; i++) R2_p[0].push_back(hash_to_field({}));
  for(int i = 0; i < N/4; i++) R2_p[1].push_back(hash_to_field({}));
  R2_K = F_ip(v[0],R1[0],R1[1],N/4,N/2,N);

  if(rank == 0){

      vector<vector<F>> L(final_r.size());
      for(int i = 0; i < degrees2.size(); i++){
          if(degrees2[i] != 1) L[i] =
  compute_lagrange_coeff(getRootOfUnity((int)log2(degrees2[i])),final_r[i+1],degrees2[i]);
          else {
              L[i].push_back(F(1)-final_r[i+1]);
              L[i].push_back(final_r[i+1]);
          }
      }
      L[degrees2.size()] =
  compute_lagrange_coeff(getRootOfUnity((int)log2(N/4)),final_r[0],N/4); for(int
  i = 0; i < L.size(); i++){ for(int j = 0; j < data.size()/L[i].size(); j++){
              data[j] = L[i][0]*data[j*L[i].size()];
              for(int k = 1; k < L[i].size(); k++){
                  data[j] += L[i][k]*data[j*L[i].size() + k];
              }
          }
          data.resize(data.size()/L[i].size());
      }

      if(data[0] != claims[0].first){
          printf("Error %lld,%lld,
  %lld,%lld\n",data[0].real,data[0].img,claims[0].first.real,claims[0].first.img);
          return {};
      }
  }

  */

  sumcheck2_ip = convert2vector(ip_vec);
  vector<F> b1 = compute_lagrange_coeff(getRootOfUnity((int)log2(N / 4)),
                                        claims[0].second[0], N / 4);
  _sumcheck2_ip.push_back(sumcheck2_ip);
  _sumcheck2_ip.push_back(b1);

  b1.resize(2 * b1.size(), F(0));
  fft(b1, (int)log2(b1.size()), true);
  b1.resize(2 * N, F(0));
  fft(b1, (int)log2(b1.size()), false);

  for (int i = 0; i < sumcheck2_ip.size(); i++) {
    sumcheck2_ip[i] = b1[2 * rank + 1] * sumcheck2_ip[i];
  }
  // Compute the ip for rL,rO,rR
  vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees2[0])),
                                       claims[0].second[1], degrees2[0]);
  for (int i = 1; i < degrees2.size(); i++) {
    vector<F> t = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees2[i])),
                                         claims[0].second[i + 1], degrees2[i]);
    for (int j = 0; j < L.size(); j++) {
      L[j] = t[0] * L[j];
    }
  }

  for (int i = 0; i < rL_old.size(); i++)
    sumcheck3_ip.push_back(a * L[i]);
  for (int i = 0; i < rR_old.size(); i++)
    sumcheck3_ip.push_back(b * L[i]);
  for (int i = 0; i < rO_old.size(); i++)
    sumcheck3_ip.push_back(c * L[i]);

  _sumcheck3_ip.push_back(sumcheck3_ip);
  _sumcheck3_ip.push_back(_sumcheck2_ip[1]);

  for (int i = 0; i < sumcheck3_ip.size(); i++)
    sumcheck3_ip[i] = b1[2 * rank + 1] * sumcheck3_ip[i];

  /*
  r_vec.clear();
  r_vec = rL_old;
  r_vec.insert(r_vec.end(),rR_old.begin(),rR_old.end());
  r_vec.insert(r_vec.end(),rO_old.begin(),rO_old.end());

  K_r = F(0);
  printf("%d,%d\n",r_vec.size(),sumcheck3_ip.size());
  for(int i = 0; i < r_vec.size(); i++){
      K_r += r_vec[i]*sumcheck3_ip[i];
  }

  K_r = sum2(K_r,N);
  if(claims[2].first != K_r){
      printf("Error\n");
      exit(-1);
  }
  */

  claims[1].first = i * claims[1].first;
  return claims;
  // TODO: REPLACE Sumcheck
  /*
  vector<pair<F,vector<F>>> claim = _quadratic_batch_sumcheck(a*yL + b*yR +
  c*yO, w, R_aggr, rL, beta_shares, R1,R2,N, _k, k, ps); claim[1].first =
  (F(1)-claim[1].second[claim[1].second.size()-1]).inv()*i*claim[1].first;
  return claim;*/
}

void get_bits(vector<short> &buff, int idx1, int idx2) {
  for (int i = 0; i < logm; i++) {
    buff[i] = 0;
    if (idx1 & 1) {
      buff[i] = 1;
    }
    idx1 = idx1 >> 1;
  }
  for (int i = logm; i < logm + logn; i++) {
    buff[i] = 0;
    if (idx2 & 1) {
      buff[i] = 1;
    }
    idx2 = idx2 >> 1;
  }
}

void prepare_matrix_data(vector<vector<pair<int, int>>> &M,
                         vector<vector<int>> &idx,
                         vector<vector<short>> &bits) {
  int mask1 = 0;
  int mask2 = 0;
  for (int i = 0; i < logm / 2; i++) {
    mask1 = mask1 << 1;
    mask1 += 1;
  }
  for (int i = 0; i < logn / 2; i++) {
    mask2 = mask2 << 1;
    mask2 += 1;
  }
  int idx_size = 0;
  for (int i = 0; i < M.size(); i++) {
    idx_size += M[i].size();
  }

  idx.resize(idx_size);
  bits.resize(idx_size);
  int ctr = 0;
  vector<short> buff(logm + logn, 0);
  for (int i = 0; i < M.size(); i++) {
    for (int j = 0; j < M[i].size(); j++) {
      idx[ctr].resize(4);
      idx[ctr][0] = i & mask1;
      idx[ctr][1] = i >> (logm / 2);
      if (idx[ctr][0] + (idx[ctr][1] << (logm / 2)) != i) {
        printf("Error in matrix preperation 1,%d,%d,%d\n", i, mask1,
               idx[ctr][1]);
        exit(-1);
      }
      idx[ctr][2] = M[i][j].first & mask2;
      idx[ctr][3] = M[i][j].first >> (logn / 2);
      if (idx[ctr][2] + (idx[ctr][3] << (logn / 2)) != M[i][j].first) {
        printf("Error in matrix preperation 2,%d\n", M[i][j].first);
        exit(-1);
      }
      get_bits(buff, i, M[i][j].first);
      bits[ctr] = buff;
      ctr++;
    }
  }
}

void compute_witness_vector(vector<F> r1, vector<F> r2,
                            vector<sparse_eval_data> &data, vector<F> &witness,
                            int N, double &pt) {
  vector<F> r11, r12, r21, r22;
  for (int i = 0; i < r1.size() / 2; i++) {
    r11.push_back(r1[i]);
  }
  for (int i = r1.size() / 2; i < r1.size(); i++) {
    r12.push_back(r1[i]);
  }
  for (int i = 0; i < r2.size() / 2; i++) {
    r21.push_back(r2[i]);
  }
  for (int i = r2.size() / 2; i < r2.size(); i++) {
    r22.push_back(r2[i]);
  }
  int mask1 = 0;
  int mask2 = 0;
  for (int i = 0; i < logm / 2; i++) {
    mask1 = mask1 << 1;
    mask1 += 1;
  }
  for (int i = 0; i < logn / 2; i++) {
    mask2 = mask2 << 1;
    mask2 += 1;
  }

  vector<F> beta11, beta12, beta21, beta22;

  clock_t t1 = clock();
  precompute_beta(r11, beta11);
  precompute_beta(r12, beta12);
  precompute_beta(r21, beta21);
  precompute_beta(r22, beta22);
  clock_t t2 = clock();
  pt += (double)(t2 - t1) * N / (double)CLOCKS_PER_SEC;
  witness.resize(8 * next_pow2(data[0].IDX1.size()), F(0));
  int ctr = 0;
  t1 = clock();
  for (int i = 0; i < data.size(); i++) {
    for (int j = 0; j < data[i].IDX1.size(); j++) {
      int idx1 = data[i].IDX1[j];
      int idx11 = idx1 & mask1;
      int idx12 = idx1 >> (logm / 2);
      witness[ctr] = beta11[idx11] * beta12[idx12];
      ctr++;
    }
    ctr += next_pow2(data[i].IDX1.size()) - data[i].IDX1.size();
    for (int j = 0; j < data[i].IDX2.size(); j++) {
      int idx2 = data[i].IDX2[j];
      int idx21 = idx2 & mask2;
      int idx22 = idx2 >> (logn / 2);
      witness[ctr] = beta21[idx21] * beta22[idx22];
      ctr++;
    }
    ctr += next_pow2(data[i].IDX1.size()) - data[i].IDX1.size();
  }
  t2 = clock();
  pt += (double)(t2 - t1) / (double)CLOCKS_PER_SEC;
}

void evaluate_sparse_matrix(size_t size, int N, vector<F> r1, vector<F> r2, F y,
                            F a, F b, F c, double &pt, double &ps, double &vt) {
  vector<F> witness;
  vector<vector<F>> R;
  vector<vector<F>> Code;
  vector<vector<vector<_hash>>> MT;
  vector<sparse_eval_data> data;

  prepare_R1CS_data(A, B, C, logm, logn, data);
  vector<F> beta1, beta2;
  precompute_beta(r1, beta1);
  precompute_beta(r2, beta2);

  compute_witness_vector(r1, r2, data, witness, N, pt);

  // COMMIT
  // plain_commit(witness,N/2,N,N,R,Code,MT,pt,vt,ps,cm);
  prove_sparse_eval(y, a, b, c, beta1, beta2, data, pt, ps, vt);

  vector<F> r, _r1, _r2;
  for (int i = 0; i < (int)log2(witness.size()); i++) {
    r.push_back(F::_random());
    if (i < (int)log2(N / 2)) {
      _r1.push_back(r[i]);
    } else {
      _r2.push_back(r[i]);
    }
  }
  clock_t t1 = clock();
  precompute_beta(_r1, beta1);
  precompute_beta(_r2, beta2);
  clock_t t2 = clock();
  pt += N * (double)(t2 - t1) / (double)CLOCKS_PER_SEC;
  y = evaluate_vector(witness, r);
  // OPEN
  // plain_open(y,Code,R,MT,beta1,beta2,N/2,pt,vt,ps,500,false);

  // r1.insert(r1.end(),r2.begin(),r2.end());
  // prepare_matrix_data(A,idx,bits);
  // prove_sparse_eval_bit(r1, evaluate_vector(RA,r2), bits, idx, logm, logn, N,
  // pt, vt, ps); printf("%lf\n",pt/N);
}

void commit_sparse_eval_witness(vector<vector<F>> &beta1,
                                vector<vector<F>> &beta2,
                                vector<vector<F>> &beta1_even_odd,
                                vector<vector<F>> &beta2_even_odd,
                                vector<vector<F>> &codeword,
                                vector<vector<F>> &_row_data, int N) {
  vector<F> data;
  vector<vector<F>> buff;
  buff = beta1;

  vector<int> order;

  sort_transcript(buff, order);

  data = convert2vector(buff);
  buff.clear();
  data.resize(next_pow2(data.size()), F(0));
  int size1 = data.size();
  // for(int i = 0; i < beta1.size(); i++)
  // data.insert(data.end(),beta1[i].begin(),beta1[i].end());
  for (int i = 0; i < beta2.size(); i++)
    data.insert(data.end(), beta2[order[i]].begin(), beta2[order[i]].end());
  data.resize(next_pow2(data.size()), F(0));

  vector<F> data_even_odd(data.size(), F(0));

  int ctr = 0;
  for (int i = 0; i < N; i++) {
    int offst = 0;
    for (int k = 0; k < 3; k++) {
      for (int j = 0; j < beta1_even_odd[order[k]].size() / N; j++)
        data_even_odd[ctr++] =
            (beta1_even_odd[order[k]]
                           [i * beta1_even_odd[order[k]].size() / N + j]);
      offst += beta1_even_odd[k].size() / N;
    }
    for (int k = offst; k < size1 / N; k++)
      data_even_odd[ctr++] = F(0);
    offst = size1 / N;
    for (int k = 0; k < 3; k++) {
      for (int j = 0; j < beta2_even_odd[order[k]].size() / N; j++)
        data_even_odd[ctr++] =
            (beta2_even_odd[order[k]]
                           [i * beta2_even_odd[order[k]].size() / N + j]);
      offst += beta2_even_odd[k].size() / N;
    }
    for (int k = offst; k < data.size() / N; k++)
      data_even_odd[ctr++] = F(0);
  }

  vector<F> row_data = data;

  vector<vector<F>> temp_data_even_odd(2);
  temp_data_even_odd[0].resize(data_even_odd.size() / 2);
  temp_data_even_odd[1].resize(data_even_odd.size() / 2);
  ctr = 0;
  int arr_size = data_even_odd.size() / 2;

  // xx zz xx zz xx zz
  // xxxxxxxx zzzzzzzz xxxxxxxx zzzzzzzz xxxxxxxx zzzzzzzz xxxxxxxx zzzzzzzz
  // 12345678 9,10,11,12,13,14,15,16     17,18,19,20
  int offset = data_even_odd.size() / (2 * N);

  for (int j = 0; j < N; j++) {
    for (int k = 0; k < 2; k++) {
      for (int i = 0; i < offset; i++) {
        temp_data_even_odd[k][j * offset + i] = data_even_odd[ctr++];
      }
    }
  }

  data_even_odd = convert2vector(temp_data_even_odd);

  Batch_dCommit(data_even_odd, row_data, _row_data, codeword, witness_Com,
                witness_commit_rate, N, witness_batch_size);
  min_seg_size = _row_data[0].size();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  // plaintext_commit(data, codeword,row_data,Com,k,N);
}

pair<F, vector<F>> sparse_matrix_evaluation_multivariate(
    F y, F a, F b, F c, vector<F> r1, vector<F> r2, vector<int> degrees1,
    vector<int> degrees2, vector<sparse_eval_data> &index,
    vector<sparse_eval_data> &index_even_odd, vector<vector<F>> &codeword,
    vector<vector<F>> &row_data, int N, double &ps) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  vector<vector<F>> beta1(3), beta2(3);
  vector<vector<F>> beta1_even_odd(3), beta2_even_odd(3);
  r1.pop_back(); // r2.pop_back();
  degrees1.pop_back();

  /*
  if(r1.size() != logm){
      printf("Wrong dimensions 1\n");
      exit(-1);
  }
  if(r2.size() != logn){
      printf("Wrong dimensions 2\n");
      exit(-1);
  }
  */

  pt_cp.start();

  timer sparse_eval_timer;
  sparse_eval_timer.start();
  /************** */
  // TODO: Compute lagrange betas instead of multilinear ones
  compute_R1CS_betas_multivariate(r1, r2, index, index_even_odd, beta1, beta2,
                                  beta1_even_odd, beta2_even_odd, degrees1,
                                  degrees2, logm, logn, N);

  for (int i = 0; i < 3; i++) {
    beta1[i].resize(next_pow2(beta1[i].size()), F(0));
    beta2[i].resize(next_pow2(beta2[i].size()), F(0));
    beta1_even_odd[i].resize(next_pow2(beta1_even_odd[i].size()), F(0));
    beta2_even_odd[i].resize(next_pow2(beta2_even_odd[i].size()), F(0));
  }

  /************** */
  // vector<vector<F>> codeword,row_data;
  vector<F> _row_data;
  // MT Com;
  pt_cp.end();
  commit_sparse_eval_witness(beta1, beta2, beta1_even_odd, beta2_even_odd,
                             codeword, row_data, N);
  sparse_eval_timer.end();
  if (rank == 0) {
    printf("      Phase 5.0 (Commit): %lf, Net: %lf\n",
           sparse_eval_timer.get_time(), network_timer.get_time());
    network_timer.reset();
  }

  sparse_eval_timer.reset();
  sparse_eval_timer.start();
  int gkr_rounds = com_rounds;
  // I do not think if we need to change something from this protocol
  _row_data = convert2vector(row_data);

  pair<F, vector<F>> claim = _prove_sparse_eval_opt(
      y, a, b, c, beta1, beta2, index, r1, r2, N, _row_data);
  sparse_eval_timer.end();
  if (rank == 0) {
    printf("      Phase 5.1 (Sumcheck): %lf, Net: %lf, Rounds: %d\n",
           sparse_eval_timer.get_time(), network_timer.get_time(),
           com_rounds - gkr_rounds);
    network_timer.reset();
  }
  sparse_eval_timer.reset();
  sparse_eval_timer.start();

  sparse_eval_timer.end();

  if (rank == 0) {
    printf("      Phase 5.2 (Open): %lf, Net: %lf\n",
           sparse_eval_timer.get_time(), network_timer.get_time());
    network_timer.reset();
  }
  return claim;
}

pair<F, vector<F>> sparse_matrix_evaluation_multivariate_LAN(
    F y, F a, F b, F c, vector<F> r1, vector<F> r2, vector<int> degrees1,
    vector<int> degrees2, vector<sparse_eval_data> &index,
    vector<sparse_eval_data> &index_even_odd, vector<vector<F>> &codeword,
    vector<vector<F>> &row_data, int N, double &ps) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  vector<vector<F>> beta1(3), beta2(3);
  vector<vector<F>> beta1_even_odd(3), beta2_even_odd(3);
  r1.pop_back(); // r2.pop_back();
  degrees1.pop_back();

  /*
  if(r1.size() != logm){
      printf("Wrong dimensions 1\n");
      exit(-1);
  }
  if(r2.size() != logn){
      printf("Wrong dimensions 2\n");
      exit(-1);
  }
  */

  pt_cp.start();

  timer sparse_eval_timer;
  sparse_eval_timer.start();
  /************** */
  // TODO: Compute lagrange betas instead of multilinear ones
  compute_R1CS_betas_multivariate(r1, r2, index, index_even_odd, beta1, beta2,
                                  beta1_even_odd, beta2_even_odd, degrees1,
                                  degrees2, logm, logn, N);

  for (int i = 0; i < 3; i++) {
    beta1[i].resize(next_pow2(beta1[i].size()), F(0));
    beta2[i].resize(next_pow2(beta2[i].size()), F(0));
    beta1_even_odd[i].resize(next_pow2(beta1_even_odd[i].size()), F(0));
    beta2_even_odd[i].resize(next_pow2(beta2_even_odd[i].size()), F(0));
  }

  /************** */
  // vector<vector<F>> codeword,row_data;
  vector<F> _row_data;
  // MT Com;
  pt_cp.end();
  commit_sparse_eval_witness(beta1, beta2, beta1_even_odd, beta2_even_odd,
                             codeword, row_data, N);
  sparse_eval_timer.end();
  if (rank == 0) {
    printf("      Phase 5.0 (Commit): %lf, Net: %lf\n",
           sparse_eval_timer.get_time(), network_timer.get_time());
    network_timer.reset();
  }

  sparse_eval_timer.reset();
  sparse_eval_timer.start();
  int gkr_rounds = com_rounds;
  // I do not think if we need to change something from this protocol
  _row_data = convert2vector(row_data);

  pair<F, vector<F>> claim = _prove_sparse_eval_opt_LAN(
      y, a, b, c, beta1, beta2, index, r1, r2, N, _row_data);
  sparse_eval_timer.end();
  if (rank == 0) {
    printf("      Phase 5.1 (Sumcheck): %lf, Net: %lf, Rounds: %d\n",
           sparse_eval_timer.get_time(), network_timer.get_time(),
           com_rounds - gkr_rounds);
    network_timer.reset();
  }
  sparse_eval_timer.reset();
  sparse_eval_timer.start();

  sparse_eval_timer.end();

  if (rank == 0) {
    printf("      Phase 5.2 (Open): %lf, Net: %lf\n",
           sparse_eval_timer.get_time(), network_timer.get_time());
    network_timer.reset();
  }
  return claim;
}

void sparse_matrix_evaluation(F y, F a, F b, F c, vector<F> r1, vector<F> r2,
                              vector<sparse_eval_data> &index,
                              vector<sparse_eval_data> &index_even_odd, int N,
                              double &ps) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  vector<vector<F>> beta1(3), beta2(3);
  vector<vector<F>> beta1_even_odd(3), beta2_even_odd(3);
  r1.pop_back();
  r2.pop_back();
  if (r1.size() != logm) {
    printf("Wrong dimensions 1\n");
    exit(-1);
  }
  if (r2.size() != logn) {
    printf("Wrong dimensions 2\n");
    exit(-1);
  }
  pt_cp.start();

  timer sparse_eval_timer;
  sparse_eval_timer.start();
  /************** */
  // TODO: Compute lagrange betas instead of multilinear ones
  compute_R1CS_betas(r1, r2, index, index_even_odd, beta1, beta2,
                     beta1_even_odd, beta2_even_odd, logm, logn, N);

  for (int i = 0; i < 3; i++) {
    beta1[i].resize(next_pow2(beta1[i].size()), F(0));
    beta2[i].resize(next_pow2(beta2[i].size()), F(0));
    beta1_even_odd[i].resize(next_pow2(beta1_even_odd[i].size()), F(0));
    beta2_even_odd[i].resize(next_pow2(beta2_even_odd[i].size()), F(0));
  }

  /************** */
  vector<F> codeword, row_data;
  // MT Com;
  pt_cp.end();
  // commit_sparse_eval_witness(beta1, beta2, beta1_even_odd,beta2_even_odd ,
  // codeword , row_data, N/2,  N);
  sparse_eval_timer.end();
  if (rank == 0) {
    printf("      Phase 5.0 (Commit): %lf, Net: %lf\n",
           sparse_eval_timer.get_time(), network_timer.get_time());
    network_timer.reset();
  }

  sparse_eval_timer.reset();
  sparse_eval_timer.start();
  int gkr_rounds = com_rounds;
  // I do not think if we need to change something from this protocol
  pair<F, vector<F>> claim = _prove_sparse_eval_opt(y, a, b, c, beta1, beta2,
                                                    index, r1, r2, N, row_data);
  sparse_eval_timer.end();
  if (rank == 0) {
    printf("      Phase 5.1 (Sumcheck): %lf, Net: %lf, Rounds: %d\n",
           sparse_eval_timer.get_time(), network_timer.get_time(),
           com_rounds - gkr_rounds);
    network_timer.reset();
  }
  sparse_eval_timer.reset();
  sparse_eval_timer.start();
  // open_sparse_eval(codeword,row_data,claim.second,Com,claim.first,796,N/2,N,ps);

  sparse_eval_timer.end();

  if (rank == 0) {
    printf("      Phase 5.2 (Open): %lf, Net: %lf\n",
           sparse_eval_timer.get_time(), network_timer.get_time());
    network_timer.reset();
  }
}
extern int Security_Level;

void batch_random_commitments(vector<F> &y, vector<vector<u32>> &queries, F a,
                              int rates, int N, int my_id, double &ps) {
  int l = (int)((double)Security_Level /
                (-log2(1 - 0.25 * (1.0 - 1.0 / (double)rates))));
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  if (rank == 0)
    vt.start();
  queries = get_indexes(l, N, R1_codeword.size(), my_id);
  if (rank == 0)
    vt.end();
  vector<F> replies(queries.size());
  vt.start();
  for (int i = 0; i < queries.size(); i++) {
    F _y = F(0);
    _y = R1_codeword[queries[i][1]] + a * R2_codeword[queries[i][1]];
    replies[i] = _y;
    y.push_back(_y);
  }
  vt.end();

  double t_ps = ps;
  ps += ((double)(replies.size() * 2 * 16)) / 1024.0;

  vector<bool> visited(2 * R1_MT.Base_MT[0].size(), false);
  for (int i = 0; i < queries.size(); i++) {
    vector<vector<_hash>> temp_MT;
    int pos = queries[i][1] / 4;
    vector<_hash> path;
    vector<F> buff = {R1_codeword[queries[i][1]], R2_codeword[queries[i][1]]};
    vt.start();
    merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(), temp_MT,
                                                     buff.size());
    vt.end();

    path = merkle_tree::merkle_tree_prover::open_tree_blake(R1_MT.Base_MT, pos);

    vt.start();
    merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(
        R1_MT.Base_MT, path.data(), pos, R1_MT.Base_MT[0].size(), visited, ps);
    vt.end();
  }
  get_evals(y, queries, l, N);
}

extern double whir_ps;
void open_randomness(vector<F> &v1, vector<F> &v1_r, F K1, int rate, int N,
                     double &ps) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  pt_cp.start();

  vector<F> v2 = sumcheck1_ip;
  v2.insert(v2.end(), sumcheck2_ip.begin(), sumcheck2_ip.end());
  v2.insert(v2.end(), sumcheck3_ip.begin(), sumcheck3_ip.end());

  v2.resize(v1_r.size(), F(0));
  // pad_vector(v2);
  F K2 = F(0);
  for (int i = 0; i < v1_r.size(); i++) {
    K2 += v1_r[i] * v2[i];
  }
  K2 = sum2(K2, N);
  F a = hash_to_field({K1, K2});
  K1 += a * K2;
  if (rank == 0)
    ps += 16.0 / 1024.0;
  for (int i = 0; i < v1.size(); i++)
    v1[i] += a * v1_r[i];

  vector<F> y;
  vector<vector<u32>> queries;
  pt_cp.end();

  batch_random_commitments(y, queries, a, rate, N, rank, ps);
  v1 = send_data(v1, N);

  if (rank == 0) {

    pt_cp.start();

    _sumcheck1_ip[1].resize(2 * _sumcheck1_ip[1].size(), F(0));
    _sumcheck2_ip[1].resize(2 * _sumcheck2_ip[1].size(), F(0));
    _sumcheck3_ip[1].resize(2 * _sumcheck3_ip[1].size(), F(0));
    fft(_sumcheck1_ip[1], (int)log2(_sumcheck1_ip[1].size()), false);
    fft(_sumcheck2_ip[1], (int)log2(_sumcheck2_ip[1].size()), false);
    fft(_sumcheck3_ip[1], (int)log2(_sumcheck3_ip[1].size()), false);

    vector<F> buff1(_sumcheck1_ip[1].size() * _sumcheck1_ip[0].size(), F(0));
    vector<F> buff2(_sumcheck2_ip[1].size() * _sumcheck2_ip[0].size(), F(0));
    vector<F> buff3(_sumcheck3_ip[1].size() * _sumcheck3_ip[0].size(), F(0));
    v2.clear();
    for (int i = 0; i < _sumcheck1_ip[0].size(); i++) {
      for (int j = 0; j < _sumcheck1_ip[1].size(); j++) {
        buff1[j + i * _sumcheck1_ip[1].size()] =
            _sumcheck1_ip[1][j] * _sumcheck1_ip[0][i];
      }
    }
    for (int i = 0; i < _sumcheck2_ip[0].size(); i++) {
      for (int j = 0; j < _sumcheck2_ip[1].size(); j++) {
        buff2[j + i * _sumcheck2_ip[1].size()] =
            _sumcheck2_ip[1][j] * _sumcheck2_ip[0][i];
      }
    }
    for (int i = 0; i < _sumcheck3_ip[0].size(); i++) {
      for (int j = 0; j < _sumcheck3_ip[1].size(); j++) {
        buff3[j + i * _sumcheck3_ip[1].size()] =
            _sumcheck3_ip[1][j] * _sumcheck3_ip[0][i];
      }
    }

    v2 = buff1;
    buff1.clear();
    v2.insert(v2.end(), buff2.begin(), buff2.end());
    buff2.clear();
    v2.insert(v2.end(), buff3.begin(), buff3.end());
    buff3.clear();

    pad_vector(v1);
    v2.resize(v1.size(), F(0));

    F s = F::_random();
    F s_pow = s;
    vector<F> old_v2 = v2;
    // v2.clear();v2.resize(old_v2.size(),F(0));
    // K1 = 0;
    for (int i = 0; i < queries.size(); i++) {
      vector<F> b = tensor_eval_vector_plaintext(
          queries[i], N / 2, R1_codeword.size() / rate, rate);
      for (int j = 0; j < b.size(); j++)
        v2[j] += s_pow * b[j];
      K1 += s_pow * y[i];
      s_pow *= s;
    }

    vector<F> codeword;
    vector<F> arr = v1;
    vector<vector<_hash>> MT;

    // Whir_Commit(v1,codeword,MT,whir_rates[sorted_indexes[0]],whir_kappas[v1.size()]);
    Whir_Commit_initial(v1, codeword, MT, whir_rates[sorted_indexes[0]],
                        whir_kappas[v1.size()]);

    whir_polys[sorted_indexes[0]] = v1;
    whir_c[sorted_indexes[0]] = codeword;
    whir_MTs[sorted_indexes[0]] = MT;
    vector<pair<F, vector<F>>> claims = quadratic_sumcheck(K1, v1, v2, a);
    vt.start();
    F y = evaluate_vector(old_v2, claims[0].second);
    // vector<F> x_pow_evals(queries.size()),y_pow_evals(queries.size());
    vector<F> r1, r2, b1;
    for (int i = 0; i < (int)log2(N / 2); i++)
      r1.push_back(claims[0].second[i]);
    for (int i = (int)log2(N / 2); i < claims[0].second.size(); i++)
      r2.push_back(claims[0].second[i]);

    F omega = getRootOfUnity((int)log2(2 * N));
    vector<F> omegas(N);
    omegas[0] = omega;
    omega = omega * omega;
    for (int i = 1; i < N; i++)
      omegas[i] = omegas[i - 1] * omega;
    s_pow = s;

    omega = getRootOfUnity((int)log2(R1_codeword.size()));
    for (int i = 0; i < queries.size(); i++) {
      F omega1 = omega.fastPow(omega, queries[i][1]);
      F x_eval = F(1), y_eval = F(1);
      for (int j = 0; j < r2.size(); j++) {
        x_eval *= (F(1) - r2[j] + r2[j] * omega1);
        omega1 *= omega1;
      }
      omega1 = omegas[queries[i][0]];
      for (int j = 0; j < r1.size(); j++) {
        y_eval *= (F(1) - r1[j] + r1[j] * omega1);
        omega1 *= omega1;
      }
      y += s_pow * x_eval * y_eval;
      s_pow *= s;
    }
    vt.end();
    if (y != claims[1].first) {
      printf("Error in open randomness check\n");
    }
    whir_ps = 0.0;
    // printf(">>>>>   Open randomness size: %d\n", arr.size());

    precompute_beta(claims[0].second, whir_betas[sorted_indexes[0]]);
    whir_claims[sorted_indexes[0]] = claims[0].first;
    // Whir_Open(arr,codeword,MT,{claims[0]},8);
    printf("Whir PCS: %lf\n", whir_ps);
    ps += whir_ps;
    pt_cp.end();
  }
}

void init_dummy_index_commitment(vector<sparse_eval_data> &data,
                                 vector<vector<F>> &row_data,
                                 vector<vector<F>> &codeword, MT &index_Com,
                                 int rate, int N) {
  int size = 0;
  for (int i = 0; i < data.size(); i++) {
    size += data[i].FINAL_FR1.size();
    size += data[i].FINAL_FR2.size();
    size += data[i].IDX1.size();
    size += data[i].IDX2.size();
    size += data[i].RD1.size();
    size += data[i].RD2.size();
  }
  size = next_pow2(size);
  for (int i = 0; i < 4 * witness_batch_size; i++) {
    row_data.push_back(generate_randomness(size / (4 * witness_batch_size)));
    codeword.push_back(
        generate_randomness(rate * size / (4 * witness_batch_size)));
  }
  distributed_MT(codeword, index_Com, N);
}

void _warmup(int M, int N, int rank) {
  if (N <= 1)
    return;

  int num_lanes = 1;
  // Minimal data to establish connections: 64KB per peer pair (8K elements * 8
  // bytes)
  int count_per_peer = min(M, 1 << 18);
  int count_per_lane = count_per_peer / num_lanes;

  vector<u64> send_buff(count_per_peer), recv_buff(count_per_peer);
  for (int i = 0; i < count_per_peer; i++)
    send_buff[i] = i + rank;

  // Ring-based warmup: each rank communicates with peers at increasing
  // distances
  for (int dist = 1; dist < N; dist++) {
    int target = (rank + dist) % N;
    int source = (rank - dist + N) % N;

    vector<MPI_Request> reqs;
    reqs.reserve(2 * num_lanes);

    for (int l = 0; l < num_lanes; l++) {
      MPI_Request sreq, rreq;
      int offset = l * count_per_lane;
      MPI_Isend(send_buff.data() + offset, count_per_lane, MPI_UINT64_T, target,
                l, MPI_COMM_WORLD, &sreq);
      MPI_Irecv(recv_buff.data() + offset, count_per_lane, MPI_UINT64_T, source,
                l, MPI_COMM_WORLD, &rreq);

      reqs.push_back(sreq);
      reqs.push_back(rreq);
    }
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

extern double dpcs_ps;
extern double pcs_ps;
extern int randomness;
extern int max_delegation_size;

void coPIOP_prove(size_t size, int N, int _k, int k, int cir_type) {
  double ps = 0.0;
  int logk = (int)log2(k);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  prepare_data(N, witness_commit_rate);
    vector<vector<int>> rate = {{4}, {4}, {4, 4}, {4, 4}};
  // vector<vector<int>> rate = {{8}, {8}, {8, 4}, {8, 4}};

  mask_l = (int)((double)Security_Level /
                 (-log2(1 - 0.25 * (1.0 - 1.0 / (double)rate[0][0]))));
  if (rank == 0)
    printf("Mask: %d\n", mask_l);
  vector<F> witness, vL, vO, vR, R, _R;
  vector<F> r_witness;
  vector<vector<F>> mask_shares, mask_data, C_mask;
  vector<F> RA, RB, RC;
  vector<sparse_eval_data> index, index_even_odd;
  // MT CR,_CR,index_Com,Com;
  // vector<MT> Com_mask;
  vector<F> codeword, row_data, codeword_R, _codeword_R;
  vector<vector<F>> index_codeword, index_data;
  distribute_index(N, size, index, cir_type);
  distribute_index_even_odd(N, size, index_even_odd, cir_type);

  distribute_proving_data(vL, vR, vO, witness, N, size, _k, k, cir_type);
  int d1 = 4;
  int d2 = ((int)log2(vL.size()) - 4) / 2;
  int d3 = ((int)log2(vL.size()) - 4) - ((int)log2(vL.size()) - 4) / 2;
  vector<int> degrees1 = {1 << d1, 1 << d2, 1 << d3, 2};
  // int d1 = ((int)log2(vL.size()))/2;
  // int d2 = ((int)log2(vL.size())) - ((int)log2(vL.size()))/2;
  // vector<int> degrees1 = {1<<d1,1<<d2,2};

  d1 = ((int)log2(2 * witness.size()) - 5) / 2;
  d2 = ((int)log2(2 * witness.size()) - 5 -
        (int)((int)log2(2 * witness.size()) - 5) / 2);
  vector<int> degrees2 = {1 << d1, 1 << d2, 32};
  vector<F> rL(4), rR(4), rO(4);
  vector<vector<F>> R1(degrees1.size()), R2(degrees2.size());

  vector<vector<vector<F>>> sumcheck_R;
  F K_r = generate_sumcheck_randomness(sumcheck_R, degrees1, degrees2, N);
  // printf("%d,%d,%d,%d\n",sumcheck_R[0].size(),sumcheck_R[1].size(),degrees1.size(),degrees2.size());

  int sumcheck_randomness_size = 0;
  for (int i = 0; i < sumcheck_R.size(); i++) {
    for (int j = 0; j < sumcheck_R[i].size(); j++)
      sumcheck_randomness_size += sumcheck_R[i][j].size();
  }
  vector<F> PCS_randomness;

  /*****************************************/
  // This emulates the masking randomness of the PCS
  // To be later replaced with the given PCS function
  prepare_parameters(rate, 4, N, witness.size() * N / 4);
  setup_randomness(PCS_randomness, N, _k, k, randomness);

  vector<vector<F>> coPCS_codeword, coPCS_Commit_data;
  MT coPCS_C;

  setup_randomness(R, N, _k, k, mask_l + 12 + 1);
  setup_randomness(_R, N, _k, k, sumcheck_randomness_size + mask_l + 12 + 1);

  vector<F> v1 = convert2vector(sumcheck_R[0]);
  vector<F> _v1 = convert2vector(sumcheck_R[1]);
  v1.insert(v1.end(), _v1.begin(), _v1.end());
  v1.insert(v1.end(), R.begin(), R.end());
  _v1 = _R;
  if (rank == 0)
    printf(">>>>>>   Open Randomness size %d\n", next_pow2(v1.size()) * N / 2);

  commit_randomness(v1, _v1, 16, N);

  init_dummy_index_commitment(index, index_data, index_codeword, index_Com,
                              index_rate, N);
  if (rank == 0)
    printf("OK\n");

  if (rank == 0)
    printf(">>>   Distributed Pcs size: %d\n", index_data[0].size());

  whir_sizes.push_back(next_pow2(v1.size()) * N / 2);
  whir_sizes.push_back(index_data[0].size());
  whir_sizes.push_back(max_delegation_size);
  whir_sizes.push_back(max_delegation_size);
  int whir_rate = 4;
  get_whir_rates(whir_rate);
  whir_betas.resize(whir_sizes.size());
  whir_polys.resize(whir_sizes.size());
  whir_c.resize(whir_sizes.size());
  whir_MTs.resize(whir_sizes.size());
  whir_claims.resize(whir_sizes.size());

  if (rank == 0) {
    printf("%d\n", whir_rates.size());
    for (int i = 0; i < 4; i++) {
      printf("%d,%d -> %d\n", whir_rates[sorted_indexes[i]], i,
             sorted_indexes[i]);
    }
  }
  // return;
  double temp_comp_time;
  double temp_ps, temp_vt;

  network_timer.start();
  MPI_Barrier(MPI_COMM_WORLD);
  network_timer.end();

  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();
  vector<double> performance_data;

  if (rank == 0) {
    const char *wan_delay = getenv("WAN_DELAY");
    const char *wan_rate = getenv("WAN_RATE");
    const char *wan_iface = getenv("WAN_IFACE");
    const char *mpi_hostfile = getenv("MPI_HOSTFILE");

    if (wan_delay && strcmp(wan_delay, "0ms") != 0) {
      char cmd[1024];
      snprintf(cmd, sizeof(cmd),
               "python3 network_simulation/driver.py --task tune_network "
               "--provers %d --iface %s --delay %s --rate %s --hostfile %s",
               N, wan_iface ? wan_iface : "lo", wan_delay,
               wan_rate ? wan_rate : "10Gbit",
               mpi_hostfile ? mpi_hostfile : "localhost");
      printf("\n[TUNING] Applying network settings: %s\n", cmd);
      int res = system(cmd);
      if (res != 0) {
        printf("[ERROR] Network tuning failed with exit code %d\n", res);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  timer tm;
  tm.start();
  _warmup(size, N, rank);
  if (rank == 0) {
    tm.end();
    printf("Warmup done in %lf secs\n", tm.get_time());
    tm.reset();
  }
  pt.start();

  PCS_commit(coPCS_C, coPCS_codeword, coPCS_Commit_data, witness,
             PCS_randomness, rate, N);
  if (!debug_mode) {
    if (rank == 0) {
      printf("Commit Phase--- Pt: %lf, Net: %lf\n",
             pt_cp.get_time() - temp_comp_time, network_timer.get_time());
      network_timer.reset();
    }
  } else {
    performance_data =
        send_performance_data({pt_cp.get_time() - temp_comp_time, ps - temp_ps,
                               vt.get_time() - temp_vt},
                              N);
    if (rank == 0) {
      printf(
          "Commit Phase--- Pt: %lf, Vt: %lf, Ps: %lf, Mean Pt: %lf, Net: %lf\n",
          pt_cp.get_time() - temp_comp_time, performance_data[1],
          performance_data[2], performance_data[3], network_timer.get_time());
      network_timer.reset();
    }
  }

  witness.resize(coPCS_Commit_data.size() * coPCS_Commit_data[0].size(), F(0));
  int batches = 0;
  for (int i = 0; i < coPCS_Commit_data.size(); i++) {
    if (coPCS_Commit_data[i].size() == 0)
      break;
    batches++;
  }
  for (int i = 0; i < batches - 1; i++) {
    for (int j = 0; j < coPCS_Commit_data[i].size() - mask_l; j++) {
      witness[i * coPCS_Commit_data[i].size() + j] = coPCS_Commit_data[i][j];
    }
  }
  pad_vector(witness);

  // To ease development, we initialize a dummy index commitment. In practice,
  // we could let the indexer generate such a commitment in a preprocessing
  // phase.
  if (witness.size() < r_witness.size()) {
    printf("Circuit too small. Select fewer parties\n");
    return;
  }

  com_rounds = 0;
  cm = 0.0;

  pt_cp.reset();
  pt_cpu.start();
  temp_pc.start();

  // Prepare randomness to mask the evaluation claims from the first sumcehck to
  // the second one (rL,rR,rO) as well as the masking  polynomials of the
  // sumcheck. For now keep rL,rR,rO but check if we can remove them to further
  // improve proof size.

  for (int i = 0; i < 4; i++) {
    rL[i] = R[i];
    rR[i] = R[i + 4];
    rO[i] = R[i + 8];
  }

  int ctr = 12;

  temp_pc.end();

  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();

  temp_pc.start();
  vector<pair<F, vector<F>>> claims1 = prove_phase1(
      vL, vR, vO, rL, rR, rO, sumcheck_R[0], degrees1, N, _k, k, ps);
  if (rank == 0) {
    printf("      Phase 1 Net Delay: %lf\n", network_timer.get_time());
    network_timer.reset();
  }
  temp_pc.end();

  if (!debug_mode) {
    if (rank == 0)
      printf("First Sumcheck--- Pt: %lf\n", pt_cp.get_time() - temp_comp_time);
  } else {
    performance_data =
        send_performance_data({pt_cp.get_time() - temp_comp_time, ps - temp_ps,
                               vt.get_time() - temp_vt},
                              N);
    if (rank == 0)
      printf("First Sumcheck--- Pt: %lf, Vt: %lf, Ps: %lf, Mean Pt: %lf\n",
             pt_cp.get_time() - temp_comp_time, performance_data[1],
             performance_data[2], performance_data[3]);
  }

  /******************************** */
  // return;
  //
  F a, b, c;

  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();

  temp_pc.start();
  vector<pair<F, vector<F>>> claims2 = prove_phase2(
      witness, rL, rR, rO, RA, RB, RC, sumcheck_R[1], claims1[0].second,
      degrees1, degrees2, claims1[0].first, claims1[1].first, claims1[2].first,
      K_r, N, size, _k, k, a, b, c, ps);
  if (rank == 0) {
    printf("      Phase 2 Net Delay: %lf\n", network_timer.get_time());
    network_timer.reset();
  }
  temp_pc.end();

  if (!debug_mode) {
    if (rank == 0)
      printf("Second Sumcheck--- Pt: %lf\n", pt_cp.get_time() - temp_comp_time);
  } else {
    performance_data =
        send_performance_data({pt_cp.get_time() - temp_comp_time, ps - temp_ps,
                               vt.get_time() - temp_vt},
                              N);
    if (rank == 0)
      printf("Second Sumcheck--- Pt: %lf, Vt: %lf, Ps: %lf, Mean Pt: %lf\n",
             pt_cp.get_time() - temp_comp_time, performance_data[1],
             performance_data[2], performance_data[3]);
  }

  vector<vector<F>> witness_data, witness_codeword;
  // return;
  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();
  open_randomness(v1, _v1,
                  claims1[3].first + claims2[4].first + claims2[2].first, 16, N,
                  ps);
  // if(rank == 0)printf("Open Randomness--- Pt:
  // %lf\n",pt_cp.get_time()-temp_comp_time);
  if (!debug_mode) {
    if (rank == 0)
      printf("Open Randomness--- Pt: %lf\n", pt_cp.get_time() - temp_comp_time);
  } else {
    performance_data =
        send_performance_data({pt_cp.get_time() - temp_comp_time, ps - temp_ps,
                               vt.get_time() - temp_vt},
                              N);
    if (rank == 0)
      printf("Open Randomness--- Pt: %lf, Vt: %lf, Ps: %lf, Mean Pt: %lf\n",
             pt_cp.get_time() - temp_comp_time, performance_data[1],
             performance_data[2], performance_data[3]);
  }
  pair<F, vector<F>> betas_claim;
  if (!data_parallel) {
    // Run sparce eval
    /**********************/
    // TODO: I have constructed a constant round protocol, but it stills
    // evaluates a mutli-linear extension. You have to adapt it in order to
    // evaluate multivariate one
    temp_pc.start();
    // sparse_matrix_evaluation(claims2[1].first,a,b,c,
    //                      claims1[0].second,claims2[0].second,index,index_even_odd,N,ps);
    betas_claim = sparse_matrix_evaluation_multivariate(
        claims2[1].first, a, b, c, claims1[0].second, claims2[0].second,
        degrees1, degrees2, index, index_even_odd, witness_codeword,
        witness_data, N, ps);
    temp_pc.end();
    if (rank == 0)
      printf("Sparse Eval PIOP: %lf\n", temp_pc.get_time());
    temp_pc.reset();
  }

  vector<F> index_r;
  for (int i = 0; i < (int)log2(N * index_data.size() * index_data[0].size());
       i++)
    index_r.push_back(hash_to_field({0}));

  vector<vector<vector<F>>> commit_data = {witness_data, index_data};
  if (mul_tree_row_data.size())
    commit_data.push_back(mul_tree_row_data);

  witness.clear();
  index_data.clear();
  vector<vector<vector<F>>> codewords = {witness_codeword, index_codeword};
  if (mul_tree_codeword.size())
    codewords.push_back(mul_tree_codeword);
  witness_codeword.clear();
  index_codeword.clear();
  vector<MT> C = {witness_Com, index_Com};
  if (mul_tree_row_data.size())
    C.push_back(MulTree_Com);

  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();
  if (MulTree_Claims.size()) {
    PIOP_dOpen_batch_new(
        commit_data, codewords, C,
        {betas_claim.second, index_r, MulTree_Claims[0].second},
        {betas_claim.first, F(0), MulTree_Claims[0].first}, {true, false}, N);
  } else {
    PIOP_dOpen_batch_new(commit_data, codewords, C,
                         {betas_claim.second, index_r},
                         {betas_claim.first, F(0)}, {true, false}, N);
  }
  ps += dpcs_ps;
  if (!debug_mode) {
    if (rank == 0)
      printf("Batch Open Witness--- Pt: %lf\n",
             pt_cp.get_time() - temp_comp_time);
  } else {
    performance_data =
        send_performance_data({pt_cp.get_time() - temp_comp_time, ps - temp_ps,
                               vt.get_time() - temp_vt},
                              N);
    if (rank == 0)
      printf("Batch Open Witness--- Pt: %lf, Vt: %lf, Ps: %lf, Mean Pt: %lf\n",
             pt_cp.get_time() - temp_comp_time, performance_data[1],
             performance_data[2], performance_data[3]);
  }

  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();

  vector<vector<F>> inputs(3);
  vector<vector<F>> Lagrange_evals;
  inputs[2] = compute_lagrange_coeff(getRootOfUnity((int)log2(N / 4)),
                                     claims2[1].second[0], N / 4);
  inputs[0] = compute_lagrange_coeff(
      getRootOfUnity(5), claims2[1].second[claims2[1].second.size() - 1], 32);
  inputs[1] = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees2[0])),
                                     claims2[1].second[1], degrees2[0]);
  Lagrange_evals.push_back(inputs[1]);
  for (int i = 1; i < degrees2.size() - 1; i++) {

    vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees2[i])),
                                         claims2[1].second[1 + i], degrees2[i]);
    Lagrange_evals.push_back(L);
    vector<F> buff(inputs[1].size() * L.size());
    for (int j = 0; j < L.size(); j++) {
      for (int k = 0; k < inputs[1].size(); k++)
        buff[k + j * inputs[1].size()] = L[j] * inputs[1][k];
    }
    inputs[1] = buff;
  }

  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();
  PCS_open_new(coPCS_C, coPCS_codeword, coPCS_Commit_data, inputs,
               Lagrange_evals, claims2[0].first, rate, N, true);
  ps += pcs_ps;
  if (!debug_mode) {
    if (rank == 0)
      printf("coPCS Open--- Pt: %lf\n", pt_cp.get_time() - temp_comp_time);
  } else {
    performance_data =
        send_performance_data({pt_cp.get_time() - temp_comp_time, ps - temp_ps,
                               vt.get_time() - temp_vt},
                              N);
    if (rank == 0)
      printf("coPCS Open--- Pt: %lf, Vt: %lf, Ps: %lf, Mean Pt: %lf\n",
             pt_cp.get_time() - temp_comp_time, performance_data[1],
             performance_data[2], performance_data[3]);
  }

  if (rank == 0) {
    for (int i = 0; i < whir_c.size(); i++) {
      printf("(%d,%d,%d)\n", whir_polys[i].size(), whir_c[i].size(),
             whir_c[i].size() / whir_polys[i].size());
    }
    whir_ps = 0.0;
    Whir_Batch_Open(whir_polys, whir_c, whir_MTs, whir_claims, whir_betas,
                    whir_rate, true);
    printf("BATCH WHIR proof: %lf\n", whir_ps);
    ps += whir_ps;
  }
  // if(rank == 0)printf("Batch Open Witness--- Pt:
  // %lf\n",pt_cp.get_time()-temp_comp_time);
  // PCS_open(coPCS_C,coPCS_codeword,coPCS_Commit_data,inputs,Lagrange_evals,claims2[0].first,rate,N);

  pt.end();
  network_timer.start();
  MPI_Barrier(MPI_COMM_WORLD);
  network_timer.end();
  if (rank == 0) {
    vector<double> buff(3);
    printf(
        "Id : %d, Pt: %lf, CPU Only Pt: %lf, Computation only: %lf, Vt: %lf\n",
        rank, pt.get_time(), pt_cpu.get_time(), pt_cp.get_time(),
        vt.get_time());
    for (int i = 1; i < N; i++) {
      MPI_Recv(buff.data(), 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      printf("Id : %d, Pt: %lf, CPU Only Pt: %lf, Computation only: %lf\n", i,
             buff[0], buff[1], buff[2]);
    }
  } else {
    vector<double> buff = {pt.get_time(), pt_cpu.get_time(), pt_cp.get_time()};
    MPI_Send(buff.data(), 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  network_timer.start();
  MPI_Barrier(MPI_COMM_WORLD);
  network_timer.end();
  vector<double> buff = {vt.get_time(), ps, cm};
  if (rank != 0) {
    MPI_Send(buff.data(), 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  } else {
    double total_vt = vt.get_time();
    for (int i = 1; i < N; i++) {
      MPI_Recv(buff.data(), 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      total_vt += buff[0];
      ps += buff[1];
      cm += buff[2];
    }
    printf("Vt : %lf sec, Ps: %lf KB, Com: %lf MB\n", total_vt, ps + ps_plain,
           cm / 1024.0);
    printf("Interaction Rounds: %d\n", com_rounds);
  }
}

void coPIOP_prove_LAN(size_t size, int N, int _k, int k, int cir_type) {
  double ps = 0.0;
  int logk = (int)log2(k);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  prepare_data(N, witness_commit_rate);
    vector<vector<int>> rate = {{4}, {4}, {4, 4}, {4, 4}};
  // vector<vector<int>> rate = {{8}, {8}, {8, 4}, {8, 4}};

  mask_l = (int)((double)Security_Level /
                 (-log2(1 - 0.25 * (1.0 - 1.0 / (double)rate[0][0]))));
  if (rank == 0)
    printf("Mask: %d\n", mask_l);
  vector<F> witness, vL, vO, vR, R, _R;
  vector<F> r_witness;
  vector<vector<F>> mask_shares, mask_data, C_mask;
  vector<F> RA, RB, RC;
  vector<sparse_eval_data> index, index_even_odd;
  // MT CR,_CR,index_Com,Com;
  // vector<MT> Com_mask;
  vector<F> codeword, row_data, codeword_R, _codeword_R;
  vector<vector<F>> index_codeword, index_data;
  distribute_index(N, size, index, cir_type);
  distribute_index_even_odd(N, size, index_even_odd, cir_type);

  distribute_proving_data(vL, vR, vO, witness, N, size, _k, k, cir_type);
  int d1 = 4;
  int d2 = ((int)log2(vL.size()) - 4) / 2;
  int d3 = ((int)log2(vL.size()) - 4) - ((int)log2(vL.size()) - 4) / 2;
  vector<int> degrees1 = {1 << d1, 1 << d2, 1 << d3, 2};
  // int d1 = ((int)log2(vL.size()))/2;
  // int d2 = ((int)log2(vL.size())) - ((int)log2(vL.size()))/2;
  // vector<int> degrees1 = {1<<d1,1<<d2,2};

  d1 = ((int)log2(2 * witness.size()) - 5) / 2;
  d2 = ((int)log2(2 * witness.size()) - 5 -
        (int)((int)log2(2 * witness.size()) - 5) / 2);
  vector<int> degrees2 = {1 << d1, 1 << d2, 32};
  vector<F> rL(4), rR(4), rO(4);
  vector<vector<F>> R1(degrees1.size()), R2(degrees2.size());

  vector<vector<vector<F>>> sumcheck_R;
  F K_r = generate_sumcheck_randomness(sumcheck_R, degrees1, degrees2, N);
  // printf("%d,%d,%d,%d\n",sumcheck_R[0].size(),sumcheck_R[1].size(),degrees1.size(),degrees2.size());

  int sumcheck_randomness_size = 0;
  for (int i = 0; i < sumcheck_R.size(); i++) {
    for (int j = 0; j < sumcheck_R[i].size(); j++)
      sumcheck_randomness_size += sumcheck_R[i][j].size();
  }
  vector<F> PCS_randomness;

  /*****************************************/
  // This emulates the masking randomness of the PCS
  // To be later replaced with the given PCS function
  prepare_parameters(rate, 4, N, witness.size() * N / 4);
  setup_randomness(PCS_randomness, N, _k, k, randomness);

  vector<vector<F>> coPCS_codeword, coPCS_Commit_data;
  MT coPCS_C;

  setup_randomness(R, N, _k, k, mask_l + 12 + 1);
  setup_randomness(_R, N, _k, k, sumcheck_randomness_size + mask_l + 12 + 1);

  vector<F> v1 = convert2vector(sumcheck_R[0]);
  vector<F> _v1 = convert2vector(sumcheck_R[1]);
  v1.insert(v1.end(), _v1.begin(), _v1.end());
  v1.insert(v1.end(), R.begin(), R.end());
  _v1 = _R;
  if (rank == 0)
    printf(">>>>>>   Open Randomness size %d\n", next_pow2(v1.size()) * N / 2);

  commit_randomness(v1, _v1, 16, N);

  init_dummy_index_commitment(index, index_data, index_codeword, index_Com,
                              index_rate, N);
  if (rank == 0)
    printf("OK\n");

  if (rank == 0)
    printf(">>>   Distributed Pcs size: %d\n", index_data[0].size());

  whir_sizes.push_back(next_pow2(v1.size()) * N / 2);
  whir_sizes.push_back(index_data[0].size());
  whir_sizes.push_back(max_delegation_size);
  whir_sizes.push_back(max_delegation_size);
  int whir_rate = 4;
  get_whir_rates(whir_rate);
  whir_betas.resize(whir_sizes.size());
  whir_polys.resize(whir_sizes.size());
  whir_c.resize(whir_sizes.size());
  whir_MTs.resize(whir_sizes.size());
  whir_claims.resize(whir_sizes.size());

  if (rank == 0) {
    printf("%d\n", whir_rates.size());
    for (int i = 0; i < 4; i++) {
      printf("%d,%d -> %d\n", whir_rates[sorted_indexes[i]], i,
             sorted_indexes[i]);
    }
  }
  // return;
  double temp_comp_time;
  double temp_ps, temp_vt;

  network_timer.start();
  MPI_Barrier(MPI_COMM_WORLD);
  network_timer.end();

  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();
  vector<double> performance_data;

  if (rank == 0) {
    const char *wan_delay = getenv("WAN_DELAY");
    const char *wan_rate = getenv("WAN_RATE");
    const char *wan_iface = getenv("WAN_IFACE");
    const char *mpi_hostfile = getenv("MPI_HOSTFILE");

    if (wan_delay && strcmp(wan_delay, "0ms") != 0) {
      char cmd[1024];
      snprintf(cmd, sizeof(cmd),
               "python3 network_simulation/driver.py --task tune_network "
               "--provers %d --iface %s --delay %s --rate %s --hostfile %s",
               N, wan_iface ? wan_iface : "lo", wan_delay,
               wan_rate ? wan_rate : "10Gbit",
               mpi_hostfile ? mpi_hostfile : "localhost");
      printf("\n[TUNING] Applying network settings: %s\n", cmd);
      int res = system(cmd);
      if (res != 0) {
        printf("[ERROR] Network tuning failed with exit code %d\n", res);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  pt.start();

  PCS_commit(coPCS_C, coPCS_codeword, coPCS_Commit_data, witness,
             PCS_randomness, rate, N);
  if (!debug_mode) {
    if (rank == 0) {
      printf("Commit Phase--- Pt: %lf, Net: %lf\n",
             pt_cp.get_time() - temp_comp_time, network_timer.get_time());
      network_timer.reset();
    }
  } else {
    performance_data =
        send_performance_data({pt_cp.get_time() - temp_comp_time, ps - temp_ps,
                               vt.get_time() - temp_vt},
                              N);
    if (rank == 0) {
      printf(
          "Commit Phase--- Pt: %lf, Vt: %lf, Ps: %lf, Mean Pt: %lf, Net: %lf\n",
          pt_cp.get_time() - temp_comp_time, performance_data[1],
          performance_data[2], performance_data[3], network_timer.get_time());
      network_timer.reset();
    }
  }

  witness.resize(coPCS_Commit_data.size() * coPCS_Commit_data[0].size(), F(0));
  int batches = 0;
  for (int i = 0; i < coPCS_Commit_data.size(); i++) {
    if (coPCS_Commit_data[i].size() == 0)
      break;
    batches++;
  }
  for (int i = 0; i < batches - 1; i++) {
    for (int j = 0; j < coPCS_Commit_data[i].size() - mask_l; j++) {
      witness[i * coPCS_Commit_data[i].size() + j] = coPCS_Commit_data[i][j];
    }
  }
  pad_vector(witness);

  // To ease development, we initialize a dummy index commitment. In practice,
  // we could let the indexer generate such a commitment in a preprocessing
  // phase.
  if (witness.size() < r_witness.size()) {
    printf("Circuit too small. Select fewer parties\n");
    return;
  }

  com_rounds = 0;
  cm = 0.0;

  pt_cp.reset();
  pt_cpu.start();
  temp_pc.start();

  // Prepare randomness to mask the evaluation claims from the first sumcehck to
  // the second one (rL,rR,rO) as well as the masking  polynomials of the
  // sumcheck. For now keep rL,rR,rO but check if we can remove them to further
  // improve proof size.

  for (int i = 0; i < 4; i++) {
    rL[i] = R[i];
    rR[i] = R[i + 4];
    rO[i] = R[i + 8];
  }

  int ctr = 12;

  temp_pc.end();

  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();

  temp_pc.start();
  vector<pair<F, vector<F>>> claims1 = prove_phase1(
      vL, vR, vO, rL, rR, rO, sumcheck_R[0], degrees1, N, _k, k, ps);
  if (rank == 0) {
    printf("      Phase 1 Net Delay: %lf\n", network_timer.get_time());
    network_timer.reset();
  }
  temp_pc.end();

  if (!debug_mode) {
    if (rank == 0)
      printf("First Sumcheck--- Pt: %lf\n", pt_cp.get_time() - temp_comp_time);
  } else {
    performance_data =
        send_performance_data({pt_cp.get_time() - temp_comp_time, ps - temp_ps,
                               vt.get_time() - temp_vt},
                              N);
    if (rank == 0)
      printf("First Sumcheck--- Pt: %lf, Vt: %lf, Ps: %lf, Mean Pt: %lf\n",
             pt_cp.get_time() - temp_comp_time, performance_data[1],
             performance_data[2], performance_data[3]);
  }

  /******************************** */
  // return;
  //
  F a, b, c;

  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();

  temp_pc.start();
  vector<pair<F, vector<F>>> claims2 = prove_phase2(
      witness, rL, rR, rO, RA, RB, RC, sumcheck_R[1], claims1[0].second,
      degrees1, degrees2, claims1[0].first, claims1[1].first, claims1[2].first,
      K_r, N, size, _k, k, a, b, c, ps);
  if (rank == 0) {
    printf("      Phase 2 Net Delay: %lf\n", network_timer.get_time());
    network_timer.reset();
  }
  temp_pc.end();

  if (!debug_mode) {
    if (rank == 0)
      printf("Second Sumcheck--- Pt: %lf\n", pt_cp.get_time() - temp_comp_time);
  } else {
    performance_data =
        send_performance_data({pt_cp.get_time() - temp_comp_time, ps - temp_ps,
                               vt.get_time() - temp_vt},
                              N);
    if (rank == 0)
      printf("Second Sumcheck--- Pt: %lf, Vt: %lf, Ps: %lf, Mean Pt: %lf\n",
             pt_cp.get_time() - temp_comp_time, performance_data[1],
             performance_data[2], performance_data[3]);
  }

  vector<vector<F>> witness_data, witness_codeword;
  // return;
  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();
  open_randomness(v1, _v1,
                  claims1[3].first + claims2[4].first + claims2[2].first, 16, N,
                  ps);
  // if(rank == 0)printf("Open Randomness--- Pt:
  // %lf\n",pt_cp.get_time()-temp_comp_time);
  if (!debug_mode) {
    if (rank == 0)
      printf("Open Randomness--- Pt: %lf\n", pt_cp.get_time() - temp_comp_time);
  } else {
    performance_data =
        send_performance_data({pt_cp.get_time() - temp_comp_time, ps - temp_ps,
                               vt.get_time() - temp_vt},
                              N);
    if (rank == 0)
      printf("Open Randomness--- Pt: %lf, Vt: %lf, Ps: %lf, Mean Pt: %lf\n",
             pt_cp.get_time() - temp_comp_time, performance_data[1],
             performance_data[2], performance_data[3]);
  }
  pair<F, vector<F>> betas_claim;
  if (!data_parallel) {
    // Run sparce eval
    /**********************/
    // TODO: I have constructed a constant round protocol, but it stills
    // evaluates a mutli-linear extension. You have to adapt it in order to
    // evaluate multivariate one
    temp_pc.start();
    // sparse_matrix_evaluation(claims2[1].first,a,b,c,
    //                      claims1[0].second,claims2[0].second,index,index_even_odd,N,ps);
    betas_claim = sparse_matrix_evaluation_multivariate_LAN(
        claims2[1].first, a, b, c, claims1[0].second, claims2[0].second,
        degrees1, degrees2, index, index_even_odd, witness_codeword,
        witness_data, N, ps);
    temp_pc.end();
    if (rank == 0)
      printf("Sparse Eval PIOP: %lf\n", temp_pc.get_time());
    temp_pc.reset();
  }

  vector<F> index_r;
  for (int i = 0; i < (int)log2(N * index_data.size() * index_data[0].size());
       i++)
    index_r.push_back(hash_to_field({0}));

  vector<vector<vector<F>>> commit_data = {witness_data, index_data};
  witness.clear();
  index_data.clear();
  vector<vector<vector<F>>> codewords = {witness_codeword, index_codeword};
  witness_codeword.clear();
  index_codeword.clear();
  vector<MT> C = {witness_Com, index_Com};

  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();

  PIOP_dOpen_batch_new(commit_data, codewords, C, {betas_claim.second, index_r},
                       {betas_claim.first, F(0)}, {true, false}, N);
  ps += dpcs_ps;
  if (!debug_mode) {
    if (rank == 0)
      printf("Batch Open Witness--- Pt: %lf\n",
             pt_cp.get_time() - temp_comp_time);
  } else {
    performance_data =
        send_performance_data({pt_cp.get_time() - temp_comp_time, ps - temp_ps,
                               vt.get_time() - temp_vt},
                              N);
    if (rank == 0)
      printf("Batch Open Witness--- Pt: %lf, Vt: %lf, Ps: %lf, Mean Pt: %lf\n",
             pt_cp.get_time() - temp_comp_time, performance_data[1],
             performance_data[2], performance_data[3]);
  }

  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();

  vector<vector<F>> inputs(3);
  vector<vector<F>> Lagrange_evals;
  inputs[2] = compute_lagrange_coeff(getRootOfUnity((int)log2(N / 4)),
                                     claims2[1].second[0], N / 4);
  inputs[0] = compute_lagrange_coeff(
      getRootOfUnity(5), claims2[1].second[claims2[1].second.size() - 1], 32);
  inputs[1] = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees2[0])),
                                     claims2[1].second[1], degrees2[0]);
  Lagrange_evals.push_back(inputs[1]);
  for (int i = 1; i < degrees2.size() - 1; i++) {

    vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees2[i])),
                                         claims2[1].second[1 + i], degrees2[i]);
    Lagrange_evals.push_back(L);
    vector<F> buff(inputs[1].size() * L.size());
    for (int j = 0; j < L.size(); j++) {
      for (int k = 0; k < inputs[1].size(); k++)
        buff[k + j * inputs[1].size()] = L[j] * inputs[1][k];
    }
    inputs[1] = buff;
  }

  temp_comp_time = pt_cp.get_time();
  temp_ps = ps;
  temp_vt = vt.get_time();
  PCS_open_new(coPCS_C, coPCS_codeword, coPCS_Commit_data, inputs,
               Lagrange_evals, claims2[0].first, rate, N, true);
  ps += pcs_ps;
  if (!debug_mode) {
    if (rank == 0)
      printf("coPCS Open--- Pt: %lf\n", pt_cp.get_time() - temp_comp_time);
  } else {
    performance_data =
        send_performance_data({pt_cp.get_time() - temp_comp_time, ps - temp_ps,
                               vt.get_time() - temp_vt},
                              N);
    if (rank == 0)
      printf("coPCS Open--- Pt: %lf, Vt: %lf, Ps: %lf, Mean Pt: %lf\n",
             pt_cp.get_time() - temp_comp_time, performance_data[1],
             performance_data[2], performance_data[3]);
  }

  if (rank == 0) {
    for (int i = 0; i < whir_c.size(); i++) {
      printf("(%d,%d,%d)\n", whir_polys[i].size(), whir_c[i].size(),
             whir_c[i].size() / whir_polys[i].size());
    }
    whir_ps = 0.0;
    Whir_Batch_Open(whir_polys, whir_c, whir_MTs, whir_claims, whir_betas,
                    whir_rate, true);
    printf("BATCH WHIR proof: %lf\n", whir_ps);
    ps += whir_ps;
  }
  // if(rank == 0)printf("Batch Open Witness--- Pt:
  // %lf\n",pt_cp.get_time()-temp_comp_time);
  // PCS_open(coPCS_C,coPCS_codeword,coPCS_Commit_data,inputs,Lagrange_evals,claims2[0].first,rate,N);

  pt.end();
  network_timer.start();
  MPI_Barrier(MPI_COMM_WORLD);
  network_timer.end();
  if (rank == 0) {
    vector<double> buff(3);
    printf(
        "Id : %d, Pt: %lf, CPU Only Pt: %lf, Computation only: %lf, Vt: %lf\n",
        rank, pt.get_time(), pt_cpu.get_time(), pt_cp.get_time(),
        vt.get_time());
    for (int i = 1; i < N; i++) {
      MPI_Recv(buff.data(), 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      printf("Id : %d, Pt: %lf, CPU Only Pt: %lf, Computation only: %lf\n", i,
             buff[0], buff[1], buff[2]);
    }
  } else {
    vector<double> buff = {pt.get_time(), pt_cpu.get_time(), pt_cp.get_time()};
    MPI_Send(buff.data(), 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  network_timer.start();
  MPI_Barrier(MPI_COMM_WORLD);
  network_timer.end();
  vector<double> buff = {vt.get_time(), ps, cm};
  if (rank != 0) {
    MPI_Send(buff.data(), 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  } else {
    double total_vt = vt.get_time();
    for (int i = 1; i < N; i++) {
      MPI_Recv(buff.data(), 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      total_vt += buff[0];
      ps += buff[1];
      cm += buff[2];
    }
    printf("Vt : %lf sec, Ps: %lf KB, Com: %lf MB\n", total_vt, ps + ps_plain,
           cm / 1024.0);
    printf("Interaction Rounds: %d\n", com_rounds);
  }
}
