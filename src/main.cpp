#include "coPIOP.h"
#include "coSumcheck_MPI.h"
#include "config_pc.hpp"
#include "constants.h"
#include "polynomial.h"
#include "utils.hpp"
#include <chrono>
#include <fcntl.h>
#include <math.h>
#include <mutex>
#include <unistd.h>
#include <vector>

#include <mpi.h>

#include "Distributed_Sumcheck.h"
#include "Fiat_Shamir.h"
#include "MPI_utils.hpp"
#include "coPCS_utils.h"
#include "timer.hpp"
#include <algorithm>

int tensor_row_size;
int mul_counter = 0;
extern timer pt, pt_cp, vt, sch_com;
extern timer_cpu pt_cpu;
timer network_timer;

extern vector<vector<pair<int, int>>> A, B, C;
extern vector<vector<pair<int, int>>> pA, pB, pC;
extern vector<vector<pair<int, int>>> tA, tB, tC;
extern int logm, logn;
extern int rate;
extern bool data_parallel;
extern int cosumcheck_offset = 0;
extern int sumcheck_offset = 4;
extern int PC_offset = 10;
extern int multree_offset = 4;
extern int com_rounds;
bool isLAN = false;
extern double cm;
extern vector<int> whir_sizes;
bool conjectured_security = false;
#include "coPCS.h"

extern double pcs_ps, cm, dpcs_ps;
extern int randomness;

void test_coPCS2(int N, int M) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  // vector<vector<int>> rate = {{8}, {8}, {8, 4}, {8, 4}};
  vector<vector<int>> rate = {{4}, {4}, {4, 4}, {4, 4}};
  int k = N / 4;
  int _k = N / 2;

  vector<vector<F>> codeword, Commit_data;
  vector<F> random_shares, shares;
  MT C;
  vector<vector<F>> mask_shares, mask_data, C_mask;
  vector<MT> Com_mask;
  prepare_parameters(rate, rate.size(), N, M);
  MPI_Barrier(MPI_COMM_WORLD);

  secret_share_coefficients(shares, M, N, _k, k);

  secret_share_coefficients(random_shares, randomness * N / 4, N, _k, k);

  MPI_Barrier(MPI_COMM_WORLD);

  pt.reset();
  pt.start();

  PCS_commit(C, codeword, Commit_data, shares, random_shares, rate, N);
  pt.end();

  if (rank == 0)
    printf("Commit: %lf,%lf\n", pt.get_time(), pt_cpu.get_time());
  MPI_Barrier(MPI_COMM_WORLD);

  vector<F> r;
  for (int i = 0; i < (int)log2(2 * M); i++)
    r.push_back(hash_to_field({0}));

  vector<F> r1, r2, r3, b1, b2, b3;
  for (int i = 0; i < (int)log2(N / 4); i++)
    r2.push_back(r[i]);
  for (int i = (int)log2(N / 4); i < r.size(); i++)
    r1.push_back(r[i]);
  precompute_beta(r1, b1);
  precompute_beta(r2, b2);

  vector<F> v(Commit_data.size() * Commit_data[0].size(), F(0));
  int batches = 0;
  for (int i = 0; i < Commit_data.size(); i++) {
    if (Commit_data[i].size() == 0)
      break;
    batches++;
  }

  for (int i = 0; i < batches - 1; i++) {
    for (int j = 0; j < Commit_data[i].size(); j++) {
      v[j * Commit_data.size() + i] = Commit_data[i][j];
    }
  }

  F K = F_ip(v, b1, b2, k, _k, N);

  b1.clear();
  b2.clear();
  r1.clear();
  r2.clear();
  int ctr = 0;
  for (int i = 0; i < (int)log2(N / 4); i++)
    r1.push_back(r[ctr++]);
  for (int i = 0; i < (int)log2(Commit_data.size()); i++)
    r2.push_back(r[ctr++]);
  for (int i = 0; i < (int)log2(Commit_data[0].size()); i++)
    r3.push_back(r[ctr++]);

  precompute_beta(r1, b1);
  precompute_beta(r2, b2);
  precompute_beta(r3, b3);

  v.clear();
  v.resize(Commit_data[0].size(), F(0));
  for (int i = 0; i < batches - 1; i++) {
    if (Commit_data[i].size() == 0)
      break;
    for (int j = 0; j < Commit_data[i].size(); j++)
      v[j] += b2[i] * Commit_data[i][j];
  }

  if (K != F_ip(v, b3, b1, k, _k, N)) {
    printf("Error\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  pt.reset();

  pt.start();
  PCS_open_new(C, codeword, Commit_data, {r}, {}, K, rate, N, false);
  pt.end();
  if (rank == 0)
    printf("Open: %lf,%lf\n", pt.get_time(), pt_cpu.get_time());

  vector<double> buff = {vt.get_time(), pcs_ps, cm};
  if (rank != 0) {
    MPI_Send(buff.data(), 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  } else {
    double total_vt = vt.get_time();
    for (int i = 1; i < N; i++) {
      MPI_Recv(buff.data(), 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      total_vt += buff[0];
      pcs_ps += buff[1];
      cm += buff[2];
    }
    printf("Vt : %lf sec, Ps: %lf KB, Com: %lf MB\n", total_vt, pcs_ps,
           cm / 1024.0);
    printf("Interaction Rounds: %d\n", com_rounds);
  }
}

void warmup_2(int M, int N, int rank) {
  int num_lanes = 16;
  size_t count_per_peer = M;
  size_t count_per_lane = count_per_peer / num_lanes;

  vector<u64> send_buff(count_per_peer), recv_buff(count_per_peer);
  for (size_t i = 0; i < count_per_peer; i++)
    send_buff[i] = i + rank;
  MPI_Request req;

  if (rank == 1) {
    MPI_Isend(send_buff.data(), send_buff.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
  }
  if (rank == 0) {
    MPI_Irecv(recv_buff.data(), recv_buff.size(), MPI_UINT64_T, 1, 0,
              MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

void warmup(int M, int N, int rank) {
  if (N <= 1)
    return;

  int num_lanes = 1;
  // Minimal data to establish connections: 64KB per peer pair (8K elements * 8
  // bytes)
  int count_per_peer = min(M, 1 << 20);
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

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  int N = size;
  int k = N / 4;
  int _k = N / 2;
  int op = atoi(argv[1]);

  int M = 1 << atoi(argv[2]);

  timer tm;
  if (op != 2 && op != 3 && op != 1) {
    tm.start();
    if (N == 2)
      warmup_2(M, N, rank);
    else
      warmup(M, N, rank);
    if (rank == 0) {
      tm.end();
      printf("Warmup done in %lf secs\n", tm.get_time());
      tm.reset();
    }
  }

  if (op == 1) {
    coPIOP_prove(M, N, _k, k, 1);
  } else if (op == 2) {
    coPIOP_prove_LAN(M, N, _k, k, 1);
  } else if (op == 4) {
    for (int reps = 0; reps < 10; reps++) {
      timer tm_rep;
      vector<u64> send_buff(M), recv_buff(M);
      MPI_Request req;
      MPI_Barrier(MPI_COMM_WORLD);
      tm_rep.start();
      MPI_Ialltoall(send_buff.data(), send_buff.size() / N, MPI_UINT64_T,
                    recv_buff.data(), recv_buff.size() / N, MPI_UINT64_T,
                    MPI_COMM_WORLD, &req);
      MPI_Wait(&req, MPI_STATUS_IGNORE);
      tm_rep.end();
      if (rank == 0)
        printf("Alltoall time: %lf sec | Comm Size: %lf KB\n",
               tm_rep.get_time(), send_buff.size() * 8.0 / 1024.0);
      tm_rep.reset();
    }
  } else if (op == 5) {
    int logn = 28;
    vector<vector<F>> arr;
    arr.resize(2);
    for (int i = 0; i < 2; i++) {
      arr[i].resize(1 << (23));
      for (int j = 0; j < arr[i].size(); j++)
        arr[i][j] = F(j);
    }
    bool flag = false;
    distributed_batch_FFT(arr, logn, flag, N);
  } else {
    test_coPCS2(N, M);
  }

  MPI_Finalize();
  return 0;
}
