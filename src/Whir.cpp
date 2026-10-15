#include "Whir.h"
#include "coPCS_utils.h"
#include "timer.hpp"
#include <algorithm>

int whir_k = 8;
double whir_ps = 0.0;
extern timer vt;

extern vector<int> whir_sizes;
extern vector<int> whir_rates;
extern vector<int> sorted_indexes;
extern vector<int> sorted_whir_rates;
extern bool conjectured_security;
#include <unordered_map>
extern unordered_map<int, int> whir_kappas;

void get_whir_rates(int rate) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  vector<pair<int, int>> size_pairs(whir_sizes.size());
  for (int i = 0; i < whir_sizes.size(); i++)
    size_pairs[i] = make_pair(whir_sizes[i], i);
  std::sort(size_pairs.begin(), size_pairs.end());
  vector<int> _sorted_indexes;
  sorted_indexes.resize(whir_sizes.size());
  for (int i = whir_sizes.size() - 1; i >= 0; i--) {
    _sorted_indexes.push_back(size_pairs[i].second);
    sorted_indexes[size_pairs[i].second] = whir_sizes.size() - 1 - i;
  }

  vector<int> sorted_sizes;
  vector<int> sorted_kappas;
  for (int i = 0; i < _sorted_indexes.size(); i++)
    sorted_sizes.push_back(whir_sizes[_sorted_indexes[i]]);
  sorted_whir_rates.push_back(rate);
  int size = sorted_sizes[0];

  vector<int> _sorted_sizes = sorted_sizes;
  vector<int> sorted_sizes2 = sorted_sizes;
  sorted_sizes.erase(sorted_sizes.begin());
  while (true) {
    if (!sorted_sizes.size())
      break;
    if (size != sorted_sizes[0]) {
      break;
    }
    sorted_whir_rates.push_back(rate);
    sorted_sizes.erase(sorted_sizes.begin());
  }

  while (sorted_sizes.size()) {
    if (size / whir_k > sorted_sizes[0]) {
      size = size / whir_k;
      rate *= 2;
    } else {
      size = sorted_sizes[0];
      rate *= 2;
      sorted_whir_rates.push_back(rate);
      int s = sorted_sizes[0];
      sorted_sizes.erase(sorted_sizes.begin());
      while (sorted_sizes.size()) {
        if (sorted_sizes[0] != s)
          break;
        sorted_whir_rates.push_back(rate);
        sorted_sizes.erase(sorted_sizes.begin());
      }
    }
    if (sorted_sizes.size() == 0) {
      break;
    }
  }
  size = sorted_sizes2[0];
  sorted_sizes2.erase(sorted_sizes2.begin());

  vector<bool> aggr;
  int ctr = 0;
  while (sorted_sizes2.size()) {
    if (size != sorted_sizes2[0])
      break;
    sorted_sizes2.erase(sorted_sizes2.begin());
    ctr++;
  }
  if (!sorted_sizes2.size()) {
    whir_kappas[size] = whir_k;
  }
  bool flag = true;
  while (sorted_sizes2.size()) {
    if (!sorted_sizes2.size() || size / whir_k > sorted_sizes2[0]) {
      if (flag) {
        whir_kappas[size] = whir_k;
        sorted_kappas.push_back(whir_k);
        flag = false;
      }
      size = size / whir_k;
    } else {
      if (flag) {
        sorted_kappas.push_back(size / sorted_sizes2[0]);
        whir_kappas[size] = size / sorted_sizes2[0];
      }
      size = sorted_sizes2[0];
      flag = true;
      int s = sorted_sizes2[0];
      while (sorted_sizes2.size()) {
        if (sorted_sizes2[0] != s)
          break;
        sorted_sizes2.erase(sorted_sizes2.begin());
      }
      if (!sorted_sizes2.size()) {
        sorted_kappas.push_back(whir_k);
        whir_kappas[size] = whir_k;
      }
    }
    if (size <= whir_k) {
      break;
    }
  }

  if (rank == 0) {
    for (int i = 0; i < sorted_whir_rates.size(); i++) {
      printf("(%d,%d) ", _sorted_sizes[i], sorted_whir_rates[i]);
    }
    printf("\n");
    printf("KAPPAS: ");
    for (int i = 0; i < sorted_kappas.size(); i++) {
      printf("(%d) ", sorted_kappas[i]);
    }
    printf("\n");
    for (int i = 0; i < whir_sizes.size(); i++) {
      printf("(%d) ", whir_kappas[whir_sizes[i]]);
    }
    printf("\n");
  }

  whir_rates = sorted_whir_rates;
}

void Whir_Commit_Codeword(vector<F> &codeword, vector<vector<_hash>> &MT,
                          int rate) {

  vector<vector<_hash>> buff_MT;

  MT.resize((int)log2(codeword.size() / whir_k) + 1);
  vector<_hash> hashes(codeword.size() / whir_k);
  vector<F> buff(whir_k);
  for (int i = 0; i < codeword.size() / whir_k; i++) {
    for (int j = 0; j < whir_k; j++)
      buff[j] = codeword[i + j * codeword.size() / whir_k];
    merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(), buff_MT,
                                                     whir_k);
  }
  MT[0] = hashes;
  merkle_tree::merkle_tree_prover::create_tree_blake(
      hashes.size(), MT, sizeof(__hhash_digest), true);
}

void Whir_Commit_initial(vector<F> &arr, vector<F> &codeword,
                         vector<vector<_hash>> &MT, int rate, int kappa) {
  codeword = arr;
  codeword.resize(rate * arr.size(), F(0));
  // fft(codeword,(int)log2(codeword.size()), false);
  pfft(codeword, (int)log2(codeword.size()), false);
  vector<vector<_hash>> buff_MT;

  // MT.resize((int)log2(codeword.size()/kappa) + 1);
  vector<_hash> hashes(codeword.size() / kappa);

  vector<F> buff(codeword.size());
  int ctr = 0;
  for (int i = 0; i < codeword.size() / kappa; i++) {
    for (int j = 0; j < kappa; j++)
      buff[ctr++] = codeword[i + j * codeword.size() / kappa];
    // merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(),buff_MT,kappa);
  }

  merkle_tree::merkle_tree_prover::MT_commit_Blake(codeword.data(), MT,
                                                   codeword.size());
  /*
  vector<F> buff(kappa);
  for(int i = 0; i < codeword.size()/kappa; i++){
      for(int j = 0; j < kappa; j++) buff[j] = codeword[i +
  j*codeword.size()/kappa];
      merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(),buff_MT,kappa);
  }
  */
  // MT[0] = hashes;
  // merkle_tree::merkle_tree_prover::create_tree_blake(hashes.size(),MT,sizeof(__hhash_digest),
  // true);
}

void Whir_Commit(vector<F> &arr, vector<F> &codeword, vector<vector<_hash>> &MT,
                 int rate, int kappa) {
  codeword = arr;
  codeword.resize(rate * arr.size(), F(0));
  // fft(codeword,(int)log2(codeword.size()), false);
  pfft(codeword, (int)log2(codeword.size()), false);

  vector<vector<_hash>> buff_MT;

  MT.resize((int)log2(codeword.size() / kappa) + 1);
  vector<_hash> hashes(codeword.size() / kappa);
  vector<F> buff(kappa);
  for (int i = 0; i < codeword.size() / kappa; i++) {
    for (int j = 0; j < kappa; j++)
      buff[j] = codeword[i + j * codeword.size() / kappa];
    merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(), buff_MT,
                                                     kappa);
  }
  MT[0] = hashes;
  merkle_tree::merkle_tree_prover::create_tree_blake(
      hashes.size(), MT, sizeof(__hhash_digest), true);
}

quadratic_poly whir_sumcheck_step1(vector<F> &arr, vector<F> &b) {
  quadratic_poly h(F_ZERO, F_ZERO, F_ZERO);
  for (int i = 0; i < arr.size() / 2; i++)
    h = h + linear_poly(arr[2 * i + 1] - arr[2 * i], arr[2 * i]) *
                linear_poly(b[2 * i + 1] - b[2 * i], b[2 * i]);
  return h;
}

void whir_sumcheck_step2(vector<F> &arr, vector<F> &b, F c) {

  for (int i = 0; i < arr.size() / 2; i++) {
    arr[i] = c * (arr[2 * i + 1] - arr[2 * i]) + arr[2 * i];
    b[i] = c * (b[2 * i + 1] - b[2 * i]) + b[2 * i];
  }
  arr.resize(arr.size() / 2);
  b.resize(b.size() / 2);
}

void whir_sumcheck(vector<F> &arr, vector<F> &b, vector<F> &r, F &K,
                   bool verify = true) {
  for (int i = 0; i < (int)log2(whir_k); i++) {
    quadratic_poly h = whir_sumcheck_step1(arr, b);
    vt.start();

    if (verify != false && K != h.eval(F(0)) + h.eval(F(1))) {
      printf("Error in Whir Sumcheck %d\n", i);
      exit(-1);
    }
    r.push_back(F::_random());
    K = h.eval(r[i]);
    vt.end();

    whir_sumcheck_step2(arr, b, r[i]);
  }
}

vector<F> whir_query(vector<F> &codeword, vector<vector<_hash>> &MT,
                     vector<F> r, vector<int> &queries, int rate) {
  int l;
  if (conjectured_security) {
    l = (int)(-100.0 / log2(1.0 - (1.0 - 1.0 / (double)rate)));
  } else {
    l = (int)(-100.0 / log2(sqrt(1.0 / (double)rate)));
  }

  queries.resize(l);

  for (int i = 0; i < l; i++) {
    if (whir_k > 4) {
      queries[i] = rand() % (codeword.size() / whir_k);
    } else {
      queries[i] = rand() % (codeword.size() / 4);
    }
  }
  vector<vector<F>> replies(l);
  for (int i = 0; i < l; i++) {
    replies[i].resize(whir_k);
    for (int j = 0; j < whir_k; j++) {
      replies[i][j] = codeword[queries[i] + j * codeword.size() / whir_k];
    }
  }

  vector<F> y;
  // Verifier checks
  vt.start();

  for (int i = 0; i < l; i++)
    y.push_back(get_RS_multifold_segment_tensor(replies[i], r, codeword.size(),
                                                queries[i]));
  vt.end();
  // Query merkle tree
  // TODO
  // printf(">> %d,%d\n",replies[0].size(),codeword.size()/whir_k);
  int log_r = (int)log2(replies[0].size());
  // printf("%d\n",MT[0].size());
  for (int i = 0; i < log_r - 2; i++) {
    //   MT.erase(MT.begin());
  }

  vector<bool> visited(2 * MT[0].size(), false);
  for (int i = 0; i < replies.size(); i++) {
    vector<vector<_hash>> MT_s;
    vector<_hash> path;
    vt.start();
    merkle_tree::merkle_tree_prover::MT_commit_Blake(replies[i].data(), MT_s,
                                                     replies[i].size());
    vt.end();

    path = merkle_tree::merkle_tree_prover::open_tree_blake(MT, queries[i]);
    vt.start();
    merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(
        MT, path.data(), queries[i], MT[0].size(), visited, whir_ps);
    vt.end();

    whir_ps += ((double)(replies[i].size() * 16)) / 1024.0;
  }

  /*****************/
  return y;
}

void compute_powers_vector(vector<F> &p, int point, int size, int n) {
  p.clear();
  p.resize(size, F(1));
  F omega = getRootOfUnity((int)log2(n));
  p[1] = omega.fastPow(omega, point);
  for (int i = 2; i < size; i++)
    p[i] = p[i - 1] * p[1];
}

vector<F> update_betas(vector<F> &b, vector<int> queries, int n) {
  // Verifier samples a random challenge
  F r = F::_random();
  vector<F> challenges;
  vector<F> powers;
  for (int i = 0; i < queries.size(); i++) {
    compute_powers_vector(powers, queries[i], b.size(), n);
    if (i == 0)
      challenges.push_back(r);
    else
      challenges.push_back(challenges[i - 1] * r);
    for (int j = 0; j < powers.size(); j++) {
      b[j] += challenges[i] * powers[j];
    }
  }

  return challenges;
}

F omega_identity(F omega, int point, vector<F> r) {
  omega = omega.fastPow(omega, point);
  F eval = F(1);
  for (int i = 0; i < r.size(); i++) {
    eval *= (F(1) - r[i] + r[i] * omega);
    omega = omega * omega;
  }
  return eval;
}

void Whir_Iteration(vector<F> &arr, vector<F> &b, vector<F> &codeword,
                    vector<vector<F>> &R, vector<vector<F>> &batch_powers,
                    vector<vector<_hash>> &MT, F &K,
                    vector<vector<int>> &all_queries,
                    vector<int> &codeword_size, int rate, bool Verify = true) {

  vector<F> r;
  vector<int> queries;
  whir_sumcheck(arr, b, r, K, Verify);

  vector<F> new_codeword;
  vector<vector<_hash>> new_MT;
  Whir_Commit_initial(arr, new_codeword, new_MT, 2 * rate, whir_k);

  vector<F> y = whir_query(codeword, MT, r, queries, rate);
  R.push_back(r);

  for (int i = 0; i < queries.size(); i++)
    queries[i] = queries[i] * (whir_k * new_codeword.size() / codeword.size());
  all_queries.push_back(queries);

  codeword = new_codeword;
  MT = new_MT;
  codeword_size.push_back(codeword.size());

  vector<F> challenges = update_betas(b, queries, new_codeword.size());
  batch_powers.push_back(challenges);
  vt.start();

  for (int i = 0; i < challenges.size(); i++)
    K += challenges[i] * y[i];
  vt.end();
}

void Whir_Open(vector<F> &arr, vector<F> &codeword, vector<vector<_hash>> &MT,
               vector<pair<F, vector<F>>> claims, int rate) {

  F K = F(0);
  vector<F> b;
  K = claims[0].first;
  precompute_beta(claims[0].second, b);

  vector<vector<F>> batch_powers(1);
  batch_powers[0].resize(claims.size(), F(1));
  if (claims.size() > 1) {
    F r = F::_random();
    F p = r;
    for (int i = 1; i < claims.size(); i++) {
      vector<F> buff;
      precompute_beta(claims[1].second, buff);
      K += p * claims[i].first;
      for (int j = 0; j < buff.size(); j++) {
        b[j] += p * buff[j];
      }
      batch_powers[0][i] = p;
      p = p * r;
    }
  }
  int iter = 0;
  vector<vector<F>> R;
  vector<vector<int>> all_queries;
  vector<int> codeword_size;
  while (true) {
    Whir_Iteration(arr, b, codeword, R, batch_powers, MT, K, all_queries,
                   codeword_size, rate);
    iter++;
    rate *= 2;
    if (arr.size() <= whir_k)
      break;
  }

  // Verification steps
  whir_ps += (arr.size() * 16 / 1024.0);
  vector<F> new_codeword;
  vector<vector<_hash>> new_MT;
  Whir_Commit(arr, new_codeword, MT, rate);
  for (int i = 0; i < new_codeword.size(); i++) {
    if (new_codeword[i] != codeword[i]) {
      printf("Error in verification step 1\n");
      exit(-1);
    }
  }
  for (int i = 0; i < arr.size(); i++) {
    K -= arr[i] * b[i];
  }
  if (K != 0) {
    printf("Error in verification step 2\n");
    exit(-1);
  }
  vector<F> r;
  for (int i = 0; i < (int)log2(arr.size()); i++)
    r.push_back(F::_random());
  F y = evaluate_vector(b, r);
  R.push_back(r);
  vector<F> total_r = convert2vector(R);
  F _y = F(0);
  for (int i = 0; i < batch_powers[0].size(); i++) {
    _y += batch_powers[0][i] * beta_identity(claims[i].second, total_r);
  }

  for (int i = 0; i < all_queries.size(); i++) {
    R.erase(R.begin());
    total_r = convert2vector(R);
    F omega = getRootOfUnity((int)log2(codeword_size[i]));
    for (int j = 0; j < all_queries[i].size(); j++) {
      _y += batch_powers[i + 1][j] *
            omega_identity(omega, all_queries[i][j], total_r);
    }
  }
  if (y != _y) {
    printf("Error in verification step 3\n");
    exit(-1);
  }
}

pair<F, vector<F>> _Whir_Open(vector<F> &arr, vector<F> &codeword,
                              vector<vector<_hash>> &MT, vector<F> &b,
                              vector<vector<pair<F, vector<F>>>> claims,
                              vector<F> challenges, F K, int rate,
                              bool Verify) {

  vector<vector<F>> batch_powers;
  vector<F> _initial_b = b;
  int iter = 0;
  vector<vector<F>> R;
  vector<vector<int>> all_queries;
  vector<int> codeword_size;
  while (true) {
    Whir_Iteration(arr, b, codeword, R, batch_powers, MT, K, all_queries,
                   codeword_size, rate, Verify);
    iter++;
    rate *= 2;
    if (arr.size() <= whir_k)
      break;
  }

  // Verification steps
  whir_ps += (arr.size() * 16 / 1024.0);
  vt.start();
  vector<F> new_codeword;
  vector<vector<_hash>> new_MT;
  Whir_Commit(arr, new_codeword, MT, rate);
  for (int i = 0; i < new_codeword.size(); i++) {
    if (new_codeword[i] != codeword[i] && Verify == true) {
      printf("Error in verification step 1\n");
      exit(-1);
    }
  }
  for (int i = 0; i < arr.size(); i++) {
    K -= arr[i] * b[i];
  }
  if (K != 0 && Verify == true) {
    printf("Error in verification step 2\n");
    exit(-1);
  }
  vector<F> r;
  for (int i = 0; i < (int)log2(arr.size()); i++)
    r.push_back(F::_random());
  F y = evaluate_vector(b, r);
  R.push_back(r);
  vector<F> total_r = convert2vector(R);
  pair<F, vector<F>> final_claim =
      make_pair(evaluate_vector(_initial_b, total_r), total_r);
  F _y = final_claim.first;
  int ctr = 0;
  for (int i = 0; i < claims.size(); i++) {
    for (int j = 0; j < claims[0].size(); j++) {
      _y += challenges[ctr++] * beta_identity(claims[i][j].second, total_r);
    }
  }

  // for(int i = 0; i < batch_powers[0].size(); i++){
  //     _y += batch_powers[0][i]*beta_identity(claims[i].second,total_r);
  // }

  for (int i = 0; i < all_queries.size(); i++) {
    R.erase(R.begin());
    total_r = convert2vector(R);
    F omega = getRootOfUnity((int)log2(codeword_size[i]));
    for (int j = 0; j < all_queries[i].size(); j++) {
      _y += batch_powers[i][j] *
            omega_identity(omega, all_queries[i][j], total_r);
    }
  }
  if (y != _y && Verify == true) {
    printf("Error in verification step 3\n");
    // exit(-1);
  }
  vt.end();

  return final_claim;
}

void aggregate_commitments(vector<F> &codeword, vector<vector<_hash>> &MT,
                           vector<vector<F>> &codewords_aggr,
                           vector<vector<vector<_hash>>> &MT_aggr, vector<F> r,
                           int rate) {

  for (int i = 0; i < codewords_aggr.size(); i++) {
    if (codewords_aggr[i].size() != codewords_aggr[0].size()) {
      printf("Incorrect codeword size %d,%d\n", codewords_aggr[0].size(),
             codewords_aggr[i].size());
      exit(-1);
    }
  }
  vector<vector<_hash>> new_MT;
  codeword.clear();
  codeword.resize(codewords_aggr[0].size());
  for (int i = 0; i < codewords_aggr.size(); i++) {
    for (int j = 0; j < codewords_aggr[i].size(); j++) {
      codeword[j] += r[i] * codewords_aggr[i][j];
    }
  }

  int l;
  if (conjectured_security) {
    l = (int)(-100.0 / log2(1.0 - (1.0 - 1.0 / (double)rate)));
  } else {
    l = (int)(-100.0 / log2(sqrt(1.0 / (double)rate)));
  }
  // int l  = (int)(-100.0/log2(sqrt(1.0/(double)rate)));
  vector<int> queries(l);
  for (int i = 0; i < l; i++)
    queries[i] = rand() % (codeword.size() / 4);

  double temp_ps = 0.0;
  vector<bool> visited(MT_aggr[0][0].size() * 2, false);
  for (int i = 0; i < l; i++) {
    vector<_hash> path = merkle_tree::merkle_tree_prover::open_tree_blake(
        MT_aggr[0], queries[i]);
    merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(
        MT_aggr[0], path.data(), queries[i], MT_aggr[0][0].size(), visited,
        temp_ps);
  }
  whir_ps += codewords_aggr.size() * temp_ps;
  whir_ps += ((double)(codewords_aggr.size() * l * sizeof(F))) / 1024.0;

  vector<vector<_hash>> buff_MT;

  new_MT.resize((int)log2(codeword.size() / whir_k) + 1);
  vector<_hash> hashes(codeword.size() / whir_k);
  vector<F> buff(whir_k);
  for (int i = 0; i < codeword.size() / whir_k; i++) {
    for (int j = 0; j < whir_k; j++)
      buff[j] = codeword[i + j * codeword.size() / whir_k];
    merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(), buff_MT,
                                                     whir_k);
  }
  new_MT[0] = hashes;
  merkle_tree::merkle_tree_prover::create_tree_blake(
      hashes.size(), new_MT, sizeof(__hhash_digest), true);
  MT = new_MT;
}

vector<vector<F>> aggregate_claims(vector<vector<F>> &arrays, vector<F> &arr,
                                   vector<F> &b_aggr,
                                   vector<vector<pair<F, vector<F>>>> claims,
                                   F &K) {
  vector<vector<F>> Sums(arrays.size());
  arr.resize(arrays[0].size(), F(0));
  vector<vector<F>> betas;
  for (int i = 0; i < claims.size(); i++) {
    for (int j = 0; j < claims[i].size(); j++) {
      vector<F> buff;
      // printf("%d\n",claims[i][j].second.size());
      precompute_beta(claims[i][j].second, buff);
      betas.push_back(buff);
    }
  }
  // printf(">OK\n");

  for (int i = 0; i < arrays.size(); i++) {
    for (int j = 0; j < betas.size(); j++) {
      F K = F(0);
      for (int k = 0; k < betas[j].size(); k++) {
        K += betas[j][k] * arrays[i][k];
      }
      Sums[i].push_back(K);
    }
  }
  whir_ps += ((double)Sums.size() * Sums[0].size() * sizeof(F)) / 1024.0;
  // Verifier's challenges to aggregate betas
  vector<F> beta_aggr_challenge;
  vector<F> arr_aggr_challenge;

  for (int i = 0; i < Sums[0].size(); i++) {
    beta_aggr_challenge.push_back(F::_random());
  }
  for (int i = 0; i < arrays.size(); i++) {
    arr_aggr_challenge.push_back(F::_random());
  }
  F K_aggr = F(0);
  b_aggr.resize(betas[0].size(), F(0));
  for (int i = 0; i < arrays.size(); i++) {
    for (int j = 0; j < Sums[i].size(); j++) {
      K_aggr += beta_aggr_challenge[j] * arr_aggr_challenge[i] * Sums[i][j];
    }
  }

  for (int i = 0; i < betas.size(); i++) {
    for (int j = 0; j < betas[i].size(); j++) {
      b_aggr[j] += beta_aggr_challenge[i] * betas[i][j];
    }
  }
  for (int i = 0; i < arrays.size(); i++) {
    for (int j = 0; j < arrays[i].size(); j++) {
      arr[j] += arr_aggr_challenge[i] * arrays[i][j];
    }
  }
  vector<vector<F>> challenges;
  K = K_aggr;
  challenges.push_back(beta_aggr_challenge);
  challenges.push_back(arr_aggr_challenge);
  return challenges;
}

F aggregate_arrays(vector<F> &arr, vector<F> &b, vector<vector<F>> &arrays,
                   vector<vector<F>> &B, vector<F> r1, vector<F> r2) {
  for (int i = 0; i < arrays.size(); i++) {
    if (arrays[i].size() != arrays[0].size()) {
      printf("Incorrect array size %d,%d\n", arrays[0].size(),
             arrays[i].size());
      exit(-1);
    }
  }

  vector<vector<F>> evals(arrays.size());
  for (int i = 0; i < arrays.size(); i++) {
    for (int j = 0; j < B.size(); j++) {
      F K = 0;
      for (int k = 0; k < B[j].size(); k++) {
        K += arrays[i][k] * B[j][k];
      }
      evals[i].push_back(K);
    }
  }
  for (int i = 1; i < arrays.size(); i++) {
    for (int j = 0; j < arrays[i].size(); j++) {
      arr[j] += r1[i] * arrays[i][j];
      b[j] += r2[i] * B[i][j];
    }
  }
  F K = F(0);
  for (int i = 0; i < evals.size(); i++) {
    for (int j = 0; j < evals[i].size(); j++) {
      K += r1[i] * r2[j] * evals[i][j];
    }
  }
  return K;
}

// Arrays are provided in sorted order
void Whir_Batch_Open(vector<vector<F>> &arrays, vector<vector<F>> &codewords,
                     vector<vector<vector<_hash>>> &MTs, vector<F> evals,
                     vector<vector<F>> &betas, int rate, bool Verify) {
  vector<F> arr = arrays[0];
  vector<F> codeword = codewords[0];
  vector<vector<_hash>> MT = MTs[0];

  F K = F(0);

  vector<F> b = betas[0];
  vector<vector<F>> old_betas = betas;
  K = evals[0];
  arrays.erase(arrays.begin());
  codewords.erase(codewords.begin());
  MTs.erase(MTs.begin());
  betas.erase(betas.begin());
  vector<vector<F>> batch_powers;
  int iter = 0;
  vector<vector<F>> R;
  vector<vector<int>> all_queries;
  vector<int> codeword_size;
  int original_whir_k = whir_k;
  vector<F> aggr_total_randomness;
  vector<int> kappas;
  int size = arr.size();
  vector<int> sizes;
  vector<bool> aggr;
  for (int i = 0; i < arrays.size(); i++) {
    sizes.push_back(arrays[i].size());
  }
  while (true) {
    if (!sizes.size() || size / whir_k > sizes[0]) {
      kappas.push_back(whir_k);
      size = size / whir_k;
      aggr.push_back(false);
    } else {
      kappas.push_back(size / sizes[0]);
      int s = sizes[0];
      while (sizes.size()) {
        if (sizes[0] != s)
          break;
        sizes.erase(sizes.begin());
      }
      size = size / kappas[kappas.size() - 1];
      aggr.push_back(true);
    }
    if (size <= whir_k) {
      break;
    }
  }
  for (int i = 0; i < kappas.size(); i++) {
    printf("%d ", kappas[i]);
  }
  printf("\n");
  for (int i = 0; i < aggr.size(); i++) {
    printf("%d ", aggr[i] == true);
  }
  printf("\n");
  while (true) {
    whir_k = kappas[iter];
    printf("Iter: %d\n", iter);
    if (!aggr[iter]) {
      Whir_Iteration(arr, b, codeword, R, batch_powers, MT, K, all_queries,
                     codeword_size, rate, Verify);
      iter++;
      rate *= 2;
    } else {
      printf("OK\n");
      // printf("Whir K: %d\n",whir_k);
      if (whir_k != 1) {
        Whir_Iteration(arr, b, codeword, R, batch_powers, MT, K, all_queries,
                       codeword_size, rate, Verify);
        printf("OK\n");
        rate *= 2;
      }
      iter++;
      vector<vector<F>> B_aggr;
      B_aggr.push_back(b);
      vector<vector<F>> arr_aggr;
      arr_aggr.push_back(arr);
      vector<vector<F>> codewords_aggr;
      codewords_aggr.push_back(codeword);
      vector<vector<vector<_hash>>> MT_aggr;
      MT_aggr.push_back(MT);
      for (int i = 0; i < arrays.size(); i++) {
        if (arrays[i].size() == arr.size()) {
          B_aggr.push_back(betas[i]);
          arr_aggr.push_back(arrays[i]);
          codewords_aggr.push_back(codewords[i]);
          MT_aggr.push_back(MTs[i]);
        }
      }
      for (int i = 1; i < arr_aggr.size(); i++) {
        arrays.erase(arrays.begin());
        codewords.erase(codewords.begin());
        MTs.erase(MTs.begin());
        betas.erase(betas.begin());
      }
      // codeword.clear();arr.clear();MT.clear();
      MT.clear();
      vector<F> aggregation_challenge1;
      vector<F> aggregation_challenge2;
      for (int i = 0; i < arr_aggr.size(); i++)
        aggregation_challenge1.push_back(F::_random());
      for (int i = 0; i < arr_aggr.size(); i++)
        aggregation_challenge2.push_back(F::_random());
      aggregation_challenge1[0] = 1;
      aggregation_challenge2[0] = 1;

      for (int i = 1; i < aggregation_challenge1.size(); i++)
        aggr_total_randomness.push_back(aggregation_challenge1[i]);
      // return;
      K = aggregate_arrays(arr, b, arr_aggr, B_aggr, aggregation_challenge2,
                           aggregation_challenge1);
      F _K = 0;
      for (int i = 0; i < arr.size(); i++)
        _K += arr[i] * b[i];
      if (_K != K)
        printf("Error\n");

      whir_k = kappas[iter + 1];
      if (arrays.size() && arr.size() / whir_k <= arrays[0].size()) {
        whir_k = arr.size() / arrays[0].size();
      }
      aggregate_commitments(codeword, MT, codewords_aggr, MT_aggr,
                            aggregation_challenge2,
                            codewords_aggr[0].size() / arr_aggr[0].size());
    }

    if (arr.size() <= whir_k)
      break;
  }

  // Verification steps
  whir_ps += (arr.size() * 16 / 1024.0);
  vt.start();
  printf("%d,%d\n", arr.size(), b.size());

  vector<F> new_codeword;
  vector<vector<_hash>> new_MT;
  Whir_Commit(arr, new_codeword, MT, rate);
  for (int i = 0; i < new_codeword.size(); i++) {
    if (new_codeword[i] != codeword[i] && Verify == true) {
      printf("Error in verification step 1\n");
      exit(-1);
    }
  }
  for (int i = 0; i < arr.size(); i++) {
    K -= arr[i] * b[i];
  }
  if (K != 0 && Verify == true) {
    printf("Error in verification step 2\n");
    exit(-1);
  }
  vector<F> r;
  for (int i = 0; i < (int)log2(arr.size()); i++)
    r.push_back(F::_random());
  F y = evaluate_vector(b, r);
  R.push_back(r);
  vt.end();

  F _y = F(0);
  aggr_total_randomness.insert(aggr_total_randomness.begin(), F(1));
  vector<F> total_r = convert2vector(R);
  for (int i = 0; i < old_betas.size(); i++) {
    vector<F> r;
    for (int j = total_r.size() - (int)log2(old_betas[i].size());
         j < total_r.size(); j++) {
      r.push_back(total_r[j]);
    }
    _y += aggr_total_randomness[i] * evaluate_vector(old_betas[i], r);
  }

  // for(int i = 0; i < batch_powers[0].size(); i++){
  //     _y += batch_powers[0][i]*beta_identity(claims[i].second,total_r);
  // }
  vt.start();

  for (int i = 0; i < all_queries.size(); i++) {
    R.erase(R.begin());
    total_r = convert2vector(R);
    F omega = getRootOfUnity((int)log2(codeword_size[i]));
    for (int j = 0; j < all_queries[i].size(); j++) {
      _y += batch_powers[i][j] *
            omega_identity(omega, all_queries[i][j], total_r);
    }
  }
  if (y != _y && Verify == true) {
    printf("Error in verification step 3\n");
    // exit(-1);
  }
  vt.end();
}
