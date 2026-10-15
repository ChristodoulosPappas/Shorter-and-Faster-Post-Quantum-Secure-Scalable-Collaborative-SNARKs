#pragma once
#include "MPI_utils.hpp"
#include "timer.hpp"
extern vector<vector<pair<int, int>>> A, B, C;
// Transposed R1CS matrixes
extern vector<vector<pair<int, int>>> tA, tB, tC;
extern vector<vector<pair<int, int>>> pA, pB, pC;
extern int logm, logn;
extern vector<int> real_idx_dim;
extern vector<int> real_idx_dim_even_odd;

extern double cm;
extern timer pt_cp, vt;
extern int com_rounds;
extern int mask_l;

void myAlltoAll(vector<u64> &in, vector<u64> &out, int N, int total_size) {
  int rank;

  vector<vector<u64>> recv_data(N);
  vector<vector<u64>> send_data(N);
  vector<MPI_Request> req1(N), req2(N);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  if (rank == 0)
    printf("Size per comm: %d\n", total_size / N);
  for (int i = 0; i < N; i++) {
    if (i == rank)
      continue;
    send_data[i].resize(total_size / N);
    recv_data[i].resize(total_size / N);
    for (int j = 0; j < total_size / N; j++) {
      send_data[i][j] = in[i * total_size / N + j];
    }
    MPI_Isend(send_data[i].data(), send_data[i].size(), MPI_UINT64_T, i, 0,
              MPI_COMM_WORLD, &req1[i]);
    MPI_Irecv(recv_data[i].data(), send_data[i].size(), MPI_UINT64_T, i, 0,
              MPI_COMM_WORLD, &req2[i]);
  }
  for (int i = 0; i < N; i++) {
    if (i == rank) {
      for (int j = 0; j < total_size / N; j++) {
        out[i * total_size / N + j] = in[i * total_size / N + j];
      }
      continue;
    }
    MPI_Wait(&req2[i], MPI_STATUS_IGNORE);
    for (int j = 0; j < total_size / N; j++) {
      out[i * total_size / N + j] = recv_data[i][j];
    }
  }
  for (int i = 0; i < N; i++) {
    if (i == rank)
      continue;
    MPI_Wait(&req1[i], MPI_STATUS_IGNORE);
  }
}

void send_pcs_data(vector<F> &arr, vector<F> &B, vector<F> &codeword,
                   vector<vector<F>> &total_arr, vector<vector<F>> &total_B,
                   vector<vector<F>> &C, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  com_rounds += 1;

  if (rank == 0) {
    C.push_back(codeword);
    total_arr.push_back(arr);
    total_B.push_back(B);
    vector<MPI_Request> req(N);
    vector<vector<u64>> recv_data(N);
    vector<u64> arr_data(2 * arr.size());
    vector<u64> b_data(2 * B.size());
    vector<u64> codeword_data(2 * codeword.size());
    vector<F> segments;
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(2 * (codeword.size() + B.size() + arr.size()));
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
      // MPI_Recv(f_element.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      int ctr = 0;
      for (int j = 0; j < arr_data.size(); j++)
        arr_data[j] = recv_data[i][ctr++];
      for (int j = 0; j < b_data.size(); j++)
        b_data[j] = recv_data[i][ctr++];
      for (int j = 0; j < codeword_data.size(); j++)
        codeword_data[j] = recv_data[i][ctr++];
      field_vector_deserialize(codeword_data, segments);
      C.push_back(segments);
      field_vector_deserialize(arr_data, segments);
      total_arr.push_back(segments);
      // arr.insert(arr.end(),segments.begin(),segments.end());
      field_vector_deserialize(b_data, segments);
      B.insert(B.end(), segments.begin(), segments.end());
      total_B.push_back(segments);
    }
  } else {
    vector<u64> msg, buff;
    field_vector_serialize(arr, buff);
    msg = buff;
    field_vector_serialize(B, buff);
    msg.insert(msg.end(), buff.begin(), buff.end());
    field_vector_serialize(codeword, buff);
    msg.insert(msg.end(), buff.begin(), buff.end());
    cm += 8 * msg.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(msg.data(), msg.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
}

void get_evals(vector<F> &y, vector<vector<u32>> &queries, int l, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  com_rounds += 2;
  vector<u64> f_element;
  if (rank != 0) {
    f_element.resize(1 + 3 * l, (0));
    cm += f_element.size() * 8.0 / 1024.0;
    f_element[0] = y.size();
    for (int i = 0; i < queries.size(); i++)
      f_element[i + 1] = queries[i][1];
    for (int i = 0; i < queries.size(); i++) {
      f_element[2 * i + 1 + l] = y[i].real;
      f_element[2 * i + 1 + 1 + l] = y[i].img;
    }
    MPI_Request req;
    MPI_Isend(f_element.data(), f_element.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
    f_element.clear();
    f_element.resize(4 * l);
  } else {
    vector<MPI_Request> req(N);
    vector<vector<u64>> recv_data(N);

    for (int i = 1; i < N; i++) {
      recv_data[i].resize(1 + 3 * l, (0));
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();

      for (int j = 0; j < recv_data[i][0]; j++) {
        queries.push_back({(u32)i, (u32)recv_data[i][j + 1]});
      }
      for (int j = 0; j < recv_data[i][0]; j++) {
        F a;
        a.real = recv_data[i][1 + l + 2 * j];
        a.img = recv_data[i][1 + l + 2 * j + 1];
        y.push_back(a);
      }
    }
    for (int i = 0; i < l; i++) {
      f_element.push_back(queries[i][0]);
      f_element.push_back(queries[i][1]);
      f_element.push_back(y[i].real);
      f_element.push_back(y[i].img);
    }
  }
  myBcast(f_element, N);
  if (rank != 0) {
    queries.clear();
    y.clear();
    for (int i = 0; i < l; i++) {
      queries.push_back({(u32)f_element[4 * i], (u32)f_element[4 * i + 1]});
      F a;
      a.real = f_element[4 * i + 2];
      a.img = f_element[4 * i + 3];
      y.push_back(a);
    }
  }
}
#include <unistd.h>
void get_segments(vector<F> &y, vector<vector<F>> &segments,
                  vector<vector<u32>> &queries, int l, int N, int degree) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> f_element;
  com_rounds++;
  // network_timer.start(); MPI_Barrier(MPI_COMM_WORLD); network_timer.end();

  if (rank != 0) {
    f_element.resize(1 + 3 * l + 2 * l * degree, (0));
    f_element[0] = segments.size();
    for (int i = 0; i < queries.size(); i++)
      f_element[i + 1] = queries[i][1];
    for (int i = 0; i < queries.size(); i++) {
      f_element[2 * i + 1 + l] = y[i].real;
      f_element[2 * i + 1 + 1 + l] = y[i].img;
    }
    int ctr = 1 + 3 * l;

    for (int i = 0; i < segments.size(); i++) {
      vector<u64> buff;
      field_vector_serialize(segments[i], buff);
      for (int j = 0; j < buff.size(); j++) {
        f_element[ctr++] = buff[j];
      }
    }
    MPI_Request req;
    MPI_Isend(f_element.data(), f_element.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  } else {
    f_element.resize(1 + 3 * l + 2 * l * degree, (0));
    cm += f_element.size() * 8.0 / 1024.0;
    vector<MPI_Request> req(N);
    vector<vector<u64>> recv_data(N);
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(1 + 3 * l + 2 * l * degree, (0));
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
    }
    for (int i = 1; i < N; i++) {
      int ctr = 1 + 3 * l;
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      for (int j = 0; j < recv_data[i][0]; j++) {
        queries.push_back({(u32)i, (u32)recv_data[i][j + 1]});
      }
      for (int j = 0; j < recv_data[i][0]; j++) {
        F a;
        a.real = recv_data[i][1 + l + 2 * j];
        a.img = recv_data[i][1 + l + 2 * j + 1];
        y.push_back(a);
      }

      for (int j = 0; j < recv_data[i][0]; j++) {
        vector<F> buff_F;
        vector<u64> buff(2 * degree);
        for (int k = 0; k < 2 * degree; k++) {
          buff[k] = recv_data[i][ctr++];
        }
        field_vector_deserialize(buff, buff_F);
        segments.push_back(buff_F);
      }
    }
  }
}

void get_segments_new(vector<F> &y, vector<vector<F>> &segments,
                      vector<vector<u32>> &queries, int l, int N, int degree) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> f_element;
  com_rounds++;
  // MPI_Barrier(MPI_COMM_WORLD);

  if (rank != 0) {
    f_element.resize(3 * segments.size() + 2 * segments.size() * degree, (0));
    u64 size = segments.size();
    MPI_Request req1, req2;
    MPI_Isend(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req2);
    for (int i = 0; i < queries.size(); i++)
      f_element[i] = queries[i][1];
    for (int i = 0; i < queries.size(); i++) {
      f_element[2 * i + size] = y[i].real;
      f_element[2 * i + 1 + size] = y[i].img;
    }
    int ctr = 3 * queries.size();

    for (int i = 0; i < segments.size(); i++) {
      vector<u64> buff;
      field_vector_serialize(segments[i], buff);
      for (int j = 0; j < buff.size(); j++) {
        f_element[ctr++] = buff[j];
      }
    }
    if (size != 0)
      MPI_Isend(f_element.data(), f_element.size(), MPI_UINT64_T, 0, 1,
                MPI_COMM_WORLD, &req1);

    if (size != 0)
      MPI_Wait(&req1, MPI_STATUS_IGNORE);
    MPI_Wait(&req2, MPI_STATUS_IGNORE);
  } else {

    // cm+= f_element.size()*8.0/1024.0;
    vector<MPI_Request> req1(N), req2(N);
    vector<vector<u64>> recv_data(N);
    vector<u64> sizes(N);
    for (int i = 1; i < N; i++)
      MPI_Irecv(&sizes[i], 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &req1[i]);
    for (int i = 1; i < N; i++)
      MPI_Wait(&req1[i], MPI_STATUS_IGNORE);
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(3 * sizes[i] + 2 * sizes[i] * degree);
    }
    for (int i = 1; i < N; i++) {
      if (sizes[i] == 0)
        continue;
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 1,
                MPI_COMM_WORLD, &req2[i]);
    }

    for (int i = 1; i < N; i++) {
      int ctr = 3 * sizes[i];
      if (sizes[i] == 0)
        continue;
      MPI_Wait(&req2[i], MPI_STATUS_IGNORE);
      for (int j = 0; j < sizes[i]; j++) {
        queries.push_back({(u32)i, (u32)recv_data[i][j]});
      }
      for (int j = 0; j < sizes[i]; j++) {
        F a;
        a.real = recv_data[i][sizes[i] + 2 * j];
        a.img = recv_data[i][sizes[i] + 2 * j + 1];
        y.push_back(a);
      }

      ctr = 3 * sizes[i];
      for (int j = 0; j < sizes[i]; j++) {
        vector<F> buff_F;
        vector<u64> buff(2 * degree);
        for (int k = 0; k < 2 * degree; k++) {
          buff[k] = recv_data[i][ctr++];
        }
        field_vector_deserialize(buff, buff_F);
        segments.push_back(buff_F);
      }
    }
  }
}

vector<F> pcs_sumcheck_aggr(vector<F> &poly, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  vector<u64> f_element;

  com_rounds += 2;
  if (rank == 0) {
    vector<vector<F>> P(N);
    P[0] = poly;
    vector<MPI_Request> req(N);
    vector<vector<u64>> recv_data(N);

    for (int i = 1; i < N; i++) {
      recv_data[i].resize(2 * poly.size());
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
      // MPI_Recv(f_element.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], P[i]);
    }
    pt_cp.start();

    P = transpose(P);
    for (int i = 0; i < P.size(); i++)
      P[i] = unpack_secret_share(P[i], N, N);
    for (int i = 0; i < poly.size(); i++)
      poly[i] = F(0);
    // P = transpose(P);

    for (int i = 0; i < P.size(); i++) {
      for (int j = 0; j < P[i].size() / 2; j++) {
        poly[i] += P[i][2 * j];
      }
    }

    pt_cp.end();
    field_vector_serialize(poly, f_element);

    // for(int i = 1; i < N; i++){
    //     cm += 8*f_element.size()/1024.0;
    //     MPI_Send(f_element.data(), 2,MPI_UINT64_T,i,0,MPI_COMM_WORLD);
    // }
  } else {
    field_vector_serialize(poly, f_element);

    cm += 8 * f_element.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(f_element.data(), f_element.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
    // MPI_Recv(f_element.data(),2,MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  myBcast(f_element, N);
  if (rank != 0) {
    poly.clear();
    field_vector_deserialize(f_element, poly);
  }
  return poly;
}
void myBcast(vector<u64> &data, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  if (rank == 0) {

    vector<MPI_Request> req(N - 1);
    for (int i = 1; i < N; i++) {
      MPI_Isend(data.data(), data.size(), MPI_UINT64_T, i, 0, MPI_COMM_WORLD,
                &req[i - 1]);
    }
    for (int i = 0; i < N - 1; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
    }
    cm += (N - 1) * data.size() * 8 / 1024.0;
  } else {
    MPI_Request req;
    MPI_Irecv(data.data(), data.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
}

vector<double> send_performance_data(vector<double> arr, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  if (rank == 0) {
    vector<double> buff(3);
    vector<double> data(4, 0.0);
    data[0] = arr[0];
    data[2] = arr[1];
    data[1] = arr[2];

    for (int i = 1; i < N; i++) {
      MPI_Recv(buff.data(), buff.size(), MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      data[1] += buff[2];
      data[2] += buff[1];
      data[3] += buff[0];
    }
    data[3] = data[3] / (double)(N - 1);
    return data;
  }
  MPI_Send(arr.data(), arr.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  return {};
}

void myBcast(vector<u8> &data, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  if (rank == 0) {
    vector<MPI_Request> req(N - 1);
    for (int i = 1; i < N; i++) {
      MPI_Isend(data.data(), data.size(), MPI_UINT8_T, i, 0, MPI_COMM_WORLD,
                &req[i - 1]);
    }
    // for(int i = 0; i < N-1; i++){
    //     network_timer.start(); MPI_Wait(&req[i],MPI_STATUS_IGNORE);
    //     network_timer.end();
    // }
    network_timer.start();
    MPI_Waitall(N - 1, req.data(), MPI_STATUSES_IGNORE);
    network_timer.end();
  } else {
    MPI_Request req;
    MPI_Irecv(data.data(), data.size(), MPI_UINT8_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
}

F F_ip(vector<F> &data, vector<F> &v1, vector<F> &v2, int k, int _k, int N){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get my process id
    F y = F(0),sum = F(0);
    //printf("%d,%d\n",data.size(),v1.size());
    pt_cp.start();
    for(int i = 0; i < data.size(); i++){
        y += data[i]*v1[i];
    }
    pt_cp.end();
    
    vector<u64> f_element(2);
    com_rounds+=2;
    if(rank == 0){
        vector<F> Y(N);Y[0] = y;
        vector<MPI_Request> req(N);
        vector<vector<u64>> recv_data(N);

        for(int i = 1; i < N; i++){
            recv_data[i].resize(2);
            MPI_Irecv(recv_data[i].data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,&req[i]);
            //MPI_Recv(f_element.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        for(int i = 1; i < N; i++){
            MPI_Wait(&req[i],MPI_STATUS_IGNORE);
            Y[i].real = recv_data[i][0];
            Y[i].img = recv_data[i][1];

        }
        pt_cp.start();
    
        fft(Y,(int)log2(Y.size()),true);
        
        F omega = getRootOfUnity(1+(int)log2(N)).inv();
        F mul = F(1); 
        for(int i = 0; i < Y.size(); i++){
            Y[i] = mul*Y[i];
            mul = mul*omega;
        }
        
        fft(Y,(int)log2(Y.size()),false);
        
        for(int i = 0; i < v2.size(); i++){
            sum += v2[i]*Y[N*i/_k];
        }
        f_element[0] = sum.real;
        f_element[1] = sum.img;
        pt_cp.end();
    
        //for(int i = 1; i < N; i++){
        //    cm += 8*f_element.size()/1024.0;
        //    MPI_Send(f_element.data(), 2,MPI_UINT64_T,i,0,MPI_COMM_WORLD);
        //}

    }else{
        f_element[0] = y.real;
        f_element[1] = y.img;
        cm += 8*f_element.size()/1024.0;
        MPI_Request req;
        MPI_Isend(f_element.data(),2,MPI_UINT64_T,0,0,MPI_COMM_WORLD,&req);
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        //MPI_Recv(f_element.data(),2,MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    myBcast(f_element,N);
    if(rank != 0){
        sum.real = f_element[0];
        sum.img = f_element[1];
    }
    return sum;
}


F F_ip_prod(vector<F> &data1, vector<F> &data2, vector<F> &v1, vector<F> &v2, int k, int _k, int N){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get my process id
    F y = F(0),sum = F(0);
    pt_cp.start();
     
    for(int i = 0; i < data1.size(); i++){
        y += data1[i]*data2[i]*v1[i];
    }
    pt_cp.end();
    
    vector<u64> f_element(2);
    com_rounds+=2;
    
    if(rank == 0){
        
        vector<F> Y(N);Y[0] = y;
        vector<MPI_Request> req(N);
        vector<vector<u64>> recv_data(N);

        for(int i = 1; i < N; i++){
            recv_data[i].resize(2);
            //MPI_Recv(f_element.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Irecv(recv_data[i].data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,&req[i]);
        }
        for(int i = 1; i < N; i++){
            MPI_Wait(&req[i],MPI_STATUS_IGNORE);
            Y[i].real = recv_data[i][0];
            Y[i].img = recv_data[i][1];
        }
        pt_cp.start();
    
        fft(Y,(int)log2(Y.size()),true);
        F omega = getRootOfUnity(1+(int)log2(N)).inv();
        F mul = F(1);
        for(int i = 0; i < Y.size(); i++){
            Y[i] = mul*Y[i];
            mul = mul*omega;
        }
        fft(Y,(int)log2(Y.size()),false);
        
        for(int i = 0; i < v2.size(); i++){
            sum += v2[i]*Y[N*i/_k];
        }
        f_element[0] = sum.real;
        f_element[1] = sum.img;
        pt_cp.end();
    
        //for(int i = 1; i < N; i++){
        //    cm += 8*f_element.size()/1024.0;
        //    MPI_Send(f_element.data(), 2,MPI_UINT64_T,i,0,MPI_COMM_WORLD);
        //}

    }else{
        f_element[0] = y.real;
        f_element[1] = y.img;
        cm += 8*f_element.size()/1024.0;
        MPI_Request req;
        MPI_Isend(f_element.data(),f_element.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD,&req);
        MPI_Wait(&req,MPI_STATUS_IGNORE);
    
        //MPI_Send(f_element.data(),2,MPI_UINT64_T,0,0,MPI_COMM_WORLD);
        //MPI_Recv(f_element.data(),2,MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    myBcast(f_element,N);
    if(rank != 0){
        sum.real = f_element[0];
        sum.img = f_element[1];
    }
    return sum;
}

quadratic_poly _batch_aggregate(vector<quadratic_poly> polys, vector<F> v2,
                                int k, bool secret_shared) {
  quadratic_poly H;
  F a = F(0), b = F(0), c = F(0);
  if (!secret_shared) {
    for (int i = 0; i < polys.size(); i++) {
      a += v2[i] * polys[i].a;
      b += v2[i] * polys[i].b;
      c += v2[i] * polys[i].c;
    }
    H = quadratic_poly(a, b, c);
  } else {
    int N = polys.size();
    vector<F> _a(N), _b(N), _c(N);
    for (int i = 0; i < N; i++) {
      _a[i] = polys[i].a;
      _b[i] = polys[i].b;
      _c[i] = polys[i].c;
    }
    pt_cp.start();

    fft(_a, (int)log2(_a.size()), true);
    fft(_b, (int)log2(_a.size()), true);
    fft(_c, (int)log2(_a.size()), true);
    F omega = getRootOfUnity(1 + (int)log2(N)).inv();
    F mul = F(1);
    for (int i = 0; i < _a.size(); i++) {
      _a[i] = mul * _a[i];
      _b[i] = mul * _b[i];
      _c[i] = mul * _c[i];
      mul = mul * omega;
    }

    fft(_a, (int)log2(_a.size()), false);
    fft(_b, (int)log2(_a.size()), false);
    fft(_c, (int)log2(_a.size()), false);

    for (int j = 0; j < v2.size(); j++) {
      a += v2[j] * _a[N * j / k];
      b += v2[j] * _b[N * j / k];
      c += v2[j] * _c[N * j / k];
    }
    H = quadratic_poly(a, b, c);
    pt_cp.end();
  }
  return H;
}

vector<quadratic_poly> batch_aggregate(vector<quadratic_poly> H,
                                       vector<vector<F>> v2,
                                       vector<bool> secret_shared,
                                       vector<int> rounds, vector<int> k,
                                       int round, int N) {
  vector<F> poly;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  for (int i = 0; i < H.size(); i++) {
    if (rounds[i] > round) {
      poly.push_back(H[i].a);
      poly.push_back(H[i].b);
      poly.push_back(H[i].c);
    }
  }
  vector<u64> coef_u;
  field_vector_serialize(poly, coef_u);
  com_rounds += 2;
  vector<quadratic_poly> final_P(rounds.size());

  if (rank == 0) {
    vector<MPI_Request> req(N - 1);
    vector<vector<u64>> recv_data(N - 1);
    vector<vector<F>> polys(N);
    polys[0] = poly;
    for (int i = 0; i < N - 1; i++) {
      recv_data[i].resize(coef_u.size());
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i + 1,
                0, MPI_COMM_WORLD, &req[i]);
    }
    for (int i = 0; i < N - 1; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], polys[i + 1]);
    }
    vector<vector<quadratic_poly>> P(rounds.size());
    int ctr = 0;
    for (int i = 0; i < rounds.size(); i++) {
      if (rounds[i] > round) {
        P[i].resize(N);
        for (int j = 0; j < N; j++) {
          P[i][j] = quadratic_poly(polys[j][3 * ctr], polys[j][3 * ctr + 1],
                                   polys[j][3 * ctr + 2]);
        }
        ctr++;
      }
      if (P[i].size() != 0) {
        final_P[i] =
            (_batch_aggregate(P[i], v2[i], 2 * k[i], secret_shared[i]));
      }
    }
    poly.clear();
    for (int i = 0; i < final_P.size(); i++) {
      if (P[i].size() != 0) {
        poly.push_back(final_P[i].a);
        poly.push_back(final_P[i].b);
        poly.push_back(final_P[i].c);
      }
    }
    // printf("%d, (%lld),%lld\n",round,final_P[1].a.real,poly[0].real);
    field_vector_serialize(poly, coef_u);
  } else {
    MPI_Request req;
    MPI_Isend(coef_u.data(), coef_u.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
  myBcast(coef_u, N);
  if (rank != 0) {
    field_vector_deserialize(coef_u, poly);
    int ctr = 0;
    for (int i = 0; i < rounds.size(); i++) {
      if (rounds[i] > round) {
        final_P[i] =
            quadratic_poly(poly[3 * ctr], poly[3 * ctr + 1], poly[3 * ctr + 2]);
        ctr++;
      }
    }
    // printf("%d, (%lld)\n",round,final_P[1].a.real);
  }
  return final_P;
}

quadratic_poly aggregate_quadratic_poly(quadratic_poly H, int k, int _k,
                                        int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  F a = F(0), b = F(0), c = F(0);
  pt_cp.start();

  vector<F> coef;
  coef.push_back(H.a);
  coef.push_back(H.b);
  coef.push_back(H.c);
  vector<u64> coef_u;

  field_vector_serialize(coef, coef_u);
  pt_cp.end();
  com_rounds += 2;
  if (rank == 0) {
    vector<F> _a(N), _b(N), _c(N);
    _a[0] = coef[0];
    _b[0] = coef[1];
    _c[0] = coef[2];
    vector<MPI_Request> req(N);
    vector<vector<u64>> recv_data(N);

    for (int i = 1; i < N; i++) {
      recv_data[i].resize(6);
      // MPI_Recv(coef_u.data(),6,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Irecv(recv_data[i].data(), 6, MPI_UINT64_T, i, 0, MPI_COMM_WORLD,
                &req[i]);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], coef);
      _a[i] = coef[0];
      _b[i] = coef[1];
      _c[i] = coef[2];
    }
    pt_cp.start();

    fft(_a, (int)log2(_a.size()), true);
    fft(_b, (int)log2(_a.size()), true);
    fft(_c, (int)log2(_a.size()), true);
    F omega = getRootOfUnity(1 + (int)log2(N)).inv();
    F mul = F(1);
    for (int i = 0; i < _a.size(); i++) {
      _a[i] = mul * _a[i];
      _b[i] = mul * _b[i];
      _c[i] = mul * _c[i];
      mul = mul * omega;
    }
    fft(_a, (int)log2(_a.size()), false);
    fft(_b, (int)log2(_a.size()), false);
    fft(_c, (int)log2(_a.size()), false);

    for (int j = 0; j < k; j++) {
      a += _a[N * j / _k];
      b += _b[N * j / _k];
      c += _c[N * j / _k];
    }
    coef[0] = a;
    coef[1] = b;
    coef[2] = c;
    field_vector_serialize(coef, coef_u);
    pt_cp.end();

    // for(int i = 1; i < N; i++){
    //     cm += 8*coef_u.size()/1024.0;
    //     MPI_Send(coef_u.data(),6,MPI_UINT64_T,i,0,MPI_COMM_WORLD);
    // }
  } else {
    cm += 8 * coef_u.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(coef_u.data(), coef_u.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();

    // MPI_Send(coef_u.data(),6,MPI_UINT64_T,0,0,MPI_COMM_WORLD);
    // MPI_Recv(coef_u.data(),6,MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    // a = coef[0];b = coef[1];c = coef[2];
  }
  myBcast(coef_u, N);
  if (rank != 0) {
    field_vector_deserialize(coef_u, coef);
    a = coef[0];
    b = coef[1];
    c = coef[2];
  }
  H = quadratic_poly(a, b, c);
  return H;
}

quadratic_poly aggregate_quadratic_poly(quadratic_poly H, vector<F> &v, int k,
                                        int _k, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  F a = F(0), b = F(0), c = F(0);
  pt_cp.start();

  vector<F> coef;
  coef.push_back(H.a);
  coef.push_back(H.b);
  coef.push_back(H.c);
  vector<u64> coef_u;

  field_vector_serialize(coef, coef_u);
  pt_cp.end();
  com_rounds += 2;
  if (rank == 0) {
    vector<F> _a(N), _b(N), _c(N);
    _a[0] = coef[0];
    _b[0] = coef[1];
    _c[0] = coef[2];
    vector<MPI_Request> req(N);
    vector<vector<u64>> recv_data(N);
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(6);
      MPI_Irecv(recv_data[i].data(), 6, MPI_UINT64_T, i, 0, MPI_COMM_WORLD,
                &req[i]);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      // MPI_Recv(coef_u.data(),6,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      field_vector_deserialize(recv_data[i], coef);
      _a[i] = coef[0];
      _b[i] = coef[1];
      _c[i] = coef[2];
    }
    pt_cp.start();

    fft(_a, (int)log2(_a.size()), true);
    fft(_b, (int)log2(_a.size()), true);
    fft(_c, (int)log2(_a.size()), true);
    F omega = getRootOfUnity(1 + (int)log2(N)).inv();
    F mul = F(1);
    for (int i = 0; i < _a.size(); i++) {
      _a[i] = mul * _a[i];
      _b[i] = mul * _b[i];
      _c[i] = mul * _c[i];
      mul = mul * omega;
    }

    fft(_a, (int)log2(_a.size()), false);
    fft(_b, (int)log2(_a.size()), false);
    fft(_c, (int)log2(_a.size()), false);

    for (int j = 0; j < v.size(); j++) {
      a += v[j] * _a[N * j / _k];
      b += v[j] * _b[N * j / _k];
      c += v[j] * _c[N * j / _k];
    }
    coef[0] = a;
    coef[1] = b;
    coef[2] = c;

    field_vector_serialize(coef, coef_u);
    pt_cp.end();
    // for(int i = 1; i < N; i++){
    // cm += 8*coef_u.size()/1024.0;
    // MPI_Send(coef_u.data(),6,MPI_UINT64_T,i,0,MPI_COMM_WORLD);
    //}
  } else {
    cm += 8 * coef_u.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(coef_u.data(), coef_u.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();

    // MPI_Send(coef_u.data(),6,MPI_UINT64_T,0,0,MPI_COMM_WORLD);
    // MPI_Recv(coef_u.data(),6,MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  myBcast(coef_u, N);
  if (rank != 0) {
    field_vector_deserialize(coef_u, coef);
    a = coef[0];
    b = coef[1];
    c = coef[2];
  }

  H = quadratic_poly(a, b, c);
  return H;
}

vector<F> zero_check_sumcheck_local(vector<F> final_v1, vector<F> final_v2,
                                    vector<F> final_v3, vector<F> h1,
                                    vector<F> h2, vector<F> &beta1,
                                    vector<F> &beta2, F b, F y, int k, int _k,
                                    int N) {

  pt_cp.start();
  vector<F> omegas;
  F omega = getRootOfUnity(1 + (int)log2(N));
  omega = omega.inv();
  F mul = F(1);
  for (int i = 0; i < N; i++) {
    omegas.push_back(mul);
    mul = mul * omega;
  }

  // for(int i = 0; i < final_v1.size(); i++){
  fft(final_v1, (int)log2(final_v1.size()), true);
  fft(final_v2, (int)log2(final_v2.size()), true);
  fft(final_v3, (int)log2(final_v3.size()), true);
  fft(h1, (int)log2(h1.size()), true);
  fft(h2, (int)log2(h2.size()), true);

  for (int j = 0; j < final_v1.size(); j++) {
    final_v1[j] = omegas[j] * final_v1[j];
    final_v2[j] = omegas[j] * final_v2[j];
    final_v3[j] = omegas[j] * final_v3[j];
    h1[j] = omegas[j] * h1[j];
    h2[j] = omegas[j] * h2[j];
  }

  fft(final_v1, (int)log2(final_v1.size()), false);
  fft(final_v2, (int)log2(final_v2.size()), false);
  fft(final_v3, (int)log2(final_v3.size()), false);
  fft(h1, (int)log2(h1.size()), false);
  fft(h2, (int)log2(h2.size()), false);
  //}

  vector<F> v1(k), v2(k), v3(k);
  vector<F> r1(k), r2(k);
  int ctr = 0;
  // for(int j = 0; j < final_v1.size(); j++){
  for (int i = 0; i < k; i++) {
    v1[ctr] = final_v1[N * i / _k];
    v2[ctr] = final_v2[N * i / _k];
    v3[ctr] = final_v3[N * i / _k];
    r1[ctr] = h1[N * i / _k];
    r2[ctr++] = h2[N * i / _k];
  }
  //}

  beta2.resize(v1.size());
  // for(int j = 0; j < final_v1.size(); j++){
  for (int i = 0; i < k; i++)
    beta2[i] = beta2[i] * beta1[0];

  //}

  int rounds = (int)log2(k);
  vector<F> challenges(rounds);
  for (int i = 0; i < rounds; i++) {
    quadratic_poly p = quadratic_poly(F_ZERO, F_ZERO, F_ZERO);
    quadratic_poly p_r = quadratic_poly(F_ZERO, F_ZERO, F_ZERO);
    linear_poly _p = linear_poly(F_ZERO, F_ZERO);
    cubic_poly poly = cubic_poly(F_ZERO, F_ZERO, F_ZERO, F_ZERO);
    linear_poly l1, l2;
    for (int j = 0; j < v1.size() / (1 << (i + 1)); j++) {
      l1 = linear_poly(v1[2 * j + 1] - v1[2 * j], v1[2 * j]);
      l2 = linear_poly(v2[2 * j + 1] - v2[2 * j], v2[2 * j]);
      p = l1 * l2;
      l1 = linear_poly(r1[2 * j + 1] - r1[2 * j], r1[2 * j]);
      l2 = linear_poly(r2[2 * j + 1] - r2[2 * j], r2[2 * j]);
      p_r = l1 * l2;
      _p = linear_poly(v3[2 * j + 1] - v3[2 * j], v3[2 * j]);
      p.a = p.a + b * p_r.a;
      p.b = p.b + b * p_r.b - _p.a;
      p.c = p.c + b * p_r.c - _p.b;
      poly =
          poly + p * linear_poly(beta2[2 * j + 1] - beta2[2 * j], beta2[2 * j]);
    }
    vt.start();
    if (poly.eval(0) + poly.eval(1) != y) {
      printf("Error in sumcheck functionality %d\n", i);
      exit(-1);
    }

    challenges[i] = F::_random();
    y = poly.eval(challenges[i]);
    vt.end();
    for (int j = 0; j < v1.size() / (1 << (i + 1)); j++) {
      v1[j] = challenges[i] * (v1[2 * j + 1] - v1[2 * j]) + v1[2 * j];
      v2[j] = challenges[i] * (v2[2 * j + 1] - v2[2 * j]) + v2[2 * j];
      v3[j] = challenges[i] * (v3[2 * j + 1] - v3[2 * j]) + v3[2 * j];
      r1[j] = challenges[i] * (r1[2 * j + 1] - r1[2 * j]) + r1[2 * j];
      r2[j] = challenges[i] * (r2[2 * j + 1] - r2[2 * j]) + r2[2 * j];
      beta2[j] =
          challenges[i] * (beta2[2 * j + 1] - beta2[2 * j]) + beta2[2 * j];
    }
  }
  vector<F> ret;
  ret.push_back(v1[0]);
  ret.push_back(v2[0]);
  ret.push_back(v3[0]);
  ret.push_back(beta2[0]);
  ret.push_back(r1[0]);
  ret.push_back(r2[0]);
  ret.insert(ret.end(), challenges.begin(), challenges.end());
  pt_cp.end();

  vt.start();
  if ((v1[0] * v2[0] - v3[0] + b * r1[0] * r2[0]) * beta2[0] != y) {
    printf("Error in final sumcheck step\n");
    exit(-1);
  }
  vt.end();

  return ret;
}

vector<F> quadratic_sumcheck_local(vector<F> final_v1, vector<F> final_v2, F y,
                                   int k, int _k, int N) {

  pt_cp.start();

  fft(final_v1, (int)log2(final_v1.size()), true);
  fft(final_v2, (int)log2(final_v2.size()), true);

  F omega = getRootOfUnity(1 + (int)log2(N));
  omega = omega.inv();
  F mul = F(1);
  for (int i = 0; i < final_v1.size(); i++) {
    final_v1[i] = mul * final_v1[i];
    final_v2[i] = mul * final_v2[i];
    mul = mul * omega;
  }

  fft(final_v1, (int)log2(final_v1.size()), false);
  fft(final_v2, (int)log2(final_v2.size()), false);

  vector<F> v1(k), v2(k);
  for (int i = 0; i < k; i++) {
    v1[i] = final_v1[N * i / _k];
    v2[i] = final_v2[N * i / _k];
  }

  int rounds = (int)log2(k);
  vector<F> challenges(rounds);
  for (int i = 0; i < rounds; i++) {
    quadratic_poly p = quadratic_poly(F_ZERO, F_ZERO, F_ZERO);
    linear_poly l1, l2;
    for (int j = 0; j < v1.size() / (1 << (i + 1)); j++) {
      l1 = linear_poly(v1[2 * j + 1] - v1[2 * j], v1[2 * j]);
      l2 = linear_poly(v2[2 * j + 1] - v2[2 * j], v2[2 * j]);
      p = p + l1 * l2;
    }
    vt.start();

    if (p.eval(0) + p.eval(1) != y) {
      printf("Error in sumcheck functionality %d\n", i);
      exit(-1);
    }

    challenges[i] = F::_random();
    y = p.eval(challenges[i]);
    vt.end();
    for (int j = 0; j < v1.size() / (1 << (i + 1)); j++) {
      v1[j] = challenges[i] * (v1[2 * j + 1] - v1[2 * j]) + v1[2 * j];
      v2[j] = challenges[i] * (v2[2 * j + 1] - v2[2 * j]) + v2[2 * j];
    }
  }
  vector<F> ret;
  ret.push_back(v1[0]);
  ret.push_back(v2[0]);
  ret.insert(ret.end(), challenges.begin(), challenges.end());
  pt_cp.end();

  return ret;
}

vector<F> batch_sumcheck_local(vector<F> final_v1, vector<F> final_v2,
                               vector<F> final_v3, vector<F> final_v4,
                               vector<F> h1, vector<F> h2, F b, F y, int k,
                               int _k, int N) {

  pt_cp.start();

  fft(final_v1, (int)log2(final_v1.size()), true);
  fft(final_v2, (int)log2(final_v2.size()), true);
  fft(final_v3, (int)log2(final_v3.size()), true);
  fft(final_v4, (int)log2(final_v4.size()), true);
  fft(h1, (int)log2(h1.size()), true);
  fft(h2, (int)log2(h2.size()), true);

  F omega = getRootOfUnity(1 + (int)log2(N));
  omega = omega.inv();
  F mul = F(1);
  for (int i = 0; i < final_v1.size(); i++) {
    final_v1[i] = mul * final_v1[i];
    final_v2[i] = mul * final_v2[i];
    final_v3[i] = mul * final_v3[i];
    final_v4[i] = mul * final_v4[i];
    h1[i] = mul * h1[i];
    h2[i] = mul * h2[i];
    mul = mul * omega;
  }

  fft(final_v1, (int)log2(final_v1.size()), false);
  fft(final_v2, (int)log2(final_v2.size()), false);
  fft(final_v3, (int)log2(final_v3.size()), false);
  fft(final_v4, (int)log2(final_v4.size()), false);
  fft(h1, (int)log2(h1.size()), false);
  fft(h2, (int)log2(h2.size()), false);
  /*
  for(int i = 0; i < N; i++){
      printf("(%lld,%lld)\n",final_v1[i].real,final_v1[i].img);
  }
  printf("=======\n");

  for(int i = 0; i < N; i++){
      printf("(%lld,%lld)\n",final_v2[i].real,final_v2[i].img);
  }
  printf("=======\n");

  for(int i = 0; i < N; i++){
      printf("(%lld,%lld)\n",final_v3[i].real,final_v3[i].img);
  }
  printf("=======\n");
  for(int i = 0; i < N; i++){
      printf("(%lld,%lld)\n",final_v4[i].real,final_v4[i].img);
  }
  printf("=======\n");
  for(int i = 0; i < N; i++){
      printf("(%lld,%lld)\n",h1[i].real,h1[i].img);
  }
  printf("=======\n");

  for(int i = 0; i < N; i++){
      printf("(%lld,%lld)\n",h2[i].real,h2[i].img);
  }
  printf("=======\n");

  printf(">> %d,%d\n",k,_k);
  */

  vector<F> v1(k), v2(k), v3(k), v4(k);
  vector<F> r1(k), r2(k);

  for (int i = 0; i < k; i++) {
    v1[i] = final_v1[N * i / _k];
    v2[i] = final_v2[N * i / _k];
    v3[i] = final_v3[N * i / _k];
    v4[i] = final_v4[N * i / _k];
    r1[i] = h1[N * i / _k];
    r2[i] = h2[N * i / _k];
  }

  int rounds = (int)log2(k);
  vector<F> challenges(rounds);
  for (int i = 0; i < rounds; i++) {
    quadratic_poly p = quadratic_poly(F_ZERO, F_ZERO, F_ZERO);
    quadratic_poly p_r = quadratic_poly(F_ZERO, F_ZERO, F_ZERO);
    linear_poly l1, l2;

    for (int j = 0; j < v1.size() / (1 << (i + 1)); j++) {

      l1 = linear_poly(v1[2 * j + 1] - v1[2 * j], v1[2 * j]);
      l2 = linear_poly(v2[2 * j + 1] - v2[2 * j], v2[2 * j]);
      p = p + l1 * l2;
      l1 = linear_poly(v3[2 * j + 1] - v3[2 * j], v3[2 * j]);
      l2 = linear_poly(v4[2 * j + 1] - v4[2 * j], v4[2 * j]);
      p = p + l1 * l2;

      l1 = linear_poly(r1[2 * j + 1] - r1[2 * j], r1[2 * j]);
      l2 = linear_poly(r2[2 * j + 1] - r2[2 * j], r2[2 * j]);
      p_r = p_r + l1 * l2;
    }
    p.a = p.a + b * p_r.a;
    p.b = p.b + b * p_r.b;
    p.c = p.c + b * p_r.c;
    vt.start();

    if (p.eval(0) + p.eval(1) != y) {
      printf("Error in sumcheck functionality %d\n", i);
      exit(-1);
    }

    challenges[i] = F::_random();
    y = p.eval(challenges[i]);
    vt.end();
    for (int j = 0; j < v1.size() / (1 << (i + 1)); j++) {
      v1[j] = challenges[i] * (v1[2 * j + 1] - v1[2 * j]) + v1[2 * j];
      v2[j] = challenges[i] * (v2[2 * j + 1] - v2[2 * j]) + v2[2 * j];
      v3[j] = challenges[i] * (v3[2 * j + 1] - v3[2 * j]) + v3[2 * j];
      v4[j] = challenges[i] * (v4[2 * j + 1] - v4[2 * j]) + v4[2 * j];
      r1[j] = challenges[i] * (r1[2 * j + 1] - r1[2 * j]) + r1[2 * j];
      r2[j] = challenges[i] * (r2[2 * j + 1] - r2[2 * j]) + r2[2 * j];
    }
  }
  vector<F> ret;
  ret.push_back(v1[0]);
  ret.push_back(v2[0]);
  ret.push_back(v3[0]);
  ret.push_back(v4[0]);
  ret.push_back(r1[0]);
  ret.push_back(r2[0]);
  ret.insert(ret.end(), challenges.begin(), challenges.end());
  pt_cp.end();
  vt.start();
  if ((v1[0] * v2[0] + v3[0] * v4[0] + b * r1[0] * r2[0]) != y) {
    printf("Error in final sumcheck step\n");
    exit(-1);
  }
  vt.end();

  return ret;
}

vector<pair<F, vector<F>>> F_zero_check_rest(vector<F> &v1, vector<F> &v2,
                                             vector<F> &v3, vector<F> &h1,
                                             vector<F> &h2, vector<F> &beta1,
                                             vector<F> &beta2, F b, F y, int k,
                                             int _k, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff_u64(10 * v1.size());
  vector<F> buff, ret;
  com_rounds += 2;

  if (rank == 0) {
    vector<vector<F>> final_v1(v1.size()), final_v2(v2.size()),
        final_v3(v3.size()), final_h1(h1.size()), final_h2(h2.size());
    for (int i = 0; i < final_v1.size(); i++) {
      final_v1[i].resize(N);
      final_v2[i].resize(N);
      final_v3[i].resize(N);
      final_h1[i].resize(N);
      final_h2[i].resize(N);
      final_v1[i][0] = v1[i];
      final_v2[i][0] = v2[i];
      final_v3[i][0] = v3[i];
      final_h1[i][0] = h1[i];
      final_h2[i][0] = h2[i];
    }
    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(10 * v1.size());
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
      // MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], buff);
      for (int j = 0; j < v1.size(); j++) {
        final_v1[j][i] = buff[j];
        final_v2[j][i] = buff[j + v1.size()];
        final_v3[j][i] = buff[j + 2 * v1.size()];
        final_h1[j][i] = buff[j + 3 * v1.size()];
        final_h2[j][i] = buff[j + 4 * v1.size()];
      }
    }
    ret = zero_check_sumcheck_local(
        convert2vector(final_v1), convert2vector(final_v2),
        convert2vector(final_v3), convert2vector(final_h1),
        convert2vector(final_h2), beta1, beta2, b, y, k, _k, N);
    // for(int i = 1; i < N; i++){
    field_vector_serialize(ret, buff_u64);
    // cm += 8*buff_u64.size()/1024.0;
    // MPI_Send(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD);
    //}
  } else {
    buff = v1;
    buff.insert(buff.end(), v2.begin(), v2.end());
    buff.insert(buff.end(), v3.begin(), v3.end());
    buff.insert(buff.end(), h1.begin(), h1.end());
    buff.insert(buff.end(), h2.begin(), h2.end());
    field_vector_serialize(buff, buff_u64);
    cm += 8 * buff_u64.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();

    // MPI_Send(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD);
    buff_u64.clear();
    buff_u64.resize(2 * (6 + (int)log2(k)));
    // MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  myBcast(buff_u64, N);
  if (rank != 0) {
    field_vector_deserialize(buff_u64, ret);
  }
  vector<F> r;
  for (int i = 6; i < ret.size(); i++)
    r.push_back(ret[i]);
  return {make_pair(ret[0], r), make_pair(ret[1], r), make_pair(ret[2], r),
          make_pair(ret[3], r), make_pair(ret[4], r), make_pair(ret[5], r)};
}

vector<pair<F, vector<F>>> F_batch_sumcheck_rest(F v1, F v2, F v3, F v4, F h1,
                                                 F h2, F b, F y, int k, int _k,
                                                 int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff_u64(12);
  vector<F> buff, ret;
  com_rounds += 2;

  if (rank == 0) {
    vector<F> final_v1(N), final_v2(N), final_v3(N), final_v4(N), final_h1(N),
        final_h2(N);
    final_v1[0] = v1;
    final_v2[0] = v2;
    final_v3[0] = v3;
    final_v4[0] = v4;
    final_h1[0] = h1;
    final_h2[0] = h2;
    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(12);
      MPI_Irecv(recv_data[i].data(), 12, MPI_UINT64_T, i, 0, MPI_COMM_WORLD,
                &req[i]);
      // MPI_Recv(buff_u64.data(),12,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], buff);
      final_v1[i] = buff[0];
      final_v2[i] = buff[1];
      final_v3[i] = buff[2];
      final_v4[i] = buff[3];
      final_h1[i] = buff[4];
      final_h2[i] = buff[5];
    }
    ret = batch_sumcheck_local(final_v1, final_v2, final_v3, final_v4, final_h1,
                               final_h2, b, y, k, _k, N);
    field_vector_serialize(ret, buff_u64);

    // for(int i = 1; i < N; i++){
    //     cm += 8*buff_u64.size()/1024.0;
    //     MPI_Send(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD);
    // }
  } else {
    buff = {v1, v2, v3, v4, h1, h2};
    field_vector_serialize(buff, buff_u64);
    cm += 8 * buff_u64.size() / 1024.0;

    MPI_Request req;
    MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();

    // MPI_Send(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD);
    buff_u64.clear();
    buff_u64.resize(2 * (6 + (int)log2(k)));
    // MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  myBcast(buff_u64, N);
  if (rank != 0) {
    field_vector_deserialize(buff_u64, ret);
  }
  vector<F> r;
  for (int i = 6; i < ret.size(); i++)
    r.push_back(ret[i]);
  return {make_pair(ret[0], r), make_pair(ret[1], r), make_pair(ret[2], r),
          make_pair(ret[3], r), make_pair(ret[4], r), make_pair(ret[5], r)};
}

vector<pair<F, vector<F>>> F_quadratic_sumcheck_rest(F v1, F v2, F y, int k,
                                                     int _k, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff_u64(4);
  vector<F> buff, ret;
  com_rounds += 2;

  if (rank == 0) {
    vector<F> final_v1(N), final_v2(N);
    final_v1[0] = v1;
    final_v2[0] = v2;
    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(4);
      MPI_Irecv(recv_data[i].data(), 4, MPI_UINT64_T, i, 0, MPI_COMM_WORLD,
                &req[i]);
      // MPI_Recv(buff_u64.data(),4,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], buff);
      final_v1[i] = buff[0];
      final_v2[i] = buff[1];
    }
    ret = quadratic_sumcheck_local(final_v1, final_v2, y, k, _k, N);
    // for(int i = 1; i < N; i++){
    field_vector_serialize(ret, buff_u64);
    // cm += 8*buff_u64.size()/1024.0;
    // MPI_Send(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD);
    //}
  } else {
    buff = {v1, v2};
    field_vector_serialize(buff, buff_u64);
    cm += 8 * buff_u64.size() / 1024.0;
    // MPI_Send(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD);
    MPI_Request req;
    MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();

    buff_u64.clear();
    buff_u64.resize(2 * (2 + (int)log2(k)));
    // MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    // field_vector_deserialize(buff_u64,ret);
  }
  myBcast(buff_u64, N);
  if (rank != 0) {
    field_vector_deserialize(buff_u64, ret);
  }
  vector<F> r;
  for (int i = 2; i < ret.size(); i++)
    r.push_back(ret[i]);
  return {make_pair(ret[0], r), make_pair(ret[1], r)};
}

cubic_poly aggregate_cubic_poly(cubic_poly H, vector<F> &v, int k, int _k,
                                int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  F a = F(0), b = F(0), c = F(0), d = F(0);
  pt_cp.start();

  vector<F> coef;
  coef.push_back(H.a);
  coef.push_back(H.b);
  coef.push_back(H.c), coef.push_back(H.d);
  vector<u64> coef_u;

  field_vector_serialize(coef, coef_u);
  pt_cp.end();
  com_rounds += 2;

  if (rank == 0) {
    vector<F> _a(N), _b(N), _c(N), _d(N);
    _a[0] = coef[0];
    _b[0] = coef[1];
    _c[0] = coef[2], _d[0] = coef[3];
    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(8);
      MPI_Irecv(recv_data[i].data(), 8, MPI_UINT64_T, i, 0, MPI_COMM_WORLD,
                &req[i]);
      // MPI_Recv(coef_u.data(),8,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], coef);
      _a[i] = coef[0];
      _b[i] = coef[1];
      _c[i] = coef[2];
      _d[i] = coef[3];
    }
    pt_cp.start();

    vector<F> test_a;

    fft(_a, (int)log2(_a.size()), true);
    fft(_b, (int)log2(_a.size()), true);
    fft(_c, (int)log2(_a.size()), true);
    fft(_d, (int)log2(_a.size()), true);
    F omega = getRootOfUnity(1 + (int)log2(N));
    omega = omega.inv();
    F mul = F(1);
    for (int i = 0; i < _a.size(); i++) {
      _a[i] = mul * _a[i];
      _b[i] = mul * _b[i];
      _c[i] = mul * _c[i];
      _d[i] = mul * _d[i];
      mul = omega * mul;
    }
    //_a.resize(_k);
    fft(_a, (int)log2(_a.size()), false);
    //_b.resize(_k);
    fft(_b, (int)log2(_a.size()), false);
    //_c.resize(_k);
    fft(_c, (int)log2(_a.size()), false);
    //_d.resize(_k);
    fft(_d, (int)log2(_a.size()), false);

    /*
    for(int i = 0; i < _a.size(); i++){
        if(_a[i] + _b[i]+_c[i]+F(2)*_d[i] == 0){
            printf("OK %d\n",i);
        }
    }
    */

    for (int j = 0; j < v.size(); j++) {

      a += v[j] * _a[N * j / _k];
      b += v[j] * _b[N * j / _k];
      c += v[j] * _c[N * j / _k];
      d += v[j] * _d[N * j / _k];
    }
    coef[0] = a;
    coef[1] = b;
    coef[2] = c;
    coef[3] = d;
    field_vector_serialize(coef, coef_u);
    pt_cp.end();

    // for(int i = 1; i < N; i++){
    // cm += 8*coef_u.size()/1024.0;
    // MPI_Send(coef_u.data(),8,MPI_UINT64_T,i,0,MPI_COMM_WORLD);
    //}
  } else {
    cm += 8 * coef_u.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(coef_u.data(), coef_u.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();

    // MPI_Send(coef_u.data(),8,MPI_UINT64_T,0,0,MPI_COMM_WORLD);
    // MPI_Recv(coef_u.data(),8,MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    // field_vector_deserialize(coef_u,coef);
    // a = coef[0];b = coef[1];c = coef[2];d = coef[3];
  }
  myBcast(coef_u, N);
  if (rank != 0) {
    field_vector_deserialize(coef_u, coef);
    a = coef[0];
    b = coef[1];
    c = coef[2];
    d = coef[3];
  }
  return cubic_poly(a, b, c, d);
}

F F_sum(F K, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff;
  if (rank == 0) {
    vector<F> Y(N);
    Y[0] = K;
    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);
    for (int i = 1; i < N; i++) {
      // MPI_Recv(buff.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      recv_data[i].resize(2);
      MPI_Irecv(recv_data[i].data(), 2, MPI_UINT64_T, i, 0, MPI_COMM_WORLD,
                &req[i]);
    }

    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      Y[i].real = recv_data[i][0];
      Y[i].img = recv_data[i][1];
    }
    pt_cp.start();

    Y = unpack_secret_share(Y, N, N);
    K = F(0);
    for (int i = 0; i < N / 2; i++) {
      K += Y[2 * i];
    }
    buff = {K.real, K.img};
    pt_cp.end();
  } else {
    buff = {K.real, K.img};
    cm += 8 * buff.size() / 1024.0;
    // MPI_Send(buff.data(),2,MPI_UINT64_T,0,0,MPI_COMM_WORLD);
    MPI_Request req;
    MPI_Isend(buff.data(), buff.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
  if (rank == 0)
    cm += (N - 1) * 8 * buff.size() / 1024.0;
  myBcast(buff, N);
  // MPI_Bcast(buff.data(),2,MPI_UINT64_T,0,MPI_COMM_WORLD);
  K.real = buff[0];
  K.img = buff[1];
  return K;
}

F distributed_eval(vector<F> &poly, vector<F> &beta1, vector<F> &beta2, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff(2);
  pt_cp.start();

  F y = F(0);
  for (int i = 0; i < poly.size(); i++) {
    y += poly[i] * beta1[i];
  }

  pt_cp.end();
  com_rounds += 2;

  if (rank == 0) {
    vector<F> Y(N);
    Y[0] = y;
    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);
    for (int i = 1; i < N; i++) {
      // MPI_Recv(buff.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      recv_data[i].resize(2);
      MPI_Irecv(recv_data[i].data(), 2, MPI_UINT64_T, i, 0, MPI_COMM_WORLD,
                &req[i]);
    }

    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      Y[i].real = recv_data[i][0];
      Y[i].img = recv_data[i][1];
    }
    pt_cp.start();

    y = 0;
    for (int i = 0; i < beta2.size(); i++) {
      y += Y[i] * beta2[i];
    }
    buff = {y.real, y.img};
    pt_cp.end();
  } else {
    buff = {y.real, y.img};
    cm += 8 * buff.size() / 1024.0;
    // MPI_Send(buff.data(),2,MPI_UINT64_T,0,0,MPI_COMM_WORLD);
    MPI_Request req;
    MPI_Isend(buff.data(), buff.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
  if (rank == 0)
    cm += (N - 1) * 8 * buff.size() / 1024.0;
  myBcast(buff, N);
  // MPI_Bcast(buff.data(),2,MPI_UINT64_T,0,MPI_COMM_WORLD);
  y.real = buff[0];
  y.img = buff[1];
  return y;
}

vector<F> send_data(vector<F> &data, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  com_rounds += 1;
  if (rank == 0) {
    vector<vector<F>> M;
    M.push_back(data);
    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(2 * data.size());
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
    }
    vector<F> buff;
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], buff);
      M.push_back(buff);
      // data.insert(data.end(),buff.begin(),buff.end());
    }

    M = transpose(M);
    for (int i = 0; i < M.size(); i++) {
      vector<F> buff = unpack_secret_share(M[i], N, N);
      M[i].resize(N / 2);
      for (int j = 0; j < N / 2; j++) {
        M[i][j] = buff[2 * j];
      }
      fft(M[i], (int)log2(M[i].size()), true);
    }
    // M = transpose(M);
    data = convert2vector(M);
  } else {
    vector<u64> send_data;
    field_vector_serialize(data, send_data);
    cm += 8 * send_data.size() / 1024.0;
    // MPI_Send(buff.data(),2,MPI_UINT64_T,0,0,MPI_COMM_WORLD);
    MPI_Request req;
    MPI_Isend(send_data.data(), send_data.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
  return data;
}

vector<pair<F, vector<F>>> compute_evaluation_claims(vector<vector<F>> &v,
                                                     vector<F> r,
                                                     vector<int> degrees,
                                                     int N) {
  vector<vector<F>> L(r.size());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff(6);

  for (int i = 0; i < degrees.size(); i++) {
    if (degrees[i] != 1)
      L[i] = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees[i])),
                                    r[i + 1], degrees[i]);
    else {
      L[i].push_back(F(1) - r[i + 1]);
      L[i].push_back(r[i + 1]);
    }
  }
  L[degrees.size()] =
      compute_lagrange_coeff(getRootOfUnity((int)log2(N / 4)), r[0], N / 4);

  vector<F> evals;

  for (int l = 0; l < v.size(); l++) {
    for (int i = 0; i < L.size() - 1; i++) {
      for (int j = 0; j < v[l].size() / L[i].size(); j++) {
        v[l][j] = L[i][0] * v[l][L[i].size() * j];
        for (int k = 1; k < L[i].size(); k++) {
          v[l][j] += L[i][k] * v[l][L[i].size() * j + k];
        }
      }
      v[l].resize(v[l].size() / L[i].size());
    }
    evals.push_back(v[l][0]);
  }
  if (rank == 0) {
    vector<vector<F>> Y(N);
    Y[0] = evals;

    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);
    for (int i = 1; i < N; i++) {
      // MPI_Recv(buff.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      recv_data[i].resize(evals.size() * 2);
      MPI_Irecv(recv_data[i].data(), evals.size() * 2, MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      Y[i].resize(evals.size());
      for (int j = 0; j < evals.size(); j++) {
        Y[i][j].real = recv_data[i][2 * j];
        Y[i][j].img = recv_data[i][2 * j + 1];
      }
    }

    pt_cp.start();
    Y = transpose(Y);
    for (int i = 0; i < evals.size(); i++) {
      Y[i] = unpack_secret_share(Y[i], N, N);
      evals[i] = 0;
      for (int j = 0; j < L[L.size() - 1].size(); j++) {
        evals[i] += Y[i][2 * j] * L[L.size() - 1][j];
      }
    }
    field_vector_serialize(evals, buff);
    pt_cp.end();
  } else {
    // evals = {vL[0],vR[0],vO[0]};
    field_vector_serialize(evals, buff);
    cm += 8 * buff.size() / 1024.0;
    // MPI_Send(buff.data(),2,MPI_UINT64_T,0,0,MPI_COMM_WORLD);
    MPI_Request req;
    MPI_Isend(buff.data(), buff.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
  myBcast(buff, N);
  // MPI_Bcast(buff.data(),2,MPI_UINT64_T,0,MPI_COMM_WORLD);
  if (rank != 0) {
    field_vector_deserialize(buff, evals);
  }
  vector<pair<F, vector<F>>> claims;
  for (int i = 0; i < evals.size(); i++) {
    claims.push_back(make_pair(evals[i], r));
  }

  return claims;
}

void get_even_odd_data(vector<F> &data, vector<F> &data_even_odd, int N) {

  vector<vector<F>> arr(N);
  for (int i = 0; i < arr.size(); i++) {
    arr[i].resize(data.size() / N);
    for (int j = 0; j < arr[i].size(); j++) {
      arr[i][j] = data[i + j * N];
    }
  }
  vector<F> temp = convert2vector(arr);
  vector<u64> send_buff(2 * temp.size()), recv_buff(2 * temp.size());
  field_vector_serialize(temp, send_buff);
  com_rounds++;
  cm += send_buff.size() * 8.0 / 1024.0;
  MPI_Request req;
  MPI_Ialltoall(send_buff.data(), send_buff.size() / N, MPI_UINT64_T,
                recv_buff.data(), recv_buff.size() / N, MPI_UINT64_T,
                MPI_COMM_WORLD, &req);
  MPI_Wait(&req, MPI_STATUS_IGNORE);
  field_vector_deserialize(recv_buff, data_even_odd);
}

vector<F> batch_ip(vector<vector<F>> &arr, vector<vector<vector<F>>> &v, int N,
                   int k, int _k) {

  pt_cp.start();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<F> Y(arr.size(), F(0));
  for (int i = 0; i < Y.size(); i++) {
    for (int j = 0; j < arr[i].size(); j++) {
      Y[i] += arr[i][j] * v[i][0][j];
    }
  }
  vector<u64> Y_int;
  pt_cp.end();
  com_rounds += 2;

  if (rank == 0) {
    vector<vector<F>> partial_Y(arr.size());
    for (int i = 0; i < partial_Y.size(); i++) {
      partial_Y[i].resize(N);
      partial_Y[i][0] = Y[i];
    }
    Y_int.resize(Y.size() * 2);
    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(Y.size() * 2);
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);

      // MPI_Recv(recv_data[i].data(),recv_data[i].size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,&req[i]);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], Y);
      for (int j = 0; j < Y.size(); j++)
        partial_Y[j][i] = Y[j];
    }

    pt_cp.start();

    Y.clear();
    Y.resize(arr.size(), F(0));
    for (int i = 0; i < partial_Y.size(); i++) {
      fft(partial_Y[i], (int)log2(partial_Y[i].size()), true);
      F omega = getRootOfUnity(1 + (int)log2(N)).inv();
      F mul = F(1);
      for (int j = 0; j < partial_Y[i].size(); j++) {
        partial_Y[i][j] = mul * partial_Y[i][j];
        mul = mul * omega;
      }
      fft(partial_Y[i], (int)log2(partial_Y[i].size()), false);
      for (int j = 0; j < k; j++) {
        Y[i] += v[i][1][j] * partial_Y[i][N * j / _k];
      }
    }
    field_vector_serialize(Y, Y_int);
    pt_cp.end();
  } else {
    field_vector_serialize(Y, Y_int);
    cm += 8 * Y_int.size() / 1024.0;
    // MPI_Send(Y_int.data(),Y_int.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD);
    MPI_Request req;
    MPI_Isend(Y_int.data(), Y_int.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
  if (rank == 0)
    cm += (N - 1) * 8 * Y_int.size() / 1024.0;
  myBcast(Y_int, N);
  // MPI_Bcast(Y_int.data(),Y_int.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
  if (rank != 0) {
    field_vector_deserialize(Y_int, Y);
  }
  return Y;
}

vector<F> batch_distributed_eval_opt(vector<vector<F>> &poly, vector<F> r1,
                                     vector<F> r2, int N) {
  int rank;
  pt_cp.start();
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff(poly.size() * 2);

  int max_size = 0;
  for (int i = 0; i < poly.size(); i++) {
    if (poly[i].size() > max_size)
      max_size = poly[i].size();
  }
  vector<F> r, y(poly.size(), F(0));
  for (int i = 0; i < (int)log2(max_size); i++)
    r.push_back(r1[i]);

  vector<F> beta1;
  precompute_beta(r, beta1);

  for (int i = 0; i < poly.size(); i++) {
    for (int j = 0; j < poly[i].size(); j++)
      y[i] += poly[i][j] * beta1[j];

    for (int j = (int)log2(poly[i].size()); j < r.size(); j++)
      y[i] *= (F(1) - r[j]).inv();
  }
  pt_cp.end();
  com_rounds += 2;

  if (rank == 0) {

    pt_cp.start();
    vector<F> beta2;
    precompute_beta(r2, beta2);
    vector<F> Y(poly.size(), F(0));
    for (int i = 0; i < poly.size(); i++) {
      Y[i] += beta2[0] * y[i];
    }
    pt_cp.end();
    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);

    for (int i = 1; i < N; i++) {
      recv_data[i].resize(2 * poly.size());
      MPI_Irecv(recv_data[i].data(), 2 * poly.size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
      // MPI_Recv(buff.data(),2*poly.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      for (int j = 0; j < poly.size(); j++) {
        F v;
        v.real = recv_data[i][2 * j];
        v.img = recv_data[i][2 * j + 1];
        Y[j] += beta2[i] * v;
      }
    }

    field_vector_serialize(Y, buff);
  } else {
    field_vector_serialize(y, buff);
    cm += 8 * buff.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(buff.data(), buff.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
  if (rank == 0)
    cm += (N - 1) * 8 * buff.size() / 1024.0;
  myBcast(buff, N);
  // MPI_Bcast(buff.data(),2*poly.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
  field_vector_deserialize(buff, y);
  return y;
}

vector<F> batch_distributed_eval(vector<vector<F>> &poly, vector<F> &beta1,
                                 vector<F> &beta2, int N) {
  int rank;
  pt_cp.start();
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff(poly.size() * 2);

  vector<F> y(poly.size(), F(0));
  for (int i = 0; i < poly.size(); i++) {
    for (int j = 0; j < poly[i].size(); j++) {
      y[i] += poly[i][j] * beta1[j];
    }
  }
  pt_cp.end();
  com_rounds += 2;

  if (rank == 0) {

    pt_cp.start();
    vector<F> Y(poly.size(), F(0));
    for (int i = 0; i < poly.size(); i++) {
      Y[i] += beta2[0] * y[i];
    }
    pt_cp.end();

    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(2 * poly.size());
      MPI_Irecv(recv_data[i].data(), 2 * poly.size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
      // MPI_Recv(recv_data[i].data(),2*poly.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,&req[i]);
    }

    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      for (int j = 0; j < poly.size(); j++) {
        F v;
        v.real = buff[2 * j];
        v.img = buff[2 * j + 1];
        Y[j] += beta2[i] * v;
      }
    }

    field_vector_serialize(Y, buff);
  } else {
    field_vector_serialize(y, buff);
    cm += 8 * buff.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(buff.data(), buff.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
  if (rank == 0)
    cm += (N - 1) * 8 * buff.size() / 1024.0;
  myBcast(buff, N);
  // MPI_Bcast(buff.data(),2*poly.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
  field_vector_deserialize(buff, y);
  return y;
}

void send_index(vector<vector<int>> &Data, vector<int> dim, int N) {
  vector<int> buff;
  vector<int> dim_buff = dim;
  dim_buff.insert(dim_buff.end(), real_idx_dim.begin(), real_idx_dim.end());
  vector<MPI_Request> req1(N - 1), req2(N - 1);
  for (int i = 1; i < N; i++) {
    MPI_Isend(dim_buff.data(), dim_buff.size(), MPI_INT, i, 0, MPI_COMM_WORLD,
              &req1[i - 1]);
    MPI_Isend(Data[i].data(), Data[i].size(), MPI_INT, i, 1, MPI_COMM_WORLD,
              &req2[i - 1]);
  }
  for (int i = 1; i < N; i++) {
    network_timer.start();
    MPI_Wait(&req1[i - 1], MPI_STATUS_IGNORE);
    network_timer.end();
    network_timer.start();
    MPI_Wait(&req2[i - 1], MPI_STATUS_IGNORE);
    network_timer.end();
  }
}

void send_matrix_dims(vector<int> dim1, vector<int> dim2, vector<int> dim3,
                      int idx, vector<MPI_Request> &req) {
  vector<int> buff = {(int)dim1.size(), (int)dim2.size(), (int)dim3.size()};
  MPI_Isend(buff.data(), 3, MPI_INT, idx, 0, MPI_COMM_WORLD, &req[0]);
  vector<int> buff2 = dim1;
  buff2.insert(buff2.end(), dim2.begin(), dim2.end());
  buff2.insert(buff2.end(), dim3.begin(), dim3.end());
  MPI_Isend(buff2.data(), buff2.size(), MPI_INT, idx, 0, MPI_COMM_WORLD,
            &req[1]);
}

void receive_matrix_dims(vector<int> &dim1, vector<int> &dim2,
                         vector<int> &dim3) {
  MPI_Request req1, req2;
  vector<int> buff(3);
  MPI_Irecv(buff.data(), 3, MPI_INT, 0, 0, MPI_COMM_WORLD, &req1);
  network_timer.start();
  MPI_Wait(&req1, MPI_STATUS_IGNORE);
  network_timer.end();
  vector<int> sizes = buff;
  buff.resize(sizes[0] + sizes[1] + sizes[2]);
  MPI_Irecv(buff.data(), buff.size(), MPI_INT, 0, 1, MPI_COMM_WORLD, &req2);
  network_timer.start();
  MPI_Wait(&req2, MPI_STATUS_IGNORE);
  network_timer.end();
  dim1.resize(sizes[0]);
  dim2.resize(sizes[1]);
  dim3.resize(sizes[2]);
  int ctr = 0;
  for (int i = 0; i < sizes[0]; i++)
    dim1[i] = buff[ctr++];
  for (int i = 0; i < sizes[1]; i++)
    dim2[i] = buff[ctr++];
  for (int i = 0; i < sizes[2]; i++)
    dim3[i] = buff[ctr++];
}

void send_R1CS_matrixes(vector<vector<int>> &Data, vector<vector<int>> tA_dim,
                        vector<vector<int>> tB_dim, vector<vector<int>> tC_dim,
                        int N) {
  vector<vector<MPI_Request>> req(N - 1);
  vector<vector<int>> buff(N), buff2(N);
  for (int i = 0; i < N - 1; i++)
    req[i].resize(3);
  for (int i = 1; i < N; i++) {
    // send_matrix_dims(tA_dim[i],tB_dim[i],tC_dim[i],i,req[i-1]);

    buff[i] = {(int)tA_dim[i].size(), (int)tB_dim[i].size(),
               (int)tC_dim[i].size()};
    MPI_Isend(buff[i].data(), 3, MPI_INT, i, 0, MPI_COMM_WORLD, &req[i - 1][0]);
    buff2[i] = tA_dim[i];
    buff2[i].insert(buff2[i].end(), tB_dim[i].begin(), tB_dim[i].end());
    buff2[i].insert(buff2[i].end(), tC_dim[i].begin(), tC_dim[i].end());
    MPI_Isend(buff2[i].data(), buff2[i].size(), MPI_INT, i, 1, MPI_COMM_WORLD,
              &req[i - 1][1]);
    MPI_Isend(Data[i].data(), Data[i].size(), MPI_INT, i, 2, MPI_COMM_WORLD,
              &req[i - 1][2]);
  }
  for (int i = 0; i < req.size(); i++) {
    network_timer.start();
    MPI_Wait(&req[i][0], MPI_STATUS_IGNORE);
    network_timer.end();
    network_timer.start();
    MPI_Wait(&req[i][1], MPI_STATUS_IGNORE);
    network_timer.end();
    network_timer.start();
    MPI_Wait(&req[i][2], MPI_STATUS_IGNORE);
    network_timer.end();
  }
}

void parse_R1CS_matrixes(vector<int> &data, vector<int> &dimsA,
                         vector<int> &dimsB, vector<int> &dimsC, int N, int M) {
  int ctr = 0;
  // pA.resize(2*M/N); pB.resize(2*M/N); pC.resize(2*M/N);
  if (pA.size() != 0) {
    return;
  }
  for (int i = 0; i < dimsA.size(); i++) {

    if (data[2 * ctr] == -1) {
      pA.push_back({});
      ctr++;
      continue;
    } else {
      vector<pair<int, int>> buff;
      for (int j = 0; j < dimsA[i]; j++) {
        buff.push_back(make_pair(data[2 * ctr], data[2 * ctr + 1]));
        ctr++;
      }
      pA.push_back(buff);
    }
    // pA[i].resize(1);
    // pA[i][0].first = data[2*i];
    // pA[i][0].second = data[2*i+1];
  }
  for (int i = 0; i < dimsB.size(); i++) {
    if (data[2 * ctr] == -1) {
      pB.push_back({});
      ctr++;
      continue;
    } else {
      vector<pair<int, int>> buff;
      for (int j = 0; j < dimsB[i]; j++) {
        buff.push_back(make_pair(data[2 * ctr], data[2 * ctr + 1]));
        ctr++;
      }
      pB.push_back(buff);
    }

    /*
    if(data[2*ctr] == -1){
        continue;
    }
    pB[i].resize(1);
    pB[i][0].first = data[2*i + 4*M/N];
    pB[i][0].second = data[2*i + 1 + 4*M/N];
    */
  }
  for (int i = 0; i < dimsC.size(); i++) {
    if (data[2 * ctr] == -1) {
      pC.push_back({});
      ctr++;
      continue;
    } else {
      vector<pair<int, int>> buff;
      for (int j = 0; j < dimsC[i]; j++) {
        buff.push_back(make_pair(data[2 * ctr], data[2 * ctr + 1]));
        ctr++;
      }
      pC.push_back(buff);
    }
    /*
    if(data[2*i + 8*M/N] == -1){
        continue;
    }
    pC[i].resize(1);
    pC[i][0].first = data[2*i + 8*M/N];
    pC[i][0].second = data[2*i + 1 + 8*M/N];*/
  }
}

void receive_R1CS_matrixes(int N, int M) {
  vector<int> dim1, dim2, dim3;

  receive_matrix_dims(dim1, dim2, dim3);
  int total_size = 0;
  for (int i = 0; i < dim1.size(); i++)
    total_size += dim1[i];
  for (int i = 0; i < dim2.size(); i++)
    total_size += dim2[i];
  for (int i = 0; i < dim3.size(); i++)
    total_size += dim3[i];
  vector<int> data(2 * total_size);
  MPI_Request req;
  MPI_Irecv(data.data(), data.size(), MPI_INT, 0, 2, MPI_COMM_WORLD, &req);
  network_timer.start();
  MPI_Wait(&req, MPI_STATUS_IGNORE);
  network_timer.end();

  parse_R1CS_matrixes(data, dim1, dim2, dim3, N, M);
}

void parse_index(vector<sparse_eval_data> &index, vector<int> &dim,
                 vector<int> &data) {
  int ctr = 0;
  for (int i = 0; i < index.size(); i++) {
    index[i].FINAL_FR1.resize(dim[4 * i]);
    index[i].FINAL_FR2.resize(dim[4 * i + 1]);
    index[i].RD1.resize(dim[4 * i + 2]);
    index[i].WR1.resize(dim[4 * i + 2]);
    index[i].IDX1.resize(dim[4 * i + 2]);
    index[i].RD2.resize(dim[4 * i + 3]);
    index[i].WR2.resize(dim[4 * i + 3]);
    index[i].IDX2.resize(dim[4 * i + 3]);
    for (int j = 0; j < index[i].FINAL_FR1.size(); j++) {
      index[i].FINAL_FR1[j] = data[ctr++];
    }
    for (int j = 0; j < index[i].FINAL_FR2.size(); j++) {
      index[i].FINAL_FR2[j] = data[ctr++];
    }
    for (int j = 0; j < index[i].RD1.size(); j++) {
      index[i].RD1[j] = data[ctr++];
      index[i].RD2[j] = data[ctr++];
      index[i].WR1[j] = data[ctr++];
      index[i].WR2[j] = data[ctr++];
      index[i].IDX1[j] = data[ctr++];
      index[i].IDX2[j] = data[ctr++];
    }
  }
}

void receive_index(vector<sparse_eval_data> &index, vector<int> &dim) {

  vector<int> buff(15);
  dim.clear();
  dim.resize(12);
  MPI_Request req1, req2;

  MPI_Irecv(buff.data(), buff.size(), MPI_INT, 0, 0, MPI_COMM_WORLD, &req1);
  network_timer.start();
  MPI_Wait(&req1, MPI_STATUS_IGNORE);
  network_timer.end();
  for (int i = 0; i < 12; i++) {
    dim[i] = buff[i];
  }
  if (!real_idx_dim.size()) {
    for (int i = 12; i < 15; i++) {
      real_idx_dim.push_back(buff[i]);
    }
  }
  int size = 0;
  for (int i = 0; i < dim.size() / 4; i++) {
    size += dim[4 * i];
    size += dim[4 * i + 1];
    size += 3 * dim[4 * i + 2];
    size += 3 * dim[4 * i + 3];
  }
  vector<int> data(size);

  MPI_Irecv(data.data(), data.size(), MPI_INT, 0, 1, MPI_COMM_WORLD, &req2);
  network_timer.start();
  MPI_Wait(&req2, MPI_STATUS_IGNORE);
  network_timer.end();
  parse_index(index, dim, data);
}

void distribute_index_even_odd(int N, int M, vector<sparse_eval_data> &index,
                               int type) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  index.resize(3);
  vector<int> dims;
  vector<int> buff(2);

  if (rank == 0) {
    // generate_R1CS_matrixes(M,type);
    buff[0] = logm;
    buff[1] = logn;
  }
  real_idx_dim.clear();

  MPI_Bcast(buff.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    vector<sparse_eval_data> data;
    vector<vector<int>> parsed_data(N);
    prepare_R1CS_data(A, B, C, logm, logn, data);
    for (int i = 0; i < data.size(); i++) {
      dims.push_back(data[i].FINAL_FR1.size() / N);
      dims.push_back(data[i].FINAL_FR2.size() / N);
      dims.push_back(data[i].RD1.size() / N);
      dims.push_back(data[i].RD2.size() / N);
    }

    for (int k = 0; k < N; k++) {
      for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].FINAL_FR1.size() / N; j++) {
          parsed_data[k].push_back(data[i].FINAL_FR1[k + j * N]);
        }
        for (int j = 0; j < data[i].FINAL_FR2.size() / N; j++) {
          parsed_data[k].push_back(data[i].FINAL_FR2[k + j * N]);
        }
        for (int j = 0; j < data[i].RD1.size() / N; j++) {
          parsed_data[k].push_back(data[i].RD1[k + j * N]);
          parsed_data[k].push_back(data[i].RD2[k + j * N]);
          parsed_data[k].push_back(data[i].WR1[k + j * N]);
          parsed_data[k].push_back(data[i].WR2[k + j * N]);
          parsed_data[k].push_back(data[i].IDX1[k + j * N]);
          parsed_data[k].push_back(data[i].IDX2[k + j * N]);
        }
      }
    }
    // for(int i = 0; i < parsed_data[0].size(); i++){
    //     printf("%d\n",parsed_data[0][i]);
    // }

    send_index(parsed_data, dims, N);

    parse_index(index, dims, parsed_data[0]);
    vector<vector<int>> parsed_R1CS_matrixes(N);
    vector<vector<int>> tA_dim(N), tB_dim(N), tC_dim(N);
    for (int i = 0; i < N; i++) {
      // parsed_R1CS_matrixes[i].resize(12*M/N,0);
      for (int j = 0; j < tA.size() / N; j++) {
        if (tA[i * tA.size() / N + j].size()) {
          for (int k = 0; k < tA[i * tA.size() / N + j].size(); k++) {
            parsed_R1CS_matrixes[i].push_back(
                tA[i * tA.size() / N + j][k].first);
            parsed_R1CS_matrixes[i].push_back(
                tA[i * tA.size() / N + j][k].second);
          }
          tA_dim[i].push_back(tA[i * tA.size() / N + j].size());
        } else {
          parsed_R1CS_matrixes[i].push_back(-1);
          parsed_R1CS_matrixes[i].push_back(-1);
          tA_dim[i].push_back(1);
        }
      }
      for (int j = 0; j < tB.size() / N; j++) {
        if (tB[i * tB.size() / N + j].size()) {
          for (int k = 0; k < tB[i * tB.size() / N + j].size(); k++) {
            parsed_R1CS_matrixes[i].push_back(
                tB[i * tB.size() / N + j][k].first);
            parsed_R1CS_matrixes[i].push_back(
                tB[i * tB.size() / N + j][k].second);
          }
          tB_dim[i].push_back(tB[i * tB.size() / N + j].size());

          // parsed_R1CS_matrixes[i][2*j + 4*M/N] = tB[i*2*M/N + j][0].first;
          // parsed_R1CS_matrixes[i][2*j+1 + 4*M/N] = tB[i*2*M/N + j][0].second;
        } else {
          parsed_R1CS_matrixes[i].push_back(-1);
          parsed_R1CS_matrixes[i].push_back(-1);
          tB_dim[i].push_back(1);
        }
      }
      for (int j = 0; j < tC.size() / N; j++) {
        if (tC[i * tC.size() / N + j].size()) {
          for (int k = 0; k < tC[i * tC.size() / N + j].size(); k++) {
            parsed_R1CS_matrixes[i].push_back(
                tC[i * tC.size() / N + j][k].first);
            parsed_R1CS_matrixes[i].push_back(
                tC[i * tC.size() / N + j][k].second);
          }
          tC_dim[i].push_back(tC[i * tC.size() / N + j].size());

          // parsed_R1CS_matrixes[i][2*j + 8*M/N] = tC[i*2*M/N + j][0].first;
          // parsed_R1CS_matrixes[i][2*j+1 + 8*M/N] = tC[i*2*M/N + j][0].second;
        } else {
          parsed_R1CS_matrixes[i].push_back(-1);
          parsed_R1CS_matrixes[i].push_back(-1);
          tC_dim[i].push_back(1);
        }
      }
    }

    send_R1CS_matrixes(parsed_R1CS_matrixes, tA_dim, tB_dim, tC_dim, N);
    parse_R1CS_matrixes(parsed_R1CS_matrixes[0], tA_dim[0], tB_dim[0],
                        tC_dim[0], N, M);
  } else {

    logm = buff[0];
    logn = buff[1];
    receive_index(index, dims);
    receive_R1CS_matrixes(N, M);
  }
}

void distribute_index(int N, int M, vector<sparse_eval_data> &index, int type) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  index.resize(3);
  vector<int> dims;
  vector<int> buff(2);

  if (rank == 0) {
    generate_R1CS_matrixes(M, type);
    buff[0] = logm;
    buff[1] = logn;
  }

  MPI_Bcast(buff.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    vector<sparse_eval_data> data;
    vector<vector<int>> parsed_data(N);
    prepare_R1CS_data(A, B, C, logm, logn, data);
    for (int i = 0; i < data.size(); i++) {
      dims.push_back(data[i].FINAL_FR1.size() / N);
      dims.push_back(data[i].FINAL_FR2.size() / N);
      dims.push_back(data[i].RD1.size() / N);
      dims.push_back(data[i].RD2.size() / N);
    }

    for (int k = 0; k < N; k++) {
      for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].FINAL_FR1.size() / N; j++) {
          parsed_data[k].push_back(
              data[i].FINAL_FR1[k * data[i].FINAL_FR1.size() / N + j]);
        }
        for (int j = 0; j < data[i].FINAL_FR2.size() / N; j++) {
          parsed_data[k].push_back(
              data[i].FINAL_FR2[k * data[i].FINAL_FR2.size() / N + j]);
        }
        for (int j = 0; j < data[i].RD1.size() / N; j++) {
          parsed_data[k].push_back(data[i].RD1[k * data[i].RD1.size() / N + j]);
          parsed_data[k].push_back(data[i].RD2[k * data[i].RD1.size() / N + j]);
          parsed_data[k].push_back(data[i].WR1[k * data[i].RD1.size() / N + j]);
          parsed_data[k].push_back(data[i].WR2[k * data[i].RD1.size() / N + j]);
          parsed_data[k].push_back(
              data[i].IDX1[k * data[i].RD1.size() / N + j]);
          parsed_data[k].push_back(
              data[i].IDX2[k * data[i].RD1.size() / N + j]);
        }
      }
    }

    // for(int i = 0; i < parsed_data[0].size(); i++){
    //     printf("%d\n",parsed_data[0][i]);
    // }
    send_index(parsed_data, dims, N);
    parse_index(index, dims, parsed_data[0]);
    vector<vector<int>> parsed_R1CS_matrixes(N);
    vector<vector<int>> tA_dim(N), tB_dim(N), tC_dim(N);
    for (int i = 0; i < N; i++) {
      // parsed_R1CS_matrixes[i].resize(12*M/N,0);
      for (int j = 0; j < tA.size() / N; j++) {
        if (tA[i * tA.size() / N + j].size()) {
          for (int k = 0; k < tA[i * tA.size() / N + j].size(); k++) {
            parsed_R1CS_matrixes[i].push_back(
                tA[i * tA.size() / N + j][k].first);
            parsed_R1CS_matrixes[i].push_back(
                tA[i * tA.size() / N + j][k].second);
          }
          tA_dim[i].push_back(tA[i * tA.size() / N + j].size());
        } else {
          parsed_R1CS_matrixes[i].push_back(-1);
          parsed_R1CS_matrixes[i].push_back(-1);
          tA_dim[i].push_back(1);
        }
      }
      for (int j = 0; j < tB.size() / N; j++) {
        if (tB[i * tB.size() / N + j].size()) {
          for (int k = 0; k < tB[i * tB.size() / N + j].size(); k++) {
            parsed_R1CS_matrixes[i].push_back(
                tB[i * tB.size() / N + j][k].first);
            parsed_R1CS_matrixes[i].push_back(
                tB[i * tB.size() / N + j][k].second);
          }
          tB_dim[i].push_back(tB[i * tB.size() / N + j].size());

          // parsed_R1CS_matrixes[i][2*j + 4*M/N] = tB[i*2*M/N + j][0].first;
          // parsed_R1CS_matrixes[i][2*j+1 + 4*M/N] = tB[i*2*M/N + j][0].second;
        } else {
          parsed_R1CS_matrixes[i].push_back(-1);
          parsed_R1CS_matrixes[i].push_back(-1);
          tB_dim[i].push_back(1);
        }
      }
      for (int j = 0; j < tC.size() / N; j++) {
        if (tC[i * tC.size() / N + j].size()) {
          for (int k = 0; k < tC[i * tC.size() / N + j].size(); k++) {
            parsed_R1CS_matrixes[i].push_back(
                tC[i * tC.size() / N + j][k].first);
            parsed_R1CS_matrixes[i].push_back(
                tC[i * tC.size() / N + j][k].second);
          }
          tC_dim[i].push_back(tC[i * tC.size() / N + j].size());

          // parsed_R1CS_matrixes[i][2*j + 8*M/N] = tC[i*2*M/N + j][0].first;
          // parsed_R1CS_matrixes[i][2*j+1 + 8*M/N] = tC[i*2*M/N + j][0].second;
        } else {
          parsed_R1CS_matrixes[i].push_back(-1);
          parsed_R1CS_matrixes[i].push_back(-1);
          tC_dim[i].push_back(1);
        }
      }
    }
    send_R1CS_matrixes(parsed_R1CS_matrixes, tA_dim, tB_dim, tC_dim, N);
    parse_R1CS_matrixes(parsed_R1CS_matrixes[0], tA_dim[0], tB_dim[0],
                        tC_dim[0], N, M);
  } else {
    logm = buff[0];
    logn = buff[1];
    receive_index(index, dims);
    receive_R1CS_matrixes(N, M);
  }
}

vector<F> sumcheck_aggr(vector<F> &poly, vector<F> &b, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  vector<u64> f_element;

  com_rounds += 2;
  if (rank == 0) {
    vector<vector<F>> P(N);
    P[0] = poly;
    vector<MPI_Request> req(N);
    vector<vector<u64>> recv_data(N);

    for (int i = 1; i < N; i++) {
      recv_data[i].resize(2 * poly.size());
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
      // MPI_Recv(f_element.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], P[i]);
    }
    pt_cp.start();

    P = transpose(P);

    for (int i = 0; i < P.size(); i++)
      P[i] = unpack_secret_share(P[i], N, N);
    for (int i = 0; i < poly.size(); i++)
      poly[i] = F(0);
    // P = transpose(P);

    for (int i = 0; i < P.size(); i++) {
      for (int j = 0; j < P[i].size() / 4; j++) {
        poly[i] += b[j] * P[i][2 * j];
      }
    }

    pt_cp.end();
    field_vector_serialize(poly, f_element);

    // for(int i = 1; i < N; i++){
    //     cm += 8*f_element.size()/1024.0;
    //     MPI_Send(f_element.data(), 2,MPI_UINT64_T,i,0,MPI_COMM_WORLD);
    // }
  } else {
    field_vector_serialize(poly, f_element);

    cm += 8 * f_element.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(f_element.data(), f_element.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
    // MPI_Recv(f_element.data(),2,MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  myBcast(f_element, N);
  if (rank != 0) {
    poly.clear();
    field_vector_deserialize(f_element, poly);
  }
  return poly;
}

vector<vector<F>> send_final_sumcheck_poly(vector<F> s, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  com_rounds += 2;
  vector<vector<F>> H(s.size());
  vector<u64> f_element;

  if (rank == 0) {
    for (int i = 0; i < H.size(); i++) {
      H[i].push_back(s[i]);
    }
    vector<MPI_Request> req(N);
    vector<vector<u64>> recv_data(N);

    for (int i = 1; i < N; i++) {
      recv_data[i].resize(s.size() * 2);
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
      // MPI_Recv(f_element.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], s);
      for (int j = 0; j < H.size(); j++) {
        H[j].push_back(s[j]);
      }
    }
    // pt_cp.start();

    for (int i = 0; i < H.size(); i++) {
      H[i] = unpack_secret_share(H[i], N / 2, N);
    }

    // for(int i = 0; i < temp.size()/2; i++){
    //     H.push_back(temp[2*i]);
    // }
    s = convert2vector(H);
    field_vector_serialize(s, f_element);
  } else {
    cm += 8 * 2 / 1024.0;
    MPI_Request req;
    field_vector_serialize(s, f_element);

    MPI_Isend(f_element.data(), f_element.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
    f_element.resize(N * s.size());
  }
  myBcast(f_element, N);
  if (rank != 0) {
    vector<F> buff;
    field_vector_deserialize(f_element, buff);
    int ctr = 0;
    for (int i = 0; i < H.size(); i++) {
      for (int j = 0; j < N / 2; j++) {
        H[i].push_back(buff[ctr++]);
      }
    }
  }
  return H;
}

void compute_secret_shares(vector<F> &v, vector<vector<F>> &v_shares, int N,
                           int k, int _k, bool privacy_preserving) {
  v_shares.resize(N);
  F y = F(0);
  int counter = 0;
  for (int i = 0; i < N; i++) {
    v_shares[i].resize(v.size() / k);
  }

  for (int i = 0; i < v.size() / k; i++) {
    vector<F> buff(_k, F(0));
    for (int j = 0; j < k; j++) {
      buff[j] = v[counter++];
    }
    for (int j = k; j < _k; j++) {
      if (privacy_preserving)
        buff[j] = random();
    }

    fft(buff, (int)log2(buff.size()), true);
    buff.resize(2 * N, F(0));
    fft(buff, (int)log2(buff.size()), false);
    for (int j = 0; j < N; j++) {
      v_shares[j][i] = buff[2 * j + 1];
    }
  }
}

#include "Distributed_Sumcheck.h"
#include "Fiat_Shamir.h"

vector<F> unpack_secret_share(vector<F> &secret, int _k, int k) {
  vector<F> v(_k);
  fft(secret, (int)log2(secret.size()), true);

  F omega = getRootOfUnity(1 + (int)log2(secret.size()));
  omega = omega.inv();
  F mul = F(1);
  for (int i = 0; i < secret.size(); i++) {
    secret[i] = mul * secret[i];
    mul = omega * mul;
  }

  fft(secret, (int)log2(secret.size()), false);
  for (int j = 0; j < v.size(); j++)
    v[j] = secret[secret.size() * j / _k];
  return v;
}

void secret_share_coefficients(vector<F> &w, int M, int N, int _k, int k) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff_u64;
  vector<F> buff;

  if (rank == 0) {
    vector<F> poly = generate_randomness(M);
    vector<vector<F>> poly_shares;
    compute_secret_shares(poly, poly_shares, N, k, _k, true);

    w = poly_shares[0];

    for (int i = 1; i < N; i++) {
      field_vector_serialize(poly_shares[i], buff_u64);
      MPI_Send(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, i, 0,
               MPI_COMM_WORLD);
    }
  } else {
    buff_u64.resize(2 * (M) / k);
    MPI_Recv(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    field_vector_deserialize(buff_u64, w);
  }
}

void distribute_proving_data(vector<F> &vL, vector<F> &vR, vector<F> &vO,
                             vector<F> &w, int N, int M, int _k, int k,
                             int cir_type) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<F> buff;

  if (rank == 0) {
    vector<F> L, R, O, W;
    vector<vector<F>> L_shares, R_shares, O_shares, W_shares;
    prepare_witness_data(M, W, L, R, O, cir_type);

    compute_secret_shares(W, W_shares, N, k, _k, true);
    W.clear(); W.shrink_to_fit();
    compute_secret_shares(L, L_shares, N, k, _k, true);
    L.clear(); L.shrink_to_fit();
    compute_secret_shares(R, R_shares, N, k, _k, true);
    R.clear(); R.shrink_to_fit();
    compute_secret_shares(O, O_shares, N, k, _k, true);
    O.clear(); O.shrink_to_fit();

    vL = std::move(L_shares[0]);
    vR = std::move(R_shares[0]);
    vO = std::move(O_shares[0]);
    w = std::move(W_shares[0]);

    for (int i = 1; i < N; i++) {
        vector<u64> prover_buff_u64;
        vector<F> prover_shares = std::move(L_shares[i]);
        prover_shares.insert(prover_shares.end(), R_shares[i].begin(), R_shares[i].end());
        R_shares[i].clear(); R_shares[i].shrink_to_fit();
        prover_shares.insert(prover_shares.end(), O_shares[i].begin(), O_shares[i].end());
        O_shares[i].clear(); O_shares[i].shrink_to_fit();
        prover_shares.insert(prover_shares.end(), W_shares[i].begin(), W_shares[i].end());
        W_shares[i].clear(); W_shares[i].shrink_to_fit();

        field_vector_serialize(prover_shares, prover_buff_u64);
        MPI_Send(prover_buff_u64.data(), prover_buff_u64.size(), MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
    }
  } else {
    vector<u64> buff_u64;
    buff_u64.resize(2 * ((1 << logn) + 3 * (1 << logm)) / k);
    MPI_Recv(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    field_vector_deserialize(buff_u64, buff);
    buff_u64.clear(); buff_u64.shrink_to_fit();

    vL.assign(buff.begin(), buff.begin() + (1 << logm) / k);
    vR.assign(buff.begin() + (1 << logm) / k, buff.begin() + 2 * (1 << logm) / k);
    vO.assign(buff.begin() + 2 * (1 << logm) / k, buff.begin() + 3 * (1 << logm) / k);
    w.assign(buff.begin() + 3 * (1 << logm) / k, buff.end());
    buff.clear(); buff.shrink_to_fit();
  }
}

void setup_randomness(vector<F> &R, int N, int _k, int k, int size) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  int logk = (int)log2(k);
  if (rank == 0) {
    vector<F> random_values(k * (size));
    vector<vector<F>> R_shares;
    for (int i = 0; i < random_values.size(); i++)
      random_values[i] = random();
    compute_secret_shares(random_values, R_shares, N, k, _k, true);
    R = R_shares[0];
    vector<MPI_Request> req(N - 1);
    vector<vector<u64>> buff_u64(N);
    for (int i = 1; i < N; i++) {

      field_vector_serialize(R_shares[i], buff_u64[i]);
      F s = F(0);
      MPI_Isend(buff_u64[i].data(), buff_u64[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i - 1]);
      R_shares[i].clear();
      // MPI_Send(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD);
    }
    for (int i = 0; i < N - 1; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
    }
  } else {
    vector<u64> buff_u64;

    buff_u64.resize(2 * (size));
    MPI_Request req;
    // MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Irecv(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
    field_vector_deserialize(buff_u64, R);
  }
}

void distributed_FFT(vector<F> &arr, int logn, bool flag, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  int len = 1 << logn;
  // vector<F> w;
  // w.resize(len);
  // w[0] = F_ONE;

  // Get root of unity
  F rou = getRootOfUnity(logn);
  // w[1] = rou;
  std::vector<u32> rev, rev2;
  rev.resize(len / N);
  rev2.resize(N);
  for (u32 i = 1; i < len / N; ++i)
    rev[i] = rev[i >> 1] >> 1 | (i & 1) << (logn - ((int)log2(N)) - 1);

  for (u32 i = 1; i < N; ++i)
    rev2[i] = rev2[i >> 1] >> 1 | (i & 1) << ((int)log2(N) - 1);

  for (int j = 0; j < len / N; j++)
    if (rev[j] < j)
      std::swap(arr[j], arr[rev[j]]);

  F u, v;
  vector<F> w1(len / (N * 2));
  w1[0] = F(1);
  w1[1] = getRootOfUnity(logn);
  w1[1] = w1[1].fastPow(w1[1], N);
  for (int i = 2; i < w1.size(); i++)
    w1[i] = w1[i - 1] * w1[1];

  for (u32 i = 2; i <= len / N; i <<= 1) {
    for (u32 j2 = 0; j2 < len / N; j2 += i) {
      for (u32 k = 0; k < (i >> 1); ++k) {
        // if(j1 == 0)printf("R2: %d: (%d,%d), w[%d]\n",i,j2+k,j2+k + (i>>1),len
        // / i * k + j2);
        u = arr[j2 + k];
        v = arr[j2 + k + (i >> 1)] * w1[(len / i * k) / N];
        arr[j2 + k] = u + v;
        arr[j2 + k + (i >> 1)] = u - v;
      }
    }
  }

  vector<u64> send_buff(2 * arr.size()), recv_buff(2 * arr.size());
  field_vector_serialize(arr, send_buff);

  MPI_Request req;
  MPI_Ialltoall(send_buff.data(), send_buff.size() / N, MPI_UINT64_T,
                recv_buff.data(), recv_buff.size() / N, MPI_UINT64_T,
                MPI_COMM_WORLD, &req);
  network_timer.start();
  MPI_Wait(&req, MPI_STATUS_IGNORE);
  network_timer.end();
  vector<F> recv_arr;
  field_vector_deserialize(recv_buff, recv_arr);

  vector<vector<F>> D(N);
  for (int i = 0; i < N; i++) {
    D[i].resize(arr.size() / N);
    for (int j = 0; j < D[i].size(); j++) {
      D[i][j] = recv_arr[i * D[i].size() + j];
    }
  }
  for (int i = 0; i < N; i++) {
    if (rev2[i] < i)
      std::swap(D[i], D[rev2[i]]);
  }

  vector<vector<F>> w2((int)log2(N));
  int ctr1 = 0;
  for (u32 i = 2; i <= N; i <<= 1) {
    w2[ctr1].resize((i >> 1) * len / (N * N), F(1));
    F omega = getRootOfUnity(logn);
    F omega_k = omega.fastPow(omega, len / i);
    F omega_j = omega.fastPow(omega, N / i);
    vector<F> omegas_j(len / (N * N), F(1));
    vector<F> omegas_k(i >> 1, F(1));
    if (rank == 0) {
      omegas_j[1] = omega_j;
      for (int j = 2; j < omegas_j.size(); j++)
        omegas_j[j] = omegas_j[j - 1] * omegas_j[1];
    } else {
      omegas_j[0] = omega.fastPow(omega, len / (i * N) * rank);
      for (int j = 1; j < omegas_j.size(); j++)
        omegas_j[j] = omegas_j[j - 1] * omega_j;
    }
    if (omegas_k.size() > 1) {
      omegas_k[1] = omega_k;
      for (int j = 2; j < omegas_k.size(); j++)
        omegas_k[j] = omegas_k[j - 1] * omegas_k[1];
    }
    int ctr = 0;
    for (int j = 0; j < len / (N * N); j++) {
      for (int k = 0; k < (i >> 1); k++) {
        w2[ctr1][ctr++] = omegas_j[j] * omegas_k[k];
      }
    }
    // printf("> %d,%d\n",ctr,w2[ctr1].size());

    ctr1++;
  }
  ctr1 = 0;

  for (u32 i = 2; i <= N; i <<= 1) {
    int ctr = 0;
    for (u32 j2 = rank * arr.size() / N; j2 < (rank + 1) * arr.size() / N;
         j2++) {
      // if(j == 0) printf("%d\n",len / (i*len/N) * k);
      for (u32 k = 0; k < (i >> 1); ++k) {
        for (u32 j1 = 0; j1 < N; j1 += i) {
          u = D[j1 + k][j2 - rank * arr.size() / N];
          v = D[j1 + k + (i >> 1)][j2 - rank * arr.size() / N] * w2[ctr1][ctr];

          // v = D[j1 + k + (i >> 1)][j2-rank*arr.size()/N] * w[len / (i*len/N)
          // * (k*len/N + j2) ];
          D[j1 + k][j2 - rank * arr.size() / N] = u + v;
          D[j1 + k + (i >> 1)][j2 - rank * arr.size() / N] = u - v;
        }
        ctr++;
      }
    }
    ctr1++;
  }
  arr = convert2vector(D);
}

vector<F> aggregate_plaintext_poly(vector<F> poly, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  com_rounds += 2;
  vector<u64> coef;

  if (rank == 0) {
    vector<MPI_Request> req(N);
    vector<vector<u64>> recv_data(N);

    for (int i = 1; i < N; i++) {
      recv_data[i].resize(2 * poly.size());
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
    }
    for (int i = 1; i < N; i++) {
      vector<F> coef;
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], coef);
      for (int j = 0; j < poly.size(); j++)
        poly[j] += coef[j];
    }
    field_vector_serialize(poly, coef);
  } else {
    field_vector_serialize(poly, coef);
    cm += 8 * coef.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(coef.data(), coef.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
  myBcast(coef, N);
  field_vector_deserialize(coef, poly);
  return poly;
}

vector<vector<F>> aggregate_plaintext_poly_batch(vector<vector<F>> poly,
                                                 int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  com_rounds += 2;
  vector<u64> coef;
  vector<F> buff;
  if (rank == 0) {
    vector<MPI_Request> req(N);
    vector<vector<u64>> recv_data(N);
    // printf("(%lld,%lld),(%lld,%lld)\n",poly[0][0].real,poly[0][0].img,poly[1][0].real,poly[1][0].img);

    for (int i = 1; i < N; i++) {
      recv_data[i].resize(2 * poly.size() * poly[0].size());
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
    }
    for (int i = 1; i < N; i++) {
      vector<F> coef;
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      field_vector_deserialize(recv_data[i], coef);
      for (int l = 0; l < poly.size(); l++) {
        for (int j = 0; j < poly[l].size(); j++)
          poly[l][j] += coef[poly[0].size() * l + j];
      }
    }
    buff = convert2vector(poly);
    field_vector_serialize(buff, coef);
  } else {
    buff = convert2vector(poly);
    field_vector_serialize(buff, coef);
    cm += 8 * coef.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(coef.data(), coef.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD,
              &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
  myBcast(coef, N);
  if (rank != 0) {
    field_vector_deserialize(coef, buff);
    for (int i = 0; i < poly.size(); i++) {
      for (int j = 0; j < poly[i].size(); j++) {
        poly[i][j] = buff[poly[0].size() * i + j];
      }
    }
  }
  return poly;
}

void distributed_batch_FFT(vector<vector<F>> &arr, int logn, bool flag, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  int len = 1 << logn;
  com_rounds++;
  // Get root of unity
  F rou = getRootOfUnity(logn);
  // if (flag) w[1] = w[1].inv();
  // for (u32 i = 2; i < len; ++i) w[i] = w[i - 1] * w[1];
  std::vector<u32> rev, rev2;
  rev.resize(len / N);
  rev2.resize(N);
  rev[0] = 0;
  rev2[0] = 0;
  for (u32 i = 1; i < len / N; ++i)
    rev[i] = rev[i >> 1] >> 1 | (i & 1) << (logn - ((int)log2(N)) - 1);

  for (u32 i = 1; i < N; ++i)
    rev2[i] = rev2[i >> 1] >> 1 | (i & 1) << ((int)log2(N) - 1);

  for (int j = 0; j < arr.size(); j++) {
    for (int i = 0; i < arr[j].size(); i++) {
      if (rev[i] < i)
        std::swap(arr[j][i], arr[j][rev[i]]);
    }
  }
  F u, v;
  vector<F> w1(len / (N * 2));
  w1[0] = F(1);
  w1[1] = getRootOfUnity(logn);
  w1[1] = w1[1].fastPow(w1[1], N);

  for (int i = 2; i < w1.size(); i++)
    w1[i] = w1[i - 1] * w1[1];

  for (int l = 0; l < arr.size(); l++) {
    for (u32 i = 2; i <= len / N; i <<= 1) {
      for (u32 j2 = 0; j2 < len / N; j2 += i) {
        for (u32 k = 0; k < (i >> 1); ++k) {
          u = arr[l][j2 + k];
          v = arr[l][j2 + k + (i >> 1)] * w1[(len / i * k) / N];
          arr[l][j2 + k] = u + v;
          arr[l][j2 + k + (i >> 1)] = u - v;
        }
      }
    }
  }

  vector<vector<u64>> send_buff(arr.size()), recv_buff(arr.size());
  vector<MPI_Request> req(arr.size());
  vector<vector<F>> recv_arr(arr.size());
  int burst_size = 2;

  // Lambda helper to trigger async Ialltoall for a given burst
  auto start_burst = [&](int start_idx) {
    for (int b = 0; b < burst_size && start_idx + b < arr.size(); b++) {
      int l = start_idx + b;
      send_buff[l].resize(2 * arr[l].size());
      recv_buff[l].resize(2 * arr[l].size());
      field_vector_serialize(arr[l], send_buff[l]);
      MPI_Ialltoall(send_buff[l].data(), send_buff[l].size() / N, MPI_UINT64_T,
                    recv_buff[l].data(), recv_buff[l].size() / N, MPI_UINT64_T,
                    MPI_COMM_WORLD, &req[l]);
      cm += send_buff[l].size() * 8.0 / 1024.0;
    }
  };

  timer nt;
  nt.start();
  if (arr.size() > 0) {
    start_burst(0);
  }

  int ctr1 = 0;
  vector<vector<F>> w2((int)log2(N));
  for (u32 i = 2; i <= N; i <<= 1) {
    w2[ctr1].resize((i >> 1) * (len / (N * N)), F(1));
    F omega = getRootOfUnity(logn);
    F omega_k = omega.fastPow(omega, len / i);
    F omega_j = omega.fastPow(omega, N / i);
    vector<F> omegas_j(len / (N * N), F(1));
    vector<F> omegas_k(i >> 1, F(1));
    if (rank == 0) {
      omegas_j[1] = omega_j;
      for (int j = 2; j < omegas_j.size(); j++)
        omegas_j[j] = omegas_j[j - 1] * omegas_j[1];
    } else {
      omegas_j[0] = omega.fastPow(omega, len / (i * N) * rank);
      for (int j = 1; j < omegas_j.size(); j++)
        omegas_j[j] = omegas_j[j - 1] * omega_j;
    }
    if (omegas_k.size() > 1) {
      omegas_k[1] = omega_k;
      for (int j = 2; j < omegas_k.size(); j++)
        omegas_k[j] = omegas_k[j - 1] * omegas_k[1];
    }
    int ctr = 0;
    for (int j = 0; j < len / (N * N); j++) {
      for (int k = 0; k < (i >> 1); k++) {
        w2[ctr1][ctr++] = omegas_j[j] * omegas_k[k];
      }
    }
    // printf("> %d,%d\n",ctr,w2[ctr1].size());
    ctr1++;
  }

  for (int j = 0; j < arr.size(); j += burst_size) {
    // Wait until the CURRENT burst is fully received
    // todo: We might be better off to distribute the twiddle factor
    // computations among bursts
    network_timer.start();
    for (int b = 0; b < burst_size && j + b < arr.size(); b++) {
      MPI_Wait(&req[j + b], MPI_STATUS_IGNORE);
    }
    network_timer.end();

    if (j + burst_size < arr.size()) {
      start_burst(j + burst_size);
    }

    // Compute the Phase 2 local FFT on the current burst
    for (int b = 0; b < burst_size && j + b < arr.size(); b++) {
      int l = j + b;
      field_vector_deserialize(recv_buff[l], recv_arr[l]);
      vector<vector<F>> D(N);

      for (int i = 0; i < N; i++) {
        D[i].resize(arr[l].size() / N);
        for (int d_idx = 0; d_idx < D[i].size(); d_idx++) {
          D[i][d_idx] = recv_arr[l][i * D[i].size() + d_idx];
        }
      }

      for (int i = 0; i < N; i++) {
        if (rev2[i] < i)
          std::swap(D[i], D[rev2[i]]);
      }

      ctr1 = 0;
      for (u32 i = 2; i <= N; i <<= 1) {
        int ctr = 0;
        for (u32 j2 = rank * arr[l].size() / N;
             j2 < (rank + 1) * arr[l].size() / N; j2++) {
          for (u32 k = 0; k < (i >> 1); ++k) {
            for (u32 j1 = 0; j1 < N; j1 += i) {
              u = D[j1 + k][j2 - rank * arr[l].size() / N];
              v = D[j1 + k + (i >> 1)][j2 - rank * arr[l].size() / N] *
                  w2[ctr1][ctr];

              D[j1 + k][j2 - rank * arr[l].size() / N] = u + v;
              D[j1 + k + (i >> 1)][j2 - rank * arr[l].size() / N] = u - v;
            }
            ctr++;
          }
        }
        ctr1++;
      }
      arr[l] = convert2vector(D);
    }
  }

  nt.end();
  if (rank == 0) {
    printf("Alltoall time: %lf sec | Comm Size: %lf KB\n", nt.get_time(),
           recv_buff[0].size() * 8.0 / 1024.0);
  }
}

F sum2(F s, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff_u64;
  F K_r = F(0);

  if (rank == 0) {
    vector<F> shares(N);
    shares[0] = s;
    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(2);
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
      // MPI_Recv(f_element.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      shares[i].real = recv_data[i][0];
      shares[i].img = recv_data[i][1];
    }

    shares = unpack_secret_share(shares, N, N);
    for (int i = 0; i < shares.size() / 2; i++) {
      K_r += shares[2 * i];
    }
    buff_u64 = {K_r.real, K_r.img};
  } else {
    buff_u64 = {s.real, s.img};
    MPI_Request req;
    MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
  myBcast(buff_u64, N);
  if (rank != 0) {
    K_r.real = buff_u64[0];
    K_r.img = buff_u64[1];
  }
  return K_r;
}

F sum3(F s, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff_u64(2);
  F K_r = s;
  if (rank == 0) {
    for (int i = 1; i < N; i++) {
      MPI_Recv(buff_u64.data(), 2, MPI_UINT64_T, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      F a;
      a.real = buff_u64[0];
      a.img = buff_u64[1];
      K_r += a;
    }
  } else {
    buff_u64[0] = s.real;
    buff_u64[1] = s.img;
    MPI_Send(buff_u64.data(), 2, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
  }
  return K_r;
}

F sum(F s, vector<F> b, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff_u64;
  F K_r = F(0);

  if (rank == 0) {
    vector<F> shares(N);
    shares[0] = s;
    vector<vector<u64>> recv_data(N);
    vector<MPI_Request> req(N);
    for (int i = 1; i < N; i++) {
      recv_data[i].resize(2);
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
      // MPI_Recv(f_element.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      shares[i].real = recv_data[i][0];
      shares[i].img = recv_data[i][1];
    }

    shares = unpack_secret_share(shares, N, N);
    for (int i = 0; i < shares.size() / 4; i++) {
      K_r += b[i] * shares[2 * i];
    }
    buff_u64 = {K_r.real, K_r.img};
  } else {
    buff_u64 = {s.real, s.img};
    MPI_Request req;
    MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0,
              MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
  }
  myBcast(buff_u64, N);
  if (rank != 0) {
    K_r.real = buff_u64[0];
    K_r.img = buff_u64[1];
  }
  return K_r;
}

void send_codeword(vector<F> &arr, vector<F> &B, vector<F> &codeword,
                   int fold_depth, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  com_rounds += 1;
  if (rank == 0) {
    vector<F> segments;

    vector<MPI_Request> req(N);
    vector<vector<u64>> recv_data(N);

    for (int i = 1; i < N; i++) {
      recv_data[i].resize(2 * (codeword.size() + B.size() + arr.size()));
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
      // MPI_Recv(f_element.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    vector<u64> b_data(2 * B.size()), arr_data(2 * arr.size()),
        codeword_data(2 * codeword.size());

    /*
    vector<F> temp_arr(N*arr.size());
    vector<F> temp_B(N*arr.size());
    for(int i = 0; i < arr.size(); i++){
        temp_arr[i*N] = arr[i];
        temp_B[i*N] = B[i];
    }*/

    vector<vector<F>> D(fold_depth);
    for (int j = 0; j < fold_depth; j++) {
      vector<F> buff(codeword.size() / fold_depth);
      for (int k = 0; k < buff.size(); k++)
        buff[k] = codeword[j * buff.size() + k];
      D[j].insert(D[j].end(), buff.begin(), buff.end());
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      int ctr = 0;
      for (int j = 0; j < arr_data.size(); j++)
        arr_data[j] = recv_data[i][ctr++];
      for (int j = 0; j < b_data.size(); j++)
        b_data[j] = recv_data[i][ctr++];
      for (int j = 0; j < codeword_data.size(); j++)
        codeword_data[j] = recv_data[i][ctr++];
      field_vector_deserialize(codeword_data, segments);
      for (int j = 0; j < fold_depth; j++) {
        vector<F> buff(codeword.size() / fold_depth);
        for (int k = 0; k < buff.size(); k++)
          buff[k] = segments[j * buff.size() + k];
        D[j].insert(D[j].end(), buff.begin(), buff.end());
      }

      // codeword.insert(codeword.end(), segments.begin(),segments.end());

      field_vector_deserialize(arr_data, segments);
      // for(int j = 0; j < segments.size(); j++){
      //     temp_arr[i+N*j] = segments[j];
      // }

      arr.insert(arr.end(), segments.begin(), segments.end());
      field_vector_deserialize(b_data, segments);
      B.insert(B.end(), segments.begin(), segments.end());
      // for(int j = 0; j < segments.size(); j++){
      //     temp_B[i+N*j] = segments[j];
      // }
    }
    codeword = convert2vector(D);
    // arr = temp_arr;
    // B = temp_B;
  } else {
    vector<u64> msg, buff;
    field_vector_serialize(arr, buff);
    msg = buff;
    field_vector_serialize(B, buff);
    msg.insert(msg.end(), buff.begin(), buff.end());
    field_vector_serialize(codeword, buff);
    msg.insert(msg.end(), buff.begin(), buff.end());
    cm += 8 * msg.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(msg.data(), msg.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
    // MPI_Recv(f_element.data(),2,MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  // return poly;
}

void send_codeword_batch(vector<vector<F>> &arr, vector<vector<F>> &B,
                         vector<vector<F>> &codeword, int fold_depth, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  com_rounds += 2;
  if (rank == 0) {
    vector<F> segments;
    int size = 0;
    for (int i = 0; i < arr.size(); i++)
      size += arr[i].size() + B[i].size() + codeword[i].size();

    vector<MPI_Request> req(N);
    vector<vector<u64>> recv_data(N);

    for (int i = 1; i < N; i++) {
      recv_data[i].resize(2 * (size));
      MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0,
                MPI_COMM_WORLD, &req[i]);
      // MPI_Recv(f_element.data(),2,MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }

    vector<vector<u64>> b_data(arr.size()), arr_data(arr.size()),
        codeword_data(arr.size());
    for (int l = 0; l < arr.size(); l++) {
      b_data[l].resize(2 * B[l].size());
      arr_data[l].resize(2 * arr[l].size());
      codeword_data[l].resize(2 * codeword[l].size());
    }
    /*
    vector<F> temp_arr(N*arr.size());
    vector<F> temp_B(N*arr.size());
    for(int i = 0; i < arr.size(); i++){
        temp_arr[i*N] = arr[i];
        temp_B[i*N] = B[i];
    }*/
    vector<vector<vector<F>>> D(arr.size());
    for (int l = 0; l < D.size(); l++) {
      D[l].resize(fold_depth);
      for (int j = 0; j < fold_depth; j++) {

        vector<F> buff(codeword[l].size() / fold_depth);
        for (int k = 0; k < buff.size(); k++)
          buff[k] = codeword[l][j * buff.size() + k];
        D[l][j].insert(D[l][j].end(), buff.begin(), buff.end());
      }
    }
    for (int i = 1; i < N; i++) {
      network_timer.start();
      MPI_Wait(&req[i], MPI_STATUS_IGNORE);
      network_timer.end();
      int ctr = 0;
      for (int l = 0; l < arr.size(); l++) {
        for (int j = 0; j < arr_data[l].size(); j++)
          arr_data[l][j] = recv_data[i][ctr++];
        for (int j = 0; j < b_data[l].size(); j++)
          b_data[l][j] = recv_data[i][ctr++];
        for (int j = 0; j < codeword_data[l].size(); j++)
          codeword_data[l][j] = recv_data[i][ctr++];
        field_vector_deserialize(codeword_data[l], segments);

        for (int j = 0; j < fold_depth; j++) {
          vector<F> buff(codeword[l].size() / fold_depth);
          for (int k = 0; k < buff.size(); k++)
            buff[k] = segments[j * buff.size() + k];
          D[l][j].insert(D[l][j].end(), buff.begin(), buff.end());
        }

        // codeword.insert(codeword.end(), segments.begin(),segments.end());

        field_vector_deserialize(arr_data[l], segments);
        // for(int j = 0; j < segments.size(); j++){
        //     temp_arr[i+N*j] = segments[j];
        // }

        arr[l].insert(arr[l].end(), segments.begin(), segments.end());
        field_vector_deserialize(b_data[l], segments);
        B[l].insert(B[l].end(), segments.begin(), segments.end());
      }
      // for(int j = 0; j < segments.size(); j++){
      //     temp_B[i+N*j] = segments[j];
      // }
    }

    vector<F> t_c = codeword[0];

    for (int l = 0; l < arr.size(); l++)
      codeword[l] = convert2vector(D[l]);

    // arr = temp_arr;
    // B = temp_B;
  } else {
    vector<F> buff;
    for (int i = 0; i < arr.size(); i++) {
      buff.insert(buff.end(), arr[i].begin(), arr[i].end());
      buff.insert(buff.end(), B[i].begin(), B[i].end());
      buff.insert(buff.end(), codeword[i].begin(), codeword[i].end());
    }
    vector<u64> msg;
    field_vector_serialize(buff, msg);

    cm += 8 * msg.size() / 1024.0;
    MPI_Request req;
    MPI_Isend(msg.data(), msg.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();
    // MPI_Recv(f_element.data(),2,MPI_UINT64_T,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  // return poly;
}

F compute_Kr(vector<vector<F>> &R, vector<F> b, int N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  F K_r = F(0);
  for (int i = 0; i < R.size(); i++) {
    for (int j = 0; j < R[i].size() / 4; j++) {
      K_r += R[i][4 * j];
    }
  }

  return sum(K_r, b, N);
}

F generate_sumcheck_randomness(vector<vector<vector<F>>> &R,
                               vector<int> degrees1, vector<int> degrees2,
                               int N) {
  R.resize(2);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
  vector<u64> buff_u64;
  F K = 0;

  if (rank == 0) {
    vector<vector<vector<F>>> R1(degrees1.size());
    vector<vector<vector<F>>> R2(degrees2.size());
    for (int i = 0; i < R1.size(); i++) {
      R1[i].resize(N / 4);
      for (int j = 0; j < R1[i].size(); j++) {
        R1[i][j] = generate_randomness(4 * degrees1[i]);
      }
    }

    for (int i = 0; i < R2.size(); i++) {
      R2[i].resize(N / 4);
      for (int j = 0; j < R2[i].size(); j++) {
        R2[i][j] = generate_randomness(2 * degrees2[i]);
      }
    }

    for (int i = 0; i < R2.size(); i++) {
      for (int j = 0; j < R2[i].size(); j++) {
        for (int k = 0; k < R2[i][j].size() / 2; k++) {
          K += R2[i][j][2 * k];
        }
      }
    }

    for (int i = 0; i < R1.size(); i++) {
      vector<F> temp = convert2vector(transpose(R1[i]));
      vector<vector<F>> temp_shares;
      compute_secret_shares(temp, temp_shares, N, N / 4, N / 2, true);
      if (temp_shares[i].size() != 4 * degrees1[i]) {
        printf("Error\n");
      }
      R[0].push_back(temp_shares[0]);
      for (int j = 1; j < N; j++) {
        field_vector_serialize(temp_shares[j], buff_u64);
        MPI_Send(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, j, 0,
                 MPI_COMM_WORLD);
      }
    }

    for (int i = 0; i < R2.size(); i++) {
      vector<F> temp = convert2vector(transpose(R2[i]));
      vector<vector<F>> temp_shares;
      compute_secret_shares(temp, temp_shares, N, N / 4, N / 2, true);
      if (temp_shares[i].size() != 2 * degrees2[i]) {
        printf("Error\n");
      }
      R[1].push_back(temp_shares[0]);
      for (int j = 1; j < N; j++) {
        field_vector_serialize(temp_shares[j], buff_u64);
        MPI_Send(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, j, 0,
                 MPI_COMM_WORLD);
      }
    }
    vector<F> b = {K};
    field_vector_serialize(b, buff_u64);
    for (int j = 1; j < N; j++) {
      MPI_Send(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, j, 0,
               MPI_COMM_WORLD);
    }
  } else {

    R[0].resize(degrees1.size());
    R[1].resize(degrees2.size());
    for (int i = 0; i < degrees1.size(); i++) {
      buff_u64.clear();
      buff_u64.resize(8 * degrees1[i]);
      MPI_Recv(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      field_vector_deserialize(buff_u64, R[0][i]);
    }

    for (int i = 0; i < degrees2.size(); i++) {
      buff_u64.clear();
      buff_u64.resize(4 * degrees2[i]);
      MPI_Recv(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      field_vector_deserialize(buff_u64, R[1][i]);
    }
    buff_u64.clear();
    buff_u64.resize(2);
    MPI_Recv(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    K.real = buff_u64[0];
    K.img = buff_u64[1];
  }
  return K;
}
