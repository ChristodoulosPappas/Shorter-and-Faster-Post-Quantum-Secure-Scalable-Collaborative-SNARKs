#include "MPI_utils.hpp"
#include "Distributed_Sumcheck.h"
#include "timer.hpp"
#include "coPCS.h"
#include <unordered_map>
#include <algorithm>
#include "distributed_PCS.h"
extern timer pt_cp, vt;
double ps_plain = 0.0;
extern double cm;
vector<vector<pair<int, int>>> pA, pB, pC;
extern vector<int> real_idx_dim;
extern int logm, logn;
extern int com_rounds;
extern int sumcheck_offset;
extern int multree_offset;
extern bool isLAN;
extern int mask_l;
vector<F> remaining_claims;
// The number of layers of the multiplciation tree that pass without commitment
int layer_offset = 3;
timer sch_com;
timer com_timer;
extern bool data_parallel;
vector<vector<F>> mul_tree_row_data, mul_tree_codeword;
MT MulTree_Com;
vector<pair<F, vector<F>>> MulTree_Claims;
extern int min_seg_size;

vector<F> aggregate_quadratic_poly_sparrow(vector<F> H, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    vector<F> buff;
    vector<u64> buff_u64(2 * H.size()), send_buff(2 * H.size());
    com_rounds += 2;
    if (rank == 0)
    {
        vector<vector<u64>> recv_data(N);
        vector<MPI_Request> req(N - 1);
        for (int i = 1; i < N; i++)
        {
            recv_data[i].resize(2 * H.size());
            MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &req[i - 1]);
            // MPI_Recv(recv_data[i].data(),recv_data[i].size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        for (int step = 0; step < N - 1; step++)
        {
            int idx;
            network_timer.start();
            MPI_Waitany(N - 1, req.data(), &idx, MPI_STATUS_IGNORE);
            network_timer.end();
            field_vector_deserialize(recv_data[idx + 1], buff);
            for (int j = 0; j < H.size(); j++)
            {
                H[j] += buff[j];
            }
        }
        buff = H;
        field_vector_serialize(buff, buff_u64);
    }
    else
    {
        buff = H;
        field_vector_serialize(buff, send_buff);
        cm += 8 * send_buff.size() / 1024.0;
        MPI_Request req;
        MPI_Isend(send_buff.data(), send_buff.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
        // MPI_Send(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD);
        network_timer.start();
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        network_timer.end();
    }

    if (rank == 0)
        cm += (N - 1) * 8 * buff_u64.size() / 1024.0;
    myBcast(buff_u64, N);
    // MPI_Bcast(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
    if (rank != 0)
    {
        field_vector_deserialize(buff_u64, buff);
    }
    return buff;
}

quadratic_poly aggregate_poly(quadratic_poly H, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    vector<F> buff;
    vector<u64> buff_u64(6);
    com_rounds += 2;

    if (rank == 0)
    {
        vector<vector<u64>> recv_data(N);
        vector<MPI_Request> req(N);
        for (int i = 1; i < N; i++)
        {
            recv_data[i].resize(buff_u64.size());
            MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &req[i]);
            // MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        for (int i = 1; i < N; i++)
        {
            network_timer.start();
            MPI_Wait(&req[i], MPI_STATUS_IGNORE);
            network_timer.end();
            field_vector_deserialize(recv_data[i], buff);
            H = H + quadratic_poly(buff[0], buff[1], buff[2]);
        }
        buff = {H.a, H.b, H.c};
        field_vector_serialize(buff, buff_u64);
    }
    else
    {
        buff = {H.a, H.b, H.c};
        field_vector_serialize(buff, buff_u64);
        cm += 8 * buff_u64.size() / 1024.0;
        // MPI_Send(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD);
        MPI_Request req;
        MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
        network_timer.start();
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        network_timer.end();
    }
    if (rank == 0)
        cm += (N - 1) * 8 * buff_u64.size() / 1024.0;
    myBcast(buff_u64, N);

    // MPI_Bcast(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
    if (rank != 0)
    {
        field_vector_deserialize(buff_u64, buff);
        H = quadratic_poly(buff[0], buff[1], buff[2]);
    }
    return H;
}

cubic_poly aggregate_poly(cubic_poly H, int N, vector<F> &v)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    vector<F> buff;
    vector<u64> buff_u64(8);
    vector<u64> send_buff(N * 8), recv_buff(N * 8);
    com_rounds += 1;
    buff = {H.a, H.b, H.c, H.d};
    field_vector_serialize(buff, buff_u64);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < 8; j++)
            send_buff[i * 8 + j] = buff_u64[j];
    }
    MPI_Request req;
    MPI_Ialltoall(send_buff.data(), send_buff.size() / N, MPI_UINT64_T, recv_buff.data(), recv_buff.size() / N, MPI_UINT64_T, MPI_COMM_WORLD, &req);
    network_timer.start();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    network_timer.end();

    H.a = 0;
    H.b = 0;
    H.c = 0, H.d = 0;
    field_vector_deserialize(recv_buff, buff);
    for (int i = 0; i < N; i++)
    {
        H.a += v[i] * buff[4 * i];
        H.b += v[i] * buff[4 * i + 1];
        H.c += v[i] * buff[4 * i + 2];
        H.d += v[i] * buff[4 * i + 3];
    }
    /*
    if(rank == 0){
        H.a  = v[0]*H.a;
        H.b  = v[0]*H.b;
        H.c  = v[0]*H.c;
        H.d  = v[0]*H.d;

        for(int i = 1; i < N; i++){
            MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            field_vector_deserialize(buff_u64,buff);
            H = H + cubic_poly(v[i]*buff[0],v[i]*buff[1],v[i]*buff[2],v[i]*buff[3]);
        }
        buff = {H.a,H.b,H.c,H.d};
        field_vector_serialize(buff,buff_u64);
    }else{
        buff = {H.a,H.b,H.c,H.d};
        field_vector_serialize(buff,buff_u64);
        cm += 8*buff_u64.size()/1024.0;
        MPI_Send(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD);
    }
    if(rank == 0) cm += (N-1)*8*buff_u64.size()/1024.0;
    myBcast(buff_u64, N);
    //MPI_Bcast(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
    if(rank != 0){
        field_vector_deserialize(buff_u64,buff);
        H = cubic_poly(buff[0],buff[1],buff[2],buff[3]);
    }
    */

    return H;
}

vector<pair<F, vector<F>>> _quadratic_sumcheck(F y, vector<F> &v1, vector<F> &v2, int N)
{
    // int offset = 4;
    // vector<F> r = generate_randomness(int(log2(v1.size())));
    int offset = sumcheck_offset;

    int rounds = int(log2(v1.size())) - offset;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    com_rounds += 2;

    F rand;
    vector<F> r;
    if (rounds > 0)
    {
        for (int i = 0; i < rounds; i++)
        {

            pt_cp.start();

            quadratic_poly poly = quadratic_poly(F_ZERO, F_ZERO, F_ZERO);
            linear_poly l1, l2;

            int L = 1 << ((int)log2(v1.size()) - 1 - i);
            for (int j = 0; j < L; j++)
            {
                l1 = linear_poly(v1[2 * j + 1] - v1[2 * j], v1[2 * j]);
                l2 = linear_poly(v2[2 * j + 1] - v2[2 * j], v2[2 * j]);
                poly = poly + (l1 * l2);
            }

            vector<F> input;
            pt_cp.end();

            poly = aggregate_poly(poly, N);
            pt_cp.start();
            if (rank == 0)
                vt.start();

            if (poly.eval(0) + poly.eval(1) != y)
            {
                printf("Error in distributed sumcheck round %d\n", i);
                exit(-1);
            }
            rand = hash_to_field({poly.a, poly.b, poly.c});
            if (rank == 0)
                ps_plain += 16 * (3) / 1024.0;

            y = poly.eval(rand);
            if (rank == 0)
                vt.end();

            r.push_back(rand);
            for (int j = 0; j < L; j++)
            {
                v1[j] = rand * (v1[2 * j + 1] - v1[2 * j]) + v1[2 * j];
                v2[j] = rand * (v2[2 * j + 1] - v2[2 * j]) + v2[2 * j];
            }
            pt_cp.end();
        }
    }
    else
    {
        offset = int(log2(v1.size()));
    }

    vector<F> final_v1(1 << (offset)), final_v2(1 << (offset)), buff;
    vector<u64> buff_u64;
    for (int i = 0; i < final_v1.size(); i++)
    {
        final_v1[i] = v1[i];
        final_v2[i] = v2[i];
    }

    vector<F> reply;
    if (rank == 0)
    {
        int idx = final_v1.size();
        buff_u64.resize(4 * final_v1.size());
        final_v1.resize(final_v1.size() * N, F(0));
        final_v2.resize(final_v2.size() * N, F(0));
        vector<vector<u64>> recv_buff(N);
        vector<MPI_Request> req(N);
        for (int i = 1; i < N; i++)
        {
            recv_buff[i].resize(4 * final_v1.size());
            // MPI_Recv(recv_buff[i].data(),recv_buff[i].size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Irecv(recv_buff[i].data(), recv_buff[i].size(), MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &req[i]);
        }
        for (int i = 1; i < N; i++)
        {
            network_timer.start();
            MPI_Wait(&req[i], MPI_STATUS_IGNORE);
            network_timer.end();
            field_vector_deserialize(recv_buff[i], buff);
            for (int j = 0; j < (1 << offset); j++)
            {
                final_v1[idx] = buff[j];
                final_v2[idx] = buff[j + (buff.size() / 2)];
                idx++;
            }
        }
        pt_cp.start();

        vector<pair<F, vector<F>>> res = quadratic_sumcheck(y, final_v1, final_v2, F(0));
        if (rank == 0)
            ps_plain += 16 * (2 + 3 * (int)log2(N)) / 1024.0;

        reply.push_back(res[0].first);
        reply.push_back(res[1].first);
        reply.insert(reply.end(), res[0].second.begin(), res[0].second.end());
        field_vector_serialize(reply, buff_u64);
        pt_cp.end();
    }
    else
    {
        buff = final_v1;
        buff.insert(buff.end(), final_v2.begin(), final_v2.end());
        field_vector_serialize(buff, buff_u64);
        cm += 8 * buff_u64.size() / 1024.0;
        MPI_Request req;
        MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
        network_timer.start();
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        network_timer.end();

        // MPI_Send(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD);
        buff_u64.clear();
        buff_u64.resize(2 * (2 + offset + (int)log2(N)));
    }
    if (rank == 0)
        cm += (N - 1) * 8 * buff_u64.size() / 1024.0;
    myBcast(buff_u64, N);
    // MPI_Bcast(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
    if (rank != 0)
    {
        field_vector_deserialize(buff_u64, reply);
    }
    for (int i = 2; i < reply.size(); i++)
    {
        r.push_back(reply[i]);
    }

    return {make_pair(reply[0], r), make_pair(reply[1], r)};
}

F compute_ip(vector<F> &v1, vector<F> &v2, int poly_degree, int size, vector<F> beta, vector<F> beta2)
{
    vector<F> poly(poly_degree * (poly_degree + 1) / 2, F(0));
    int ctr = 0;
    F sum = F(0);
    F sum2 = 0;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    for (int i = 0; i < v1.size() / size; i++)
    {
        // for(int l = 0; l < poly_degree; l++){
        if (size != poly_degree)
        {
            printf("Abort %d,%d\n", size, poly_degree);
            exit(-1);
        }
        vector<F> temp_v1(poly_degree), temp_v2(poly_degree);
        for (int l = 0; l < poly_degree; l++)
        {
            temp_v1[l] = v1[poly_degree * i + l];
            temp_v2[l] = v2[poly_degree * i + l];
            sum += beta[ctr] * temp_v2[l] * temp_v1[l];
        }

        //}

        if (((i != 0 || v1.size() / (size * beta.size()) == 1)) && beta.size() && !(i % (v1.size() / (size * beta.size()))))
        {
            ctr++;
        }
    }

    // printf("Size: %d,%d, (%lld,%lld)\n",beta.size(),poly_degree*v1.size()/size,sum2.real,sum2.img);
    return sum;
}

vector<F> _sparrow_quadratic_sumcheck_step1(vector<F> &v1, vector<F> &v2, int poly_degree, int size, vector<F> beta = {})
{
    vector<F> poly;
    if (poly_degree <= 16)
        poly.resize(poly_degree * (poly_degree + 1) / 2, F(0));
    else
        poly.resize(2 * poly_degree, F(0));

    int ctr = 0;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    F sum = 0;
    for (int i = 0; i < v1.size() / size; i++)
    {
        if ((i != 0) && beta.size() && !(i % (v1.size() / (size * beta.size()))))
        {
            ctr++;
        }
        vector<F> temp_v1(poly_degree), temp_v2(poly_degree);
        for (int l = 0; l < poly_degree; l++)
        {
            temp_v1[l] = v1[poly_degree * i + l];
            temp_v2[l] = v2[poly_degree * i + l];
            sum += beta[ctr] * temp_v2[l] * temp_v1[l];
        }
        bool is_zeroes = true;
        for (int j = 0; j < temp_v1.size(); j++)
        {
            if (temp_v1[j] != 0)
            {
                is_zeroes = false;
                break;
            }
        }
        for (int j = 0; j < temp_v1.size(); j++)
        {
            if (is_zeroes == false)
                break;
            if (temp_v2[j] != 0)
            {
                is_zeroes = false;
                break;
            }
        }
        if (is_zeroes)
            continue;
        if (poly_degree <= 16)
        {
            int idx = 0;
            for (int l = 0; l < poly_degree; l++)
            {
                for (int h = l + 1; h < poly_degree; h++)
                {
                    if (beta.size())
                        poly[idx] += beta[ctr] * (temp_v1[l] * temp_v2[h] + temp_v1[h] * temp_v2[l]);
                    else
                        poly[idx] += temp_v1[l] * temp_v2[h] + temp_v1[h] * temp_v2[l];
                    idx++;
                }
            }

            for (int l = 0; l < poly_degree; l++)
            {
                if (beta.size())
                    poly[idx] += beta[ctr] * temp_v1[l] * temp_v2[l];
                else
                    poly[idx] += temp_v1[l] * temp_v2[l];
                idx++;
            }
        }
        else
        {
            fft(temp_v1, (int)log2(temp_v1.size()), true);
            fft(temp_v2, (int)log2(temp_v2.size()), true);
            temp_v1.resize(2 * temp_v1.size(), F(0));
            temp_v2.resize(2 * temp_v2.size(), F(0));
            fft(temp_v1, (int)log2(temp_v2.size()), false);
            fft(temp_v2, (int)log2(temp_v2.size()), false);
            for (int l = 0; l < poly.size(); l++)
            {
                if (beta.size())
                    poly[l] += beta[ctr] * temp_v1[l] * temp_v2[l];
                else
                    poly[l] += temp_v1[l] * temp_v2[l];
            }
        }
    }
    return poly;
}

F _sparrow_quadratic_sumcheck_step2(vector<F> &v1, vector<F> &v2, F rand, int poly_degree, int size)
{
    vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(poly_degree)), rand, poly_degree);
    int i = 0;
    F sum = F(0);
    for (i = 0; i < v1.size() / ((size)); i++)
    {
        F sum1 = F(0), sum2 = F(0);
        for (int l = 0; l < poly_degree; l++)
        {
            sum1 += L[l] * v1[poly_degree * i + l];
            sum2 += L[l] * v2[poly_degree * i + l];
        }
        v1[i] = sum1;
        v2[i] = sum2;
        sum += v1[i] * v2[i];
    }
    return sum;
}

vector<pair<F, vector<F>>> _cubic_sumcheck_sparrow(F y, vector<F> &v1, vector<F> &v2, vector<F> &beta1, vector<F> &beta2, vector<F> &beta3, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    com_rounds += 4;
    int offset = sumcheck_offset;
    vector<int> degrees;
    vector<F> original_v1 = v1, original_v2 = v2;
    int rounds = int(log2(v1.size())) - offset;
    if (rounds <= 8)
    {
        if (rounds % 4 != 0)
            degrees.push_back(rounds - 4 * ((int)rounds / 4));

        for (int i = 0; i < rounds / 4; i++)
        {
            degrees.push_back(4);
        }
    }
    else
    {
        degrees.push_back(4);
        degrees.push_back(rounds - 4);
    }
    F rand;
    vector<F> r;
    vector<F> Lambdas(1, 1);
    for (int i = 0; i < v1.size(); i++)
        v1[i] = beta1[i % beta1.size()] * v1[i];
    if (rounds > 0)
    {
        int _s = 1;
        vector<F> challenges;
        for (int i = 0; i < degrees.size(); i++)
        {
            _s *= (1 << degrees[i]);
            pt_cp.start();
            vector<F> poly = _sparrow_quadratic_sumcheck_step1(v1, v2, 1 << degrees[i], _s, beta2);
            for (int j = 0; j < poly.size(); j++)
            {
                poly[j] = beta3[rank] * poly[j];
            }
            pt_cp.end();
            sch_com.start();
            poly = aggregate_quadratic_poly_sparrow(poly, N);
            sch_com.end();
            pt_cp.start();
            if (rank == 0)
                vt.start();
            if (sparrow_V_check(poly, 1 << degrees[i]) != y)
            {
                printf("Error in distributed sumcheck round %d, %d, (%lld,%lld),(%lld,%lld)\n", i, poly.size(), poly[0].real, poly[0].img, y.real, y.img);
                exit(-1);
            }
            rand = hash_to_field({0});
            challenges.push_back(rand);
            if (rank == 0)
                ps_plain += 16 * 4 / 1024.0;

            y = evaluate_poly_extended(poly, {}, rand, degrees[i]);
            if (rank == 0)
                vt.end();
            F sum = _sparrow_quadratic_sumcheck_step2(v1, v2, rand, 1 << degrees[i], _s);
            pt_cp.end();
        }
        if (rank == 0)
        {
            pt_cp.start();
            for (int i = 0; i < challenges.size(); i++)
            {
                vector<F> L = compute_lagrange_coeff(getRootOfUnity(degrees[i]), challenges[i], 1 << degrees[i]);
                vector<F> buff = Lambdas;
                Lambdas.resize(Lambdas.size() * L.size());
                for (int j = 0; j < L.size(); j++)
                {
                    for (int k = 0; k < buff.size(); k++)
                    {
                        Lambdas[j * buff.size() + k] = buff[k] * L[j];
                    }
                }
            }
            if (Lambdas.size() != _s)
            {
                printf("Error\n");
            }
            pt_cp.end();
        }
    }
    else
    {
        offset = int(log2(v1.size()));
    }
    pt_cp.start();

    vector<F> final_v1(1 << (offset)), final_v2(1 << (offset)), buff;
    vector<u64> buff_u64;
    for (int i = 0; i < final_v1.size(); i++)
    {
        final_v1[i] = v1[i];
        final_v2[i] = v2[i];
    }
    pt_cp.end();

    vector<F> reply;
    if (rank == 0)
    {
        int final_v1_size = final_v1.size();
        pt_cp.start();

        vector<vector<u64>> recv_buff(N - 1);
        for (int i = 0; i < N - 1; i++)
            recv_buff[i].resize(4 * final_v1.size());
        // buff_u64.resize(6*final_v1.size());
        final_v1.resize(final_v1.size() * N, F(0));
        final_v2.resize(final_v2.size() * N, F(0));
        vector<MPI_Request> req(N - 1);
        pt_cp.end();

        sch_com.start();

        for (int i = 1; i < N; i++)
        {
            MPI_Irecv(recv_buff[i - 1].data(), recv_buff[i - 1].size(), MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &req[i - 1]);
        }
        // for(int i = 1; i < N; i++){
        //     //MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //     network_timer.start(); MPI_Wait(&req[i-1],MPI_STATUS_IGNORE); network_timer.end();
        //     field_vector_deserialize(recv_buff[i-1],buff);
        //     for(int j = 0; j < (1<<offset); j++){
        //         final_v1[idx] = buff[j];
        //         final_v2[idx] = buff[j+ (buff.size()/2)];
        //         idx++;
        //     }
        // }
        for (int step = 0; step < N - 1; step++)
        {
            int mpi_idx;
            network_timer.start();
            MPI_Waitany(N - 1, req.data(), &mpi_idx, MPI_STATUS_IGNORE);
            network_timer.end();

            int node_offset = final_v1_size + (mpi_idx * (1 << offset));
            field_vector_deserialize(recv_buff[mpi_idx], buff);
            for (int j = 0; j < (1 << offset); j++)
            {
                final_v1[node_offset + j] = buff[j];
                final_v2[node_offset + j] = buff[j + (buff.size() / 2)];
            }
        }
        sch_com.end();
        pt_cp.start();

        vector<F> B(beta2.size() * beta3.size());
        for (int i = 0; i < beta3.size(); i++)
        {
            for (int j = 0; j < beta2.size(); j++)
            {
                B[i * beta2.size() + j] = beta3[i] * beta2[j];
            }
        }
        vector<pair<F, vector<F>>> res = cubic_sumcheck(y, final_v1, final_v2, B, F(0));
        ps_plain += 16 * (2 + 4 * (int)log2(N)) / 1024.0;

        reply.push_back(res[0].first);
        reply.push_back(res[1].first);
        reply.insert(reply.end(), res[0].second.begin(), res[0].second.end());
        field_vector_serialize(reply, buff_u64);
        pt_cp.end();
    }
    else
    {
        buff = final_v1;
        buff.insert(buff.end(), final_v2.begin(), final_v2.end());
        field_vector_serialize(buff, buff_u64);
        cm += 8 * buff_u64.size() / 1024.0;
        MPI_Request req;
        MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
        network_timer.start();
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        network_timer.end();
        buff_u64.clear();
        buff_u64.resize(2 * (2 + offset + (int)log2(N)));
    }
    if (rank == 0)
        cm += (N - 1) * 8 * buff_u64.size() / 1024.0;
    sch_com.start();

    myBcast(buff_u64, N);
    sch_com.end();
    // MPI_Bcast(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
    pt_cp.start();
    if (rank != 0)
    {
        field_vector_deserialize(buff_u64, reply);
    }
    F y1_claim = reply[0];
    F y2_claim = reply[1];

    for (int i = 2; i < reply.size(); i++)
    {
        r.push_back(reply[i]);
    }
    vector<F> r_last = r;
    vector<F> r1, r2;
    for (int i = 0; i < offset; i++)
        r1.push_back(r[i]);
    for (int i = (offset); i < r.size(); i++)
        r2.push_back(r[i]);
    vector<F> beta;
    precompute_beta(r1, beta);

    vector<F> aggr_v1(v1.size() / (1 << offset), F(0)), aggr_v2(v1.size() / (1 << offset), F(0));
    for (int j = 0; j < beta.size(); j++)
    {
        for (int i = 0; i < aggr_v1.size(); i++)
        {
            aggr_v1[i] += original_v1[aggr_v1.size() * j + i] * beta[j];
            aggr_v2[i] += original_v2[aggr_v2.size() * j + i] * beta[j];
        }
    }
    pt_cp.end();

    if (rank == 0)
    {
        int aggr_v1_size = aggr_v1.size();

        sch_com.start();

        vector<vector<u64>> recv_buff(N - 1);
        for (int i = 0; i < N - 1; i++)
            recv_buff[i].resize(4 * aggr_v1.size());
        // buff_u64.resize(6*final_v1.size());
        aggr_v1.resize(aggr_v1.size() * N, F(0));
        aggr_v2.resize(aggr_v2.size() * N, F(0));
        vector<MPI_Request> req(N - 1);
        for (int i = 1; i < N; i++)
        {
            MPI_Irecv(recv_buff[i - 1].data(), recv_buff[i - 1].size(), MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &req[i - 1]);
        }
        // for(int i = 1; i < N; i++){
        //     //MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //     network_timer.start(); MPI_Wait(&req[i-1],MPI_STATUS_IGNORE); network_timer.end();
        //     field_vector_deserialize(recv_buff[i-1],buff);
        //     for(int j = 0; j < aggr_v1.size()/N; j++){
        //         aggr_v1[idx] = buff[j];
        //         aggr_v2[idx] = buff[j+ (buff.size()/2)];
        //         idx++;
        //     }
        // }
        int block_size = aggr_v1.size() / N;
        for (int step = 0; step < N - 1; step++)
        {
            int mpi_idx;
            network_timer.start();
            MPI_Waitany(N - 1, req.data(), &mpi_idx, MPI_STATUS_IGNORE);
            network_timer.end();

            int node_offset = aggr_v1_size + (mpi_idx * block_size);
            field_vector_deserialize(recv_buff[mpi_idx], buff);
            for (int j = 0; j < block_size; j++)
            {
                aggr_v1[node_offset + j] = buff[j];
                aggr_v2[node_offset + j] = buff[j + (buff.size() / 2)];
            }
        }
        sch_com.end();

        pt_cp.start();
        vector<F> extended_lambdas, extended_beta;
        for (int i = 0; i < N; i++)
        {
            extended_lambdas.insert(extended_lambdas.end(), Lambdas.begin(), Lambdas.end());
            extended_beta.insert(extended_beta.end(), beta1.begin(), beta1.end());
        }
        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < Lambdas.size(); i++)
            {
                extended_lambdas[Lambdas.size() * j + i] = _beta(j, r2) * extended_lambdas[Lambdas.size() * j + i];
            }
        }

        F a = F::_random();

        vector<pair<F, vector<F>>> res = batch_cubic_sumcheck(aggr_v1, extended_beta, aggr_v2, extended_lambdas, y1_claim + a * y2_claim, a);

        vector<F> temp_buff = remaining_claims;
        remaining_claims.clear();
        remaining_claims = {temp_buff[0], temp_buff[1], temp_buff[4], temp_buff[5]};
        ps_plain += 16 * (3 + 4 * (int)log2(N)) / 1024.0;
        reply.clear();
        reply.push_back(res[0].first);

        reply.push_back(res[2].first);
        reply.insert(reply.end(), res[0].second.begin(), res[0].second.end());
        field_vector_serialize(reply, buff_u64);
        pt_cp.end();
    }
    else
    {
        buff = aggr_v1;
        buff.insert(buff.end(), aggr_v2.begin(), aggr_v2.end());
        field_vector_serialize(buff, buff_u64);
        cm += 8 * buff_u64.size() / 1024.0;
        MPI_Request req;

        MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
        network_timer.start();
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        network_timer.end();
        buff_u64.clear();
        buff_u64.resize(2 * (2 + (int)log2(N * aggr_v1.size())));
        // printf(">> %d\n",buff_u64.size());
    }
    if (rank == 0)
        cm += (N - 1) * 8 * buff_u64.size() / 1024.0;
    sch_com.start();

    myBcast(buff_u64, N);
    sch_com.end();

    if (rank != 0)
    {
        field_vector_deserialize(buff_u64, reply);
    }

    r.clear();
    for (int i = 2; i < reply.size() - (int)log2(N); i++)
        r.push_back(reply[i]);

    r.insert(r.end(), r1.begin(), r1.end());
    for (int i = reply.size() - (int)log2(N); i < reply.size(); i++)
        r.push_back(reply[i]);

    return {make_pair(reply[0], r), make_pair(reply[1], r)};
}

vector<pair<F, vector<F>>> _quadratic_sumcheck_sparrow(F y, vector<F> &v1, vector<F> &v2, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    com_rounds += 2;
    int offset = sumcheck_offset;
    vector<int> degrees;
    // int offset = 4;
    // vector<F> r = generate_randomness(int(log2(v1.size())));
    int rounds = int(log2(v1.size())) - offset;
    if (rounds % 4 != 0)
        degrees.push_back(rounds - 4 * ((int)rounds / 4));
    vector<F> original_v1 = v1, original_v2 = v2;
    for (int i = 0; i < rounds / 4; i++)
    {
        degrees.push_back(4);
    }

    F rand;
    vector<F> r;
    vector<F> Lambdas(1, 1);

    if (rounds > 0)
    {
        int s = 1;
        vector<F> challenges;
        for (int i = 0; i < degrees.size(); i++)
        {
            s *= (1 << degrees[i]);

            pt_cp.start();

            vector<F> poly = _sparrow_quadratic_sumcheck_step1(v1, v2, 1 << degrees[i], s);

            pt_cp.end();

            poly = aggregate_quadratic_poly_sparrow(poly, N);
            pt_cp.start();

            if (rank == 0)
                vt.start();

            if (sparrow_V_check(poly, 1 << degrees[i]) != y)
            {
                printf("Error in distributed sumcheck round %d\n", i);
                exit(-1);
            }
            rand = hash_to_field(poly);
            challenges.push_back(rand);
            if (rank == 0)
                ps_plain += 16 * 4 / 1024.0;

            y = evaluate_poly_extended(poly, {}, rand, degrees[i]);
            if (rank == 0)
                vt.end();
            // r.push_back(rand);
            F sum = _sparrow_quadratic_sumcheck_step2(v1, v2, rand, 1 << degrees[i], s);
            pt_cp.end();
        }
        if (rank == 0)
        {
            pt_cp.start();

            for (int i = 0; i < challenges.size(); i++)
            {
                vector<F> L = compute_lagrange_coeff(getRootOfUnity(degrees[i]), challenges[i], 1 << degrees[i]);
                vector<F> buff = Lambdas;
                Lambdas.resize(Lambdas.size() * L.size());
                for (int j = 0; j < L.size(); j++)
                {
                    for (int k = 0; k < buff.size(); k++)
                    {
                        Lambdas[j * buff.size() + k] = buff[k] * L[j];
                    }
                }
            }
            if (Lambdas.size() != s)
            {
                printf("Error\n");
            }
            pt_cp.end();
        }
    }
    else
    {
        offset = int(log2(v1.size()));
    }
    pt_cp.start();

    vector<F> final_v1(1 << (offset)), final_v2(1 << (offset)), buff;
    vector<u64> buff_u64;
    for (int i = 0; i < final_v1.size(); i++)
    {
        final_v1[i] = v1[i];
        final_v2[i] = v2[i];
    }
    pt_cp.end();

    vector<F> reply;
    if (rank == 0)
    {
        int idx = final_v1.size();

        vector<vector<u64>> recv_buff(N - 1);
        for (int i = 0; i < N - 1; i++)
            recv_buff[i].resize(4 * final_v1.size());
        // buff_u64.resize(6*final_v1.size());
        final_v1.resize(final_v1.size() * N, F(0));
        final_v2.resize(final_v2.size() * N, F(0));
        vector<MPI_Request> req(N - 1);
        for (int i = 1; i < N; i++)
        {
            MPI_Irecv(recv_buff[i - 1].data(), recv_buff[i - 1].size(), MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &req[i - 1]);
        }
        for (int i = 1; i < N; i++)
        {
            // MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            network_timer.start();
            MPI_Wait(&req[i - 1], MPI_STATUS_IGNORE);
            network_timer.end();
            field_vector_deserialize(recv_buff[i - 1], buff);
            for (int j = 0; j < (1 << offset); j++)
            {
                final_v1[idx] = buff[j];
                final_v2[idx] = buff[j + (buff.size() / 2)];
                idx++;
            }
        }
        pt_cp.start();

        vector<pair<F, vector<F>>> res = quadratic_sumcheck(y, final_v1, final_v2, F(0));
        ps_plain += 16 * (2 + 4 * (int)log2(N)) / 1024.0;

        reply.push_back(res[0].first);
        reply.push_back(res[1].first);
        reply.insert(reply.end(), res[0].second.begin(), res[0].second.end());
        field_vector_serialize(reply, buff_u64);
        pt_cp.end();
    }
    else
    {
        buff = final_v1;
        buff.insert(buff.end(), final_v2.begin(), final_v2.end());
        field_vector_serialize(buff, buff_u64);
        cm += 8 * buff_u64.size() / 1024.0;
        MPI_Request req;
        MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
        network_timer.start();
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        network_timer.end();
        buff_u64.clear();
        buff_u64.resize(2 * (2 + offset + (int)log2(N)));
    }
    if (rank == 0)
        cm += (N - 1) * 8 * buff_u64.size() / 1024.0;
    myBcast(buff_u64, N);

    // MPI_Bcast(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
    if (rank != 0)
    {
        field_vector_deserialize(buff_u64, reply);
    }

    pt_cp.start();
    F y1_claim = reply[0];
    F y2_claim = reply[1];

    for (int i = 2; i < reply.size(); i++)
    {
        r.push_back(reply[i]);
    }
    vector<F> r_last = r;
    vector<F> r1, r2;
    for (int i = 0; i < offset; i++)
        r1.push_back(r[i]);
    for (int i = (offset); i < r.size(); i++)
        r2.push_back(r[i]);
    vector<F> beta;
    precompute_beta(r1, beta);

    // for(int i = 0; i < beta.size(); i++){
    //     beta[i] = _beta(rank,r2)*beta[i];
    // }
    vector<F> aggr_v1(v1.size() / (1 << offset), F(0)), aggr_v2(v1.size() / (1 << offset), F(0));
    for (int j = 0; j < beta.size(); j++)
    {
        for (int i = 0; i < aggr_v1.size(); i++)
        {
            aggr_v1[i] += original_v1[aggr_v1.size() * j + i] * beta[j];
            aggr_v2[i] += original_v2[aggr_v2.size() * j + i] * beta[j];
        }
    }
    pt_cp.end();

    if (rank == 0)
    {
        int idx = aggr_v1.size();

        vector<vector<u64>> recv_buff(N - 1);
        for (int i = 0; i < N - 1; i++)
            recv_buff[i].resize(4 * aggr_v1.size());
        // buff_u64.resize(6*final_v1.size());
        aggr_v1.resize(aggr_v1.size() * N, F(0));
        aggr_v2.resize(aggr_v2.size() * N, F(0));
        vector<MPI_Request> req(N - 1);
        for (int i = 1; i < N; i++)
        {
            MPI_Irecv(recv_buff[i - 1].data(), recv_buff[i - 1].size(), MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &req[i - 1]);
        }
        for (int i = 1; i < N; i++)
        {
            // MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            network_timer.start();
            MPI_Wait(&req[i - 1], MPI_STATUS_IGNORE);
            network_timer.end();
            field_vector_deserialize(recv_buff[i - 1], buff);
            for (int j = 0; j < aggr_v1.size() / N; j++)
            {
                aggr_v1[idx] = buff[j];
                aggr_v2[idx] = buff[j + (buff.size() / 2)];
                idx++;
            }
        }
        pt_cp.start();
        vector<F> extended_lambdas;
        for (int i = 0; i < N; i++)
        {
            extended_lambdas.insert(extended_lambdas.end(), Lambdas.begin(), Lambdas.end());
        }
        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < Lambdas.size(); i++)
            {
                extended_lambdas[Lambdas.size() * j + i] = _beta(j, r2) * extended_lambdas[Lambdas.size() * j + i];
            }
        }

        F a = F::_random();
        vector<F> aggr_v(aggr_v1.size());
        for (int i = 0; i < aggr_v.size(); i++)
        {
            aggr_v[i] = aggr_v1[i] + a * aggr_v2[i];
        }
        vector<pair<F, vector<F>>> res = quadratic_sumcheck(y1_claim + a * y2_claim, extended_lambdas, aggr_v, F(0));

        ps_plain += 16 * (3 + 4 * (int)log2(N)) / 1024.0;
        reply.clear();
        reply.push_back(evaluate_vector(aggr_v1, res[0].second));

        reply.push_back(evaluate_vector(aggr_v2, res[0].second));
        reply.insert(reply.end(), res[0].second.begin(), res[0].second.end());
        field_vector_serialize(reply, buff_u64);
        pt_cp.end();
    }
    else
    {
        buff = aggr_v1;
        buff.insert(buff.end(), aggr_v2.begin(), aggr_v2.end());
        field_vector_serialize(buff, buff_u64);
        cm += 8 * buff_u64.size() / 1024.0;
        MPI_Request req;
        MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
        network_timer.start();
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        network_timer.end();
        buff_u64.clear();
        buff_u64.resize(2 * (2 + (int)log2(N * aggr_v1.size())));
        // printf(">> %d\n",buff_u64.size());
    }
    if (rank == 0)
        cm += (N - 1) * 8 * buff_u64.size() / 1024.0;
    myBcast(buff_u64, N);
    if (rank != 0)
    {
        field_vector_deserialize(buff_u64, reply);
    }

    r.clear();
    for (int i = 2; i < reply.size() - (int)log2(N); i++)
        r.push_back(reply[i]);

    r.insert(r.end(), r1.begin(), r1.end());
    for (int i = reply.size() - (int)log2(N); i < reply.size(); i++)
        r.push_back(reply[i]);

    return {make_pair(reply[0], r), make_pair(reply[1], r)};
}

vector<pair<F, vector<F>>> _cubic_sumcheck(F y, vector<F> &v1, vector<F> &v2, vector<F> &v3, vector<F> &v, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    com_rounds += 2;
    int offset = sumcheck_offset;
    // int offset = 4;
    // vector<F> r = generate_randomness(int(log2(v1.size())));
    int rounds = int(log2(v1.size())) - offset;
    F rand;
    vector<F> r;
    if (rounds > 0)
    {
        for (int i = 0; i < rounds; i++)
        {
            pt_cp.start();

            cubic_poly poly = cubic_poly(F_ZERO, F_ZERO, F_ZERO, F_ZERO);
            linear_poly l1, l2, l3;

            int L = 1 << ((int)log2(v1.size()) - 1 - i);
            for (int j = 0; j < L; j++)
            {
                l1 = linear_poly(v1[2 * j + 1] - v1[2 * j], v1[2 * j]);
                l2 = linear_poly(v2[2 * j + 1] - v2[2 * j], v2[2 * j]);
                l3 = linear_poly(v3[2 * j + 1] - v3[2 * j], v3[2 * j]);
                poly = poly + (l1 * l2 * l3);
            }

            vector<F> input;
            pt_cp.end();

            poly = aggregate_poly(poly, N, v);
            pt_cp.start();

            if (rank == 0)
                vt.start();

            if (poly.eval(0) + poly.eval(1) != y)
            {
                printf("Error in distributed sumcheck round %d\n", i);
                exit(-1);
            }
            rand = hash_to_field({poly.a, poly.b, poly.c, poly.d});
            if (rank == 0)
                ps_plain += 16 * 4 / 1024.0;
            y = poly.eval(rand);
            if (rank == 0)
                vt.end();
            r.push_back(rand);
            for (int j = 0; j < L; j++)
            {
                v1[j] = rand * (v1[2 * j + 1] - v1[2 * j]) + v1[2 * j];
                v2[j] = rand * (v2[2 * j + 1] - v2[2 * j]) + v2[2 * j];
                v3[j] = rand * (v3[2 * j + 1] - v3[2 * j]) + v3[2 * j];
            }
            pt_cp.end();
        }
    }
    else
    {
        offset = int(log2(v1.size()));
    }

    vector<F> final_v1(1 << (offset)), final_v2(1 << (offset)), final_v3(1 << (offset)), buff;
    vector<u64> buff_u64;
    for (int i = 0; i < final_v1.size(); i++)
    {
        final_v1[i] = v1[i];
        final_v2[i] = v2[i];
        final_v3[i] = v[rank] * v3[i];
    }

    vector<F> reply;
    if (rank == 0)
    {
        int idx = final_v1.size();

        vector<vector<u64>> recv_buff(N - 1);
        for (int i = 0; i < N - 1; i++)
            recv_buff[i].resize(6 * final_v1.size());
        // buff_u64.resize(6*final_v1.size());
        final_v1.resize(final_v1.size() * N, F(0));
        final_v2.resize(final_v2.size() * N, F(0));
        final_v3.resize(final_v3.size() * N, F(0));
        vector<MPI_Request> req(N - 1);
        for (int i = 1; i < N; i++)
        {
            MPI_Irecv(recv_buff[i - 1].data(), recv_buff[i - 1].size(), MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &req[i - 1]);
        }
        for (int i = 1; i < N; i++)
        {
            // MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            network_timer.start();
            MPI_Wait(&req[i - 1], MPI_STATUS_IGNORE);
            network_timer.end();
            field_vector_deserialize(recv_buff[i - 1], buff);
            for (int j = 0; j < (1 << offset); j++)
            {
                final_v1[idx] = buff[j];
                final_v2[idx] = buff[j + (buff.size() / 3)];
                final_v3[idx] = buff[j + 2 * (buff.size() / 3)];
                idx++;
            }
        }
        pt_cp.start();

        vector<pair<F, vector<F>>> res = cubic_sumcheck(y, final_v1, final_v2, final_v3, F(0));
        ps_plain += 16 * (3 + 4 * (int)log2(N)) / 1024.0;

        reply.push_back(res[0].first);
        reply.push_back(res[1].first);
        reply.push_back(res[2].first);
        reply.insert(reply.end(), res[0].second.begin(), res[0].second.end());
        field_vector_serialize(reply, buff_u64);
        pt_cp.end();
    }
    else
    {
        buff = final_v1;
        buff.insert(buff.end(), final_v2.begin(), final_v2.end());
        buff.insert(buff.end(), final_v3.begin(), final_v3.end());
        field_vector_serialize(buff, buff_u64);
        cm += 8 * buff_u64.size() / 1024.0;
        MPI_Request req;
        MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
        network_timer.start();
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        network_timer.end();
        buff_u64.clear();
        buff_u64.resize(2 * (3 + offset + (int)log2(N)));
    }
    if (rank == 0)
        cm += (N - 1) * 8 * buff_u64.size() / 1024.0;
    myBcast(buff_u64, N);

    // MPI_Bcast(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
    if (rank != 0)
    {
        field_vector_deserialize(buff_u64, reply);
    }
    for (int i = 3; i < reply.size(); i++)
    {
        r.push_back(reply[i]);
    }
    return {make_pair(reply[0], r), make_pair(reply[1], r), make_pair(reply[2], r)};
}

vector<pair<F, vector<F>>> _cubic_sumcheck2(F y, vector<F> &v1, vector<F> &v2, vector<F> &v3, vector<F> &v, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    com_rounds += 2;
    // int offset = 4;
    // vector<F> r = generate_randomness(int(log2(v1.size())));
    int rounds = int(log2(N)) + 1;
    int offset = (int)log2(v1.size()) - rounds;
    F rand;
    vector<F> r;
    if (rounds > 0)
    {
        for (int i = 0; i < rounds; i++)
        {
            pt_cp.start();

            cubic_poly poly = cubic_poly(F_ZERO, F_ZERO, F_ZERO, F_ZERO);
            linear_poly l1, l2, l3;

            int L = 1 << ((int)log2(v1.size()) - 1 - i);
            for (int j = 0; j < L; j++)
            {
                l1 = linear_poly(v1[2 * j + 1] - v1[2 * j], v1[2 * j]);
                l2 = linear_poly(v2[2 * j + 1] - v2[2 * j], v2[2 * j]);
                l3 = linear_poly(v3[2 * j + 1] - v3[2 * j], v3[2 * j]);
                poly = poly + (l1 * l2 * l3);
            }

            vector<F> input;
            pt_cp.end();

            poly = aggregate_poly(poly, N, v);
            pt_cp.start();

            if (rank == 0)
                vt.start();

            if (poly.eval(0) + poly.eval(1) != y)
            {
                printf("Error in distributed sumcheck round %d\n", i);
                exit(-1);
            }
            rand = hash_to_field({poly.a, poly.b, poly.c, poly.d});
            if (rank == 0)
                ps_plain += 16 * 4 / 1024.0;
            y = poly.eval(rand);
            if (rank == 0)
                vt.end();
            r.push_back(rand);
            for (int j = 0; j < L; j++)
            {
                v1[j] = rand * (v1[2 * j + 1] - v1[2 * j]) + v1[2 * j];
                v2[j] = rand * (v2[2 * j + 1] - v2[2 * j]) + v2[2 * j];
                v3[j] = rand * (v3[2 * j + 1] - v3[2 * j]) + v3[2 * j];
            }
            pt_cp.end();
        }
    }
    else
    {
        offset = int(log2(v1.size()));
    }

    vector<F> final_v1(1 << (offset)), final_v2(1 << (offset)), final_v3(1 << (offset)), buff;
    vector<u64> buff_u64;
    for (int i = 0; i < final_v1.size(); i++)
    {
        final_v1[i] = v1[i];
        final_v2[i] = v2[i];
        final_v3[i] = v[rank] * v3[i];
    }

    vector<F> reply;
    if (rank == 0)
    {
        int idx = final_v1.size();

        vector<vector<u64>> recv_buff(N - 1);
        for (int i = 0; i < N - 1; i++)
            recv_buff[i].resize(6 * final_v1.size());
        // buff_u64.resize(6*final_v1.size());
        final_v1.resize(final_v1.size() * N, F(0));
        final_v2.resize(final_v2.size() * N, F(0));
        final_v3.resize(final_v3.size() * N, F(0));
        vector<MPI_Request> req(N - 1);
        for (int i = 1; i < N; i++)
        {
            MPI_Irecv(recv_buff[i - 1].data(), recv_buff[i - 1].size(), MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &req[i - 1]);
        }
        for (int i = 1; i < N; i++)
        {
            // MPI_Recv(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            network_timer.start();
            MPI_Wait(&req[i - 1], MPI_STATUS_IGNORE);
            network_timer.end();
            field_vector_deserialize(recv_buff[i - 1], buff);
            for (int j = 0; j < (1 << offset); j++)
            {
                final_v1[idx] = buff[j];
                final_v2[idx] = buff[j + (buff.size() / 3)];
                final_v3[idx] = buff[j + 2 * (buff.size() / 3)];
                idx++;
            }
        }
        pt_cp.start();

        vector<pair<F, vector<F>>> res = pcubic_sumcheck(y, final_v1, final_v2, final_v3, F(0));
        ps_plain += 16 * (3 + 4 * (int)log2(N)) / 1024.0;

        reply.push_back(res[0].first);
        reply.push_back(res[1].first);
        reply.push_back(res[2].first);
        reply.insert(reply.end(), res[0].second.begin(), res[0].second.end());
        field_vector_serialize(reply, buff_u64);
        pt_cp.end();
    }
    else
    {
        buff = final_v1;
        buff.insert(buff.end(), final_v2.begin(), final_v2.end());
        buff.insert(buff.end(), final_v3.begin(), final_v3.end());
        field_vector_serialize(buff, buff_u64);
        cm += 8 * buff_u64.size() / 1024.0;
        MPI_Request req;
        MPI_Isend(buff_u64.data(), buff_u64.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
        network_timer.start();
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        network_timer.end();
        buff_u64.clear();
        buff_u64.resize(2 * (3 + offset + (int)log2(N)));
    }
    if (rank == 0)
        cm += (N - 1) * 8 * buff_u64.size() / 1024.0;
    myBcast(buff_u64, N);

    // MPI_Bcast(buff_u64.data(),buff_u64.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
    if (rank != 0)
    {
        field_vector_deserialize(buff_u64, reply);
    }
    for (int i = 3; i < reply.size(); i++)
    {
        r.push_back(reply[i]);
    }
    return {make_pair(reply[0], r), make_pair(reply[1], r), make_pair(reply[2], r)};
}

vector<F> compute_eval_claims(vector<vector<F>> commitment_data, vector<F> &r1, vector<F> &r2, F old_y, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

    vector<vector<F>> r(commitment_data.size());
    for (int i = commitment_data.size() - 1; i >= 0; i--)
    {
        r[i] = r1;
        int offset = (int)log2(commitment_data[i].size()) - r1.size();
        for (int j = 0; j < offset; j++)
        {
            F p = hash_to_field({0});
            r1.push_back(p);
            r[i].push_back(p);
        }
    }
    vector<F> evals = batch_distributed_eval_opt(commitment_data, r[0], r2, N);

    F p = hash_to_field({0});
    r1.push_back(p);

    evals.push_back(old_y);
    return evals;
}

F compute_new_sum(vector<F> eval, vector<int> sizes, vector<F> r)
{
    F y = F(0);
    int pos = 0;
    for (int i = 0; i < eval.size(); i++)
    {

        y += get_offset_product(sizes[i], pos, r) * eval[i];
        pos += sizes[i];
    }
    return y;
}

F eval_commitment_data(vector<vector<F>> r, vector<vector<F>> commitment_data, int N)
{
    vector<F> r1 = r[2];
    vector<F> r2 = r[0];

    for (int i = 0; i < (int)log2(commitment_data.size() * N * commitment_data[0].size()) - r[2].size() - r[0].size(); i++)
    {
        r2.push_back(r[1][i]);
    }

    // r2.insert(r2.end(),r[1].begin(),r[1].end());

    vector<F> comm_data_vector = convert2vector(commitment_data);
    pad_vector(comm_data_vector);
    vector<F> beta1, beta2;
    while (r2.size() != (int)log2(comm_data_vector.size()))
    {
        r2.pop_back();
    }
    precompute_beta(r1, beta1);
    precompute_beta(r2, beta2);

    beta2.resize(comm_data_vector.size());

    F y = distributed_eval(comm_data_vector, beta2, beta1, N);

    vector<F> _r = r1;
    _r.insert(_r.begin(), r2.begin(), r2.end());
    MulTree_Claims.push_back(make_pair(y, _r));
    return y;
}
extern int witness_commit_rate;
void commit_tree_witness(vector<vector<F>> &data, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    vector<F> buff = convert2vector(data);
    buff.resize(next_pow2(buff.size()), F(0));
    vector<F> even_odd_data;
    get_even_odd_data(buff, even_odd_data, N);
    int Batch_size = buff.size() / min_seg_size;
    vector<vector<F>> temp_data_even_odd(Batch_size);
    for (int i = 0; i < Batch_size; i++)
    {
        temp_data_even_odd[i].resize(even_odd_data.size() / Batch_size);
    }
    // temp_data_even_odd[1].resize(even_odd_data.size()/2);
    int ctr = 0;
    int arr_size = even_odd_data.size() / 2;

    // xx zz xx zz xx zz
    // xxxxxxxx zzzzzzzz xxxxxxxx zzzzzzzz xxxxxxxx zzzzzzzz xxxxxxxx zzzzzzzz
    // 12345678 9,10,11,12,13,14,15,16     17,18,19,20
    int offset = even_odd_data.size() / (Batch_size * N);

    for (int j = 0; j < N; j++)
    {
        for (int k = 0; k < Batch_size; k++)
        {
            for (int i = 0; i < offset; i++)
            {
                temp_data_even_odd[k][j * offset + i] = even_odd_data[ctr++];
            }
        }
    }
    even_odd_data = convert2vector(temp_data_even_odd);

    Batch_dCommit(even_odd_data, buff, mul_tree_row_data, mul_tree_codeword, MulTree_Com, witness_commit_rate, N, buff.size() / min_seg_size);
    // dCommit(even_odd_data, mul_tree_codeword, MulTree_Com, 4, N);
}

pair<F, vector<vector<F>>> prove_product_new(vector<vector<F>> &input, vector<F> &output, F y, vector<F> r, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

    multree_offset = (int)log2(input[0].size() / N);
    int depth = layer_offset * ((int)(log2(next_pow2(input[0].size())) - multree_offset) / layer_offset + 1);
    multree_offset = (int)log2(input[0].size()) - depth;

    if (rank == 0)
        printf("%d,%d\n", multree_offset, depth);
    double temp_time = pt_cp.get_time();
    pt_cp.start();
    timer smch_timer;
    smch_timer.start();
    size_t vectors = input.size();

    size_t size = input[0].size();
    for (size_t i = 0; i < input.size(); i++)
    {
        if (input[i].size() != size)
        {
            printf("Error in mul tree sumcheck, no equal size vectors %zu,%zu\n", input[i].size(), size);
            exit(-1);
        }
        input[i].resize(1 << depth, F(1));
    }

    if (vectors != 1 << ((int)log2(vectors)))
    {
        vectors = 1 << ((int)log2(vectors) + 1);
        size_t old_vector_size = input.size();
        vector<F> temp_vector(1 << depth, F(0));
        for (size_t i = 0; i < vectors - old_vector_size; i++)
        {
            input.push_back(temp_vector);
        }
    }
    // Initialize total input
    size_t total_input_size = (size_t)vectors * size;

    vector<F> total_input(total_input_size);
    size_t counter = 0;
    for (size_t j = 0; j < input.size(); j++)
    {
        for (size_t i = 0; i < input[0].size(); i++)
        {
            total_input[counter] = input[j][i];
            counter++;
        }
    }

    vector<vector<F>> transcript(depth);
    vector<vector<F>> in1(depth), in2(depth);
    for (size_t i = 0; i < depth; i++)
    {
        transcript[i].resize(total_input_size / ((size_t)1 << (i + 1)));
        in1[i].resize(total_input_size / ((size_t)1 << (i + 1)));
        in2[i].resize(total_input_size / ((size_t)1 << (i + 1)));
    }

    counter = 0;
    for (size_t i = 0; i < total_input_size / 2; i++)
    {
        transcript[0][counter] = total_input[2 * i] * total_input[2 * i + 1];
        in1[0][counter] = total_input[2 * i];
        in2[0][counter] = total_input[2 * i + 1];
        counter++;
    }

    size_t len = total_input_size / 2;
    for (size_t i = 1; i < transcript.size(); i++)
    {
        counter = 0;
        for (size_t j = 0; j < len / 2; j++)
        {
            transcript[i][counter] = transcript[i - 1][2 * j] * transcript[i - 1][2 * j + 1];
            in1[i][counter] = transcript[i - 1][2 * j];
            in2[i][counter] = transcript[i - 1][2 * j + 1];
            counter++;
        }
        len = len / 2;
    }

    vector<vector<F>> commitment_data;
    // Commit selected layers
    for (int i = 0; i < depth / layer_offset - 1; i++)
    {
        commitment_data.push_back(transcript[layer_offset * (i + 1) - 1]);
        // if(rank == 0) printf("%d,%d\n",transcript[0].size(),commitment_data[i].size());
    }
    pt_cp.end();
    // Need to run commitment aglorithm
    // get_even_odd_data(commitment_data);
    if (commitment_data.size())
        commit_tree_witness(commitment_data, N);
    pt_cp.start();
    vector<vector<F>> batched_in1(layer_offset), batched_in2(layer_offset);
    for (int i = 0; i < depth; i++)
    {
        batched_in1[i % (layer_offset)].insert(batched_in1[i % (layer_offset)].end(), in1[i].begin(), in1[i].end());
        batched_in2[i % (layer_offset)].insert(batched_in2[i % (layer_offset)].end(), in2[i].begin(), in2[i].end());
        in1[i].clear();
        in2[i].clear();
    }
    for (int i = 0; i < batched_in1.size(); i++)
    {
        pad_vector(batched_in1[i]);
        pad_vector(batched_in2[i]);
    }

    // Send outputs to the leader
    vector<u64> buff;
    pair<F, vector<F>> eval_claim;
    vector<F> buff_reply;
    com_rounds += 2;
    pt_cp.end();

    if (rank == 0)
    {
        size_t ctr = 0;
        sch_com.start();
        vector<vector<F>> local_input(vectors);
        for (size_t i = 0; i < local_input.size(); i++)
        {
            local_input[i].resize(N * ((transcript[depth - 1].size()) / vectors));
            for (size_t j = 0; j < transcript[depth - 1].size() / vectors; j++)
                local_input[i][j] = transcript[depth - 1][ctr++];

            // local_input[i][0] = transcript[depth-1][i];
        }
        // buff.resize(2*transcript[depth-1].size());
        vector<vector<u64>> recv_data(N - 1);
        vector<MPI_Request> req(N - 1);
        for (int i = 0; i < N - 1; i++)
        {
            recv_data[i].resize(2 * transcript[depth - 1].size());
            MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i + 1, 0, MPI_COMM_WORLD, &req[i]);
        }
        for (size_t i = 0; i < N - 1; i++)
        {
            vector<F> _buff;
            ctr = 0;
            MPI_Wait(&req[i], MPI_STATUS_IGNORE);
            // MPI_Recv(buff.data(),buff.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            field_vector_deserialize(recv_data[i], _buff);

            size_t chunk_size = _buff.size() / vectors;
            for (size_t j = 0; j < vectors; j++)
            {
                for (size_t k = 0; k < chunk_size; k++)
                {
                    // if(j == 0) printf("%d,%d\n",i*_buff.size()/vectors + k,vectors);
                    size_t target_idx = (i + 1) * chunk_size + k;
                    if (target_idx >= local_input[j].size())
                    {
                        printf("[CRITICAL] Out of bounds: trying to access index %zu on local_input[%zu] (max size: %zu)\n",
                               target_idx, j, local_input[j].size());
                        fflush(stdout);
                        MPI_Abort(MPI_COMM_WORLD, 1); // Fail fast
                    }
                    // local_input[j][(i + 1) * _buff.size() / vectors + k] = _buff[ctr++];
                    local_input[j][target_idx] = _buff[ctr++];
                }
            }
            // for(int j = 0; j < buff.size()/2; j++){
            //     local_input[j][i].real = buff[2*j];
            //     local_input[j][i].img = buff[2*j+1];
            // }
        }
        sch_com.end();

        pt_cp.start();
        eval_claim = prove_multiplication_tree_new(local_input, output, F(0), y, {});
        buff_reply.push_back(eval_claim.first);

        buff_reply.insert(buff_reply.end(), eval_claim.second.begin(), eval_claim.second.end());

        field_vector_serialize(buff_reply, buff);

        pt_cp.end();
    }
    else
    {
        field_vector_serialize(transcript[depth - 1], buff);
        cm += 8 * buff.size() / 1024.0;
        MPI_Request req;
        MPI_Isend(buff.data(), buff.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        buff.clear();
        buff.resize(2 * (1 + log2(N * transcript[depth - 1].size())));
    }

    if (rank == 0)
        cm += (N - 1) * 8 * buff.size() / 1024.0;
    // todo: track this
    myBcast(buff, N);
    smch_timer.end();
    if (rank == 0)
        printf("          Step 1: %lf, %lf,%lf\n", smch_timer.get_time(), pt_cp.get_time() - temp_time, sch_com.get_time());
    temp_time = pt_cp.get_time();
    // MPI_Bcast(buff.data(),buff.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
    if (rank != 0)
    {
        field_vector_deserialize(buff, buff_reply);
        eval_claim.first = buff_reply[0];
        for (int i = 1; i < buff_reply.size(); i++)
        {
            eval_claim.second.push_back(buff_reply[i]);
        }
    }

    // printf("Final prod len : %d\n",transcript[depth-1].size());

    F sum = eval_claim.first; // evaluate_vector(transcript[depth-1],r);
    r = eval_claim.second;
    vector<F> r1, r2;
    for (int i = 0; i < multree_offset; i++)
    {
        r2.push_back(r[i]);
    }
    for (int i = multree_offset; i < (int)log2(N) + multree_offset; i++)
    {
        r1.push_back(r[i]);
    }
    for (int i = multree_offset + r1.size(); i < r.size(); i++)
    {
        r2.push_back(r[i]);
    }
    int prev_size = 1 << r2.size();

    vector<F> eval;
    if (commitment_data.size())
        eval = compute_eval_claims(commitment_data, r2, r1, sum, N);

    vector<int> sizes;
    for (int i = 0; i < commitment_data.size(); i++)
        sizes.push_back(commitment_data[i].size());
    sizes.push_back(prev_size);
    if (commitment_data.size())
        sum = compute_new_sum(eval, sizes, r2);
    double smc_time = smch_timer.get_time();
    for (int i = layer_offset - 1; i >= 0; i--)
    {
        pt_cp.start();
        smch_timer.start();
        sumcheck_offset = 4;

        pt_cp.end();
        vector<pair<F, vector<F>>> claims;
        // if(rank == 0)printf("%d,%d,%d,%d\n",batched_in1.size(),i,batched_in1[i].size(),r2.size());
        if ((r2.size() < sumcheck_offset + 1) || isLAN)
        {
            vector<F> beta1, beta2;
            pt_cp.start();
            precompute_beta(r2, beta2);
            precompute_beta(r1, beta1);
            pt_cp.end();
            claims = _cubic_sumcheck(sum, batched_in1[i], batched_in2[i], beta2, beta1, N);
        }
        else
        {
            sumcheck_offset = r2.size() / 2;
            vector<F> beta1, beta2, beta3, r21, r22;
            for (int j = 0; j < r2.size() - sumcheck_offset; j++)
                r21.push_back(r2[j]);
            for (int j = r2.size() - sumcheck_offset; j < r2.size(); j++)
                r22.push_back(r2[j]);
            pt_cp.start();
            precompute_beta(r21, beta2);
            precompute_beta(r22, beta3);
            precompute_beta(r1, beta1);
            pt_cp.end();
            claims = _cubic_sumcheck_sparrow(sum, batched_in1[i], batched_in2[i], beta2, beta3, beta1, N);
        }
        smch_timer.end();
        if (rank == 0)
            printf("          Round %d: %lf, %lf\n", i, smch_timer.get_time(), sch_com.get_time());

        pt_cp.start();

        F new_rand = hash_to_field(claims[0].second);
        r = claims[0].second;
        r.insert(r.begin(), new_rand);
        sum = (F(1) - new_rand) * claims[0].first + new_rand * claims[1].first;
        r1.clear();
        r2.clear();
        for (int j = 0; j < r.size() - (int)log2(N); j++)
        {
            r2.push_back(r[j]);
        }
        for (int j = r2.size(); j < r.size(); j++)
        {
            r1.push_back(r[j]);
        }
        pt_cp.end();
    }
    if (rank == 0)
        printf("          Step 2: %lf, %lf,%lf\n", smch_timer.get_time() - smc_time, pt_cp.get_time() - temp_time, sch_com.get_time());
    pt_cp.start();

    vector<vector<F>> eval_points(3);

    for (int i = 0; i < r.size() - 1 - (int)log2(N) - (int)log2(vectors); i++)
    {
        eval_points[0].push_back(r[i]);
    }
    for (int i = eval_points[0].size(); i < r.size() - 1 - (int)log2(N); i++)
    {
        eval_points[1].push_back(r[i]);
    }
    for (int i = r.size() - (int)log2(N); i < r.size(); i++)
    {
        eval_points[2].push_back(r[i]);
    }
    pt_cp.end();
    if (commitment_data.size())
        sum -= r[r.size() - (int)log2(N) - 1] * eval_commitment_data(eval_points, commitment_data, N);

    return make_pair(sum, eval_points);
}

pair<F, vector<vector<F>>> prove_product(vector<vector<F>> &input, vector<F> &output, F y, vector<F> r, int N)
{

    double temp_time = pt_cp.get_time();
    pt_cp.start();
    timer smch_timer;
    smch_timer.start();
    size_t vectors = input.size();
    int depth = (int)log2(next_pow2(input[0].size())) - multree_offset;
    size_t size = input[0].size();
    for (size_t i = 0; i < input.size(); i++)
    {
        if (input[i].size() != size)
        {
            printf("Error in mul tree sumcheck, no equal size vectors %zu,%zu\n", input[i].size(), size);
            exit(-1);
        }
        input[i].resize(1 << depth, F(1));
    }

    if (vectors != 1 << ((int)log2(vectors)))
    {
        vectors = 1 << ((int)log2(vectors) + 1);
        size_t old_vector_size = input.size();
        vector<F> temp_vector(1 << depth, F(0));
        for (size_t i = 0; i < vectors - old_vector_size; i++)
        {
            input.push_back(temp_vector);
        }
    }
    // Initialize total input
    size_t total_input_size = (size_t)vectors * size;

    // printf("total input %d\n",total_input_size );
    //  todo: this can be optimized (redundant total_input)
    vector<F> total_input(total_input_size);
    size_t counter = 0;
    for (size_t j = 0; j < input.size(); j++)
    {
        for (size_t i = 0; i < input[0].size(); i++)
        {
            total_input[counter] = input[j][i];
            counter++;
        }
    }

    vector<vector<F>> transcript(depth);
    vector<vector<F>> in1(depth), in2(depth);
    for (int i = 0; i < depth; i++)
    {
        transcript[i].resize(total_input_size / ((size_t)1 << (i + 1)));
        in1[i].resize(total_input_size / ((size_t)1 << (i + 1)));
        in2[i].resize(total_input_size / ((size_t)1 << (i + 1)));
    }

    counter = 0;
    for (size_t i = 0; i < total_input_size / 2; i++)
    {
        transcript[0][counter] = total_input[2 * i] * total_input[2 * i + 1];
        in1[0][counter] = total_input[2 * i];
        in2[0][counter] = total_input[2 * i + 1];
        counter++;
    }

    size_t len = total_input_size / 2;
    for (int i = 1; i < transcript.size(); i++)
    {
        counter = 0;
        for (size_t j = 0; j < len / 2; j++)
        {
            transcript[i][counter] = transcript[i - 1][2 * j] * transcript[i - 1][2 * j + 1];
            in1[i][counter] = transcript[i - 1][2 * j];
            in2[i][counter] = transcript[i - 1][2 * j + 1];
            counter++;
        }
        len = len / 2;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    // Send outputs to the leader
    vector<u64> buff;
    pair<F, vector<F>> eval_claim;
    vector<F> buff_reply;
    pt_cp.end();
    com_rounds += 2;

    if (rank == 0)
    {
        size_t ctr = 0;
        sch_com.start();
        vector<vector<F>> local_input(vectors);
        for (size_t i = 0; i < local_input.size(); i++)
        {
            local_input[i].resize(N * (transcript[depth - 1].size()) / vectors);
            for (size_t j = 0; j < transcript[depth - 1].size() / vectors; j++)
                local_input[i][j] = transcript[depth - 1][ctr++];

            // local_input[i][0] = transcript[depth-1][i];
        }
        // buff.resize(2*transcript[depth-1].size());
        vector<vector<u64>> recv_data(N - 1);
        vector<MPI_Request> req(N - 1);
        for (int i = 0; i < N - 1; i++)
        {
            recv_data[i].resize(2 * transcript[depth - 1].size());
            MPI_Irecv(recv_data[i].data(), recv_data[i].size(), MPI_UINT64_T, i + 1, 0, MPI_COMM_WORLD, &req[i]);
        }

        vector<F> _buff;
        for (int i = 0; i < N - 1; i++)
        {
            ctr = 0;
            network_timer.start();
            MPI_Wait(&req[i], MPI_STATUS_IGNORE);
            network_timer.end();
            // MPI_Recv(buff.data(),buff.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            field_vector_deserialize(recv_data[i], _buff);
            for (size_t j = 0; j < vectors; j++)
            {
                for (size_t k = 0; k < _buff.size() / vectors; k++)
                {
                    // if(j == 0) printf("%d,%d\n",i*_buff.size()/vectors + k,vectors);
                    local_input[j][(i + 1) * _buff.size() / vectors + k] = _buff[ctr++];
                }
            }
            // for(int j = 0; j < buff.size()/2; j++){
            //     local_input[j][i].real = buff[2*j];
            //     local_input[j][i].img = buff[2*j+1];
            // }
        }
        sch_com.end();

        pt_cp.start();
        printf("Size : %zu\n", local_input.size() * local_input[0].size());
        eval_claim = prove_multiplication_tree_new(local_input, output, F(0), y, {});
        buff_reply.push_back(eval_claim.first);

        buff_reply.insert(buff_reply.end(), eval_claim.second.begin(), eval_claim.second.end());

        field_vector_serialize(buff_reply, buff);

        pt_cp.end();
    }
    else
    {

        field_vector_serialize(transcript[depth - 1], buff);
        cm += 8 * buff.size() / 1024.0;
        MPI_Request req;
        MPI_Isend(buff.data(), buff.size(), MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, &req);
        network_timer.start();
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        network_timer.end();
        buff.clear();
        buff.resize(2 * (1 + log2(N * transcript[depth - 1].size())));
    }
    if (rank == 0)
        cm += (N - 1) * 8 * buff.size() / 1024.0;
    // printf("%d ?? %d\n",rank,buff.size());
    myBcast(buff, N);
    smch_timer.end();
    if (rank == 0)
        printf("          Step 1: %lf, %lf,%lf\n", smch_timer.get_time(), pt_cp.get_time() - temp_time, sch_com.get_time());
    temp_time = pt_cp.get_time();
    // MPI_Bcast(buff.data(),buff.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
    if (rank != 0)
    {
        field_vector_deserialize(buff, buff_reply);
        eval_claim.first = buff_reply[0];
        for (int i = 1; i < buff_reply.size(); i++)
        {
            eval_claim.second.push_back(buff_reply[i]);
        }
    }

    // printf("Final prod len : %d\n",transcript[depth-1].size());

    F sum = eval_claim.first; // evaluate_vector(transcript[depth-1],r);
    r = eval_claim.second;
    vector<F> r1, r2;
    for (int i = 0; i < multree_offset; i++)
    {
        r2.push_back(r[i]);
    }
    for (int i = multree_offset; i < (int)log2(N) + multree_offset; i++)
    {
        r1.push_back(r[i]);
    }
    for (int i = multree_offset + r1.size(); i < r.size(); i++)
    {
        r2.push_back(r[i]);
    }
    double smc_time = smch_timer.get_time();
    for (int i = depth - 1; i >= 0; i--)
    {
        pt_cp.start();
        smch_timer.start();

        pt_cp.end();
        vector<pair<F, vector<F>>> claims;

        if ((r2.size() < sumcheck_offset + 1) || isLAN)
        {
            vector<F> beta1, beta2;
            pt_cp.start();
            precompute_beta(r2, beta2);
            precompute_beta(r1, beta1);
            pt_cp.end();
            claims = _cubic_sumcheck(sum, in1[i], in2[i], beta2, beta1, N);
        }
        else
        {
            vector<F> beta1, beta2, beta3, r21, r22;
            for (int j = 0; j < r2.size() - sumcheck_offset; j++)
                r21.push_back(r2[j]);
            for (int j = r2.size() - sumcheck_offset; j < r2.size(); j++)
                r22.push_back(r2[j]);
            pt_cp.start();
            precompute_beta(r21, beta2);
            precompute_beta(r22, beta3);
            precompute_beta(r1, beta1);
            pt_cp.end();
            claims = _cubic_sumcheck_sparrow(sum, in1[i], in2[i], beta2, beta3, beta1, N);
        }
        smch_timer.end();
        if (rank == 0)
            printf("          Round %d: %lf, %lf\n", i, smch_timer.get_time(), sch_com.get_time());

        pt_cp.start();

        F new_rand = hash_to_field(claims[0].second);
        r = claims[0].second;
        r.insert(r.begin(), new_rand);
        sum = (F(1) - new_rand) * claims[0].first + new_rand * claims[1].first;
        r1.clear();
        r2.clear();
        for (int j = 0; j < r.size() - (int)log2(N); j++)
        {
            r2.push_back(r[j]);
        }
        for (int j = r2.size(); j < r.size(); j++)
        {
            r1.push_back(r[j]);
        }
        pt_cp.end();
    }
    if (rank == 0)
        printf("          Step 2: %lf, %lf,%lf\n", smch_timer.get_time() - smc_time, pt_cp.get_time() - temp_time, sch_com.get_time());
    pt_cp.start();

    vector<vector<F>> eval_points(3);

    for (int i = 0; i < r.size() - (int)log2(N) - (int)log2(vectors); i++)
    {
        eval_points[0].push_back(r[i]);
    }
    for (int i = eval_points[0].size(); i < r.size() - (int)log2(N); i++)
    {
        eval_points[1].push_back(r[i]);
    }
    for (int i = r.size() - (int)log2(N); i < r.size(); i++)
    {
        eval_points[2].push_back(r[i]);
    }
    pt_cp.end();

    return make_pair(sum, eval_points);
}

// New version of multiplication tree prover that takes as input vectors of different size
pair<F, vector<vector<F>>> prove_product_opt_new(vector<vector<F>> &input, vector<F> &output, int N, vector<F> &evals)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    size_t total_size = 0;
    size_t size;
    pt_cp.start();

    for (size_t i = 0; i < input.size(); i++)
    {
        if (i > 0 && input[i].size() > input[i - 1].size())
        {
            printf("Input is not sorted, exiting \n");
            if (rank == 0)
                for (int j = 0; j < input.size(); j++)
                    printf("%d\n", input[j].size());
            exit(-1);
        }
        input[i].resize(next_pow2(input[i].size()), F(1));
        total_size += input[i].size();
        size = input[i].size();
    }

    vector<vector<F>> new_input;
    vector<F> buff(size);
    for (size_t i = 0; i < input.size(); i++)
    {
        size_t ctr = 0;
        for (size_t j = 0; j < input[i].size() / size; j++)
        {
            for (size_t k = 0; k < size; k++)
                buff[k] = input[i][ctr++];
            new_input.push_back(buff);
        }
    }
    pt_cp.end();
    vector<F> temp_out;
    pair<F, vector<vector<F>>> claim = prove_product_new(new_input, temp_out, F(0), {}, N);
    size_t ctr = 0;
    if (rank == 0)
    {
        output.resize(input.size(), F(1));
        for (size_t i = 0; i < input.size(); i++)
        {
            for (size_t j = 0; j < input[i].size() / size; j++)
                output[i] *= temp_out[ctr++];
        }
    }
    vector<F> r = claim.second[0];
    r.insert(r.end(), claim.second[1].begin(), claim.second[1].end());
    // evals = batch_distributed_eval_opt(input, r, claim.second[2], N);

    return claim;
}

// New version of multiplication tree prover that takes as input vectors of different size
pair<F, vector<vector<F>>> prove_product_opt(vector<vector<F>> &input, vector<F> &output, int N, vector<F> &evals)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    size_t total_size = 0;
    size_t size;
    pt_cp.start();

    for (size_t i = 0; i < input.size(); i++)
    {
        if (i > 0 && input[i].size() > input[i - 1].size())
        {
            printf("Input is not sorted, exiting \n");
            if (rank == 0)
                for (int j = 0; j < input.size(); j++)
                    printf("%d\n", input[j].size());
            exit(-1);
        }
        input[i].resize(next_pow2(input[i].size()), F(1));
        total_size += input[i].size();
        size = input[i].size();
    }

    vector<vector<F>> new_input;
    vector<F> buff(size);
    for (size_t i = 0; i < input.size(); i++)
    {
        size_t ctr = 0;
        for (size_t j = 0; j < input[i].size() / size; j++)
        {
            for (size_t k = 0; k < size; k++)
                buff[k] = input[i][ctr++];
            new_input.push_back(buff);
        }
    }
    pt_cp.end();
    vector<F> temp_out;
    pair<F, vector<vector<F>>> claim = prove_product(new_input, temp_out, F(0), {}, N);
    size_t ctr = 0;
    if (rank == 0)
    {
        output.resize(input.size(), F(1));
        for (size_t i = 0; i < input.size(); i++)
        {
            for (size_t j = 0; j < input[i].size() / size; j++)
                output[i] *= temp_out[ctr++];
        }
    }
    vector<F> r = claim.second[0];
    r.insert(r.end(), claim.second[1].begin(), claim.second[1].end());
    // evals = batch_distributed_eval_opt(input, r, claim.second[2], N);

    return claim;
}

void _reduce_R1CS_matrixes(size_t size, vector<F> r, vector<F> &RA, vector<F> &RB, vector<F> &RC, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

    vector<F> r1, r2, beta1, beta2;
    for (int i = 0; i < r.size() - (int)log2(N); i++)
    {
        r1.push_back(r[i]);
    }
    for (int i = r.size() - (int)log2(N); i < r.size(); i++)
    {
        r2.push_back(r[i]);
    }

    precompute_beta(r1, beta1);
    precompute_beta(r2, beta2);

    int M = (1 << logn) / N;
    RA.resize(M, F(0));
    RB.resize(M, F(0));
    RC.resize(M, F(0));

    for (int i = 0; i < pA.size(); i++)
    {
        if (pA[i].size())
        {
            // if(rank == 1) printf("%d,(%d,%d),(%d,%d,%d)\n",rank,pA[i][0].first,pA[i][0].first%RA.size(),pA[i][0].second,pA[i][0].second/(pA.size()/2),pA[i][0].second%(pA.size()/2));
            for (int j = 0; j < pA[i].size(); j++)
            {
                RA[pA[i][j].first % RA.size()] += beta1[pA[i][j].second % ((1 << logm) / N)] * beta2[pA[i][j].second / ((1 << logm) / N)];
            }
        }
        if (pB[i].size())
        {
            // printf("> %d, %d\n",rank,pB[i][0].second);
            for (int j = 0; j < pB[i].size(); j++)
            {
                RB[pB[i][j].first % RB.size()] += beta1[pB[i][j].second % ((1 << logm) / N)] * beta2[pB[i][j].second / ((1 << logm) / N)];
            }
        }
        if (pC[i].size())
        {
            for (int j = 0; j < pC[i].size(); j++)
            {
                RC[pC[i][j].first % RC.size()] += beta1[pC[i][j].second % ((1 << logm) / N)] * beta2[pC[i][j].second / ((1 << logm) / N)];
            }
        }
    }
}

void _reduce_R1CS_matrixes_multivariate(size_t size, vector<F> r, vector<F> &RA, vector<F> &RB, vector<F> &RC, vector<int> degrees, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

    vector<F> r1, r2, beta1, beta2;
    /*
    for(int i = 0; i < r.size() - (int)log2(N); i++){
        r1.push_back(r[i]);
    }
    for(int i = r.size() - (int)log2(N); i < r.size(); i++){
        r2.push_back(r[i]);
    }
    return;
    */

    degrees.insert(degrees.begin(), N / 4);
    degrees.pop_back();

    // precompute_beta(r1,beta1);precompute_beta(r2,beta2);
    beta1 = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees[0])), r[0], degrees[0]);
    for (int i = 1; i < degrees.size() - 1; i++)
    {
        vector<F> buff(degrees[i] * beta1.size());
        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees[i])), r[i], degrees[i]);
        for (int k = 0; k < L.size(); k++)
        {
            for (int j = 0; j < buff.size() / L.size(); j++)
            {
                buff[j + k * buff.size() / L.size()] = L[k] * beta1[j];
            }
        }
        beta1 = buff;
    }
    beta2 = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees[r.size() - 1])), r[r.size() - 1], degrees[r.size() - 1]);
    int M = (1 << logn) / N;
    RA.resize(M, F(0));
    RB.resize(M, F(0));
    RC.resize(M, F(0));
    for (int i = 0; i < pA.size(); i++)
    {
        if (pA[i].size())
        {
            // if(rank == 1) printf("%d,(%d,%d),(%d,%d,%d)\n",rank,pA[i][0].first,pA[i][0].first%RA.size(),pA[i][0].second,pA[i][0].second/(pA.size()/2),pA[i][0].second%(pA.size()/2));
            for (int j = 0; j < pA[i].size(); j++)
            {
                // if(rank == 0) printf("%d,%d\n",pA[i][j].first,pA[i][j].second);
                RA[pA[i][j].first % RA.size()] += beta1[pA[i][j].second % ((1 << logm) / beta2.size())] * beta2[pA[i][j].second / ((1 << logm) / beta2.size())];
            }
        }
        if (pB[i].size())
        {
            // printf("> %d, %d\n",rank,pB[i][0].second);
            for (int j = 0; j < pB[i].size(); j++)
            {

                RB[pB[i][j].first % RB.size()] += beta1[pB[i][j].second % ((1 << logm) / beta2.size())] * beta2[pB[i][j].second / ((1 << logm) / beta2.size())];
            }
        }
        if (pC[i].size())
        {
            for (int j = 0; j < pC[i].size(); j++)
            {
                RC[pC[i][j].first % RC.size()] += beta1[pC[i][j].second % ((1 << logm) / beta2.size())] * beta2[pC[i][j].second / ((1 << logm) / beta2.size())];
            }
        }
    }
}

void secret_share_vector(vector<F> &v, int _k, int k, int N)
{
    int ctr = 0;
    pt_cp.start();

    vector<vector<F>> data(N);
    for (int i = 0; i < data.size(); i++)
    {
        data[i].resize(v.size() / k);
    }

    for (int i = 0; i < v.size() / k; i++)
    {
        vector<F> buff(_k);
        for (int j = 0; j < k; j++)
        {
            buff[j] = v[ctr++];
        }
        for (int j = k; j < _k; j++)
        {
            buff[j] = 0;
        }
        fft(buff, (int)log2(buff.size()), true);
        buff.resize(2 * N, F(0));
        fft(buff, (int)log2(buff.size()), false);
        for (int j = 0; j < N; j++)
        {
            data[j][i] = buff[2 * j + 1];
        }
    }

    vector<u64> buff_u64, buff_recv_u64;
    vector<F> buff = convert2vector(data);
    field_vector_serialize(buff, buff_u64);
    buff_recv_u64.resize(buff_u64.size(), (0));
    pt_cp.end();
    if (!data_parallel)
        cm += 8 * buff_u64.size() / 1024.0;
    com_rounds++;

    MPI_Alltoall(buff_u64.data(), buff_u64.size() / N, MPI_UINT64_T, buff_recv_u64.data(), buff_u64.size() / N, MPI_UINT64_T, MPI_COMM_WORLD);

    pt_cp.start();

    field_vector_deserialize(buff_recv_u64, v);
    pt_cp.end();
}

void compute_tensor_lagrange(vector<F> r, vector<F> r1, vector<F> r2, vector<int> degree, vector<F> &beta1, vector<F> &beta2)
{
    beta1 = compute_lagrange_coeff(getRootOfUnity((int)log2(degree[0])), r1[0], degree[0]);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

    for (int i = 1; i < r1.size(); i++)
    {
        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(degree[i])), r1[i], degree[i]);
        vector<F> buff(beta1.size() * L.size());
        for (int j = 0; j < L.size(); j++)
        {
            for (int k = 0; k < beta1.size(); k++)
            {
                buff[k + j * beta1.size()] = beta1[k] * L[j];
            }
        }
        beta1 = buff;
    }

    beta2 = compute_lagrange_coeff(getRootOfUnity((int)log2(degree[r1.size()])), r2[0], degree[r1.size()]);

    for (int i = r1.size() + 1; i < r.size(); i++)
    {

        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(degree[i])), r[i], degree[i]);
        vector<F> buff(beta2.size() * L.size());
        for (int j = 0; j < L.size(); j++)
        {
            for (int k = 0; k < beta2.size(); k++)
            {
                buff[k + j * beta2.size()] = beta2[k] * L[j];
            }
        }
        beta2 = buff;
    }
}

vector<F> _beta1, _beta2;

void compute_R1CS_betas_multivariate(vector<F> r1, vector<F> r2, vector<sparse_eval_data> &data, vector<sparse_eval_data> &data_even_odd,
                                     vector<vector<F>> &beta1, vector<vector<F>> &beta2,
                                     vector<vector<F>> &beta1_even_odd, vector<vector<F>> &beta2_even_odd, vector<int> degrees1, vector<int> degrees2,
                                     int logm, int logn, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

    // Create a map for masking

    int ctr = 0;
    int segment_size = (1 << (logn - 4));
    int segment_size2 = (1 << (logn - 4)) - (N / 4) * mask_l;

    degrees1.insert(degrees1.begin(), N / 4);
    degrees2.insert(degrees2.begin(), N / 4);
    // degrees1.pop_back();
    // r1.pop_back();

    assert(degrees1.size() == r1.size());
    assert(degrees2.size() == r2.size());

    /*
    int eval1 = 1;
    for(int i = 0; i < degrees2.size(); i++){
        eval1 *= degrees2[i];
    }

    if(1<<logn != eval1){
        printf("ERROR %d,%d\n",1<<logn, eval1);
        exit(-1);
    }*/

    vector<F> r11, r12, r21, r22;
    for (int i = 0; i < r1.size() / 2 + 1; i++)
    {
        r11.push_back(r1[i]);
    }
    for (int i = r1.size() / 2 + 1; i < r1.size(); i++)
    {
        r12.push_back(r1[i]);
    }
    for (int i = 0; i < r2.size() / 2 + 1; i++)
    {
        r21.push_back(r2[i]);
    }
    for (int i = r2.size() / 2 + 1; i < r2.size(); i++)
    {
        r22.push_back(r2[i]);
    }

    vector<F> beta11, beta12, beta21, beta22;

    compute_tensor_lagrange(r1, r11, r12, degrees1, beta11, beta12);

    compute_tensor_lagrange(r2, r21, r22, degrees2, beta21, beta22);
    _beta1.resize(beta11.size() * beta12.size() / N);
    _beta2.resize(beta21.size() * beta22.size() / N);

    vector<F> b = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees1[0])), r1[0], degrees1[0]);
    for (int i = 1; i < r1.size(); i++)
    {
        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees1[i])), r1[i], degrees1[i]);
        vector<F> buff(L.size() * b.size());
        for (int j = 0; j < L.size(); j++)
        {
            for (int k = 0; k < b.size(); k++)
            {
                buff[b.size() * j + k] = b[k] * L[j];
            }
        }
        b = buff;
    }
    for (int i = 0; i < beta12.size(); i++)
    {
        for (int j = 0; j < beta11.size(); j++)
        {
            if (b[i * beta11.size() + j] != beta12[i] * beta11[j])
            {
                printf("ERROR\n");
                exit(-1);
            }
        }
    }

    int mask1 = 0;
    int mask2 = 0;
    for (int i = 0; i < (int)log2(beta11.size()); i++)
    {
        mask1 = mask1 << 1;
        mask1 += 1;
    }

    for (int i = 0; i < (int)log2(beta21.size()); i++)
    {
        mask2 = mask2 << 1;
        mask2 += 1;
    }
    // precompute_beta(r11,beta11);
    // precompute_beta(r12,beta12);
    // precompute_beta(r21,beta21);
    // precompute_beta(r22,beta22);

    ctr = 0;

    // vector<F> b; precompute_beta(r2,b);
    int mask11 = (int)log2(beta11.size());
    int mask21 = (int)log2(beta21.size());

    for (int i = 0; i < _beta1.size(); i++)
    {
        int idx = rank * _beta1.size() + i;
        int idx11 = (idx)&mask1;
        int idx12 = idx >> (mask11);
        _beta1[i] = beta11[idx11] * beta12[idx12];
    }
    for (int i = 0; i < _beta2.size(); i++)
    {
        int idx = rank * _beta2.size() + i;
        int idx11 = (idx)&mask2;
        int idx12 = idx >> (mask21);
        _beta2[i] = beta21[idx11] * beta22[idx12];
    }

    for (int i = 0; i < data.size(); i++)
    {

        beta1[i].resize(data[i].IDX1.size(), F(0));
        beta1_even_odd[i].resize(data[i].IDX1.size(), F(0));
        for (int j = 0; j < data[i].IDX1.size(); j++)
        {
            int idx1 = data[i].IDX1[j];
            int idx11 = idx1 & mask1;
            int idx12 = idx1 >> (mask11);
            beta1[i][j] = beta11[idx11] * beta12[idx12];
            // ctr++;
        }

        for (int j = 0; j < data[i].IDX1.size(); j++)
        {
            int idx1 = data_even_odd[i].IDX1[j];
            int idx11 = idx1 & mask1;
            int idx12 = idx1 >> (mask11);
            beta1_even_odd[i][j] = beta11[idx11] * beta12[idx12];
            // ctr++;
        }

        beta2[i].resize(data[i].IDX2.size(), F(0));
        beta2_even_odd[i].resize(data[i].IDX2.size(), F(0));
        for (int j = 0; j < data[i].IDX2.size(); j++)
        {
            int idx2 = segment_size * (int)(data[i].IDX2[j] / segment_size2) + (data[i].IDX2[j] % segment_size2);
            // int idx2 = map[data[i].IDX2[j]];
            // if(rank == 0) printf("%d,%d\n",data[i].IDX2[j],map[data[i].IDX2[j]]);
            int idx21 = idx2 & mask2;
            int idx22 = idx2 >> (mask21);
            if (beta21.size() <= idx21 || beta22.size() <= idx22)
            {
                printf("Error\n");
                exit(-1);
            }
            beta2[i][j] = beta21[idx21] * beta22[idx22];
            // ctr++;
        }
        for (int j = 0; j < data[i].IDX2.size(); j++)
        {
            int idx2 = segment_size * (int)(data_even_odd[i].IDX2[j] / segment_size2) + (data_even_odd[i].IDX2[j] % segment_size2);
            // int idx2 = map[data[i].IDX2[j]];
            // if(rank == 0) printf("%d,%d\n",data[i].IDX2[j],map[data[i].IDX2[j]]);
            int idx21 = idx2 & mask2;
            int idx22 = idx2 >> (mask21);
            if (beta21.size() <= idx21 || beta22.size() <= idx22)
            {
                printf("Error\n");
                exit(-1);
            }
            beta2_even_odd[i][j] = beta21[idx21] * beta22[idx22];
            // ctr++;
        }
    }
}

void compute_R1CS_betas(vector<F> r1, vector<F> r2, vector<sparse_eval_data> &data, vector<sparse_eval_data> &data_even_odd,
                        vector<vector<F>> &beta1, vector<vector<F>> &beta2,
                        vector<vector<F>> &beta1_even_odd, vector<vector<F>> &beta2_even_odd,
                        int logm, int logn, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

    // Create a map for masking

    int ctr = 0;
    int segment_size = (1 << (logn - 4));
    int segment_size2 = (1 << (logn - 4)) - (N / 4) * mask_l;

    vector<F> r11, r12, r21, r22;
    for (int i = 0; i < r1.size() / 2; i++)
    {
        r11.push_back(r1[i]);
    }
    for (int i = r1.size() / 2; i < r1.size(); i++)
    {
        r12.push_back(r1[i]);
    }
    for (int i = 0; i < r2.size() / 2; i++)
    {
        r21.push_back(r2[i]);
    }
    for (int i = r2.size() / 2; i < r2.size(); i++)
    {
        r22.push_back(r2[i]);
    }
    int mask1 = 0;
    int mask2 = 0;
    for (int i = 0; i < logm / 2; i++)
    {
        mask1 = mask1 << 1;
        mask1 += 1;
    }

    for (int i = 0; i < logn / 2; i++)
    {
        mask2 = mask2 << 1;
        mask2 += 1;
    }

    vector<F> beta11, beta12, beta21, beta22;

    precompute_beta(r11, beta11);
    precompute_beta(r12, beta12);
    precompute_beta(r21, beta21);
    precompute_beta(r22, beta22);

    ctr = 0;

    vector<F> b;
    precompute_beta(r2, b);
    for (int i = 0; i < data.size(); i++)
    {

        beta1[i].resize(data[i].IDX1.size(), F(0));
        beta1_even_odd[i].resize(data[i].IDX1.size(), F(0));
        for (int j = 0; j < data[i].IDX1.size(); j++)
        {
            int idx1 = data[i].IDX1[j];
            int idx11 = idx1 & mask1;
            int idx12 = idx1 >> (logm / 2);
            beta1[i][j] = beta11[idx11] * beta12[idx12];
            // ctr++;
        }

        for (int j = 0; j < data[i].IDX1.size(); j++)
        {
            int idx1 = data_even_odd[i].IDX1[j];
            int idx11 = idx1 & mask1;
            int idx12 = idx1 >> (logm / 2);
            beta1_even_odd[i][j] = beta11[idx11] * beta12[idx12];
            // ctr++;
        }

        beta2[i].resize(data[i].IDX2.size(), F(0));
        beta2_even_odd[i].resize(data[i].IDX2.size(), F(0));
        for (int j = 0; j < data[i].IDX2.size(); j++)
        {
            int idx2 = segment_size * (int)(data[i].IDX2[j] / segment_size2) + (data[i].IDX2[j] % segment_size2);
            // int idx2 = map[data[i].IDX2[j]];
            // if(rank == 0) printf("%d,%d\n",data[i].IDX2[j],map[data[i].IDX2[j]]);
            int idx21 = idx2 & mask2;
            int idx22 = idx2 >> (logn / 2);
            if (beta21.size() <= idx21 || beta22.size() <= idx22)
            {
                printf("Error\n");
                exit(-1);
            }
            beta2[i][j] = beta21[idx21] * beta22[idx22];
            // ctr++;
        }
        for (int j = 0; j < data[i].IDX2.size(); j++)
        {
            int idx2 = segment_size * (int)(data_even_odd[i].IDX2[j] / segment_size2) + (data_even_odd[i].IDX2[j] % segment_size2);
            // int idx2 = map[data[i].IDX2[j]];
            // if(rank == 0) printf("%d,%d\n",data[i].IDX2[j],map[data[i].IDX2[j]]);
            int idx21 = idx2 & mask2;
            int idx22 = idx2 >> (logn / 2);
            if (beta21.size() <= idx21 || beta22.size() <= idx22)
            {
                printf("Error\n");
                exit(-1);
            }
            beta2_even_odd[i][j] = beta21[idx21] * beta22[idx22];
            // ctr++;
        }
    }
}

void compute_base_betas(vector<F> r1, vector<F> r2, vector<F> &base_beta1, vector<F> &base_beta2, int N)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    vector<F> r11, r12, r21, r22;

    /*
    for(int i  = 0; i < r1.size()-(int)log2(N); i++){
        r12.push_back(r1[i]);
    }
    for(int i = r12.size(); i < r1.size(); i++){
        r11.push_back(r1[i]);
    }

    for(int i  = 0; i < r2.size()-(int)log2(N); i++){
        r22.push_back(r2[i]);
    }
    for(int i = r22.size(); i < r2.size(); i++){
        r21.push_back(r2[i]);
    }
    */

    // precompute_beta(r12,base_beta1);
    // precompute_beta(r22,base_beta2);
    F b1 = compute_lagrange_coeff(getRootOfUnity((int)log2(N)), r1[0], N)[rank]; // _beta(rank,  r11);
    F b2 = compute_lagrange_coeff(getRootOfUnity((int)log2(N)), r2[0], N)[rank]; //_beta(rank,  r21);
    r1.erase(r1.begin());
    r2.erase(r2.begin());
    // base_beta1 =

    for (int i = 0; i < base_beta1.size(); i++)
    {
        base_beta1[i] = b1 * base_beta1[i];
    }
    for (int i = 0; i < base_beta2.size(); i++)
    {
        base_beta2[i] = b2 * base_beta2[i];
    }
}

vector<pair<F, vector<F>>> distributed_accumulation(vector<vector<F>> &polys, vector<vector<vector<F>>> eval_points, vector<F> evals, int N)
{
    int rank, ctr = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    pt_cp.start();
    vector<F> v1, v2, b;
    vector<F> r;
    F y;
    r.push_back(hash_to_field(evals));
    y = r[0] * evals[0];
    for (int i = 1; i < polys.size(); i++)
    {
        r.push_back(hash_to_field({r[i - 1]}));
        y += r[i] * evals[i];
    }

    for (int i = 0; i < polys.size(); i++)
    {
        v1.insert(v1.end(), polys[i].begin(), polys[i].end());
    }
    v2.resize(v1.size());
    precompute_beta(eval_points[0][0], b);
    F _b = _beta(rank, eval_points[0][2]);
    for (int i = 0; i < b.size(); i++)
    {
        b[i] = _b * b[i];
    }
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < polys[i].size(); j++)
        {
            v2[ctr] = r[i] * b[j];
            ctr++;
        }
    }
    b.clear();
    precompute_beta(eval_points[1][0], b);
    _b = _beta(rank, eval_points[1][2]);
    for (int i = 0; i < b.size(); i++)
    {
        b[i] = _b * b[i];
    }
    for (int i = 3; i < 15; i++)
    {
        for (int j = 0; j < polys[i].size(); j++)
        {
            v2[ctr] = r[i] * b[j];
            ctr++;
        }
    }
    b.clear();
    precompute_beta(eval_points[2][0], b);
    _b = _beta(rank, eval_points[2][2]);
    for (int i = 0; i < b.size(); i++)
    {
        b[i] = _b * b[i];
    }
    for (int i = 15; i < 18; i++)
    {
        for (int j = 0; j < polys[i].size(); j++)
        {
            v2[ctr] = r[i] * b[j];
            ctr++;
        }
    }
    v1.resize(next_pow2(v1.size()), F(0));
    v2.resize(next_pow2(v2.size()), F(0));
    pt_cp.end();

    vector<pair<F, vector<F>>> claim = _quadratic_sumcheck(y, v1, v2, N);
    pt_cp.start();

    if (rank == 0)
    {
        vector<F> claimed_r1, claimed_r2;
        for (int i = 0; i < (int)log2(polys[0].size()); i++)
        {
            claimed_r1.push_back(claim[0].second[i]);
        }
        for (int i = 0; i < (int)log2(N); i++)
        {
            claimed_r1.push_back(claim[0].second[claim[0].second.size() - (int)log2(N) + i]);
        }
        for (int i = (int)log2(polys[0].size()); i < claim[0].second.size() - (int)log2(N); i++)
        {
            claimed_r1.push_back(claim[0].second[i]);
        }

        for (int i = 0; i < (int)log2(polys[3].size()); i++)
        {
            claimed_r2.push_back(claim[0].second[i]);
        }
        for (int i = 0; i < (int)log2(N); i++)
        {
            claimed_r2.push_back(claim[0].second[claim[0].second.size() - (int)log2(N) + i]);
        }
        for (int i = (int)log2(polys[3].size()); i < claim[0].second.size() - (int)log2(N); i++)
        {
            claimed_r2.push_back(claim[0].second[i]);
        }

        vector<F> partial_poly(32, F(0));

        vector<F> _r = eval_points[0][0];
        _r.insert(_r.end(), eval_points[0][2].begin(), eval_points[0][2].end());
        _b = beta_identity(_r, claimed_r1);
        for (int i = 0; i < 3; i++)
        {
            partial_poly[i] = r[i] * _b;
        }
        _r = eval_points[1][0];
        _r.insert(_r.end(), eval_points[1][2].begin(), eval_points[1][2].end());
        _b = beta_identity(_r, claimed_r2);
        for (int i = 3; i < 15; i++)
        {
            partial_poly[i] = r[i] * _b;
        }
        _r = eval_points[2][0];
        _r.insert(_r.end(), eval_points[2][2].begin(), eval_points[2][2].end());
        _b = beta_identity(_r, claimed_r2);
        for (int i = 15; i < 18; i++)
        {
            partial_poly[i] = r[i] * _b;
        }

        int pos = 0;
        for (int i = 0; i < 3; i++)
        {
            F prod = get_offset_product(N * polys[i].size(), pos, claimed_r1);
            pos += N * polys[i].size();
            partial_poly[i] *= prod;
        }
        for (int i = 3; i < 18; i++)
        {
            F prod = get_offset_product(N * polys[i].size(), pos, claimed_r2);
            pos += N * polys[i].size();
            partial_poly[i] *= prod;
        }
        for (int i = 0; i < partial_poly.size(); i++)
        {
            claim[1].first -= partial_poly[i];
        }
        printf("%lld,%lld\n", claim[1].first.real, claim[1].first.img);
        if (F(0) != claim[1].first)
        {
            printf("Error in final eval\n");
        }
    }
    pt_cp.end();

    return claim;
}

void compute_transcript(vector<vector<F>> &Tr, vector<sparse_eval_data> &data, vector<F> challenges, vector<vector<F>> &beta1, vector<vector<F>> &beta2,
                        vector<F> r1, vector<F> r2, int N)
{
    int rank;
    vector<F> base_beta1, base_beta2;
    // compute_base_betas(r1,r2,base_beta1,base_beta2,N);
    base_beta1 = _beta1;
    base_beta2 = _beta2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    for (int j = 0; j < data.size(); j++)
    {
        Tr[2 * j].resize(next_pow2(data[j].IDX1.size()), F(1));
        Tr[2 * j + 1].resize(next_pow2(data[j].IDX1.size()), F(1));
        for (int i = 0; i < data[j].IDX1.size(); i++)
        {
            Tr[2 * j][i] = challenges[0] * beta1[j][i] + challenges[1] * F(data[j].RD1[i]) + challenges[2] * F(data[j].IDX1[i]) + F(1);
            Tr[2 * j + 1][i] = challenges[0] * beta1[j][i] + challenges[1] * F(data[j].WR1[i]) + challenges[2] * F(data[j].IDX1[i]) + F(1);
        }
    }
    for (int j = 0; j < data.size(); j++)
    {
        Tr[2 * j + 6].resize(next_pow2(data[j].IDX2.size()), F(1));
        Tr[2 * j + 1 + 6].resize(next_pow2(data[j].IDX2.size()), F(1));
        for (int i = 0; i < data[j].IDX2.size(); i++)
        {
            Tr[2 * j + 6][i] = challenges[0] * beta2[j][i] + challenges[1] * F(data[j].RD2[i]) + challenges[2] * F(data[j].IDX2[i]) + F(1);
            Tr[2 * j + 1 + 6][i] = challenges[0] * beta2[j][i] + challenges[1] * F(data[j].WR2[i]) + challenges[2] * F(data[j].IDX2[i]) + F(1);
        }
    }
    for (int j = 0; j < data.size(); j++)
    {
        Tr[2 * j + 12].resize(next_pow2(data[j].FINAL_FR1.size()), F(1));
        Tr[2 * j + 1 + 12].resize(next_pow2(data[j].FINAL_FR1.size()), F(1));
        // printf("%d,%d\n",data[j].FINAL_FR1.size(),_beta1.size());
        for (int i = 0; i < data[j].FINAL_FR1.size(); i++)
        {
            Tr[2 * j + 0 + 12][i] = challenges[0] * base_beta1[i] + F(1) + challenges[2] * F(rank * data[j].FINAL_FR1.size() + i);
            Tr[2 * j + 1 + 12][i] = challenges[0] * base_beta1[i] + challenges[1] * F(data[j].FINAL_FR1[i]) + challenges[2] * F(rank * data[j].FINAL_FR1.size() + i) + F(1);
        }
    }
    for (int j = 0; j < data.size(); j++)
    {
        Tr[2 * j + 18].resize(next_pow2(data[j].FINAL_FR2.size()), F(1));
        Tr[2 * j + 1 + 18].resize(next_pow2(data[j].FINAL_FR2.size()), F(1));
        // printf("%d,%d\n",data[j].FINAL_FR2.size(),_beta2.size());
        for (int i = 0; i < data[j].FINAL_FR2.size(); i++)
        {
            Tr[2 * j + 18][i] = challenges[2] * F(rank * data[j].FINAL_FR2.size() + i) + F(1) + challenges[0] * base_beta2[i];
            Tr[2 * j + 1 + 18][i] = challenges[0] * base_beta2[i] + challenges[1] * F(data[j].FINAL_FR2[i]) + challenges[2] * F(rank * data[j].FINAL_FR2.size() + i) + F(1);
        }
    }
}

void sort_transcript(vector<vector<F>> &Tr, vector<int> &order)
{
    // Create a vector of pairs to store (value, original_index)
    std::vector<std::pair<int, int>> sizes;
    for (int i = 0; i < Tr.size(); i++)
    {
        sizes.push_back(make_pair(Tr[i].size(), i));
    }

    // Sort the indexed_vector based on the values (first element of the pair)
    // By default, std::sort sorts pairs based on the first element, then the second if first elements are equal.
    std::sort(sizes.begin(), sizes.end(), [](const std::pair<int, int> &a, const std::pair<int, int> &b)
              {
                  return a.first > b.first; // For descending order
              });

    // reverse(sizes.begin(),sizes.end());
    order.resize(sizes.size());

    for (int i = 0; i < sizes.size(); i++)
        order[sizes[i].second] = i;

    vector<vector<F>> temp;
    for (int i = 0; i < sizes.size(); i++)
        temp.push_back(Tr[sizes[i].second]);
    Tr = temp;
}

double get_time()
{
    // Returns elapsed wall-clock time in seconds since some time in the past
    return MPI_Wtime();
}

pair<F, vector<F>> _prove_sparse_eval_opt(F y, F a, F b, F c, vector<vector<F>> &beta1, vector<vector<F>> &beta2, vector<sparse_eval_data> &data,
                                          vector<F> r1, vector<F> r2, int N, vector<F> &debug_data)
{

    double temp = pt_cp.get_time();
    timer t;
    t.start();
    pt_cp.start();
    vector<vector<F>> Tr(24);
    vector<F> output;
    vector<int> order;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    if (rank == 0)
        printf("Timestamp: %lf | Starting _prove_sparse_eval_opt", get_time());

    // pt_cp.start();
    vector<F> challenges(3);
    clock_t t1 = clock();
    for (int i = 0; i < 3; i++)
        challenges[i] = hash_to_field({0});

    compute_transcript(Tr, data, challenges, beta1, beta2, r1, r2, N);
    // vector<F> test_v = Tr[13];
    sort_transcript(Tr, order);
    vector<F> debug_evals;
    pt_cp.end();
    if (rank == 0)
        printf("Timestamp: %lf | Starting prove_product_opt_new", get_time());
    pair<F, vector<vector<F>>> claim = prove_product_opt_new(Tr, output, N, debug_evals);
    if (rank == 0)
        printf("Timestamp: %lf | Finished prove_product_opt_new", get_time());
    t.end();

    // if(rank == 0) printf("> %lf,%lf\n",t.get_time(),pt_cp.get_time()-temp);
    if (rank == 0)
    {
        vector<F> organized_output(output.size());
        for (int i = 0; i < output.size(); i++)
        {
            organized_output[i] = output[order[i]];
        }
        for (int i = 0; i < 6; i++)
        {
            if (organized_output[2 * i] * organized_output[2 * i + 1 + 12] != organized_output[2 * i + 1] * organized_output[2 * i + 12])
            {
                printf("Error phase 2 %d\n", i);
            }
        }
    }

    vector<F> r = claim.second[0];
    r.insert(r.end(), claim.second[1].begin(), claim.second[1].end());
    vector<vector<F>> polys;
    for (int i = 0; i < data.size(); i++)
    {
        polys.push_back(beta1[i]);
        polys.push_back(convert_to_field(data[i].RD1));
        polys.push_back(convert_to_field(data[i].IDX1));
    }
    for (int i = 0; i < data.size(); i++)
    {
        polys.push_back(beta2[i]);
        polys.push_back(convert_to_field(data[i].RD2));
        polys.push_back(convert_to_field(data[i].IDX2));
    }
    for (int i = 0; i < data.size(); i++)
        polys.push_back(convert_to_field(data[i].FINAL_FR1));
    for (int i = 0; i < data.size(); i++)
        polys.push_back(convert_to_field(data[i].FINAL_FR2));

    if (rank == 0)
        printf("Timestamp: %lf | Starting batch_distributed_eval_opt", get_time());
    vector<F> evals = batch_distributed_eval_opt(polys, r, claim.second[2], N);
    if (rank == 0)
        printf("Timestamp: %lf | Finished batch_distributed_eval_opt", get_time());
    pt_cp.start();

    vector<F> beta_evals;

    for (int i = 0; i < data.size(); i++)
    {
        beta_evals.push_back(evals[3 * i]);
        beta_evals.push_back(evals[3 * i + 9]);
    }

    if (rank == 0)
    {
        // TO FIX
        /*
        vector<F> Tr_evals(24,F(0));
        vector<F> r11,r12,r21,r22,_r;

        for(int i  = 0; i < r1.size()-(int)log2(N); i++){
            r11.push_back(r1[i]);
        }
        for(int i = r11.size(); i < r1.size(); i++){
            r12.push_back(r1[i]);
        }

        for(int i  = 0; i < r2.size()-(int)log2(N); i++){
            r21.push_back(r2[i]);
        }
        for(int i = r21.size(); i < r2.size(); i++){
            r22.push_back(r2[i]);
        }


        for(int i = 0; i < data.size();i++){
            Tr_evals[2*i] = challenges[0]*evals[3*i] +  challenges[1]*evals[3*i+1] + challenges[2]*evals[3*i+2] + F(1);
            Tr_evals[2*i+1] = Tr_evals[2*i] +  challenges[1];
        }
        for(int i = 0; i < data.size();i++){
            Tr_evals[2*i+6] = challenges[0]*evals[3*i+9] +  challenges[1]*evals[3*i+1+9] + challenges[2]*evals[3*i+2+9] + F(1);
            Tr_evals[2*i+1+6] = Tr_evals[2*i+6] +  challenges[1];
        }
        for(int i = 0; i < data.size(); i++){
            Tr_evals[2*i + 12] =  challenges[0]*betas_eval(data[i].FINAL_FR1.size(),r11,r12,r,claim.second[2]) + F(1) + challenges[2]*sequence_eval(N*next_pow2(data[i].FINAL_FR1.size()),r,claim.second[2]);
            Tr_evals[2*i + 13] = Tr_evals[2*i + 12] + challenges[1]*evals[i+18];
        }
        for(int i = 0; i < data.size(); i++){
            Tr_evals[2*i + 18] =  challenges[2]*sequence_eval(N*next_pow2(data[i].FINAL_FR2.size()),r,claim.second[2]) + F(1) + challenges[0]*betas_eval(data[i].FINAL_FR2.size(),r21,r22,r,claim.second[2]);
            Tr_evals[2*i + 19] = Tr_evals[2*i + 18] + challenges[1]*evals[i+3+18];
        }*/
    }
    order.clear();
    vector<F> v1, v2, v3, ones(N, 1);
    vector<vector<F>> _beta2_sorted, _beta1_sorted = beta1;
    vector<F> aggr_challenges = {a, b, c};
    sort_transcript(_beta1_sorted, order);
    for (int i = 0; i < beta2.size(); i++)
        _beta2_sorted.push_back(beta2[order[i]]);

    for (int i = 0; i < beta1.size(); i++)
    {
        v1.insert(v1.end(), _beta1_sorted[i].begin(), _beta1_sorted[i].end());
        v2.insert(v2.end(), _beta2_sorted[i].begin(), _beta2_sorted[i].end());
        // vector<F> c_buff(_beta1_sorted[i].size(),aggr_challenges[order[i]]);
        // v3.insert(v3.end(),c_buff.begin(),c_buff.end());
    }
    v1.resize(next_pow2(v1.size()), F(0));
    v2.resize(next_pow2(v2.size()), F(0));
    // v3.resize(next_pow2(v3.size()),F(0));

    int ctr = 0;
    v3.resize(next_pow2(real_idx_dim[0] + real_idx_dim[1] + real_idx_dim[2]) / N, F(0));

    for (int j = 0; j < 3; j++)
    {
        if ((rank + 1) * next_pow2(real_idx_dim[order[j]]) / N <= real_idx_dim[order[j]])
        {
            for (int i = 0; i < next_pow2(real_idx_dim[order[j]]) / N; i++)
            {
                v3[ctr] = aggr_challenges[order[j]];
                ctr++;
            }
        }
        else if (rank * next_pow2(real_idx_dim[order[j]]) / N < real_idx_dim[order[j]])
        {

            int offset = real_idx_dim[order[j]] - (rank)*next_pow2(real_idx_dim[order[j]]) / N;
            for (int i = 0; i < real_idx_dim[order[j]] - (rank)*next_pow2(real_idx_dim[order[j]]) / N; i++)
            {
                v3[ctr] = aggr_challenges[order[j]];
                ctr++;
            }
            for (int i = 0; i < (next_pow2(real_idx_dim[order[j]]) / N) - offset; i++)
                ctr++;
            // ctr = (j+1)*next_pow2(real_idx_dim[order[j]])/N;
        }
        else
        {
            for (int i = 0; i < next_pow2(real_idx_dim[order[j]]) / N; i++)
            {
                v3[ctr] = 0;
                ctr++;
            }
        }
    }

    pt_cp.end();
    timer eval;
    eval.reset();

    eval.start();

    if (rank == 0)
        printf("Timestamp: %lf | Starting final _cubic_sumcheck2", get_time());
    vector<pair<F, vector<F>>> beta_evals2 = _cubic_sumcheck2(y, v1, v2, v3, ones, N);
    if (rank == 0)
        printf("Timestamp: %lf | Finished final _cubic_sumcheck2", get_time());
    eval.end();
    if (rank == 0)
        printf("          Final Sumcheck: %lf\n", eval.get_time());
    // for(int i = 0; i < )

    F __r = F(0); // hash_to_field({0});

    r1.clear();
    r2.clear();

    for (int i = 0; i < beta_evals2[0].second.size() - (int)log2(N); i++)
    {
        r1.push_back(beta_evals2[0].second[i]);
    }
    for (int i = r1.size(); i < beta_evals2[0].second.size(); i++)
    {
        r2.push_back(beta_evals2[0].second[i]);
    }
    r1.push_back(__r);

    r2.insert(r2.begin(), r1.begin(), r1.end());

    return make_pair((F(1) - __r) * beta_evals2[0].first + __r * beta_evals2[1].first, r2);
}

pair<F, vector<F>> _prove_sparse_eval_opt_LAN(F y, F a, F b, F c, vector<vector<F>> &beta1, vector<vector<F>> &beta2, vector<sparse_eval_data> &data,
                                              vector<F> r1, vector<F> r2, int N, vector<F> &debug_data)
{

    double temp = pt_cp.get_time();
    timer t;
    t.start();
    pt_cp.start();
    vector<vector<F>> Tr(24);
    vector<F> output;
    vector<int> order;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    if (rank == 0)
        printf("Timestamp: %lf | Starting _prove_sparse_eval_opt_LAN", get_time());
    // pt_cp.start();
    vector<F> challenges(3);
    clock_t t1 = clock();
    for (int i = 0; i < 3; i++)
        challenges[i] = hash_to_field({0});

    compute_transcript(Tr, data, challenges, beta1, beta2, r1, r2, N);
    // vector<F> test_v = Tr[13];
    sort_transcript(Tr, order);
    vector<F> debug_evals;
    pt_cp.end();
    if (rank == 0)
        printf("Timestamp: %lf | Starting prove_product_opt", get_time());
    pair<F, vector<vector<F>>> claim = prove_product_opt(Tr, output, N, debug_evals);
    t.end();
    if (rank == 0)
        printf("Timestamp: %lf | Finished prove_product_opt", get_time());

    // if(rank == 0) printf("> %lf,%lf\n",t.get_time(),pt_cp.get_time()-temp);
    if (rank == 0)
    {
        vector<F> organized_output(output.size());
        for (int i = 0; i < output.size(); i++)
        {
            organized_output[i] = output[order[i]];
        }
        for (int i = 0; i < 6; i++)
        {
            if (organized_output[2 * i] * organized_output[2 * i + 1 + 12] != organized_output[2 * i + 1] * organized_output[2 * i + 12])
            {
                printf("Error phase 2 %d\n", i);
            }
        }
    }

    vector<F> r = claim.second[0];
    r.insert(r.end(), claim.second[1].begin(), claim.second[1].end());
    vector<vector<F>> polys;
    for (int i = 0; i < data.size(); i++)
    {
        polys.push_back(beta1[i]);
        polys.push_back(convert_to_field(data[i].RD1));
        polys.push_back(convert_to_field(data[i].IDX1));
    }
    for (int i = 0; i < data.size(); i++)
    {
        polys.push_back(beta2[i]);
        polys.push_back(convert_to_field(data[i].RD2));
        polys.push_back(convert_to_field(data[i].IDX2));
    }
    for (int i = 0; i < data.size(); i++)
        polys.push_back(convert_to_field(data[i].FINAL_FR1));
    for (int i = 0; i < data.size(); i++)
        polys.push_back(convert_to_field(data[i].FINAL_FR2));

    if (rank == 0)
        printf("Timestamp: %lf | Starting batch_distributed_eval_opt", get_time());
    vector<F> evals = batch_distributed_eval_opt(polys, r, claim.second[2], N);
    if (rank == 0)
        printf("Timestamp: %lf | Finished batch_distributed_eval_opt", get_time());
    pt_cp.start();

    vector<F> beta_evals;

    for (int i = 0; i < data.size(); i++)
    {
        beta_evals.push_back(evals[3 * i]);
        beta_evals.push_back(evals[3 * i + 9]);
    }

    order.clear();
    vector<F> v1, v2, v3, ones(N, 1);
    vector<vector<F>> _beta2_sorted, _beta1_sorted = beta1;
    vector<F> aggr_challenges = {a, b, c};
    sort_transcript(_beta1_sorted, order);
    for (int i = 0; i < beta2.size(); i++)
        _beta2_sorted.push_back(beta2[order[i]]);

    for (int i = 0; i < beta1.size(); i++)
    {
        v1.insert(v1.end(), _beta1_sorted[i].begin(), _beta1_sorted[i].end());
        v2.insert(v2.end(), _beta2_sorted[i].begin(), _beta2_sorted[i].end());
        // vector<F> c_buff(_beta1_sorted[i].size(),aggr_challenges[order[i]]);
        // v3.insert(v3.end(),c_buff.begin(),c_buff.end());
    }
    v1.resize(next_pow2(v1.size()), F(0));
    v2.resize(next_pow2(v2.size()), F(0));
    // v3.resize(next_pow2(v3.size()),F(0));

    int ctr = 0;
    v3.resize(next_pow2(real_idx_dim[0] + real_idx_dim[1] + real_idx_dim[2]) / N, F(0));

    for (int j = 0; j < 3; j++)
    {
        if ((rank + 1) * next_pow2(real_idx_dim[order[j]]) / N <= real_idx_dim[order[j]])
        {
            for (int i = 0; i < next_pow2(real_idx_dim[order[j]]) / N; i++)
            {
                v3[ctr] = aggr_challenges[order[j]];
                ctr++;
            }
        }
        else if (rank * next_pow2(real_idx_dim[order[j]]) / N < real_idx_dim[order[j]])
        {

            int offset = real_idx_dim[order[j]] - (rank)*next_pow2(real_idx_dim[order[j]]) / N;
            for (int i = 0; i < real_idx_dim[order[j]] - (rank)*next_pow2(real_idx_dim[order[j]]) / N; i++)
            {
                v3[ctr] = aggr_challenges[order[j]];
                ctr++;
            }
            for (int i = 0; i < (next_pow2(real_idx_dim[order[j]]) / N) - offset; i++)
                ctr++;
            // ctr = (j+1)*next_pow2(real_idx_dim[order[j]])/N;
        }
        else
        {
            for (int i = 0; i < next_pow2(real_idx_dim[order[j]]) / N; i++)
            {
                v3[ctr] = 0;
                ctr++;
            }
        }
    }

    pt_cp.end();
    timer eval;
    eval.reset();

    eval.start();

    if (rank == 0)
        printf("Timestamp: %lf | Starting final _cubic_sumcheck", get_time());
    vector<pair<F, vector<F>>> beta_evals2 = _cubic_sumcheck(y, v1, v2, v3, ones, N);
    eval.end();
    if (rank == 0)
        printf("Timestamp: %lf | Finished final _cubic_sumcheck", get_time());
    if (rank == 0)
        printf("          Final Sumcheck: %lf\n", eval.get_time());
    // for(int i = 0; i < )

    F __r = F(0); // hash_to_field({0});

    r1.clear();
    r2.clear();

    for (int i = 0; i < beta_evals2[0].second.size() - (int)log2(N); i++)
    {
        r1.push_back(beta_evals2[0].second[i]);
    }
    for (int i = r1.size(); i < beta_evals2[0].second.size(); i++)
    {
        r2.push_back(beta_evals2[0].second[i]);
    }
    r1.push_back(__r);

    r2.insert(r2.begin(), r1.begin(), r1.end());

    return make_pair((F(1) - __r) * beta_evals2[0].first + __r * beta_evals2[1].first, r2);
}

pair<F, vector<F>> _prove_sparse_eval(F y, F a, F b, F c, vector<vector<F>> &beta1, vector<vector<F>> &beta2, vector<sparse_eval_data> &data,
                                      vector<F> r1, vector<F> r2,
                                      int N)
{

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id
    pt_cp.start();
    if (rank == 0)
        printf("Timestamp: %lf | Entering _prove_sparse_eval\n", get_time());
    vector<F> challenges(3);
    clock_t t1 = clock();
    for (int i = 0; i < 3; i++)
        challenges[i] = hash_to_field({0});

    vector<vector<F>> Tr(12);
    vector<vector<F>> polys, acc_polys(18);
    vector<vector<vector<F>>> acc_eval_points;
    vector<F> acc_evals;
    for (int j = 0; j < data.size(); j++)
    {
        Tr[2 * j].resize(next_pow2(data[j].IDX1.size()), F(1));
        Tr[2 * j + 1].resize(next_pow2(data[j].IDX1.size()), F(1));
        for (int i = 0; i < data[j].IDX1.size(); i++)
        {
            Tr[2 * j][i] = challenges[0] * beta1[j][i] + challenges[1] * F(data[j].RD1[i]) + challenges[2] * F(data[j].IDX1[i]) + F(1);
            Tr[2 * j + 1][i] = challenges[0] * beta1[j][i] + challenges[1] * F(data[j].WR1[i]) + challenges[2] * F(data[j].IDX1[i]) + F(1);
        }
    }
    for (int j = 0; j < data.size(); j++)
    {
        Tr[2 * j + 6].resize(next_pow2(data[0].IDX2.size()), F(1));
        Tr[2 * j + 1 + 6].resize(next_pow2(data[0].IDX2.size()), F(1));
        for (int i = 0; i < data[j].IDX2.size(); i++)
        {
            Tr[2 * j + 6][i] = challenges[0] * beta2[j][i] + challenges[1] * F(data[j].RD2[i]) + challenges[2] * F(data[j].IDX2[i]) + F(1);
            Tr[2 * j + 1 + 6][i] = challenges[0] * beta2[j][i] + challenges[1] * F(data[j].WR2[i]) + challenges[2] * F(data[j].IDX2[i]) + F(1);
        }
    }
    vector<F> output1, output2, output3;
    pt_cp.end();

    if (rank == 0)
        printf("Timestamp: %lf | Starting prove_product\n", get_time());
    pair<F, vector<vector<F>>> claims = prove_product(Tr, output1, F(0), {}, N);
    if (rank == 0)
        printf("Timestamp: %lf | Finished prove_product\n", get_time());
    vector<F> v1, v2;
    pt_cp.start();

    precompute_beta(claims.second[0], v1);
    precompute_beta(claims.second[2], v2);
    vector<vector<F>> eval_points1 = claims.second;
    for (int i = 0; i < data.size(); i++)
    {
        polys.push_back(beta1[i]);
        polys.push_back(convert_to_field(data[i].RD1));
        polys.push_back(convert_to_field(data[i].IDX1));
        acc_polys[2 * i + 3] = polys[3 * i + 1];
        acc_polys[2 * i + 1 + 3] = polys[3 * i + 2];
    }
    for (int i = 0; i < data.size(); i++)
    {
        polys.push_back(beta2[i]);
        polys.push_back(convert_to_field(data[i].RD2));
        polys.push_back(convert_to_field(data[i].IDX2));
        acc_polys[2 * i + 6 + 3] = polys[3 * i + 1 + 9];
        acc_polys[2 * i + 1 + 6 + 3] = polys[3 * i + 2 + 9];
    }
    pt_cp.end();

    if (rank == 0)
        printf("Timestamp: %lf | Starting batch_distributed_eval (phase1)\n", get_time());
    vector<F> evals = batch_distributed_eval(polys, v1, v2, N);
    if (rank == 0)
        printf("Timestamp: %lf | Finished batch_distributed_eval (phase1)\n", get_time());
    pt_cp.start();

    vector<F> beta_evals, evals1;
    vector<F> betas_r = claims.second[0];
    vector<F> temp_r;

    for (int i = 0; i < (int)log2(next_pow2(2 * beta1.size())); i++)
    {
        betas_r.push_back(hash_to_field({0}));
        temp_r.push_back(betas_r[betas_r.size() - 1]);
    }
    betas_r.insert(betas_r.end(), claims.second[2].begin(), claims.second[2].end());

    // vector<F> evals2 = batch_distributed_eval(Tr,v1,v2,N);
    for (int i = 0; i < data.size(); i++)
    {
        beta_evals.push_back(evals[3 * i]);
        evals1.push_back(evals[3 * i + 1]);
        evals1.push_back(evals[3 * i + 2]);
    }
    for (int i = 0; i < data.size(); i++)
    {
        beta_evals.push_back(evals[3 * i + 9]);
        evals1.push_back(evals[3 * i + 1 + 9]);
        evals1.push_back(evals[3 * i + 2 + 9]);
    }

    if (rank == 0)
    {
        vector<F> Tr_evals(12, F(0));
        for (int i = 0; i < data.size(); i++)
        {
            Tr_evals[2 * i] = challenges[0] * evals[3 * i] + challenges[1] * evals[3 * i + 1] + challenges[2] * evals[3 * i + 2] + F(1);
            Tr_evals[2 * i + 1] = Tr_evals[2 * i] + challenges[1];
        }
        for (int i = 0; i < data.size(); i++)
        {
            Tr_evals[2 * i + 6] = challenges[0] * evals[3 * i + 9] + challenges[1] * evals[3 * i + 1 + 9] + challenges[2] * evals[3 * i + 2 + 9] + F(1);
            Tr_evals[2 * i + 1 + 6] = Tr_evals[2 * i + 6] + challenges[1];
        }

        Tr_evals.resize(16, F(0));
        if (evaluate_vector(Tr_evals, claims.second[1]) != claims.first)
        {
            printf("Sparse Eval Error 1\n");
        }
    }

    vector<F> base_beta1, base_beta2;
    compute_base_betas(r1, r2, base_beta1, base_beta2, N);

    Tr.clear();
    Tr.resize(6);
    for (int j = 0; j < data.size(); j++)
    {
        Tr[2 * j].clear();
        Tr[2 * j + 1].clear();
        Tr[2 * j].resize(next_pow2(data[0].FINAL_FR1.size()), F(1));
        Tr[2 * j + 1].resize(next_pow2(data[0].FINAL_FR1.size()), F(1));

        for (int i = 0; i < data[j].FINAL_FR1.size(); i++)
        {
            Tr[2 * j + 0][i] = challenges[0] * base_beta1[i] + challenges[2] * F(rank * data[j].FINAL_FR1.size() + i) + F(1);
            Tr[2 * j + 1][i] = challenges[0] * base_beta1[i] + challenges[1] * F(data[j].FINAL_FR1[i]) + challenges[2] * F(rank * data[j].FINAL_FR1.size() + i) + F(1);
        }
    }
    pt_cp.end();

    if (rank == 0)
        printf("Timestamp: %lf | Starting prove_product (phase1)\n", get_time());
    claims = prove_product(Tr, output2, F(0), {}, N);
    if (rank == 0)
        printf("Timestamp: %lf | Finished prove_product (FINAL_FR1)\n", get_time());
    pt_cp.start();

    polys.clear();
    for (int i = 0; i < data.size(); i++)
    {
        polys.push_back(convert_to_field(data[i].FINAL_FR1));
        acc_polys[i + 15] = polys[i];
    }
    v1.clear();
    v2.clear();
    precompute_beta(claims.second[0], v1);
    precompute_beta(claims.second[2], v2);
    pt_cp.end();
    vector<F> evals2 = batch_distributed_eval(polys, v1, v2, N);
    vector<vector<F>> eval_points2 = claims.second;

    pt_cp.start();

    Tr.clear();
    Tr.resize(6);
    for (int j = 0; j < data.size(); j++)
    {
        Tr[2 * j].clear();
        Tr[2 * j + 1].clear();
        Tr[2 * j].resize(next_pow2(data[0].FINAL_FR2.size()), F(1));
        Tr[2 * j + 1].resize(next_pow2(data[0].FINAL_FR2.size()), F(1));
        for (int i = 0; i < data[j].FINAL_FR2.size(); i++)
        {
            Tr[2 * j][i] = challenges[0] * base_beta2[i] + challenges[2] * F(rank * data[j].FINAL_FR2.size() + i) + F(1);
            Tr[2 * j + 1][i] = challenges[0] * base_beta2[i] + challenges[1] * F(data[j].FINAL_FR2[i]) + challenges[2] * F(rank * data[j].FINAL_FR2.size() + i) + F(1);
        }
    }
    pt_cp.end();

    if (rank == 0)
        printf("Timestamp: %lf | Starting prove_product (FINAL_FR2)\n", get_time());
    claims = prove_product(Tr, output3, F(0), {}, N);
    if (rank == 0)
        printf("Timestamp: %lf | Finished prove_product (FINAL_FR2)\n", get_time());
    pt_cp.start();

    polys.clear();
    for (int i = 0; i < data.size(); i++)
    {
        polys.push_back(convert_to_field(data[i].FINAL_FR2));
        acc_polys[i] = polys[i];
    }
    v1.clear();
    v2.clear();
    timer eval;
    eval.start();
    precompute_beta(claims.second[0], v1);
    precompute_beta(claims.second[2], v2);
    pt_cp.end();
    vector<F> evals3 = batch_distributed_eval(polys, v1, v2, N);
    eval.end();
    vector<vector<F>> eval_points3 = claims.second;
    pt_cp.start();
    if (rank == 0)
        printf("          Distributed Eval: %lf\n", eval.get_time());

    if (rank == 0)
    {
        // printf("%d\n",output1.size());
        for (int i = 0; i < 3; i++)
        {
            if (output1[2 * i] * output2[2 * i + 1] != output1[2 * i + 1] * output2[2 * i + 0])
            {
                printf("Error phase 2 %d\n", i);
            }
        }
        for (int i = 0; i < 3; i++)
        {
            if (output1[2 * i + 6] * output3[2 * i + 1] != output1[2 * i + 1 + 6] * output3[2 * i + 0])
            {
                printf("Error phase 2 %d\n", i);
            }
        }
    }
    acc_eval_points.push_back(eval_points3);
    acc_eval_points.push_back(eval_points1);
    acc_eval_points.push_back(eval_points2);
    acc_evals = evals3;
    acc_evals.insert(acc_evals.end(), evals1.begin(), evals1.end());
    acc_evals.insert(acc_evals.end(), evals2.begin(), evals2.end());
    pt_cp.end();

    if (rank == 0)
        printf("Timestamp: %lf | Starting distributed_accumulation\n", get_time());
    distributed_accumulation(acc_polys, acc_eval_points, acc_evals, N);
    if (rank == 0)
        printf("Timestamp: %lf | Finished distributed_accumulation\n", get_time());

    pt_cp.start();
    /// Need to do the final sumcheck
    vector<F> v3, ones(N, 1);

    v1.clear();
    v2.clear();
    for (int i = 0; i < beta1.size(); i++)
    {
        v1.insert(v1.end(), beta1[i].begin(), beta1[i].end());
        v2.insert(v2.end(), beta2[i].begin(), beta2[i].end());
    }
    v1.resize(next_pow2(v1.size()), F(0));
    v2.resize(next_pow2(v2.size()), F(0));

    int ca = 0, cb = 0, cc = 0;
    v3.resize(4 * next_pow2(real_idx_dim[0]) / N, F(0));
    if (rank != N - 1)
    {
        int ctr = 0;
        for (int i = 0; i < next_pow2(real_idx_dim[0]) / N; i++)
        {
            v3[ctr] = a;
            ctr++;
        }
        for (int i = 0; i < next_pow2(real_idx_dim[1]) / N; i++)
        {
            v3[ctr] = b;
            ctr++;
        }
        for (int i = 0; i < next_pow2(real_idx_dim[2]) / N; i++)
        {
            v3[ctr] = c;
            ctr++;
        }
    }
    else
    {
        int ctr = 0;
        for (int i = 0; i < real_idx_dim[0] - (N - 1) * next_pow2(real_idx_dim[0]) / N; i++)
        {
            v3[ctr] = a;
            ctr++;
        }
        ctr = next_pow2(real_idx_dim[0]) / N;
        for (int i = 0; i < real_idx_dim[1] - (N - 1) * next_pow2(real_idx_dim[1]) / N; i++)
        {
            v3[ctr] = b;
            ctr++;
        }
        ctr = 2 * next_pow2(real_idx_dim[1]) / N;

        for (int i = 0; i < real_idx_dim[2] - (N - 1) * next_pow2(real_idx_dim[2]) / N; i++)
        {
            v3[ctr] = c;
            ctr++;
        }
    }
    pt_cp.end();
    eval.reset();
    eval.start();
    if (rank == 0)
        printf("Timestamp: %lf | Starting final _cubic_sumcheck\n", get_time());
    vector<pair<F, vector<F>>> beta_evals2 = _cubic_sumcheck(y, v1, v2, v3, ones, N);
    eval.end();
    if (rank == 0)
        printf("Timestamp: %lf | Finished final _cubic_sumcheck\n", get_time());
    if (rank == 0)
        printf("          Final Sumcheck: %lf\n", eval.get_time());

    beta_evals.resize(next_pow2(beta_evals.size()), F(0));
    return make_pair(evaluate_vector(beta_evals, temp_r), betas_r);
}