#include "coPCS.h"
#include "New_PCS.h"
#include "timer.hpp"
#include "coPCS_utils.h"
extern int BATCH_FACTOR;
extern vector<int> _queries;
extern int Security_Level;
extern vector<int> _degrees;
extern vector<vector<int>> _rates;
extern vector<F> poly;
extern double pcs_ps;
extern vector<int> delegation_size;
extern vector<vector<F>> whir_codewords;
extern vector<vector<F>> whir_arrays;
extern vector<vector<vector<_hash>>> whir_MT;
extern vector<vector<pair<F,vector<F>>>> claims;
extern int max_delegation_size; 
int randomness =0;
int my_id = 0;
extern timer pt,pt_cp,vt,sch_com;
extern double whir_ps;
unsigned int rate_offset = 1;

extern vector<vector<F>> whir_polys,whir_c,whir_betas;
extern vector<vector<vector<_hash>>> whir_MTs;
extern vector<F> whir_claims;
extern vector<int> whir_sizes;
extern vector<int> whir_rates;
extern vector<int> sorted_indexes;
extern vector<int> sorted_whir_rates;

extern int com_rounds;
extern double cm;

#include <unordered_map>
extern unordered_map<int, int> whir_kappas;

void prepare_parameters(vector<vector<int>> rates,int rounds,int N, int M){
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id); //get my process id
    
    
    get_degrees(rounds, 2*(4*M/N));
    get_queries(rounds,rates);
    
    for(int i  = 1; i < rates.size(); i++){
        rate_offset *= rates[i][0];
    }
    for(int i = 2; i  <_degrees.size(); i++){
        delegation_size.push_back(next_pow2(_degrees[i]*_queries[i]));
    }
    for(int i = 0; i < _degrees.size(); i++){
        randomness += _queries[i]*_degrees[i];
    }
    max_delegation_size = delegation_size[0];
    if(!my_id){
        for(int i = 0; i < delegation_size.size(); i++){
            printf("%d, ",delegation_size[i]);
        }
        printf("\n");
    }
    
    for(int i = 0; i < delegation_size.size(); i++){
        if(max_delegation_size <= delegation_size[i]) max_delegation_size = delegation_size[i]; 
    }
    
    whir_codewords.resize(delegation_size.size());
    whir_MT.resize(delegation_size.size());
    whir_arrays.resize(delegation_size.size());
    claims.resize(delegation_size.size());
}



// This function is invoked by every prover to mask the data
// Here we mask them naively, but in practice they should be masked the same way as described in the paper
void prepare_data(vector<vector<F>> &Commit_data, vector<F> &Shares, vector<F> &Random){
    Commit_data.resize(BATCH_FACTOR);
    int Batch_size = Shares.size()/(BATCH_FACTOR/2);
    int data_ctr = 0;
    int rand_ctr = 0;
    int N = Shares.size();
    int last_batch = 0;
    
    for(int i = 0; i < BATCH_FACTOR; i++){
        Commit_data[i].resize(Batch_size,F(0));
        
        for(int k = 0; k < Batch_size-_queries[0]; k++){
            Commit_data[i][k] = Shares[data_ctr];
            data_ctr++;
            if(data_ctr == Shares.size()){
                break;
            }
        }
        for(int k = Batch_size-_queries[0]; k < Batch_size; k++){
            Commit_data[i][k] = Random[rand_ctr];
            rand_ctr++;
        }
        if(data_ctr == Shares.size()){
            last_batch = i;
            break;
        }

    }
    if(data_ctr < Shares.size()){
        printf("Error more batches are needed %d,%d\n",data_ctr,Shares.size());
        exit(-1);
    }
    last_batch++;
    Commit_data[last_batch].resize(Batch_size,F(0));
    
    if(Shares.size()-rand_ctr > Batch_size){
        //if(!my_id) printf("Case 1\n");
        for(int i = 0; i < Batch_size; i++){
            Commit_data[last_batch][i] = Random[rand_ctr]; 
            rand_ctr++;
            if(rand_ctr == Random.size()){
                break;
            }
        }
    }else{
        if(!my_id) printf("Case 2\n");
        for(int i = 0; i < Batch_size- (Shares.size()-rand_ctr); i++){
            Commit_data[last_batch][i] = Random[rand_ctr]; 
            rand_ctr++;
        }
    }
    
}


void Encode(vector<vector<F>> &Codeword, int N, vector<int> rate){
    for(int i = 0; i < BATCH_FACTOR; i++){
        if(Codeword[i].size() == 0) return;
        fft(Codeword[i],(int)log2(Codeword[i].size()),true);
        
        F omega = getRootOfUnity(1+(int)log2(rate_offset)+(int)log2(Codeword[i].size())).inv();
        F mul = F(1);
        for(int j = 0; j < Codeword[i].size(); j++){
            Codeword[i][j] = mul*Codeword[i][j];
            mul = mul*omega;
        }
        
        
        Codeword[i].resize(rate[0]*Codeword[i].size(),F(0));
        fft(Codeword[i],(int)log2(Codeword[i].size()),false);
    }
}   




void distributed_MT(vector<vector<F>> &Codeword, MT &Com, int N){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get my process id
    
    pt_cp.start();
    
    int batches = 0;
    for(int i = 0; i < Codeword.size(); i++){
        if(!Codeword[i].size()) break;
        batches++;
    }
    
    vector<u8> root(32);
    vector<vector<_hash>> H;
    Com.Base_MT.resize((int)log2(Codeword[0].size())+1);
    Com.Base_MT[0].resize(Codeword[0].size());
    for(int j = 0; j < Codeword[0].size(); j++){
        vector<F> batch_vec(Codeword.size(),F(0));
        for(int k  = 0; k < batches; k++){
            batch_vec[k] = Codeword[k][j];
        }
        // MT is not required and is overkill. Switch to chain hashing
        merkle_tree::merkle_tree_prover::MT_commit_Blake(batch_vec.data(),H,batch_vec.size());
        Com.Base_MT[0][j] = H[H.size()-1][0];
        H.clear();
    }   
    
    merkle_tree::merkle_tree_prover::create_tree_blake(Com.Base_MT[0].size(),Com.Base_MT,sizeof(__hhash_digest), true);
    pt_cp.end();
    

    vector<uint8_t> recv_buff(32*N), send_buff(32*N);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < 32; j++) send_buff[32*i + j] = 0;
    }
    MPI_Request req;
    MPI_Ialltoall(send_buff.data(),send_buff.size()/N,MPI_UINT8_T,recv_buff.data(),recv_buff.size()/N,MPI_UINT8_T,MPI_COMM_WORLD,&req);
    com_rounds++;
    cm += 32*N/1024.0;

    MPI_Wait(&req,MPI_STATUS_IGNORE);
    if(rank == 0){
        vector<_hash> recv_hashes(N);
        recv_hashes[0] = Com.Base_MT[Com.Base_MT.size()-1][0];
        Com.Root_MT.resize((int)log2(N)+1);
        Com.Root_MT[0] = recv_hashes;
        merkle_tree::merkle_tree_prover::create_tree_blake(N,Com.Root_MT,sizeof(__hhash_digest),true);
        for(int i = 0; i  <32; i++){
            root[i] = Com.Root_MT[Com.Root_MT.size()-1][0].arr[i];
        }
    }   
    /*
    if(rank != 0){
        MPI_Request req;
        MPI_Isend(Com.Base_MT[Com.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD,&req);
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        //MPI_Send(Com.Base_MT[Com.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD);
    }else{
        vector<_hash> recv_hashes(N);
        vector<MPI_Request> req(N);
        recv_hashes[0] = Com.Base_MT[Com.Base_MT.size()-1][0];
        for(int i = 1; i < N; i++){
            MPI_Irecv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,&req[i]);
            //MPI_Recv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);            
        }
        for(int i = 1; i < N; i++){
            MPI_Wait(&req[i],MPI_STATUS_IGNORE);
        }
        Com.Root_MT.resize((int)log2(N)+1);
        Com.Root_MT[0] = recv_hashes;
        merkle_tree::merkle_tree_prover::create_tree_blake(N,Com.Root_MT,sizeof(__hhash_digest),true);
        for(int i = 0; i  <32; i++){
            root[i] = Com.Root_MT[Com.Root_MT.size()-1][0].arr[i];
        }
    }
    myBcast(root,N);
    */
    
}





void PCS_commit(MT &C, 
                vector<vector<F>> &Codeword, 
                vector<vector<F>> &Commit_data, 
                vector<F> &Shares, vector<F> &Random_Shares, 
                vector<vector<int>> rate, int N){
    prepare_data(Commit_data, Shares, Random_Shares);
    Codeword = Commit_data;
    
    pt_cp.start();
    Encode(Codeword, Shares.size(), rate[0]);
    pt_cp.end();
    distributed_MT(Codeword, C, N);
    Shares.clear();
    Random_Shares.clear();
}

void distributed_commit2(MT &Com, int N){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get my process id
    

    vector<u8> root(32);
    
    vector<uint8_t> recv_buff(32*N), send_buff(32*N);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < 32; j++) send_buff[32*i + j] = 0;
    }
    MPI_Request req;
    MPI_Ialltoall(send_buff.data(),send_buff.size()/N,MPI_UINT8_T,recv_buff.data(),recv_buff.size()/N,MPI_UINT8_T,MPI_COMM_WORLD,&req);
    com_rounds++;
    cm += 32*N/1024.0;

    MPI_Wait(&req,MPI_STATUS_IGNORE);
    if(rank == 0){
        vector<_hash> recv_hashes(N);
        recv_hashes[0] = Com.Base_MT[Com.Base_MT.size()-1][0];
        Com.Root_MT.resize((int)log2(N)+1);
        Com.Root_MT[0] = recv_hashes;
        merkle_tree::merkle_tree_prover::create_tree_blake(N,Com.Root_MT,sizeof(__hhash_digest),true);
        for(int i = 0; i  <32; i++){
            root[i] = Com.Root_MT[Com.Root_MT.size()-1][0].arr[i];
        }
    }
    //com_rounds++;
    //cm += 32*N/1024.0;

    /*
    if(rank != 0){
        MPI_Request req;
        MPI_Isend(Com.Base_MT[Com.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD,&req);
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        //MPI_Send(Com.Base_MT[Com.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD);
    }else{
        vector<_hash> recv_hashes(N);
        vector<MPI_Request> req(N);
        recv_hashes[0] = Com.Base_MT[Com.Base_MT.size()-1][0];
        for(int i = 1; i < N; i++){
            MPI_Irecv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,&req[i]);
            //MPI_Recv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);            
        }
        for(int i = 1; i < N; i++){
            MPI_Wait(&req[i],MPI_STATUS_IGNORE);
        }
        Com.Root_MT.resize((int)log2(N)+1);
        Com.Root_MT[0] = recv_hashes;
        merkle_tree::merkle_tree_prover::create_tree_blake(N,Com.Root_MT,sizeof(__hhash_digest),true);
        for(int i = 0; i  <32; i++){
            root[i] = Com.Root_MT[Com.Root_MT.size()-1][0].arr[i];
        }
    }
    myBcast(root,N);*/
    

}

void distributed_commit(MT &Com, int N){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get my process id
    pt_cp.start();
    
    merkle_tree::merkle_tree_prover::create_tree_blake(Com.Base_MT[0].size(),Com.Base_MT,sizeof(__hhash_digest), true);
    pt_cp.end();
    vector<u8> root(32);
    
    vector<uint8_t> recv_buff(32*N), send_buff(32*N);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < 32; j++) send_buff[32*i + j] = 0;
    }
    MPI_Request req;
    MPI_Ialltoall(send_buff.data(),send_buff.size()/N,MPI_UINT8_T,recv_buff.data(),recv_buff.size()/N,MPI_UINT8_T,MPI_COMM_WORLD,&req);
    com_rounds++;
    cm += 32*N/1024.0;

    MPI_Wait(&req,MPI_STATUS_IGNORE);
    if(rank == 0){
        vector<_hash> recv_hashes(N);
        recv_hashes[0] = Com.Base_MT[Com.Base_MT.size()-1][0];
        Com.Root_MT.resize((int)log2(N)+1);
        Com.Root_MT[0] = recv_hashes;
        merkle_tree::merkle_tree_prover::create_tree_blake(N,Com.Root_MT,sizeof(__hhash_digest),true);
        for(int i = 0; i  <32; i++){
            root[i] = Com.Root_MT[Com.Root_MT.size()-1][0].arr[i];
        }
    }   
    /*
    if(rank != 0){
        MPI_Request req;
        MPI_Isend(Com.Base_MT[Com.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD,&req);
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        //MPI_Send(Com.Base_MT[Com.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD);
    }else{
        vector<_hash> recv_hashes(N);
        vector<MPI_Request> req(N);
        recv_hashes[0] = Com.Base_MT[Com.Base_MT.size()-1][0];
        for(int i = 1; i < N; i++){
            MPI_Irecv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,&req[i]);
            //MPI_Recv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);            
        }
        for(int i = 1; i < N; i++){
            MPI_Wait(&req[i],MPI_STATUS_IGNORE);
        }
        Com.Root_MT.resize((int)log2(N)+1);
        Com.Root_MT[0] = recv_hashes;
        merkle_tree::merkle_tree_prover::create_tree_blake(N,Com.Root_MT,sizeof(__hhash_digest),true);
        for(int i = 0; i  <32; i++){
            root[i] = Com.Root_MT[Com.Root_MT.size()-1][0].arr[i];
        }
    }
    myBcast(root,N);
    */
    
}

void distributed_commit_batch(vector<MT> &Com, int N){
    pt_cp.start();
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get my process id
    for(int i = 0; i < Com.size(); i++){
        merkle_tree::merkle_tree_prover::create_tree_blake(Com[i].Base_MT[0].size(),Com[i].Base_MT,sizeof(__hhash_digest), true);
    }
    
    vector<u8> roots(32*Com.size());
    pt_cp.end();
    vector<u8> data(32*Com.size());
    for(int i = 0; i < Com.size(); i++){
        for(int j = 0; j < 32; j++){
            data[i*32 + j] = Com[i].Base_MT[Com[i].Base_MT.size()-1][0].arr[j];
        }
    }
    
    
    data.resize(N*data.size());
    for(int i = 32*Com.size(); i < data.size(); i++) data[i] = data[i%(32*Com.size())];
    vector<u8> recv_data(data.size());
    MPI_Request req;
    MPI_Ialltoall(data.data(),data.size()/N,MPI_UINT8_T,recv_data.data(),recv_data.size()/N,MPI_UINT8_T,MPI_COMM_WORLD,&req);
    MPI_Wait(&req,MPI_STATUS_IGNORE);
    com_rounds++;
    cm += data.size()/1024.0;
    
    
    for(int l  = 0; l < Com.size(); l++){
        Com[l].Root_MT.resize((int)log2(N)+1);
        Com[l].Root_MT[0].resize(N);
        for(int j = 0; j < N; j++){
            for(int i = 0; i < 32; i++){
                Com[l].Root_MT[0][j].arr[i] = 0;
            }
            //Com[l].Root_MT[0].push_back();    
        }
        merkle_tree::merkle_tree_prover::create_tree_blake(N,Com[l].Root_MT,sizeof(__hhash_digest),true);
    }
    for(int i = 0; i < Com.size(); i++){
        for(int j = 0; j < 32; j++){
            roots[32*i + j] = Com[i].Root_MT[Com[i].Root_MT.size()-1][0].arr[j];
        }
    }
    
    //exit(-1);
    
    /*
    if(rank != 0){
        MPI_Request req;
        vector<u8> data(32*Com.size());
        for(int i = 0; i < Com.size(); i++){
            for(int j = 0; j < 32; j++){
                data[i*32 + j] = Com[i].Base_MT[Com[i].Base_MT.size()-1][0].arr[j];
            }
        }
        MPI_Isend(data.data(),data.size(),MPI_UINT8_T,0,0,MPI_COMM_WORLD,&req);
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        //MPI_Send(Com.Base_MT[Com.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD);
    }else{
        
        vector<vector<_hash>> recv_hashes(Com.size());
        for(int i  =0 ; i < recv_hashes.size(); i++) {
            recv_hashes[i].resize(N);
            recv_hashes[i][0] = Com[i].Base_MT[Com[i].Base_MT.size()-1][0];
        }
        vector<MPI_Request> req(N);
        
        vector<vector<u8>> recv_data(N);
        for(int i = 0; i < recv_data.size(); i++){
            recv_data[i].resize(32*Com.size());
        }
        for(int i = 1; i < N; i++){
            MPI_Irecv(recv_data[i].data(),recv_data[i].size(),MPI_UINT8_T,i,0,MPI_COMM_WORLD,&req[i]);
            //MPI_Recv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);            
        }
        for(int i = 1; i < N; i++){
            MPI_Wait(&req[i],MPI_STATUS_IGNORE);
            for(int j = 0; j < Com.size(); j++){
                vector<u32> buff(32);
                for(int l = 0; l < 32; l++){
                    buff[l] = recv_data[i][32*j+l];
                }
                
                //recv_hashes[j][i] = recv_hashes[j][0];
                memcpy(&recv_hashes[j][i].arr,buff.data(),32);
            }
        }
        for(int l  = 0; l < Com.size(); l++){
            Com[l].Root_MT.resize((int)log2(N)+1);
            Com[l].Root_MT[0] = recv_hashes[l];
            merkle_tree::merkle_tree_prover::create_tree_blake(N,Com[l].Root_MT,sizeof(__hhash_digest),true);
        }
        for(int i = 0; i < Com.size(); i++){
            for(int j = 0; j < 32; j++){
                roots[32*i + j] = Com[i].Root_MT[Com[i].Root_MT.size()-1][0].arr[j];
            }
        }
    }
    myBcast(roots, N);   
    */
     
}

void batch_commitments(MT &C, vector<vector<F>> &Codeword, vector<vector<F>> &Commit_data, 
           MT &batched_C, vector<F> &batched_Codeword, vector<F> &batched_commit_data, 
           vector<F> b1, int N){
    
  
    pt_cp.start();
    
    batched_Codeword.resize(Codeword[0].size(),F(0));
    batched_commit_data.resize(Commit_data[0].size(),F(0));
    for(int i = 0; i < Codeword.size(); i++){
        if(Codeword[i].size() == 0) break;
        for(int k  = 0; k < batched_Codeword.size(); k++) batched_Codeword[k] += b1[i]*Codeword[i][k];
        for(int k  = 0; k < batched_commit_data.size(); k++) batched_commit_data[k] += b1[i]*Commit_data[i][k];
    }

    vector<F> buff(_degrees[1]);
    vector<vector<_hash>> MT_buff;
    vector<_hash> hash_leaves(batched_Codeword.size()/_degrees[1]);
    for(int j = 0; j < batched_Codeword.size()/_degrees[1]; j++){
        for(int k = 0; k < buff.size(); k++){
            buff[k] = batched_Codeword[j + k*batched_Codeword.size()/_degrees[1]];
        }
        merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(),MT_buff,buff.size());
        hash_leaves[j] = MT_buff[MT_buff.size()-1][0];
    }
    
    batched_C.Base_MT.resize((int)log2(hash_leaves.size())+1);
    batched_C.Base_MT[0] = hash_leaves;
    pt_cp.end();
    
    distributed_commit(batched_C,  N);
    
    // TO DO: RUN Batch verification algorithm. Here we do not have to query the oracle of C_new, we will 
    // query it in the next step
    vector<vector<u32>> queries;
    vt.start();
    queries = get_indexes(_queries[0],N,Codeword[0].size(),my_id);
    vector<vector<F>> replies(queries.size());


  
    for(int i = 0; i < queries.size(); i++){
        F y = F(0);
        for(int k = 0; k < Codeword.size(); k++){
            if(Codeword[k].size() == 0) break;
            replies[i].push_back(Codeword[k][queries[i][1]]);
            y += b1[k]*replies[i][k];
        }
    }
    vt.end();
    
    pcs_ps += ((double)(replies.size()*replies[0].size()*16))/1024.0;
    
    vector<bool> visited(2*C.Base_MT[0].size(),false);
    for(int i = 0; i<queries.size(); i++){
        vector<vector<_hash>> temp_MT;
        int pos = queries[i][1];
        vector<_hash> path;
        replies[i].resize((next_pow2(replies[i].size())),F(0));
        vt.start();
        merkle_tree::merkle_tree_prover::MT_commit_Blake(replies[i].data(),temp_MT,replies[i].size());
        vt.end();

        path = merkle_tree::merkle_tree_prover::open_tree_blake(C.Base_MT,pos);
        
        vt.start();
        merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(C.Base_MT,path.data(),pos,C.Base_MT[0].size(),visited,pcs_ps);
        vt.end();
    
    }

}

vector<F> pcs_sumcheck_phase1(vector<F> &Commit_data, vector<F> &B, int degree, int N){
    int M = Commit_data.size();
    vector<F> poly(2*degree,F(0));
    pt_cp.start();
    
        for(int j = 0; j < M/degree; j++){
            vector<F> h1(degree),h2(degree);    
            for(int k = 0; k < degree; k++){
                h1[k] = Commit_data[degree*j+k];
                h2[k] = B[degree*j+k];
            }
            fft(h1,(int)log2(h1.size()),true);
            fft(h2,(int)log2(h1.size()),true);
            h1.resize(2*h1.size(),F(0));h2.resize(2*h1.size(),F(0));
            fft(h1,(int)log2(h1.size()),false);fft(h2,(int)log2(h1.size()),false);
            
            for(int k = 0; k < 2*degree; k++) poly[k] += h1[k]*h2[k];
            
        }
    
    pt_cp.end();
    return pcs_sumcheck_aggr(poly, N);    
}

void pcs_sumcheck_phase2(vector<F> &Commit_data, vector<F> &B, int degree, F r){
    int M = Commit_data.size();
    
    vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(degree)),r,degree);
    
    for(int j = 0; j < Commit_data.size()/degree; j++){
        F s1 = F(0), s2 = F(0);
        for(int k = 0; k < degree; k++){
            s1 += L[k]*Commit_data[j*degree+k];
            s2 += L[k]*B[j*degree+k];
        }
        Commit_data[j] = s1;
        B[j] = s2;
    }
    Commit_data.resize(Commit_data.size()/degree);
    B.resize(B.size()/degree);   
}

vector<F> tensor_eval_vector_plaintext(vector<u32> point, int cols, int rows, int rate){
    int N = 2*cols;
    F omega1 = getRootOfUnity((int)log2(rate*rows));
    F omega2 = getRootOfUnity((int)log2(4*cols));
    omega1 = omega1.fastPow(omega1,point[1]);
    omega2 = omega2.fastPow(omega2,2*point[0]+1);
    vector<F> pow1(rows,F(1)),pow2(cols,F(1));
    vector<F> B(rows*cols);
    for(int i = 1; i < rows; i++) pow1[i] = pow1[i-1]*omega1;
    for(int i = 1; i < cols; i++) pow2[i] = pow2[i-1]*omega2;
    //fft(pow2,(int)log2(pow2.size()),true);
    for(int j = 0; j <  rows; j++){
        for(int i = 0; i < cols; i++){
            B[cols*j + i] = pow1[j]*pow2[i];
        }
    }

    return B;
}


vector<F> tensor_eval_vector(vector<u32> point, int cols, int rows, int rate){
    int N = 2*cols;
    F omega1 = getRootOfUnity((int)log2(rate*rows));
    F omega2 = getRootOfUnity((int)log2(4*cols));
    omega1 = omega1.fastPow(omega1,point[1]);
    omega2 = omega2.fastPow(omega2,2*point[0]+1);
    vector<F> pow1(rows,F(1)),pow2(cols,F(1));
    vector<F> B;
    for(int i = 1; i < rows; i++) pow1[i] = pow1[i-1]*omega1;
    for(int i = 1; i < cols; i++) pow2[i] = pow2[i-1]*omega2;
    fft(pow2,(int)log2(pow2.size()),true);
    fft(pow2,(int)log2(pow2.size()),true);
    vector<F> buff = pow2;
    buff.resize(2*N,F(0));
    fft(buff,(int)log2(buff.size()),false);
    pow2.resize(N);
    for(int i = 0; i  < N; i++){
        pow2[i] = buff[2*i+1];
    }
    
    B = pow1;
    for(int j = 0; j < B.size(); j++){
        B[j] = pow2[my_id]*B[j];
    }
    return B;
}

void pcs_sumcheck_phase3(vector<F> &B, vector<vector<u32>> queries, F s, int iter, int N){
    
    if(iter+1 >= _rates.size()) return;
    F p = s;
    
    for(int i = 0; i < queries.size(); i++){
        vector<F> temp_B = tensor_eval_vector(queries[i],N/2,B.size(),_rates[iter+1][0]);
        for(int j = 0; j < temp_B.size(); j++){
            B[j] += p*temp_B[j];
            
        }
        p *= s;   
    }

}

void encode_concatenated(vector<F> &Commit_Data, vector<F> &C_prime, vector<vector<F>> &C, vector<int> rate, int degree){
    //if(my_id == 0) printf("New Rate: %d\n",rate[0]);
        
    C_prime = Commit_Data;
    /*
    F omega = getRootOfUnity(1+(int)log2(_rates[0][0])+(int)log2(C_prime.size())).inv();
    F mul = F(1);
    for(int j = 0; j < C_prime.size(); j++){
        C_prime[j] = mul*C_prime[j];
        mul = mul*omega;
    }*/
    
    
    C_prime.resize(rate[0]*C_prime.size(),F(0));
    fft(C_prime,(int)log2(C_prime.size()),false);
    C.resize(C_prime.size()/degree);
    for(int j = 0; j < C.size(); j++){
        C[j].resize(degree);
        for(int k = 0; k < degree; k++){
            C[j][k] = C_prime[j + k*C_prime.size()/C[j].size()];
        }
        C[j].resize(degree*rate[1],F(0));
        fft(C[j],(int)log2(C[j].size()),false);
    }
}

void commit_concatenated(vector<vector<F>> &C, MT &Com, int N){
    vector<F> leaves = convert2vector(C);
    Com.Base_MT.clear();
    Com.Root_MT.clear();
    
    pt_cp.start();
    merkle_tree::merkle_tree_prover::MT_commit_Blake(leaves.data(),Com.Base_MT,C.size()*C[0].size());
    pt_cp.end();
    vector<u8> root(32);            
    
    
    //vector<u8> root(32);
    
    vector<uint8_t> recv_buff(32*N), send_buff(32*N);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < 32; j++) send_buff[32*i + j] = 0;
    }
    MPI_Request req;
    MPI_Ialltoall(send_buff.data(),send_buff.size()/N,MPI_UINT8_T,recv_buff.data(),recv_buff.size()/N,MPI_UINT8_T,MPI_COMM_WORLD,&req);
    com_rounds++;
    cm += 32*N/1024.0;

    MPI_Wait(&req,MPI_STATUS_IGNORE);
    vector<_hash> recv_hashes(N);
    recv_hashes[0] = Com.Base_MT[Com.Base_MT.size()-1][0];
    Com.Root_MT.resize((int)log2(N)+1);
    Com.Root_MT[0] = recv_hashes;
    merkle_tree::merkle_tree_prover::create_tree_blake(N,Com.Root_MT,sizeof(__hhash_digest),true);
    for(int i = 0; i  <32; i++){
        root[i] = Com.Root_MT[Com.Root_MT.size()-1][0].arr[i];
    }
    
    /*
    if(my_id != 0){
        MPI_Request req;
        MPI_Isend(Com.Base_MT[Com.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD,&req);
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        //MPI_Send(Com.Base_MT[Com.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD);
    }else{
        vector<_hash> recv_hashes(N);
        vector<MPI_Request> req(N);
        recv_hashes[0] = Com.Base_MT[Com.Base_MT.size()-1][0];
        for(int i = 1; i < N; i++){
            MPI_Irecv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,&req[i]);
            //MPI_Recv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);            
        }
        for(int i = 1; i < N; i++){
            MPI_Wait(&req[i],MPI_STATUS_IGNORE);
        }
        Com.Root_MT.resize((int)log2(N)+1);
        Com.Root_MT[0] = recv_hashes;
        merkle_tree::merkle_tree_prover::create_tree_blake(N,Com.Root_MT,sizeof(__hhash_digest),true);
        for(int i = 0; i < 32; i++){
            root[i] = Com.Root_MT[Com.Root_MT.size()-1][0].arr[i];
        }
    }
    myBcast(root,N);
    */
    
}

vector<F> delegate_checks(vector<F> &C_prime, vector<vector<F>> &C, 
                     MT &MT_C, vector<F> L, 
                     vector<vector<u32>> &queries, int iter, int N){
    
    
    int Segments = C_prime.size()/_degrees[iter+1];
    // Sample segments
    int l = _queries[iter+1];
    if(my_id == 0) vt.start();
    queries = get_indexes(l,N,Segments,my_id);
    if(my_id == 0) vt.end();
    
    // In the first interation the verifier directly queries the tensorcode, no delegation required
    vector<vector<F>> segments(queries.size());
    vector<F> y(queries.size());
    
    pt_cp.start();
    for(int i = 0; i < segments.size(); i++){
        for(int j = 0; j < _degrees[iter+1]; j++) segments[i].push_back(C_prime[queries[i][1] + j*Segments]);
        if(iter == 0) vt.start();
        y[i] = RS_multifold_segment(segments[i],L,C_prime.size(),queries[i][1]);
        if(iter == 0) vt.end();
    }
    // SEND SEGMENTS
    pt_cp.end();
    get_segments(y,segments, queries, l, N, _degrees[iter+1]);
    vector<int> queries2;
    if(my_id == 0){
        vector<MPI_Request> req(N-1);
        queries2.resize(l,0);
        if(iter != 0){
            pt_cp.start();
            
            whir_arrays[iter-1] = convert2vector(segments);
            whir_arrays[iter-1].resize(max_delegation_size,F(0));
            //printf(" >>>>  RATES: %d,%d\n",whir_rates[sorted_indexes[2+iter-1]],2+iter-1);
            if(whir_rates.size()){
                Whir_Commit_initial(whir_arrays[iter-1],whir_codewords[iter-1],whir_MT[iter-1],whir_rates[sorted_indexes[2+iter-1]],whir_kappas[whir_arrays[iter-1].size()]);
            }else{
                Whir_Commit_initial(whir_arrays[iter-1],whir_codewords[iter-1],whir_MT[iter-1],4,8);
            }
    
            // Compute the commitment using WHIR
            //printf("Delegation starting\n");
            
            if(next_pow2(segments.size()*segments[0].size()) != delegation_size[iter-1]){
                printf("Inconsistent Delegation size\n");
                exit(-1);
            }
            for(int i = 0; i < l; i++) queries2[i] = rand()%C[0].size();
            pt_cp.end();

        }else{
            pcs_ps += ((double)l*segments[0].size()*16)/1024.0;
        }
        com_rounds++;
        
        vector<u64> randomness(5*l);
        for(int i = 0; i < l; i++){
            
            
            randomness[5*i] = queries[i][0];
            randomness[5*i+1] = queries[i][1];
            randomness[5*i+2] = queries2[i];
            randomness[5*i+3] = y[i].real;
            randomness[5*i+4] = y[i].img;
        }
       
        for(int i = 1; i < N; i++){
            MPI_Isend(randomness.data(),randomness.size(),MPI_UINT64_T,i,0,MPI_COMM_WORLD,&req[i-1]);
        }
        cm += (N-1)*randomness.size()*8.0/1024.0;
        if(iter != 0){
            pt_cp.start();


            claims[iter-1].push_back(prove_multi_folding(segments, L, y,  queries, C_prime.size(),pcs_ps));

            vector<vector<F>> extended_segments(l);
            for(int i = 0; i < queries.size(); i++){
                extended_segments[i] = segments[i];
                extended_segments[i].resize(C[0].size(),F(0));
                fft(extended_segments[i],(int)log2(extended_segments[i].size()),false);
            } //extended_segments.push_back(C[queries[i][0]][queries[i][1]]);
            
            vector<F> o; 
            for(int i = 0; i < extended_segments.size(); i++){
                o.push_back(extended_segments[i][queries2[i]]);
            }
            //printf("%d\n",C[0].size()/_degrees[iter+1]);
            //claims[iter-1].push_back(prove_extension(segments, extended_segments, o, queries2, C[0].size()/_degrees[iter+1],pcs_ps));
            prove_extension(segments, extended_segments, o, queries2, C[0].size()/_degrees[iter+1],pcs_ps);
            pcs_ps += ((double)(3*o.size())*16)/1024.0;
            // Point query MT
            pt_cp.end();

        }
        
        for(int i = 0; i < N-1; i++){
            MPI_Wait(&req[i],MPI_STATUS_IGNORE);
        }

        if(iter != 0){
            //printf("Claims[%d], Size : %d, %d\n",iter-1,claims[iter-1].size(),(int)log2(delegation_size[iter-1]));
            for(int i = 0; i < claims[iter-1].size(); i++){
                if(claims[iter-1][i].second.size() != (int)log2(delegation_size[iter-1])){
                    printf("Error in Claims %d,%d\n",i,iter-1);
                    exit(-1);
                }
                int cur_size = claims[iter-1][i].second.size();
                for(int j = cur_size; j < (int)log2(max_delegation_size); j++){
                    claims[iter-1][i].second.push_back(F(0));
                }
            }
        
        }
    }
    else{
        y.clear();
        vector<u64> data(5*l);
        MPI_Request req;
        MPI_Irecv(data.data(),data.size(),MPI_UINT64_T,0,0,MPI_COMM_WORLD,&req);
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        queries.clear();
        for(int i = 0; i < data.size()/5; i++){
            queries.push_back({(u32)data[5*i],(u32)data[5*i+1]});
            queries2.push_back((int)data[5*i+2]);    
                
            F a;
            a.real = data[5*i+3];
            a.img = data[5*i+4];
            y.push_back(a);    
        }
    }
    
    vector<bool> visited(2*MT_C.Base_MT[0].size(),false);
    
    for(int i = 0; i < l; i++){
        if(queries[i][0] == my_id){
            int pos = queries[i][1];
            if(iter != 0) pos = queries[i][1]*C[0].size() + queries2[i];
            vector<_hash> path = merkle_tree::merkle_tree_prover::open_tree_blake(MT_C.Base_MT,pos/4);
            vt.start();
            merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(MT_C.Base_MT,path.data(),pos/4,MT_C.Base_MT[0].size(),visited,pcs_ps);
            vt.end();
        }
    }
    

    return y;

}

void commit_tensorcode(vector<F> &C, MT &MT, int N){
    pt_cp.start();
            
    merkle_tree::merkle_tree_prover::MT_commit_Blake(C.data(),MT.Base_MT,C.size());
    pt_cp.end();
            
    if(my_id != 0){
        MPI_Request req;
        MPI_Isend(MT.Base_MT[MT.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD,&req);
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        //MPI_Send(Com.Base_MT[Com.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD);
    }else{
        vector<_hash> recv_hashes(N);
        vector<MPI_Request> req(N);
        recv_hashes[0] = MT.Base_MT[MT.Base_MT.size()-1][0];
        for(int i = 1; i < N; i++){
            MPI_Irecv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,&req[i]);
            //MPI_Recv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);            
        }
        for(int i = 1; i < N; i++){
            MPI_Wait(&req[i],MPI_STATUS_IGNORE);
        }
        MT.Root_MT.resize((int)log2(N)+1);
        MT.Root_MT[0] = recv_hashes;
        merkle_tree::merkle_tree_prover::create_tree_blake(N,MT.Root_MT,sizeof(__hhash_digest),true);
    }
}

void PCS_open(MT &MT_C, vector<vector<F>> &Codeword, vector<vector<F>> &Commit_data, vector<vector<F>> input_vec,vector<vector<F>> _Lagrange_Evals_input, F K, vector<vector<int>> rates, int N){
    
    pt_cp.start();
    
    vector<F> r1,r2,r3;
    vector<F> b1,b2,b3;
    vector<F> aggr_rand;
    vector<F> r;
    // The individual instance size of the sumcheck
    vector<int> instance_size;
    
    int ctr = 0;
    if(input_vec.size() == 1){
        r = input_vec[0];
        vector<F> _r;
        for(int i = 0; i < (int)log2(N/4); i++) r3.push_back(r[ctr++]);   
        for(int i = 0; i < (int)log2(Commit_data.size()); i++) r1.push_back(r[ctr++]);   
        for(int i = 0; i < (int)log2(Commit_data[0].size()); i++) r2.push_back(r[ctr++]);   
        _r = r1; _r.insert(_r.end(),r2.begin(),r2.end());
        precompute_beta(r1,b1);precompute_beta(r2,b2);precompute_beta(r3,b3);
        
    }else{
        b1 = input_vec[0];
        b2 = input_vec[1];
        b3 = input_vec[2];
    }

    // First Step: Batching 
  
    MT MT_new_C;
    vector<F> C_prime,batched_commit_data;
    pt_cp.end();
    
    batch_commitments(MT_C,Codeword,Commit_data,MT_new_C,C_prime,batched_commit_data,b1,N);
   
    

    pt_cp.start();
    
    fft(b2,(int)log2(b2.size()),false);
    instance_size.push_back(Commit_data[0].size());

    
    // Discard old merkle tree once checks finished
    MT_C = MT_new_C;
    /*************************/
    fft(batched_commit_data,(int)log2(batched_commit_data.size()),true);
    
    
    // Need to do verification checks to ensure the validity of batched C
    
    // Second Step: Start Protocol 1 
    // Each prover computes its own beta
    vector<F> buff = b3,old_b3 = b3;
    buff.resize(N/2,F(0));
    fft(buff,(int)log2(buff.size()),true);
    buff.resize(2*N,F(0));
    fft(buff,(int)log2(buff.size()),false);
    b3.clear();
    for(int i = 0; i < N; i++) b3.push_back(buff[2*i+1]);
    // Betas used by each prover
    vector<F> B(b2.size());
    for(int j = 0; j < B.size(); j++) B[j] = b2[j]*b3[my_id];
    
    
    vector<F> c;
    vector<vector<vector<u32>>> queries(rates.size()-1);
    
    vector<F> C_prime_new;
    vector<vector<F>> C,C_new;
    vector<vector<F>> Lagrange_Evals;    
    pt_cp.end();
    
    F omega = getRootOfUnity(1+(int)log2(rate_offset)+(int)log2(batched_commit_data.size())).inv();
    F mul = F(1);
    for(int j = 0; j < batched_commit_data.size(); j++){
        batched_commit_data[j] = mul*batched_commit_data[j];
        mul = mul*omega;
    }
    omega = getRootOfUnity(1+(int)log2(rate_offset)+(int)log2(batched_commit_data.size()));
    //printf(">> %d\n",1+(int)log2(rate_offset)+(int)log2(batched_commit_data.size()));
    mul = F(1);
    for(int j = 0; j < batched_commit_data.size(); j++){
        B[j] = mul*B[j];
        b2[j] = mul*b2[j];
        mul = mul*omega;
    }

    //send_pcs_data(batched_commit_data,B,C_prime,recv_data,recv_B,recv_C,N);
    //int _d = 1;
    //for(int i = 1; i < _degrees.size(); i++) _d *= _degrees[i];
    //printf("%d,%d,%d\n",B.size(),batched_commit_data.size(),_d);
    
    //return;
    for(int i = 0; i < rates.size()-1; i++){
        //printf("Iter %lf Ps \n",pcs_ps);
        instance_size.push_back(batched_commit_data.size());
        // Sumcheck
        vector<F> poly = pcs_sumcheck_phase1(batched_commit_data, B, _degrees[i+1],N);
        
        if(my_id == 0){ 
            pcs_ps += ((double)poly.size()*16)/1024.0;
            vt.start();
        }
        F eval = F(0);
         
        // Sumcheck Verification
        for(int j = 0; j < poly.size()/2; j++) eval+= poly[2*j];
        if(eval != K){
            printf("Sumcheck Error Round %d\n",i);
            return;
        }
        
        // Sample a random challenge
        c.push_back(hash_to_field(poly));
        
        K = F(0);
        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(2*_degrees[i+1])),c[i],2*_degrees[i+1]);
	    for(int j = 0; j < L.size(); j++){
            K +=  poly[j]*L[j];
        }
        if(my_id == 0) vt.end();
        
        // Folding
        //L = compute_lagrange_coeff((int)log2(_degrees[i+1]),c[i],_degrees[i+1]);
        pt_cp.start();
        pcs_sumcheck_phase2(batched_commit_data, B, _degrees[i+1], c[i]);
        pt_cp.end();
        
            
        if(i < rates.size()-2){
            if(my_id == 0) vt.start();
            L = compute_lagrange_coeff((int)log2(_degrees[i+1]),c[i],_degrees[i+1]);
            if(my_id == 0) vt.end();
            if(my_id == 0){
                for(int j = 0; j < b2.size()/L.size(); j++){
                    b2[j] =  L[0]*b2[L.size()*j];
                    for(int k = 1; k < L.size(); k++){
                        b2[j] += L[k]*b2[L.size()*j+k];   
                    }
                }
                b2.resize(b2.size()/L.size());
            }
        
            Lagrange_Evals.push_back(L);
            pt_cp.start();
            encode_concatenated(batched_commit_data, C_prime_new, C_new, _rates[i+2], _degrees[i+2]);    
            pt_cp.end();
            
            // Commit
            commit_concatenated(C_new,MT_new_C,N);
            // Check Consistency
            if(my_id == 0)printf(">>>> %d\n",i);
            vector<F> y = delegate_checks(C_prime,C,MT_C,L,queries[i],i,N);
            
            // Update commitment
            C_prime = C_prime_new;
            C = C_new;
            MT_C = MT_new_C ;
            for(int j = 0; j < queries[i].size(); j++){
                queries[i][j][1] = _rates[i+2][0]*queries[i][j][1]/_rates[i+1][0];
                if(queries[i][j][0] == my_id){
                    if(C_prime[queries[i][j][1]] != y[j]){
                        printf(">>> Error %d\n",j);
                        exit(-1);
                    }
                }
            }
            
            // Update Sumcheck instance
            // The verifier samples the random challenge s
            //F s = F(0);//hash_to_field({});
            F s = hash_to_field({0});
            aggr_rand.push_back(s);
            pt_cp.start();
            pcs_sumcheck_phase3(B, queries[i], s, i+1,N);
            pt_cp.end();
            if(my_id== 0 )vt.start();
            F p = F(s);
            for(int j = 0; j < y.size(); j++){
                K += p*y[j];
                p*= s;
            }
            if(my_id== 0 )vt.end();
            
            
        }else{
            if(my_id == 0) vt.start();
            L = compute_lagrange_coeff((int)log2(_degrees[i+1]),c[i],_degrees[i+1]);
            if(my_id == 0) vt.end();
            if(my_id == 0){
                for(int j = 0; j < b2.size()/L.size(); j++){
                    b2[j] =  L[0]*b2[L.size()*j];
                    for(int k = 1; k < L.size(); k++){
                        b2[j] += L[k]*b2[L.size()*j+k];   
                    }
                }
                b2.resize(b2.size()/L.size());
            }
            Lagrange_Evals.push_back(L);
            
            C_prime_new = C_prime;
            pt_cp.start();
            RS_multifold5(C_prime_new,L,L.size());
            pt_cp.end();
            
            commit_tensorcode(C_prime_new, MT_new_C,N);
            // delegate checks, In the last round query complexity can be significatnly reduced
            if(my_id == 0)printf(">>>> %d\n",i);
            vector<F> y = delegate_checks(C_prime,C,MT_C,L,queries[i],i,N);
            
            /*
            for(int j = 0; j < queries[i].size(); j++){
                queries[i][j][1] = _rates[i+2][0]*queries[i][j][1]/_rates[i+1][0];
                if(queries[i][j][0] == my_id){
                    if(C_prime[queries[i][j][1]] != y[j]){
                        printf(">>> Error %d\n",j);
                        exit(-1);
                    }
                }
            }*/
            
            C_prime = C_prime_new;
            C = C_new;
            MT_C = MT_new_C;
            //F s = F(0);//
            F s = hash_to_field({0});
            aggr_rand.push_back(s);
            pt_cp.start();
            pcs_sumcheck_phase3(B, queries[i], s, i+1,N);
            pt_cp.end();
            if(my_id== 0 )vt.start();
            F p = F(s);
            for(int j = 0; j < y.size(); j++){
                //K += p*y[j];
                p*= s;
            }
            if(my_id== 0 )vt.end();
            
        }
        //return;
    }
    vector<vector<F>> recv_data,recv_B,recv_C;
    pcs_ps += ((double)2*N*16)/1024.0;
    
    send_pcs_data(batched_commit_data,B,C_prime,recv_data,recv_B,recv_C,N);
    if(my_id == 0){
        vector<F> aggr_codeword,aggr_arr,b_aggr;
        vector<vector<_hash>> aggr_MT;
        F whir_K = F(0);
        
        //Whir_Open(whir_arrays[1],whir_codewords[1],whir_MT[1],{claims[1][1]},4);
        vector<vector<F>> aggr_challenge = aggregate_claims(whir_arrays, aggr_arr, b_aggr, claims, whir_K );
        aggregate_commitments(aggr_codeword,aggr_MT,whir_codewords,whir_MT,aggr_challenge[1],4);        
        _Whir_Open(aggr_arr, aggr_codeword, aggr_MT, b_aggr, claims, aggr_challenge[0], whir_K, 4);
        pcs_ps += whir_ps; 
        vt.start();
        for(int i = 0; i < recv_C.size(); i++){
            fft(recv_C[i],(int)log2(recv_C[i].size()),true);
            for(int j = 0; j < recv_data[i].size(); j++){
                if(recv_C[i][j] != recv_data[i][j]){
                    printf("ERROR in decoding %d,%d\n",i,j);
                    return;
                }
            }
        }

        vector<F> final_poly = convert2vector(recv_data);
        vector<F> final_B = convert2vector(recv_B);
        final_poly = unpack_secret_share(final_poly,N/2,N);
        final_B = unpack_secret_share(final_B,N/2,N);
        
        for(int i = 0 ; i < final_poly.size(); i++) K -= final_poly[i]*final_B[i];
        if(K != 0) printf("> Error in final verification step 1\n");
        F B_eval_claim = 0;
        F p = F(1);
        F r_c = F::_random();
        
        for(int i = 0; i < final_B.size(); i++){
            B_eval_claim += p*final_B[i];
            p *= r_c;
        }
        p = r_c;
        b3 = old_b3;
        for(int i = 1; i < b3.size(); i++){
            b3[0] += p*b3[i];
            p *= r_c;
        }
       
        // Validate last step
        vector<F> partial_evals_x,partial_evals_y;
        partial_evals_x.push_back(b2[0]);
        partial_evals_y.push_back(b3[0]);
        //printf("%d,%d\n",Lagrange_Evals.size(),aggr_rand.size());
        vector<vector<F>> _Lagrange_Evals = Lagrange_Evals;
        for(int i = 2; i < _rates.size(); i++){
            p = aggr_rand[i-2];
            Lagrange_Evals.erase(Lagrange_Evals.begin());
            for(int j = 0; j < queries[i-2].size(); j++){
                partial_evals_x.push_back(p*fast_eval_powers_x(Lagrange_Evals,queries[i-2][j][1],instance_size[i],_rates[i][0]));
                p *= aggr_rand[i-2];
            }
            for(int j = 0; j < queries[i-2].size(); j++){
                partial_evals_y.push_back(fast_eval_powers_y(r_c, queries[i-2][j][0], N));
            }
            
        }
        //printf("%d,%d\n",partial_evals_x.size(),Lagrange_Evals.size());
        F final_eval = F(0);
        for(int i = 0; i < partial_evals_y.size(); i++) final_eval += partial_evals_x[i]*partial_evals_y[i];
        if(final_eval != B_eval_claim){
            printf("Error in final verification step\n");
            exit(-1);
        }
        vt.end();
        //delegate_beta_eval(r2, _Lagrange_Evals, b2[0], 1+(int)log2(rate_offset)+(int)log2(instance_size[0]), pcs_ps);
        delegate_lagrange_eval(_Lagrange_Evals_input, _Lagrange_Evals, b2[0], 1+(int)log2(rate_offset)+(int)log2(instance_size[0]), pcs_ps);
    }
}


void batch_commitments_new(MT &C, vector<vector<F>> &Codeword, 
           MT &batched_C, vector<F> &batched_Codeword, vector<F> b1, int N){
    
  
    pt_cp.start();
    
    
    vector<F> buff(_degrees[1]);
    vector<vector<_hash>> MT_buff;
    vector<_hash> hash_leaves(batched_Codeword.size()/_degrees[1]);
    for(int j = 0; j < batched_Codeword.size()/_degrees[1]; j++){
        for(int k = 0; k < buff.size(); k++){
            buff[k] = batched_Codeword[j + k*batched_Codeword.size()/_degrees[1]];
        }
        merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(),MT_buff,buff.size());
        hash_leaves[j] = MT_buff[MT_buff.size()-1][0];
    }
    
    batched_C.Base_MT.resize((int)log2(hash_leaves.size())+1);
    batched_C.Base_MT[0] = hash_leaves;
    pt_cp.end();
    
    distributed_commit(batched_C,  N);
    
    // TO DO: RUN Batch verification algorithm. Here we do not have to query the oracle of C_new, we will 
    // query it in the next step
    vector<vector<u32>> queries;
    if(my_id == 0) vt.start();
    queries = get_indexes(_queries[0],N,Codeword[0].size(),my_id);
    if(my_id == 0) vt.end();
    vector<vector<F>> replies(queries.size());
    vt.start();
    

  
    for(int i = 0; i < queries.size(); i++){
        F y = F(0);
        for(int k = 0; k < Codeword.size(); k++){
            if(Codeword[k].size() == 0) break;
            replies[i].push_back(Codeword[k][queries[i][1]]);
            y += b1[k]*replies[i][k];
        }
    }
    vt.end();
    if(queries.size() == 0) return;
    
    pcs_ps += ((double)(replies.size()*replies[0].size()*16))/1024.0;
    
    vector<bool> visited(2*C.Base_MT[0].size(),false);
    for(int i = 0; i<queries.size(); i++){
        vector<vector<_hash>> temp_MT;
        int pos = queries[i][1];
        vector<_hash> path;
        replies[i].resize((next_pow2(replies[i].size())),F(0));
        vt.start();
        merkle_tree::merkle_tree_prover::MT_commit_Blake(replies[i].data(),temp_MT,replies[i].size());
        vt.end();

        path = merkle_tree::merkle_tree_prover::open_tree_blake(C.Base_MT,pos);
        
        vt.start();
        merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(C.Base_MT,path.data(),pos,C.Base_MT[0].size(),visited,pcs_ps);
        vt.end();
    
    }

}

void PCS_open_new(MT &MT_C, vector<vector<F>> &Codeword, vector<vector<F>> &Commit_data, vector<vector<F>> input_vec, vector<vector<F>> _Lagrange_Evals_input, F K, vector<vector<int>> rates, int N, bool isPIOP ){
    
    pt_cp.start();
    
    vector<F> r1,r2,r3;
    vector<F> b1,b2,b3;
    vector<F> aggr_rand;
    vector<F> r;
    // The individual instance size of the sumcheck
    vector<int> instance_size;
    
    int ctr = 0;
    if(input_vec.size() == 1){
        r = input_vec[0];
        vector<F> _r;
        for(int i = 0; i < (int)log2(N/4); i++) r3.push_back(r[ctr++]);   
        for(int i = 0; i < (int)log2(Commit_data.size()); i++) r1.push_back(r[ctr++]);   
        for(int i = 0; i < (int)log2(Commit_data[0].size()); i++) r2.push_back(r[ctr++]);   
        _r = r1; _r.insert(_r.end(),r2.begin(),r2.end());
        precompute_beta(r1,b1);precompute_beta(r2,b2);precompute_beta(r3,b3);
        
    }else{
        b1 = input_vec[0];
        b2 = input_vec[1];
        b3 = input_vec[2];
        for(int i =  b2.size() - _queries[0]; i < b2.size(); i++){
            b2[i] = 0;
        }
    }

    // First Step: Batching 
  
    MT MT_new_C;
    vector<F> C_prime,batched_commit_data;
    
    int batches = 0;
    for(int i = 0; i < Codeword.size(); i++){
        if(Codeword[i].size() == 0) break;
        batches++;
    }
    b1.resize(batches);
    C_prime.resize(Codeword[0].size(),F(0));    
    batched_commit_data.resize(Commit_data[0].size(),F(0));
    for(int i = 0; i < batches-1; i++){
        for(int k  = 0; k < C_prime.size(); k++) C_prime[k] += b1[i]*Codeword[i][k];
        for(int k  = 0; k < batched_commit_data.size(); k++) batched_commit_data[k] += b1[i]*Commit_data[i][k];
    }
        
    fft(b2,(int)log2(b2.size()),false);
    instance_size.push_back(Commit_data[0].size());

    
    // Discard old merkle tree once checks finished
    //MT_C = MT_new_C;
    /*************************/
    fft(batched_commit_data,(int)log2(batched_commit_data.size()),true);
    fft(Commit_data[batches-1],(int)log2(batched_commit_data.size()),true);
    
    
    // Need to do verification checks to ensure the validity of batched C
    
    // Second Step: Start Protocol 1 
    // Each prover computes its own beta
    vector<F> buff = b3,old_b3 = b3;
    buff.resize(N/2,F(0));
    fft(buff,(int)log2(buff.size()),true);
    buff.resize(2*N,F(0));
    fft(buff,(int)log2(buff.size()),false);
    b3.clear();
    for(int i = 0; i < N; i++) b3.push_back(buff[2*i+1]);
    // Betas used by each prover
    vector<F> B(b2.size());
    for(int j = 0; j < B.size(); j++) B[j] = b2[j]*b3[my_id];
    
    
    vector<F> c;
    vector<vector<vector<u32>>> queries(rates.size()-1);
    
    vector<F> C_prime_new;
    vector<vector<F>> C,C_new;
    vector<vector<F>> Lagrange_Evals;    
    
    
    F omega = getRootOfUnity(1+(int)log2(rate_offset)+(int)log2(batched_commit_data.size())).inv();
    F mul = F(1);
    for(int j = 0; j < batched_commit_data.size(); j++){
        batched_commit_data[j] = mul*batched_commit_data[j];
        Commit_data[batches-1][j] = mul*Commit_data[batches-1][j];
        mul = mul*omega;
    }
    omega = getRootOfUnity(1+(int)log2(rate_offset)+(int)log2(batched_commit_data.size()));
    mul = F(1);
    for(int j = 0; j < batched_commit_data.size(); j++){
        B[j] = mul*B[j];
        b2[j] = mul*b2[j];
        mul = mul*omega;
    }
    
    //vector<F> ones(N,1);
    //F s = F_ip(batched_commit_data,B,ones,N/4,N/2,N);
    
    //if(s != K)printf("ERROR\n");
    //for(int i = 0; i  < batched_commit_data.size(); i++){
    //    s += batched_commit_data[i]*B[i];
    //}

    
    // Run first batching step
    quadratic_poly H = quadratic_poly(F_ZERO,F_ZERO,F_ZERO);
    linear_poly l1,l2;
    for(int i = 0; i < batched_commit_data.size(); i++){
        l1 = linear_poly(batched_commit_data[i]-Commit_data[batches-1][i],Commit_data[batches-1][i]);
        l2 = linear_poly(B[i],F(0));
        H = H + l1*l2;//quadratic_poly(B[i]*(batched_commit_data[i]-Commit_data[batches-1][i]),B[i]*Commit_data[batches-1][i],F_ZERO);
    }
    vector<F> poly = {H.a,H.b,H.c};
    pt_cp.end();
    poly = pcs_sumcheck_aggr(poly,N);
    if(my_id == 0) vt.start();
    if(poly[0]+poly[1] != K){
        printf("Error coPCS in first round\n");
        return;
    }
    F r_first = hash_to_field(poly);
    K = r_first*(poly[1] + r_first*poly[0]);
    if(my_id == 0) vt.end();
    pt_cp.start();
    if(my_id == 0) pcs_ps += ((double)poly.size()*16)/1024.0;
    for(int i = 0; i < batched_commit_data.size(); i++){
        //l1 = linear_poly(batched_commit_data[i]-Commit_data[batches][i],Commit_data[batches][i]);
        //l2 = linear_poly(B[i],F(0));
        B[i] = r_first*B[i];
        batched_commit_data[i] = r_first*batched_commit_data[i] + (F(1)-r_first)*Commit_data[batches-1][i];
        //H = H + quadratic_poly(B[i]*(batched_commit_data[i]-Commit_data[batches][i]),B[i]*Commit_data[batches][i],F_ZERO);
    }
    for(int i = 0; i < C_prime.size(); i++){
        C_prime[i] = r_first*C_prime[i] + (F(1)-r_first)*Codeword[batches-1][i];
    }
    for(int i = 0; i < b1.size(); i++){
        b1[i] = r_first*b1[i];
    }
    b1.push_back(F(1)-r_first);
    pt_cp.end();
    
    batch_commitments_new(MT_C, Codeword, MT_new_C, C_prime,  b1, N);
    for(int i = 0; i < rates.size()-1; i++){


        //printf("Iter %lf Ps \n",pcs_ps);
        instance_size.push_back(batched_commit_data.size());
        // Sumcheck
        vector<F> poly = pcs_sumcheck_phase1(batched_commit_data, B, _degrees[i+1],N);
        
        if(my_id == 0){ 
            pcs_ps += ((double)poly.size()*16)/1024.0;
            vt.start();
        }
        F eval = F(0);
         
        // Sumcheck Verification
        for(int j = 0; j < poly.size()/2; j++) eval+= poly[2*j];
        if(eval != K){
            printf("Sumcheck Error Round %d\n",i);
            return;
        }
        
        // Sample a random challenge
        c.push_back(hash_to_field(poly));
        
        K = F(0);
        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(2*_degrees[i+1])),c[i],2*_degrees[i+1]);
	    for(int j = 0; j < L.size(); j++){
            K +=  poly[j]*L[j];
        }
        if(my_id == 0) vt.end();
        
        // Folding
        //L = compute_lagrange_coeff((int)log2(_degrees[i+1]),c[i],_degrees[i+1]);
        pt_cp.start();
        pcs_sumcheck_phase2(batched_commit_data, B, _degrees[i+1], c[i]);
        pt_cp.end();
        
        if(i < rates.size()-2){
            if(my_id == 0) vt.start();
            L = compute_lagrange_coeff((int)log2(_degrees[i+1]),c[i],_degrees[i+1]);
            if(my_id == 0) vt.end();
            if(my_id == 0){
                for(int j = 0; j < b2.size()/L.size(); j++){
                    b2[j] =  L[0]*b2[L.size()*j];
                    for(int k = 1; k < L.size(); k++){
                        b2[j] += L[k]*b2[L.size()*j+k];   
                    }
                }
                b2.resize(b2.size()/L.size());
            }
        
            Lagrange_Evals.push_back(L);
            pt_cp.start();
            encode_concatenated(batched_commit_data, C_prime_new, C_new, _rates[i+2], _degrees[i+2]);    
            pt_cp.end();
            
           
            // Commit

            commit_concatenated(C_new,MT_new_C,N);
            // Check Consistency
            vector<F> y = delegate_checks(C_prime,C,MT_C,L,queries[i],i,N);
           
            
            // Update commitment_Lagrange_Evals_input
            C_prime = C_prime_new;
            C = C_new;
            MT_C = MT_new_C ;
            for(int j = 0; j < queries[i].size(); j++){
                queries[i][j][1] = _rates[i+2][0]*queries[i][j][1]/_rates[i+1][0];
                if(queries[i][j][0] == my_id){
                    if(C_prime[queries[i][j][1]] != y[j]){
                        printf(">>> Error %d\n",j);
                        exit(-1);
                    }
                }
            }
            
            // Update Sumcheck instance
            // The verifier samples the random challenge s
            //F s = F(0);//hash_to_field({});
            F s = hash_to_field({0});
            aggr_rand.push_back(s);
            pt_cp.start();
            pcs_sumcheck_phase3(B, queries[i], s, i+1,N);
            pt_cp.end();
            if(my_id== 0 )vt.start();
            F p = F(s);
            for(int j = 0; j < y.size(); j++){
                K += p*y[j];
                p*= s;
            }
            if(my_id== 0 )vt.end();
            
            
        }else{
          
          
            if(my_id == 0) vt.start();
            L = compute_lagrange_coeff((int)log2(_degrees[i+1]),c[i],_degrees[i+1]);
            if(my_id == 0) vt.end();
            if(my_id == 0){
                for(int j = 0; j < b2.size()/L.size(); j++){
                    b2[j] =  L[0]*b2[L.size()*j];
                    for(int k = 1; k < L.size(); k++){
                        b2[j] += L[k]*b2[L.size()*j+k];   
                    }
                }
                b2.resize(b2.size()/L.size());
            }
            Lagrange_Evals.push_back(L);
            
            C_prime_new = C_prime;
            pt_cp.start();
            RS_multifold5(C_prime_new,L,L.size());
            pt_cp.end();
            
            commit_tensorcode(C_prime_new, MT_new_C,N);
            // delegate checks, In the last round query complexity can be significatnly reduced
            vector<F> y = delegate_checks(C_prime,C,MT_C,L,queries[i],i,N);
            
            /*
            for(int j = 0; j < queries[i].size(); j++){
                queries[i][j][1] = _rates[i+2][0]*queries[i][j][1]/_rates[i+1][0];
                if(queries[i][j][0] == my_id){
                    if(C_prime[queries[i][j][1]] != y[j]){
                        printf(">>> Error %d\n",j);
                        exit(-1);
                    }
                }
            }*/
            
            C_prime = C_prime_new;
            C = C_new;
            MT_C = MT_new_C;
            //F s = F(0);//
            F s = hash_to_field({0});
            aggr_rand.push_back(s);
            pt_cp.start();
            pcs_sumcheck_phase3(B, queries[i], s, i+1,N);
            pt_cp.end();
            if(my_id== 0 )vt.start();
            F p = F(s);
            for(int j = 0; j < y.size(); j++){
                //K += p*y[j];
                p*= s;
            }
            if(my_id== 0 )vt.end();
            
        }
        
        //return;
    }
    
    vector<vector<F>> recv_data,recv_B,recv_C;
    if(my_id == 0)pcs_ps += ((double)N*16)/1024.0;
    send_pcs_data(batched_commit_data,B,C_prime,recv_data,recv_B,recv_C,N);
    
    if(my_id == 0){
        vector<F> aggr_codeword,aggr_arr,b_aggr;
        vector<vector<_hash>> aggr_MT;
        F whir_K = F(0);
        whir_ps = 0.0;
        
        if(isPIOP){
            whir_polys[sorted_indexes[2]] = whir_arrays[0];
            whir_polys[sorted_indexes[3]] = whir_arrays[1];
            whir_c[sorted_indexes[2]] = whir_codewords[0];
            whir_c[sorted_indexes[3]] = whir_codewords[1];
            whir_MTs[sorted_indexes[2]] = whir_MT[0];
            whir_MTs[sorted_indexes[3]] = whir_MT[1];
            precompute_beta( claims[0][0].second,whir_betas[sorted_indexes[2]]);
            precompute_beta( claims[1][0].second,whir_betas[sorted_indexes[3]]);
            whir_claims[sorted_indexes[2]] = claims[0][0].first;
            whir_claims[sorted_indexes[3]] = claims[1][0].first;
            vt.start();
            for(int i = 0; i < recv_C.size(); i++){
                fft(recv_C[i],(int)log2(recv_C[i].size()),true);
                for(int j = 0; j < recv_data[i].size(); j++){
                    if(recv_C[i][j] != recv_data[i][j]){
                        printf("ERROR in decoding %d,%d\n",i,j);
                        return;
                    }
                }
            }
            vector<F> final_poly = convert2vector(recv_data);
            vector<F> final_B = convert2vector(recv_B);
            final_poly = unpack_secret_share(final_poly,N/2,N);
            final_B = unpack_secret_share(final_B,N/2,N);
            
            for(int i = 0 ; i < final_poly.size(); i++) K -= final_poly[i]*final_B[i];
            if(K != 0) printf("Error in final verification step 1\n");
            F B_eval_claim = 0;
            F p = F(1);
            F r_c = F::_random();
        
            for(int i = 0; i < final_B.size(); i++){
                B_eval_claim += p*final_B[i];
                p *= r_c;
            }
            p = r_c;
            b3 = old_b3;
            for(int i = 1; i < b3.size(); i++){
                b3[0] += p*b3[i];
                p *= r_c;
            }
        
            // Validate last step
            vector<F> partial_evals_x,partial_evals_y;
            partial_evals_x.push_back(r_first*b2[0]);
            partial_evals_y.push_back(b3[0]);
            //printf("%d,%d\n",Lagrange_Evals.size(),aggr_rand.size());
            vector<vector<F>> _Lagrange_Evals = Lagrange_Evals;
            for(int i = 2; i < _rates.size(); i++){
                p = aggr_rand[i-2];
                Lagrange_Evals.erase(Lagrange_Evals.begin());
                for(int j = 0; j < queries[i-2].size(); j++){
                    partial_evals_x.push_back(p*fast_eval_powers_x(Lagrange_Evals,queries[i-2][j][1],instance_size[i],_rates[i][0]));
                    p *= aggr_rand[i-2];
                }
                for(int j = 0; j < queries[i-2].size(); j++){
                    partial_evals_y.push_back(fast_eval_powers_y(r_c, queries[i-2][j][0], N));
                }
                
            }
            //printf("%d,%d\n",partial_evals_x.size(),Lagrange_Evals.size());
            F final_eval = F(0);
            for(int i = 0; i < partial_evals_y.size(); i++) final_eval += partial_evals_x[i]*partial_evals_y[i];
            if(final_eval != B_eval_claim){
                printf("Error in final verification step\n");
                exit(-1);
            }
            vt.end();
            delegate_lagrange_eval(_Lagrange_Evals_input, _Lagrange_Evals, b2[0], 1+(int)log2(rate_offset)+(int)log2(instance_size[0]), _queries[0], pcs_ps);
        
            //delegate_beta_eval(r2, _Lagrange_Evals, b2[0], 1+(int)log2(rate_offset)+(int)log2(instance_size[0]), pcs_ps);    
        }else{
            vector<vector<F>> aggr_challenge = aggregate_claims(whir_arrays, aggr_arr, b_aggr, claims, whir_K );
            aggregate_commitments(aggr_codeword,aggr_MT,whir_codewords,whir_MT,aggr_challenge[1],4);        
            _Whir_Open(aggr_arr, aggr_codeword, aggr_MT, b_aggr, claims, aggr_challenge[0], whir_K, 4);
            pcs_ps += whir_ps; 
            vt.start();
            for(int i = 0; i < recv_C.size(); i++){
                fft(recv_C[i],(int)log2(recv_C[i].size()),true);
                for(int j = 0; j < recv_data[i].size(); j++){
                    if(recv_C[i][j] != recv_data[i][j]){
                        printf("ERROR in decoding %d,%d\n",i,j);
                        return;
                    }
                }
            }
            vector<F> final_poly = convert2vector(recv_data);
            vector<F> final_B = convert2vector(recv_B);
            final_poly = unpack_secret_share(final_poly,N/2,N);
            final_B = unpack_secret_share(final_B,N/2,N);
            
            for(int i = 0 ; i < final_poly.size(); i++) K -= final_poly[i]*final_B[i];
            if(K != 0) printf("Error in final verification step 1\n");
            F B_eval_claim = 0;
            F p = F(1);
            F r_c = F::_random();
        
            for(int i = 0; i < final_B.size(); i++){
                B_eval_claim += p*final_B[i];
                p *= r_c;
            }
            p = r_c;
            b3 = old_b3;
            for(int i = 1; i < b3.size(); i++){
                b3[0] += p*b3[i];
                p *= r_c;
            }
        
            // Validate last step
            vector<F> partial_evals_x,partial_evals_y;
            partial_evals_x.push_back(r_first*b2[0]);
            partial_evals_y.push_back(b3[0]);
            //printf("%d,%d\n",Lagrange_Evals.size(),aggr_rand.size());
            vector<vector<F>> _Lagrange_Evals = Lagrange_Evals;
            for(int i = 2; i < _rates.size(); i++){
                p = aggr_rand[i-2];
                Lagrange_Evals.erase(Lagrange_Evals.begin());
                for(int j = 0; j < queries[i-2].size(); j++){
                    partial_evals_x.push_back(p*fast_eval_powers_x(Lagrange_Evals,queries[i-2][j][1],instance_size[i],_rates[i][0]));
                    p *= aggr_rand[i-2];
                }
                for(int j = 0; j < queries[i-2].size(); j++){
                    partial_evals_y.push_back(fast_eval_powers_y(r_c, queries[i-2][j][0], N));
                }
                
            }
            //printf("%d,%d\n",partial_evals_x.size(),Lagrange_Evals.size());
            F final_eval = F(0);
            for(int i = 0; i < partial_evals_y.size(); i++) final_eval += partial_evals_x[i]*partial_evals_y[i];
            if(final_eval != B_eval_claim){
                printf("Error in final verification step\n");
                exit(-1);
            }
            vt.end();
            delegate_beta_eval(r2, _Lagrange_Evals, b2[0], 1+(int)log2(rate_offset)+(int)log2(instance_size[0]), pcs_ps);
        }
        //Whir_Open(whir_arrays[1],whir_codewords[1],whir_MT[1],{claims[1][1]},4);
        /*
        vector<vector<F>> aggr_challenge = aggregate_claims(whir_arrays, aggr_arr, b_aggr, claims, whir_K );
        aggregate_commitments(aggr_codeword,aggr_MT,whir_codewords,whir_MT,aggr_challenge[1],4);        
        _Whir_Open(aggr_arr, aggr_codeword, aggr_MT, b_aggr, claims, aggr_challenge[0], whir_K, 4);
        pcs_ps += whir_ps; 
        vt.start();
        for(int i = 0; i < recv_C.size(); i++){
            fft(recv_C[i],(int)log2(recv_C[i].size()),true);
            for(int j = 0; j < recv_data[i].size(); j++){
                if(recv_C[i][j] != recv_data[i][j]){
                    printf("ERROR in decoding %d,%d\n",i,j);
                    return;
                }
            }
        }
        vector<F> final_poly = convert2vector(recv_data);
        vector<F> final_B = convert2vector(recv_B);
        final_poly = unpack_secret_share(final_poly,N/2,N);
        final_B = unpack_secret_share(final_B,N/2,N);
        
        for(int i = 0 ; i < final_poly.size(); i++) K -= final_poly[i]*final_B[i];
        if(K != 0) printf("Error in final verification step 1\n");
        F B_eval_claim = 0;
        F p = F(1);
        F r_c = F::_random();
    
        for(int i = 0; i < final_B.size(); i++){
            B_eval_claim += p*final_B[i];
            p *= r_c;
        }
        p = r_c;
        b3 = old_b3;
        for(int i = 1; i < b3.size(); i++){
            b3[0] += p*b3[i];
            p *= r_c;
        }
       
        // Validate last step
        vector<F> partial_evals_x,partial_evals_y;
        partial_evals_x.push_back(r_first*b2[0]);
        partial_evals_y.push_back(b3[0]);
        //printf("%d,%d\n",Lagrange_Evals.size(),aggr_rand.size());
        vector<vector<F>> _Lagrange_Evals = Lagrange_Evals;
        for(int i = 2; i < _rates.size(); i++){
            p = aggr_rand[i-2];
            Lagrange_Evals.erase(Lagrange_Evals.begin());
            for(int j = 0; j < queries[i-2].size(); j++){
                partial_evals_x.push_back(p*fast_eval_powers_x(Lagrange_Evals,queries[i-2][j][1],instance_size[i],_rates[i][0]));
                p *= aggr_rand[i-2];
            }
            for(int j = 0; j < queries[i-2].size(); j++){
                partial_evals_y.push_back(fast_eval_powers_y(r_c, queries[i-2][j][0], N));
            }
            
        }
        //printf("%d,%d\n",partial_evals_x.size(),Lagrange_Evals.size());
        F final_eval = F(0);
        for(int i = 0; i < partial_evals_y.size(); i++) final_eval += partial_evals_x[i]*partial_evals_y[i];
        if(final_eval != B_eval_claim){
            printf("Error in final verification step\n");
            exit(-1);
        }
        vt.end();
        //return;
        delegate_lagrange_eval(_Lagrange_Evals_input, _Lagrange_Evals, b2[0], 1+(int)log2(rate_offset)+(int)log2(instance_size[0]), _queries[0], pcs_ps);
        */
        
        //delegate_beta_eval(r2, _Lagrange_Evals, b2[0], 1+(int)log2(rate_offset)+(int)log2(instance_size[0]), pcs_ps);
    }

}


