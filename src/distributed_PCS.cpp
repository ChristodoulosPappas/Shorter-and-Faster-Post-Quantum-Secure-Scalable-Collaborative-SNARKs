#include "distributed_PCS.h"
#include "timer.hpp"
int distributed_rate = 8;
int BATCH_SIZE = 8;
extern int my_id;
vector<int> plaintext_degrees;
vector<int> plaintext_queries;
extern double whir_ps;
double dpcs_ps = 0.0;
extern timer vt,pt_cp;
extern vector<F> mul_tree_row_data,mul_tree_codeword;
extern MT MulTree_Com;
extern vector<pair<F,vector<F>>> MulTree_Claims;
extern vector<vector<F>> whir_polys,whir_c,whir_betas;
extern vector<vector<vector<_hash>>> whir_MTs;
extern vector<F> whir_claims;
extern vector<int> whir_sizes;
extern vector<int> whir_rates;
extern vector<int> sorted_indexes;
extern vector<int> sorted_whir_rates;
extern bool conjectured_security;
extern int com_rounds;
extern double cm;

#include <unordered_map>
extern unordered_map<int, int> whir_kappas;

void prepare_data(int N, int rate){
    distributed_rate = rate; 
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id); //get my process id
    
    plaintext_degrees.push_back(BATCH_SIZE);
    int l; 
    if(conjectured_security){
        l = (int)(-100.0/log2(1.0- (1.0-1.0/(double)rate)));
    }else{
        l = (int)(-100.0/log2(sqrt(1.0/(double)rate)));
    }
    //int l = (int)(-100.0/(log2(sqrt(1.0/(double)rate))));
    
    int logn = (int)log2(N);
    int d1 = 1<<(logn - logn/2);
    plaintext_degrees.push_back(d1);
    int d2 = 1<<(logn/2);
    
    plaintext_queries.push_back(l);
    plaintext_queries.push_back(l);
    
    if(d2 > 1){
        plaintext_degrees.push_back(d2);
        plaintext_queries.push_back(l);
    }
}


void Batch_dCommit(vector<F> &arr, vector<F> &arr2, vector<vector<F>> &commit_data, vector<vector<F>> &Codeword, MT &Com, int rate, int N, int BATCH_SIZE){
    commit_data.clear();
    Codeword.clear();
    commit_data.resize(BATCH_SIZE);
    Codeword.resize(BATCH_SIZE);
    for(int i = 0; i < commit_data.size(); i++){
      
        commit_data[i].resize(arr.size()/BATCH_SIZE);
        for(int j = 0; j < commit_data[i].size(); j++) commit_data[i][j] = arr[i*commit_data[i].size()+j];
        
        Codeword[i] = commit_data[i];
        Codeword[i].resize(rate*Codeword[i].size(),F(0));
        for(int j = 0; j < commit_data[i].size(); j++) commit_data[i][j] = arr2[i*commit_data[i].size()+j];
    }

    distributed_batch_FFT(Codeword,(int)log2(N*Codeword[0].size()),false,N);
    vector<u8> root(32);
    

    Com.Base_MT.resize((int)log2(Codeword[0].size())+1);
    Com.Base_MT[0].resize(Codeword[0].size());
    vector<vector<_hash>> H;
    
    for(int j = 0; j < Codeword[0].size(); j++){
        vector<F> batch_vec(Codeword.size(),F(0));
        for(int k  = 0; k < BATCH_SIZE; k++){
            batch_vec[k] = Codeword[k][j];
        }
        // MT is not required and is overkill. Switch to chain hashing
        merkle_tree::merkle_tree_prover::MT_commit_Blake(batch_vec.data(),H,batch_vec.size());
        Com.Base_MT[0][j] = H[H.size()-1][0];
        H.clear();
        
        
    }   
    
    merkle_tree::merkle_tree_prover::create_tree_blake(Com.Base_MT[0].size(),Com.Base_MT,sizeof(__hhash_digest), true);
    
    vector<uint8_t> recv_buff(32*N), send_buff(32*N);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < 32; j++) send_buff[32*i + j] = 0;
    }
    MPI_Request req;
    MPI_Ialltoall(send_buff.data(),send_buff.size()/N,MPI_UINT8_T,recv_buff.data(),recv_buff.size()/N,MPI_UINT8_T,MPI_COMM_WORLD,&req);
    com_rounds++;
    cm += 32*N/1024.0;

    MPI_Wait(&req,MPI_STATUS_IGNORE);
    if(my_id == 0){
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
    if(my_id != 0){
        MPI_Request req;
        MPI_Isend(Com.Base_MT[Com.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD,&req);
        network_timer.start(); MPI_Wait(&req,MPI_STATUS_IGNORE); network_timer.end();
        //MPI_Send(Com.Base_MT[Com.Base_MT.size()-1][0].arr,32,MPI_UINT8_T,0,0,MPI_COMM_WORLD);
    }else{
        
        vector<_hash> recv_hashes(N);
        vector<MPI_Request> req(N);
        recv_hashes[0] = Com.Base_MT[Com.Base_MT.size()-1][0];
        for(int i = 1; i < N; i++){
            MPI_Irecv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,&req[i]);
            //network_timer.start(); MPI_Recv(recv_hashes[i].arr,32,MPI_UINT8_T,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); network_timer.end();            
        }
        
        
        for(int i = 1; i < N; i++){
            network_timer.start(); MPI_Wait(&req[i],MPI_STATUS_IGNORE); network_timer.end();
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




void dCommit(vector<F> &arr, vector<F> &codeword, MT &C, int rate, int N, int batch_size){
    codeword = arr;
    codeword.resize(rate*codeword.size(),F(0));
    distributed_FFT(codeword,(int)log2(N*codeword.size()),false,N);
    
    vector<_hash> hashes(codeword.size()/plaintext_degrees[1]);
    vector<vector<_hash>> temp_tree;

    // Distributed MT
    C.Base_MT.clear();
    C.Base_MT.resize((int)log2(codeword.size()/plaintext_degrees[1])+1);
    for(int i = 0; i < codeword.size()/plaintext_degrees[1]; i++){
        vector<F> buff(plaintext_degrees[1]);
        for(int j = 0; j < plaintext_degrees[1]; j++){
            buff[j] = codeword[j*(codeword.size()/plaintext_degrees[1]) + i];
        }
        merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(),temp_tree,buff.size());
        hashes[i] = temp_tree[temp_tree.size()-1][0];
    }
    C.Base_MT[0] = hashes;
    distributed_commit(C,  N);
}

void dBatch(vector<vector<F>> &commit_data, vector<vector<F>> &codeword, MT &C, 
            vector<F> &batched_data, vector<F> &batched_codeword, MT &C_batched,
            vector<F> &b, int N){
    
    batched_data.resize(commit_data[0].size(),F(0));
    batched_codeword.resize(codeword[0].size(),F(0));
    for(int i = 0; i < commit_data.size(); i++){
        for(int j = 0; j < commit_data[0].size(); j++) batched_data[j] += b[i]*commit_data[i][j];
        for(int j = 0; j < codeword[0].size(); j++) batched_codeword[j] += b[i]*codeword[i][j];
    }

    // Commit Batch

    vector<_hash> hashes(batched_codeword.size()/plaintext_degrees[1]);
    vector<vector<_hash>> temp_tree;
    C_batched.Base_MT.resize((int)log2(batched_codeword.size()/plaintext_degrees[1])+1);
    for(int i = 0; i < batched_codeword.size()/plaintext_degrees[1]; i++){
        vector<F> buff(plaintext_degrees[1]);
        for(int j = 0; j < plaintext_degrees[1]; j++){
            buff[j] = batched_codeword[j*(batched_codeword.size()/plaintext_degrees[1]) + i];
        }
        merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(),temp_tree,buff.size());
        hashes[i] = temp_tree[temp_tree.size()-1][0];
    }
    C_batched.Base_MT[0] = hashes;
    distributed_commit(C_batched,  N);

    // Note that the query part must be done after the verifier recieved all codeword commitments.  
    // This step is solely done to ease code development 
    
    vector<vector<u32>> queries;
    queries = get_indexes(plaintext_queries[0],N,codeword[0].size(),my_id);
    vector<vector<F>> replies(queries.size());


  
    for(int i = 0; i < queries.size(); i++){
        F y = F(0);
        for(int k = 0; k < codeword.size(); k++){
            if(codeword[k].size() == 0) break;
            replies[i].push_back(codeword[k][queries[i][1]]);
            y += b[k]*replies[i][k];
        }
    }

    dpcs_ps += ((double)(replies.size()*replies[0].size()*16))/1024.0;
    
    vector<bool> visited(2*C.Base_MT[0].size(),false);
    for(int i = 0; i<queries.size(); i++){
        vector<vector<_hash>> temp_MT;
        int pos = queries[i][1];
        vector<_hash> path;
        replies[i].resize((next_pow2(replies[i].size())),F(0));
        merkle_tree::merkle_tree_prover::MT_commit_Blake(replies[i].data(),temp_MT,replies[i].size());

        path = merkle_tree::merkle_tree_prover::open_tree_blake(C.Base_MT,pos);
        
        merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(C.Base_MT,path.data(),pos,C.Base_MT[0].size(),visited,dpcs_ps);
    }

}


void dBatch_batched(vector<vector<vector<F>>> &commit_data, vector<vector<vector<F>>> &codeword, vector<MT> &C, 
            vector<vector<F>> &batched_data, vector<vector<F>> &batched_codeword, vector<MT> &C_batched,
            vector<vector<F>> &b, int N){
    
    batched_data.resize(commit_data.size());
    batched_codeword.resize(commit_data.size());
    C_batched.resize(commit_data.size());
    for(int l = 0; l < commit_data.size(); l++){
        batched_data[l].resize(commit_data[l][0].size(),F(0));
        batched_codeword[l].resize(codeword[l][0].size(),F(0));
        for(int i = 0; i < commit_data[l].size(); i++){
            for(int j = 0; j < commit_data[l][0].size(); j++) batched_data[l][j] += b[l][i]*commit_data[l][i][j];
            for(int j = 0; j < codeword[l][0].size(); j++) batched_codeword[l][j] += b[l][i]*codeword[l][i][j];
        }

        // Commit Batch

        vector<_hash> hashes(batched_codeword[l].size()/plaintext_degrees[1]);
        vector<vector<_hash>> temp_tree;
        C_batched[l].Base_MT.resize((int)log2(batched_codeword[l].size()/plaintext_degrees[1])+1);
        for(int i = 0; i < batched_codeword[l].size()/plaintext_degrees[1]; i++){
            vector<F> buff(plaintext_degrees[1]);
            for(int j = 0; j < plaintext_degrees[1]; j++){
                buff[j] = batched_codeword[l][j*(batched_codeword[l].size()/plaintext_degrees[1]) + i];
            }
            merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(),temp_tree,buff.size());
            hashes[i] = temp_tree[temp_tree.size()-1][0];
        }
        C_batched[l].Base_MT[0] = hashes;
    }
    
    distributed_commit_batch(C_batched,  N);
    
    // Note that the query part must be done after the verifier recieved all codeword commitments.  
    // This step is solely done to ease code development 
    for(int l = 0; l < commit_data.size(); l++){
        vector<vector<u32>> queries;
        if(my_id == 0) vt.start();
        queries = get_indexes(plaintext_queries[0],N,codeword[l][0].size(),my_id);
        if(my_id == 0) vt.end();

        vector<vector<F>> replies(queries.size());

        vt.start();
        for(int i = 0; i < queries.size(); i++){
            F y = F(0);
            for(int k = 0; k < codeword[l].size(); k++){
                if(codeword[l][k].size() == 0) break;
                replies[i].push_back(codeword[l][k][queries[i][1]]);
                y += b[l][k]*replies[i][k];
                
            }
        }
        vt.end();
        

        dpcs_ps += ((double)(replies.size()*replies[0].size()*16))/1024.0;
        vector<bool> visited(2*C[l].Base_MT[0].size(),false);
        for(int i = 0; i<queries.size(); i++){
            vector<vector<_hash>> temp_MT;
            int pos = queries[i][1];
            vector<_hash> path;
            replies[i].resize((next_pow2(replies[i].size())),F(0));
            vt.start();
            merkle_tree::merkle_tree_prover::MT_commit_Blake(replies[i].data(),temp_MT,replies[i].size());
            vt.end();
        
            path = merkle_tree::merkle_tree_prover::open_tree_blake(C[l].Base_MT,pos);
            vt.start();
            merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(C[l].Base_MT,path.data(),pos,C[l].Base_MT[0].size(),visited,dpcs_ps);
            vt.end();
        }
        
    }

}

void PIOP_dBatch_batched_new(vector<vector<vector<F>>> &commit_data, vector<vector<vector<F>>> &codeword, vector<MT> &C, 
            vector<vector<F>> &batched_data, vector<vector<F>> &batched_codeword, vector<MT> &C_batched,
            vector<vector<F>> &b, vector<F> a, int N){
    
    pt_cp.start();
    
    batched_data.resize(1);
    batched_codeword.resize(1);
    C_batched.resize(1);


    vector<F> B = {1,a[0],a[1]};
    /*
    for(int i = 1; i < b.size(); i++){
        for(int j = 0; j < b[i].size(); j++) B.push_back( a[i-1]*b[i][j]);
    }
    */
    
    batched_data[0].resize(commit_data[0][0].size(),F(0));
    batched_codeword[0].resize(codeword[0][0].size(),F(0));
    int ctr = 0;
    for(int l = 0; l < commit_data.size(); l++){
        //for(int i = 0; i < commit_data[l].size(); i++){
            for(int j = 0; j < commit_data[l][0].size(); j++) batched_data[0][j] += B[l]*commit_data[l][0][j];
            for(int j = 0; j < codeword[l][0].size(); j++) batched_codeword[0][j] += B[l]*codeword[l][0][j];
            ctr++;
        //}
    }
    // Commit Batch
    vector<_hash> hashes(batched_codeword[0].size()/plaintext_degrees[1]);
    vector<vector<_hash>> temp_tree;
    C_batched[0].Base_MT.resize((int)log2(batched_codeword[0].size()/plaintext_degrees[1])+1);
    for(int i = 0; i < batched_codeword[0].size()/plaintext_degrees[1]; i++){
        vector<F> buff(plaintext_degrees[1]);
        for(int j = 0; j < plaintext_degrees[1]; j++){
            buff[j] = batched_codeword[0][j*(batched_codeword[0].size()/plaintext_degrees[1]) + i];
        }
        merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(),temp_tree,buff.size());
        hashes[i] = temp_tree[temp_tree.size()-1][0];
    }
    C_batched[0].Base_MT[0] = hashes;
    
    pt_cp.end();
    
    distributed_commit_batch(C_batched,  N);
    // Note that the query part must be done after the verifier recieved all codeword commitments.  
    // This step is solely done to ease code development 
    vector<vector<u32>> queries;
    if(my_id == 0) vt.start();
    queries = get_indexes(plaintext_queries[0],N,codeword[0][0].size(),my_id);
    if(my_id == 0) vt.end();
    if(queries.size() == 0) return;
    for(int l = 0; l < commit_data.size(); l++){
        
        vector<vector<F>> replies(queries.size());

        vt.start();
        for(int i = 0; i < queries.size(); i++){
            F y = F(0);
            for(int k = 0; k < codeword[l].size(); k++){
                if(codeword[l][k].size() == 0) break;
                replies[i].push_back(codeword[l][k][queries[i][1]]);
                y += b[l][k]*replies[i][k];
                
            }
        }
        vt.end();
        
        
        dpcs_ps += ((double)(replies.size()*replies[0].size()*16))/1024.0;
        vector<bool> visited(2*C[l].Base_MT[0].size(),false);
        for(int i = 0; i<queries.size(); i++){
            vector<vector<_hash>> temp_MT;
            int pos = queries[i][1];
            vector<_hash> path;
            replies[i].resize((next_pow2(replies[i].size())),F(0));
            vt.start();
            merkle_tree::merkle_tree_prover::MT_commit_Blake(replies[i].data(),temp_MT,replies[i].size());
            vt.end();
            path = merkle_tree::merkle_tree_prover::open_tree_blake(C[l].Base_MT,pos);
            vt.start();
            merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(C[l].Base_MT,path.data(),pos,C[l].Base_MT[0].size(),visited,dpcs_ps);
            vt.end();
        }
 
    }
    
}

void PIOP_dBatch_batched(vector<vector<vector<F>>> &commit_data, vector<vector<vector<F>>> &codeword, vector<MT> &C, 
            vector<vector<F>> &batched_data, vector<vector<F>> &batched_codeword, vector<MT> &C_batched,
            vector<vector<F>> &b, int N){
    
    pt_cp.start();
    
    batched_data.resize(2);
    batched_codeword.resize(2);
    C_batched.resize(2);
    for(int l = 0; l < 2; l++){
        batched_data[l].resize(commit_data[l][0].size(),F(0));
        batched_codeword[l].resize(codeword[l][0].size(),F(0));
        if(b[l].size() != commit_data[l].size()){
            printf("ERROR\n");
            exit(-1);
        }
        for(int i = 0; i < commit_data[l].size(); i++){
            for(int j = 0; j < commit_data[l][0].size(); j++) batched_data[l][j] += b[l][i]*commit_data[l][i][j];
            for(int j = 0; j < codeword[l][0].size(); j++) batched_codeword[l][j] += b[l][i]*codeword[l][i][j];
        }
        if(l == 0){
            for(int i = 0; i < commit_data[2].size(); i++){
                for(int j = 0; j < commit_data[2][0].size(); j++) batched_data[l][j] += b[2][i]*commit_data[2][i][j];
                for(int j = 0; j < codeword[2][0].size(); j++) batched_codeword[l][j] += b[2][i]*codeword[2][i][j];
            }
        }
        
        // Commit Batch
        
        vector<_hash> hashes(batched_codeword[l].size()/plaintext_degrees[1]);
        vector<vector<_hash>> temp_tree;
        C_batched[l].Base_MT.resize((int)log2(batched_codeword[l].size()/plaintext_degrees[1])+1);
        for(int i = 0; i < batched_codeword[l].size()/plaintext_degrees[1]; i++){
            vector<F> buff(plaintext_degrees[1]);
            for(int j = 0; j < plaintext_degrees[1]; j++){
                buff[j] = batched_codeword[l][j*(batched_codeword[l].size()/plaintext_degrees[1]) + i];
            }
            merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(),temp_tree,buff.size());
            hashes[i] = temp_tree[temp_tree.size()-1][0];
        }
        C_batched[l].Base_MT[0] = hashes;
    }
    pt_cp.end();
    
    distributed_commit_batch(C_batched,  N);
    
    // Note that the query part must be done after the verifier recieved all codeword commitments.  
    // This step is solely done to ease code development 
    for(int l = 0; l < commit_data.size(); l++){
        vector<vector<u32>> queries;
        if(my_id == 0) vt.start();
        queries = get_indexes(plaintext_queries[0],N,codeword[l][0].size(),my_id);
        if(my_id == 0) vt.end();

        vector<vector<F>> replies(queries.size());

        vt.start();
        for(int i = 0; i < queries.size(); i++){
            F y = F(0);
            for(int k = 0; k < codeword[l].size(); k++){
                if(codeword[l][k].size() == 0) break;
                replies[i].push_back(codeword[l][k][queries[i][1]]);
                y += b[l][k]*replies[i][k];
                
            }
        }
        vt.end();
        

        dpcs_ps += ((double)(replies.size()*replies[0].size()*16))/1024.0;
        vector<bool> visited(2*C[l].Base_MT[0].size(),false);
        for(int i = 0; i<queries.size(); i++){
            vector<vector<_hash>> temp_MT;
            int pos = queries[i][1];
            vector<_hash> path;
            replies[i].resize((next_pow2(replies[i].size())),F(0));
            vt.start();
            merkle_tree::merkle_tree_prover::MT_commit_Blake(replies[i].data(),temp_MT,replies[i].size());
            vt.end();
        
            path = merkle_tree::merkle_tree_prover::open_tree_blake(C[l].Base_MT,pos);
            vt.start();
            merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(C[l].Base_MT,path.data(),pos,C[l].Base_MT[0].size(),visited,dpcs_ps);
            vt.end();
        }
        
    }

}

vector<F> distributed_pcs_sumcheck_phase1(vector<F> &Commit_data, vector<F> &B, int degree, int N){
    int M = Commit_data.size();
    vector<F> poly(2*degree,F(0));
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
    
    //if(my_id == 0)printf("@ %lld,%lld\n",poly[0].real,poly[0].img);
    return aggregate_plaintext_poly(poly,  N);
}


vector<vector<F>> distributed_pcs_sumcheck_phase1_batch(vector<vector<F>> &Commit_data, vector<vector<F>> &B, int degree, int N){
    vector<vector<F>> poly(Commit_data.size());
    pt_cp.start();
    
    for(int l = 0; l < Commit_data.size(); l++){
        int M = Commit_data[l].size();
        poly[l].resize(2*degree,F(0));
        for(int j = 0; j < M/degree; j++){
            
            vector<F> h1(degree),h2(degree);    
            for(int k = 0; k < degree; k++){
                h1[k] = Commit_data[l][degree*j+k];
                h2[k] = B[l][degree*j+k];
            }
            fft(h1,(int)log2(h1.size()),true);
            fft(h2,(int)log2(h1.size()),true);
            h1.resize(2*h1.size(),F(0));h2.resize(2*h1.size(),F(0));
            fft(h1,(int)log2(h1.size()),false);fft(h2,(int)log2(h1.size()),false);
            
            for(int k = 0; k < 2*degree; k++) poly[l][k] += h1[k]*h2[k];
            
        }
    }

    pt_cp.end();
    
    return aggregate_plaintext_poly_batch(poly,  N);
}



void distributed_pcs_sumcheck_phase2(vector<F> &Commit_data, vector<F> &B, int degree, F r){
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

void commit_FRI_codeword(vector<F> &codeword, vector<F> b, MT &C, int degree, int N){
    vector<_hash> hashes(codeword.size()/degree);
    vector<vector<_hash>> temp_tree;
    C.Base_MT.clear();
    C.Base_MT.resize((int)log2(codeword.size()/degree)+1);
    for(int i = 0; i < codeword.size()/degree; i++){
        vector<F> buff(degree);
        for(int j = 0; j < degree; j++){
            buff[j] = codeword[j*(codeword.size()/degree) + i];
        }
        merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(),temp_tree,buff.size());
        hashes[i] = temp_tree[temp_tree.size()-1][0];
    }
    C.Base_MT[0] = hashes;
    distributed_commit(C,  N);

    vector<vector<u32>> queries;
    
    // Note that the query part must be done after the verifier recieved all codeword commitments.  
    // This step is solely done to ease code development 
    queries = get_indexes(plaintext_queries[0],N,codeword.size()/degree,my_id);
    vector<vector<F>> replies(queries.size());
    
    for(int i = 0; i < queries.size(); i++){
        F y = F(0);
        for(int k = 0; k < degree; k++){
            replies[i].push_back(codeword[k*codeword.size()/degree + queries[i][1]]);
            y += b[k]*replies[i][k];
        }
        y = RS_multifold_segment(replies[i],b,codeword.size()*N,0);
    }
    
    dpcs_ps += ((double)(replies.size()*replies[0].size()*16))/1024.0;
    
    vector<bool> visited(2*C.Base_MT[0].size(),false);
    for(int i = 0; i<queries.size(); i++){
        vector<vector<_hash>> temp_MT;
        int pos = queries[i][1];
        vector<_hash> path;
        replies[i].resize((next_pow2(replies[i].size())),F(0));
        merkle_tree::merkle_tree_prover::MT_commit_Blake(replies[i].data(),temp_MT,replies[i].size());

        path = merkle_tree::merkle_tree_prover::open_tree_blake(C.Base_MT,pos);
        
        merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(C.Base_MT,path.data(),pos,C.Base_MT[0].size(),visited,dpcs_ps);
    }
}

void commit_FRI_codeword_batch(vector<vector<F>> &codeword, vector<F> b, vector<MT> &C, int degree, int N){
    vector<vector<_hash>> temp_tree;
    pt_cp.start();
    
    for(int l  = 0; l < codeword.size(); l++){
        vector<_hash> hashes(codeword[l].size()/degree);
        
        C[l].Base_MT.clear();
        C[l].Base_MT.resize((int)log2(codeword[l].size()/degree)+1);
        for(int i = 0; i < codeword[l].size()/degree; i++){
            vector<F> buff(degree);
            for(int j = 0; j < degree; j++){
                buff[j] = codeword[l][j*(codeword[l].size()/degree) + i];
            }
            merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(),temp_tree,buff.size());
            hashes[i] = temp_tree[temp_tree.size()-1][0];
        }
        C[l].Base_MT[0] = hashes;
        
    }
    pt_cp.end();
    
    distributed_commit_batch(C,  N);
   
    vector<vector<u32>> queries;
    // Note that the query part must be done after the verifier recieved all codeword commitments.  
    // This step is solely done to ease code development 
    for(int l = 0; l < C.size(); l++){
        if(my_id == 0) vt.start();
        queries = get_indexes(plaintext_queries[0],N,codeword[l].size()/degree,my_id);
        if(my_id == 0) vt.end();
        if(queries.size() == 0) return;
        vector<vector<F>> replies(queries.size());
        for(int i = 0; i < queries.size(); i++){
            F y = F(0);
            vt.start();
            for(int k = 0; k < degree; k++){
                if(codeword[l].size() <= k*codeword[l].size()/degree + queries[i][1]){
                    printf("Incorrect dim\n",codeword[l].size(),k*codeword[l].size()/degree + queries[i][1]);
                    exit(-1);
                }
                replies[i].push_back(codeword[l][k*codeword[l].size()/degree + queries[i][1]]);
                y += b[k]*replies[i][k];
            }
            //y = RS_multifold_segment(replies[i],b,codeword[l].size()*N,0);
            vt.end();
        }
        
        dpcs_ps += ((double)(replies.size()*replies[0].size()*16))/1024.0;
        
        vector<bool> visited(2*C[l].Base_MT[0].size(),false);
        for(int i = 0; i<queries.size(); i++){
            vector<vector<_hash>> temp_MT;
            int pos = queries[i][1];
            vector<_hash> path;
            replies[i].resize((next_pow2(replies[i].size())),F(0));
            vt.start();
            merkle_tree::merkle_tree_prover::MT_commit_Blake(replies[i].data(),temp_MT,replies[i].size());
            vt.end();

            path = merkle_tree::merkle_tree_prover::open_tree_blake(C[l].Base_MT,pos);
            vt.start();
            merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(C[l].Base_MT,path.data(),pos,C[l].Base_MT[0].size(),visited,dpcs_ps);
            vt.end();

        }
    }
    
}

void dOpen(vector<vector<F>> &commit_data, vector<vector<F>> &codewords, MT &C, vector<F> r, F y, int N){
    
    vector<F> r1,r2,r3,b1,b2,b3;
    int ctr = 0;
    for(int i = 0; i < (int)log2(commit_data[0].size()); i++){
        r2.push_back(r[ctr++]);
    }
    for(int i = 0; i < (int)log2(N); i++){
        r1.push_back(r[ctr++]);
    }
    for(int i = 0; i < (int)log2(BATCH_SIZE); i++){
        r3.push_back(r[ctr++]);
    }
    

    precompute_beta(r1,b1);
    precompute_beta(r2,b2);
    precompute_beta(r3,b3);

    vector<F> arr,codeword;

    vector<MT> FRI_C(plaintext_degrees.size()-1);
    vector<vector<F>> FRI_codewords(plaintext_degrees.size()-1);
    MT C_new;
    dBatch(commit_data,codewords,C,arr,codeword,C_new,b3,N);
    C = C_new;
    vector<F> B = b2;
    
    int fold_depth = N;

    for(int i = 0; i  < B.size(); i++) B[i] *= b1[my_id];
    F K = y;
    vector<F> c;
    for(int i = 0; i < plaintext_degrees.size()-1; i++){
        FRI_codewords[i] = codeword;
        FRI_C[i] = C;
        vector<F> poly = distributed_pcs_sumcheck_phase1(arr, B, plaintext_degrees[i+1], N);
        
        F eval = F(0); 
        // Sumcheck Verification
        for(int j = 0; j < poly.size()/2; j++) eval+= poly[2*j];
        if(eval != K){
            if(my_id == 0) printf("Sumcheck Error Round %d\n",i);
            return;
        }
        // Sample a random challenge
        c.push_back(hash_to_field(poly));
        
        K = F(0);
        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(2*plaintext_degrees[i+1])),c[i],2*plaintext_degrees[i+1]);
	    for(int j = 0; j < L.size(); j++){
            K +=  poly[j]*L[j];
        }
        
        // Folding
        L = compute_lagrange_coeff((int)log2(plaintext_degrees[i+1]),c[i],plaintext_degrees[i+1]);
        distributed_pcs_sumcheck_phase2(arr, B, plaintext_degrees[i+1], c[i]);
        //RS_multifold5(codeword, L, L.size());
        RS_distributed_fold(codeword, L, L.size(), codeword.size()/fold_depth, N, my_id);
        fold_depth = fold_depth/L.size();
        
        //send_codeword(arr,B,codeword,fold_depth,N);
        if(i < plaintext_degrees.size()-2){
            commit_FRI_codeword(codeword,L,C,plaintext_degrees[i+1],N);   
        }else{
            send_codeword(arr,B,codeword,fold_depth,N);
        }       
    }
    if(my_id == 0){
        vector<vector<_hash>> MT;
        
        Whir_Commit_Codeword(codeword, MT, distributed_rate);
        pair<F,vector<F>> claim = _Whir_Open(arr, codeword, MT, B, {}, {}, K, distributed_rate);
        vector<vector<F>> L(c.size());
        vector<F> _r = r2;
        _r.insert(_r.end(),r1.begin(),r1.end());
        
        int ctr = 0;
        F _y = F(1);
        for(int i = 0; i < L.size(); i++){
            vector<F> temp_r;
            L[i] = compute_lagrange_coeff((int)log2(plaintext_degrees[i+1]),c[i],plaintext_degrees[i+1]);
            for(int j = 0; j < (int)log2(L[i].size()); j++) temp_r.push_back(_r[ctr++]);
            _y *= evaluate_vector(L[i],temp_r);
        }
        vector<F> temp_r;
        for(int j = ctr; j < _r.size(); j++) temp_r.push_back(_r[j]);
        _y *= beta_identity(claim.second,temp_r);
        if(_y != claim.first){
            printf("Error in verification 4");
            return;
        }
    }
}




void dOpen_batch(vector<vector<vector<F>>> &commit_data, vector<vector<vector<F>>> &codewords, vector<MT> &C, vector<vector<F>> r, vector<F> y, int N){
    int batches = commit_data.size();
    vector<vector<F>> r1(batches),r2(batches),r3(batches),b1(batches),b2(batches),b3(batches);
    for(int l = 0; l < batches; l++){
      int ctr = 0;
        for(int i = 0; i < (int)log2(commit_data[l][0].size()); i++){
            r2[l].push_back(r[l][ctr++]);
        }
        for(int i = 0; i < (int)log2(N); i++){
            r1[l].push_back(r[l][ctr++]);
        }
        for(int i = 0; i < (int)log2(next_pow2(commit_data[l].size())); i++){
            r3[l].push_back(r[l][ctr++]);
        }
        
        precompute_beta(r1[l],b1[l]);
        precompute_beta(r2[l],b2[l]);
        precompute_beta(r3[l],b3[l]);

    }

    vector<vector<F>> arr(batches),codeword(batches);
    vector<MT> C_new(batches);
    vector<vector<F>> B = b2;
    dBatch_batched(commit_data,codewords,C,arr,codeword,C_new,b3,N);
    for(int l = 0; l < batches; l++){
        //dBatch(commit_data[l],codewords[l],C[l],arr[l],codeword[l],C_new[l],b3[l],N);
        for(int i = 0; i  < B[l].size(); i++) B[l][i] *= b1[l][my_id];
    }
    C = C_new;
    vector<vector<MT>> FRI_C(batches);
    vector<vector<vector<F>>> FRI_codewords(batches);
    for(int l = 0; l < batches; l++){
        FRI_C[l].resize(plaintext_degrees.size()-1);
        FRI_codewords[l].resize(plaintext_degrees.size()-1);
    }
    
    int fold_depth = N;

    vector<F> K = y;
    vector<F> c;
    
    for(int i = 0; i < plaintext_degrees.size()-1; i++){
        for(int l = 0; l  < batches; l++){
            FRI_codewords[l][i] = codeword[l];
            FRI_C[l][i] = C[l];
        }

        vector<vector<F>> poly = distributed_pcs_sumcheck_phase1_batch(arr, B, plaintext_degrees[i+1], N);
        
        if(my_id == 0)vt.start();
        for(int l = 0 ;l < batches; l++){
            F eval = F(0); 
            // Sumcheck Verification
            if(my_id == 0)dpcs_ps += poly[l].size()*16/1024.0;
            //if(my_id == 0)printf("%lld,%lld\n",K[l].real,K[l].img);
            for(int j = 0; j < poly[l].size()/2; j++) eval += poly[l][2*j];
            if(eval != K[l]){
                if(my_id == 0) printf("Sumcheck Error Round %d, %d\n",i,l);
                return;
            }    
        }
        //return;
        // Sample a random challenge
        vector<F> buff = convert2vector(poly);
        c.push_back(hash_to_field(buff));
        pt_cp.start();
    
        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(2*plaintext_degrees[i+1])),c[i],2*plaintext_degrees[i+1]);
	    for(int l = 0; l  < batches; l++){
            K[l] = 0;  
            for(int j = 0; j < L.size(); j++){
                K[l] +=  poly[l][j]*L[j];
            }
        }
        // Folding
        L = compute_lagrange_coeff((int)log2(plaintext_degrees[i+1]),c[i],plaintext_degrees[i+1]);
        if(my_id == 0)vt.end();
        
        
        for(int l  =0 ; l < batches; l++){
            distributed_pcs_sumcheck_phase2(arr[l], B[l], plaintext_degrees[i+1], c[i]);
            RS_distributed_fold(codeword[l], L, L.size(), codeword[l].size()/fold_depth, N, my_id);
        }
        fold_depth = fold_depth/L.size();
        pt_cp.end();
    
        //send_codeword(arr,B,codeword,fold_depth,N);
        if(i < plaintext_degrees.size()-2){
            commit_FRI_codeword_batch(codeword,L,C,plaintext_degrees[i+1],N);   
           
        }else{
            //for(int l  =0 ; l < arr.size(); l++){
            //    send_codeword(arr[l],B[l],codeword[l],fold_depth,N);
            //}
            send_codeword_batch(arr,B,codeword,fold_depth,N);
        }  
      
    }
    if(my_id == 0){
        vector<vector<_hash>> MT;
        
       pt_cp.start();
    
        for(int l = 0; l < codeword.size(); l++){
            whir_ps = 0;
            Whir_Commit_Codeword(codeword[l], MT, distributed_rate);
            pair<F,vector<F>> claim = _Whir_Open(arr[l], codeword[l], MT, B[l], {}, {}, K[l], distributed_rate);
            dpcs_ps += whir_ps;
            vector<vector<F>> L(c.size());
            vector<F> _r = r2[l];
            vt.start();
            _r.insert(_r.end(),r1[l].begin(),r1[l].end());
            
            int ctr = 0;
            F _y = F(1);
            for(int i = 0; i < L.size(); i++){
                vector<F> temp_r;
                L[i] = compute_lagrange_coeff((int)log2(plaintext_degrees[i+1]),c[i],plaintext_degrees[i+1]);
                for(int j = 0; j < (int)log2(L[i].size()); j++) temp_r.push_back(_r[ctr++]);
                _y *= evaluate_vector(L[i],temp_r);
            }
            vector<F> temp_r;
            for(int j = ctr; j < _r.size(); j++) temp_r.push_back(_r[j]);
            _y *= beta_identity(claim.second,temp_r);
            if(_y != claim.first){
                printf("Error in verification 4");
                return;
            }
            vt.end();
            
        }

        pt_cp.end();
    
    }

}



void PIOP_dOpen_batch(vector<vector<vector<F>>> &commit_data, vector<vector<vector<F>>> &codewords, vector<MT> &C, vector<vector<F>> r, vector<F> y, vector<bool> Verify, int N){
    int batches = commit_data.size()-1;
    vector<vector<F>> r1(batches+1),r2(batches+1),r3(batches+1),b1(batches+1),b2(batches+1),b3(batches+1);
    if(commit_data[0][0].size() != commit_data[2][0].size()){
        printf("Incorrect sizes %d,%d\n",commit_data[0][0].size(),commit_data[2][0].size());
        exit(-1);
    }
    
    pt_cp.start();
    
    for(int l = 0; l < batches+1; l++){
        int ctr = 0;
        
        for(int i = 0; i < (int)log2(commit_data[l][0].size()); i++){
            r2[l].push_back(r[l][ctr++]);
        }
        for(int i = 0; i < (int)log2(next_pow2(commit_data[l].size())); i++){
            r3[l].push_back(r[l][ctr++]);
        }
        
        
        for(int i = 0; i < (int)log2(N); i++){
            r1[l].push_back(r[l][ctr++]);
        }
        
        precompute_beta(r1[l],b1[l]);
        precompute_beta(r2[l],b2[l]);
        precompute_beta(r3[l],b3[l]);

    }
    
    F K1 = F(0),K2 = F(0);
    for(int i = 0; i < commit_data[0][0].size(); i++){
        F t = 0;
        for(int j = 0; j < commit_data[2].size(); j++){
            t += commit_data[2][j][i]*b3[2][j];
        }
        K1 +=  b2[0][i]*t;
        t = 0;
        for(int j = 0; j < commit_data[0].size(); j++){
            t += commit_data[0][j][i]*b3[0][j];
        }
        K2 += b2[2][i]*t;
    }
    K1 = K1*b1[0][my_id];
    K2 = K2*b1[2][my_id];
    
    vector<u64> send_buff(N*4),recv_buff(N*4);
    for(int i = 0; i < N; i++){
        send_buff[4*i] = K1.real;
        send_buff[4*i+1] = K1.img;
        send_buff[4*i+2] = K2.real;
        send_buff[4*i+3] = K2.img;
    }
    MPI_Request req;
    pt_cp.end();
    
    MPI_Ialltoall(send_buff.data(),4,MPI_UINT64_T , recv_buff.data(),4,MPI_UINT64_T,MPI_COMM_WORLD,&req);
    network_timer.start(); MPI_Wait(&req,MPI_STATUS_IGNORE); network_timer.end();
    pt_cp.start();
    
    K1 = 0;K2 = 0;
    for(int i = 0; i < N; i++){
        F t; 
        t.real = recv_buff[4*i];
        t.img = recv_buff[4*i+1];
        K1 += t;
        t.real = recv_buff[4*i+2];
        t.img = recv_buff[4*i+3];
        K2 += t;
    }
    
    
    F a = hash_to_field({K1,K2});
    F b = hash_to_field({0});
    y[0] = (y[0] + a*K2) + b*(K1 + a*y[2]);
    
    vector<vector<F>> arr(batches),codeword(batches);
    vector<MT> C_new(batches);
    vector<vector<F>> B;
    for(int i = 0; i < 2; i++){
        B.push_back(b2[i]);
    }
    
    for(int i = 0; i < codewords[2].size(); i++){
        
        for(int j = 0; j < codewords[2][i].size(); j++) codewords[2][i][j] *= b;
        for(int j = 0; j < commit_data[2][i].size(); j++) commit_data[2][i][j] *= b;
        
    }
    
    pt_cp.end();
    
    
    PIOP_dBatch_batched(commit_data,codewords,C,arr,codeword,C_new,b3,N);
    
    pt_cp.start();
    for(int l = 0; l < batches; l++){
        //dBatch(commit_data[l],codewords[l],C[l],arr[l],codeword[l],C_new[l],b3[l],N);
        for(int i = 0; i  < B[l].size(); i++){
            if(l == 0){
                B[l][i] = B[l][i]*b1[l][my_id] + a*b2[2][i]*b1[2][my_id];
            }else{
                B[l][i] *= b1[l][my_id];
            }
        } 
    }
    
    C = C_new;
    vector<vector<MT>> FRI_C(batches);
    vector<vector<vector<F>>> FRI_codewords(batches);
    for(int l = 0; l < batches; l++){
        FRI_C[l].resize(plaintext_degrees.size()-1);
        FRI_codewords[l].resize(plaintext_degrees.size()-1);
    }
    
    int fold_depth = N;

    vector<F> K = y;
    vector<F> c;
    pt_cp.end();
    for(int i = 0; i < plaintext_degrees.size()-1; i++){
        for(int l = 0; l  < batches; l++){
            FRI_codewords[l][i] = codeword[l];
            FRI_C[l][i] = C[l];
        }

        vector<vector<F>> poly = distributed_pcs_sumcheck_phase1_batch(arr, B, plaintext_degrees[i+1], N);
        
        if(my_id == 0)vt.start();
        for(int l = 0 ;l < batches; l++){
            F eval = F(0); 
            // Sumcheck Verification
            if(my_id == 0)dpcs_ps += poly[l].size()*16/1024.0;
            //if(my_id == 0)printf("%lld,%lld\n",K[l].real,K[l].img);
            for(int j = 0; j < poly[l].size()/2; j++) eval += poly[l][2*j];
            if(eval != K[l] && Verify[l] == true){
                if(my_id == 0) printf("Sumcheck Error Round %d, %d\n",i,l);
                return;
            }    
        }
        //return;
        // Sample a random challenge
        vector<F> buff = convert2vector(poly);
        c.push_back(hash_to_field(buff));
        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(2*plaintext_degrees[i+1])),c[i],2*plaintext_degrees[i+1]);
	    for(int l = 0; l  < batches; l++){
            K[l] = 0;  
            for(int j = 0; j < L.size(); j++){
                K[l] +=  poly[l][j]*L[j];
            }
        }
        // Folding
        L = compute_lagrange_coeff((int)log2(plaintext_degrees[i+1]),c[i],plaintext_degrees[i+1]);
        if(my_id == 0)vt.end();
        
        
        for(int l  =0 ; l < batches; l++){
            distributed_pcs_sumcheck_phase2(arr[l], B[l], plaintext_degrees[i+1], c[i]);
            RS_distributed_fold(codeword[l], L, L.size(), codeword[l].size()/fold_depth, N, my_id);
        }
        fold_depth = fold_depth/L.size();
        
        //send_codeword(arr,B,codeword,fold_depth,N);
        if(i < plaintext_degrees.size()-2){
            commit_FRI_codeword_batch(codeword,L,C,plaintext_degrees[i+1],N);   
           
        }else{
            //for(int l  =0 ; l < arr.size(); l++){
            //    send_codeword(arr[l],B[l],codeword[l],fold_depth,N);
            //}
            send_codeword_batch(arr,B,codeword,fold_depth,N);
        }  
      
    }
    
    if(my_id == 0){
        vector<vector<_hash>> MT;

        
        for(int l = 0; l < codeword.size(); l++){
            
            whir_ps = 0;
            
            /*
            F _k = 0;
            vector<F> buff = arr[l];
            printf("%d,%d\n",codeword[l].size(),arr[l].size());
            buff.resize(codeword[l].size(),F(0));
            fft(codeword[l],(int)log2(codeword[l].size()),true);
            for(int i = codeword[l].size()/4; i < codeword[l].size(); i++){
                if(codeword[l][i] != F(0)){
                    printf("Error %d\n",i);
                    exit(-1);
                }
            }
            */
            
            Whir_Commit_Codeword(codeword[l], MT, distributed_rate);
            pair<F,vector<F>> claim = _Whir_Open(arr[l], codeword[l], MT, B[l], {}, {}, K[l], distributed_rate, Verify[l]);
            dpcs_ps += whir_ps;
            vector<vector<F>> L(c.size());
            vector<F> _r = r2[l];
            vector<F> _r2;

            vt.start();
            _r.insert(_r.end(),r1[l].begin(),r1[l].end());
            if(l == 0){
                _r2 = r2[2];
                _r2.insert(_r2.end(),r1[2].begin(),r1[2].end());
            }
            int ctr = 0;
            int ctr2=  0;
            F _y = F(1);
            F _y2 = F(1);
            for(int i = 0; i < L.size(); i++){
                vector<F> temp_r;
                vector<F> temp_r2;

                L[i] = compute_lagrange_coeff((int)log2(plaintext_degrees[i+1]),c[i],plaintext_degrees[i+1]);
                for(int j = 0; j < (int)log2(L[i].size()); j++) temp_r.push_back(_r[ctr++]);

                _y *= evaluate_vector(L[i],temp_r);
                if(l == 0){
                    for(int j = 0; j < (int)log2(L[i].size()); j++) temp_r2.push_back(_r2[ctr2++]);
                    _y2 *= evaluate_vector(L[i],temp_r2);
                }
            }
            vector<F> temp_r;
            vector<F> temp_r2;
            for(int j = ctr; j < _r.size(); j++) temp_r.push_back(_r[j]);
            if(l == 0){
                for(int j = ctr2; j < _r.size(); j++) temp_r2.push_back(_r2[j]);
                _y2 *= beta_identity(claim.second,temp_r2);
            }else{
                _y2 = 0;
            }
            _y *= beta_identity(claim.second,temp_r);
            if(_y + a*_y2 != claim.first && Verify[l] == true){
                printf("Error in verification 4\n");
                return;
            }
            vt.end();
            
        }

        
    }

}




void PIOP_dOpen_batch_new(vector<vector<vector<F>>> &commit_data, vector<vector<vector<F>>> &codewords, vector<MT> &C, vector<vector<F>> r, vector<F> y, vector<bool> Verify, int N){
    int batches = 1;
    int polys = commit_data.size();
    vector<vector<F>> r1(polys),r2(polys),r3(polys),b1(polys),b2(polys),b3(polys);
    if(polys == 3){
        if(commit_data[0][0].size() != commit_data[2][0].size()){
            printf("Incorrect sizes %d,%d\n",commit_data[0][0].size(),commit_data[2][0].size());
            exit(-1);
        }
    }
    
    pt_cp.start();
    
    for(int l = 0; l < polys; l++){
        int ctr = 0;
        
        for(int i = 0; i < (int)log2(commit_data[l][0].size()); i++){
            r2[l].push_back(r[l][ctr++]);
        }
        for(int i = 0; i < (int)log2(next_pow2(commit_data[l].size())); i++){
            r3[l].push_back(r[l][ctr++]);
        }
        
        
        for(int i = 0; i < (int)log2(N); i++){
            r1[l].push_back(r[l][ctr++]);
        }
        
        precompute_beta(r1[l],b1[l]);
        precompute_beta(r2[l],b2[l]);
        precompute_beta(r3[l],b3[l]);
    }
    vector<F> eval_claims;
    for(int i = 0; i < commit_data.size(); i++){
        for(int k  = 0; k < commit_data[0][0].size(); k++){
            commit_data[i][0][k] = b3[i][0]*commit_data[i][0][k];
            for(int h = 1; h < commit_data[i].size(); h++){
                commit_data[i][0][k] += b3[i][h]*commit_data[i][h][k];
            }
        }
        for(int k = 0; k < codewords[0][0].size(); k++){
            codewords[i][0][k] = b3[i][0]*codewords[i][0][k];
            for(int h = 1; h < commit_data[i].size(); h++){
                codewords[i][0][k] += b3[i][h]*codewords[i][h][k];
            }
        }
    }
 

    for(int i = 0; i < commit_data.size(); i++){
        for(int j = 0; j < commit_data.size(); j++){
            F K1 = 0;
            if(j == i) continue;
            for(int k  = 0; k < commit_data[0][0].size(); k++){
                K1 += b2[j][k]*commit_data[i][0][k];
            }
            K1 *= b1[j][my_id];
            eval_claims.push_back(K1);
        }
    }
    
    vector<u64> send_buff,recv_buff;
    field_vector_serialize(eval_claims,send_buff);
    vector<u64> _buff;
    for(int i = 0; i < N; i++) _buff.insert(_buff.end(),send_buff.begin(),send_buff.end());
    send_buff = _buff;
    recv_buff.resize(send_buff.size());

    MPI_Request req;
    pt_cp.end();
   
    com_rounds++;
    cm+= send_buff.size()*8.0/1024.0;
    MPI_Ialltoall(send_buff.data(),2*eval_claims.size(),MPI_UINT64_T , recv_buff.data(),2*eval_claims.size(),MPI_UINT64_T,MPI_COMM_WORLD,&req);
    network_timer.start(); MPI_Wait(&req,MPI_STATUS_IGNORE); network_timer.end();
    pt_cp.start();
    eval_claims.clear();field_vector_deserialize(recv_buff,eval_claims);
    vector<vector<F>> _K(polys);
    for(int i = 0; i < polys; i++) _K[i].resize(polys-1,F(0));
    if(polys == 3){
        for(int i = 0; i < N; i++){
            _K[0][0] += eval_claims[6*i];
            _K[0][1] += eval_claims[6*i+1];
            _K[1][0] += eval_claims[6*i+2];
            _K[1][1] += eval_claims[6*i+3];
            _K[2][0] += eval_claims[6*i+4];
            _K[2][1] += eval_claims[6*i+5];
        }
    }else{
        for(int i = 0; i < N; i++){
            _K[0][0] += eval_claims[2*i];
            _K[1][0] += eval_claims[2*i+1];
        }
    }
    
    

   
   
    vector<F> a(2);
    vector<F> b(2);
    a[0] =hash_to_field(convert2vector(_K));
    a[1] =  hash_to_field({a[1]});
    for(int i = 0; i < b.size(); i++) b[i] = hash_to_field({0});
    b[0] = F(0);
    //b[1] = F(0);
    //a[0] = F(0);
    //a[1] = F(0);
    if(polys==3){
        y[0] = (y[0] + a[0]*_K[0][0] + a[1]*_K[0][1]) + 
            b[0]*(_K[1][0] + a[0]*y[1] + a[1]*_K[1][1]) + 
            b[1]*(_K[2][0] + a[0]*_K[2][1] + a[1]*y[2]);
    }else{
        y[0] = (y[0] + a[0]*_K[0][0]) + b[0]*(_K[1][0] + a[0]*y[1]);

    }
    
    vector<vector<F>> arr(batches),codeword(batches);
    vector<MT> C_new(batches);
    
    
    vector<vector<F>> B;B.push_back(b2[0]);
    /*
    for(int i = 1; i < codewords.size(); i++){
        for(int k = 0; k < codewords[i].size(); k++){
            for(int j = 0; j < codewords[i][k].size(); j++) codewords[i][k][j] *= b[i-1];
            for(int j = 0; j < commit_data[i][k].size(); j++) commit_data[i][k][j] *= b[i-1];
        }
    }*/
    
    
    pt_cp.end();
    
    PIOP_dBatch_batched_new(commit_data,codewords,C,arr,codeword,C_new,b3,b,N);
    pt_cp.start();
    if(polys == 3){
        for(int i = 0; i < B[0].size(); i++){
            B[0][i] = B[0][i]*b1[0][my_id] + a[0]*b2[1][i]*b1[1][my_id] + a[1]*b2[2][i]*b1[2][my_id];
        }
    }else{
        for(int i = 0; i < B[0].size(); i++){
            B[0][i] = B[0][i]*b1[0][my_id] + a[0]*b2[1][i]*b1[1][my_id];
        }

    }
    
    
    /*
    send_codeword_batch(arr,B,codeword,N,N);
    
    if(my_id == 0){
        arr[0].resize(codeword[0].size(),F(0));
        fft(arr[0],(int)log2(arr[0].size()),false);
        for(int i = 0; i < arr[0].size(); i++){
            if(codeword[0][i] != arr[0][i]){
                printf(">>> Error\n");
                return;
            }
        }
    }
    return;
    */
    
    /*
    //dBatch(commit_data[l],codewords[l],C[l],arr[l],codeword[l],C_new[l],b3[l],N);
    
    F sum = F(0);
    for(int i = 0; i < arr[0].size(); i++){
        sum += arr[0][i]*B[0][i];
    }
    sum = sum3(sum, N);
    if(my_id == 0){
        if(sum != y[0])printf("Error\n");
    }
    return;
    */
    
    
    C = C_new;
    vector<vector<MT>> FRI_C(batches);
    vector<vector<vector<F>>> FRI_codewords(batches);
    for(int l = 0; l < batches; l++){
        FRI_C[l].resize(plaintext_degrees.size()-1);
        FRI_codewords[l].resize(plaintext_degrees.size()-1);
    }
    
    int fold_depth = N;

    vector<F> K = y;
    vector<F> c;
    pt_cp.end();
   
    for(int i = 0; i < plaintext_degrees.size()-1; i++){
        
        for(int l = 0; l  < batches; l++){
            FRI_codewords[l][i] = codeword[l];
            FRI_C[l][i] = C[l];
        }

        vector<vector<F>> poly = distributed_pcs_sumcheck_phase1_batch(arr, B, plaintext_degrees[i+1], N);
        
        if(my_id == 0)vt.start();
        for(int l = 0 ;l < batches; l++){
            F eval = F(0); 
            // Sumcheck Verification
            if(my_id==0)dpcs_ps += 16.0*poly[l].size()/1024.0;
            //if(my_id == 0)printf("%lld,%lld\n",K[l].real,K[l].img);
            for(int j = 0; j < poly[l].size()/2; j++) eval += poly[l][2*j];
            if(eval != K[l] && Verify[l] == true){
                if(my_id == 0) printf("Sumcheck Error Round %d, %d\n",i,l);
                return;
            }    
        }
        //return;
        // Sample a random challenge
        vector<F> buff = convert2vector(poly);
        c.push_back(hash_to_field(buff));
        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(2*plaintext_degrees[i+1])),c[i],2*plaintext_degrees[i+1]);
	    for(int l = 0; l  < batches; l++){
            K[l] = 0;  
            for(int j = 0; j < L.size(); j++){
                K[l] +=  poly[l][j]*L[j];
            }
        }
        // Folding
        L = compute_lagrange_coeff((int)log2(plaintext_degrees[i+1]),c[i],plaintext_degrees[i+1]);
        if(my_id == 0)vt.end();
        
        
        for(int l  =0 ; l < batches; l++){
            distributed_pcs_sumcheck_phase2(arr[l], B[l], plaintext_degrees[i+1], c[i]);
            RS_distributed_fold(codeword[l], L, L.size(), codeword[l].size()/fold_depth, N, my_id);
        }
        fold_depth = fold_depth/L.size();
        
        
        //send_codeword(arr,B,codeword,fold_depth,N);
        if(i < plaintext_degrees.size()-2){
            commit_FRI_codeword_batch(codeword,L,C,plaintext_degrees[i+1],N);   
           
        }else{
            //for(int l  =0 ; l < arr.size(); l++){
            //    send_codeword(arr[l],B[l],codeword[l],fold_depth,N);
            //}
            send_codeword_batch(arr,B,codeword,fold_depth,N);
        }  
      
    }
    
    if(my_id == 0){
        vector<vector<_hash>> MT;

        
        for(int l = 0; l < 1; l++){
            
            whir_ps = 0;
            /*
            F _k = 0;
            vector<F> buff = arr[l];
            buff.resize(codeword[l].size(),F(0));
            fft(codeword[l],(int)log2(codeword[l].size()),true);
            for(int i = 0; i < codeword[l].size()/8; i++){
                if(codeword[l][i] != arr[l][i]){
                    printf("Errorr %d\n",i);
                    exit(-1);
                }
            }
            for(int i = codeword[l].size()/8; i < codeword[l].size(); i++){
                if(codeword[l][i] != F(0)){
                    printf("Error %d\n",i);
                    exit(-1);
                }
            }
            return;
            */
            
            
            //printf("Distributed PCS Size: %d\n",arr[l].size());
            codeword[l].clear();
            //printf("        >>> %d,%d\n",arr[l].size(),whir_kappas[arr[l].size()]);
        
            //Whir_Commit(arr[l],codeword[l],MT,whir_rates[sorted_indexes[1]],whir_kappas[arr[l].size()]);
            Whir_Commit_initial(arr[l],codeword[l],MT,whir_rates[sorted_indexes[1]],whir_kappas[arr[l].size()]);
            
            whir_polys[sorted_indexes[1]] = arr[l];
            whir_c[sorted_indexes[1]] = codeword[l];
            whir_MTs[sorted_indexes[1]] = MT;
            whir_betas[sorted_indexes[1]]  = B[l];
            whir_claims[sorted_indexes[1]] = K[l];
            //Whir_Commit_Codeword(codeword[l], MT, distributed_rate);
            
            /*
            pair<F,vector<F>> claim = _Whir_Open(arr[l], codeword[l], MT, B[l], {}, {}, K[l], whir_rates[sorted_indexes[1]], Verify[l]);
            
            dpcs_ps += whir_ps;
            vector<vector<F>> L(c.size());
            vector<F> _r = r2[l];
            vector<F> _r2,_r3;

            vt.start();
            _r.insert(_r.end(),r1[l].begin(),r1[l].end());
            if(l == 0){
                _r2 = r2[2];
                _r2.insert(_r2.end(),r1[2].begin(),r1[2].end());
                _r3 = r2[1];
                _r3.insert(_r3.end(),r1[1].begin(),r1[1].end());
                
            }
            int ctr = 0;
            int ctr2=  0;
            
            F _y = F(1);
            F _y2 = F(1);
            F _y3 = F(1);
            for(int i = 0; i < L.size(); i++){
                vector<F> temp_r;
                vector<F> temp_r2,temp_r3;

                L[i] = compute_lagrange_coeff((int)log2(plaintext_degrees[i+1]),c[i],plaintext_degrees[i+1]);
                for(int j = 0; j < (int)log2(L[i].size()); j++) temp_r.push_back(_r[ctr++]);

                _y *= evaluate_vector(L[i],temp_r);
                if(l == 0){
                    for(int j = 0; j < (int)log2(L[i].size()); j++){
                        temp_r2.push_back(_r2[ctr2]);
                        temp_r3.push_back(_r3[ctr2++]);
                    }
                    _y2 *= evaluate_vector(L[i],temp_r2);
                    _y3 *= evaluate_vector(L[i],temp_r3);
                }
            }
            vector<F> temp_r;
            vector<F> temp_r2,temp_r3;
            for(int j = ctr; j < _r.size(); j++) temp_r.push_back(_r[j]);
            if(l == 0){
                for(int j = ctr2; j < _r.size(); j++){
                    temp_r2.push_back(_r2[j]);
                    temp_r3.push_back(_r3[j]);
                } 
                _y2 *= beta_identity(claim.second,temp_r2);
                _y3 *= beta_identity(claim.second,temp_r3);
            }else{
                _y2 = 0;
            }
            _y *= beta_identity(claim.second,temp_r);
            // TO DOOOO
            if(_y + a[1]*_y2 + a[0]*_y3 != claim.first && Verify[l] == true){
                printf("Error in verification 4\n");
                return;
            }
            vt.end();
            */
            
        }

        
    }

}

