#include "New_PCS.h"
#include "coPCS_utils.h"
int Security_Level = 100;
vector<int> _degrees;
vector<int> _queries;
vector<vector<int>> _rates;
vector<F> poly;
double pcs_ps = 0.0;
int BATCH_FACTOR = 32;
vector<int> delegation_size;
vector<vector<F>> whir_codewords;
vector<vector<F>> whir_arrays;
vector<vector<vector<_hash>>> whir_MT;
vector<vector<pair<F,vector<F>>>> claims;
extern int my_id;

int max_delegation_size; 

// Degree: The degree of folding on each round. The first round is simply batching of codewords.
// Rates: is a vector of pairs. The first element of the pair corresponds to the rate of the tesnor and the second one is 
//              is the rate of the RS code use to further encode the row.  
// It returns the number of queries for each round
vector<int> get_queries(int rounds, vector<vector<int>> rates){
    for(int i = 0; i < rounds; i++){
        if(i <= 1) _queries.push_back((int)((double)Security_Level/(-log2(1-0.25*(1.0-1.0/(double)rates[0][0])))));
        else {
            int l1 = (int)((double)Security_Level/(-log2(1-0.5*0.5*(1.0-1.0/(double)rates[i][0])*(1-sqrt((1.0/(double)rates[i][1]))))));
            int l2 = (int)((double)Security_Level/(-log2(1-0.5*0.5*(1.0-1.0/(double)rates[i][0]))) + (double)Security_Level/(-log2((sqrt((1.0/(double)rates[i][1]))))));
            if(l1 > l2){
                _queries.push_back(l2);
            }else{
                _queries.push_back(l1);

            }
        }
    }
    if(!my_id){
        printf("Queries: ");
        for(int i = 0; i < _queries.size(); i++){
            printf("%d, ",_queries[i]);
        }
        printf("\n");
    
    }
    _rates = rates;
    return _queries;
}

// Computes the folding degree.
// Row_Size: The size of the polynomial each prover holds
// The first is set to be constant e.g., 16. For the first round we use batching and not folding
// All other are set by default
vector<int> get_degrees(int rounds, int Row_Size){
    int log_size = (int)log2(Row_Size);
    int deg;
    _degrees.push_back(BATCH_FACTOR);
    log_size -= (int)log2(BATCH_FACTOR);
    _degrees.push_back(8);
    log_size -= (int)log2(8);
    
    deg = (int)(log_size/(rounds-2));
    for(int i = 1; i < rounds-2; i++) _degrees.push_back(1<<deg);    
    _degrees.push_back(1<<(log_size - deg*(rounds-3)));


    if(!my_id){
        printf("Degrees: ");
        for(int i = 0; i < _degrees.size(); i++){
            printf("%d, ",_degrees[i]);
        }
        printf("\n");
    
    }
    
    return _degrees;
}

// Generate the random columns needed for masking
void generate_randomness(vector<vector<F>> &Random, int N){
    int total_rand = 0;
    for(int i = 0; i < _degrees.size(); i++){
        total_rand += _queries[i]*_degrees[i];
    }
    vector<F> random_data = generate_randomness(total_rand*N/4);
    compute_secret_shares(random_data, Random, N,  N/4, N/2,true);
    printf("Total Number of randomness: %d,%d\n",total_rand,Random[0].size());
}


// Shares: The secret shares held by provers 
// Random: The random values used to mask the polynomial
// Rates: is a vector of pairs. The first element of the pair corresponds to the rate of the tesnor and the second one is 
//              is the rate of the RS code use to further encode the row.  
// rounds: Number of rounds
// N: Number of provers
// M: polynomial Size
void get_witness(vector<vector<F>> &Shares, vector<vector<F>> &Random, vector<vector<int>> rates,int rounds,int N, int M){
    printf("%d\n",M);
    vector<F> data = generate_randomness(M);
    compute_secret_shares(data, Shares, N,  N/4, N/2,true);
    printf("%d\n",Shares[0].size());
    get_degrees(rounds, 2*Shares[0].size());
    get_queries(rounds,rates);
    for(int i = 2; i  <_degrees.size(); i++){
        delegation_size.push_back(next_pow2(_degrees[i]*_queries[i]));
    }
    
    max_delegation_size = delegation_size[0];
    printf("Delegation Size: ");
    for(int i = 0; i < delegation_size.size(); i++){
        printf("%d, ",delegation_size[i]);
        if(max_delegation_size >= delegation_size[i]) max_delegation_size = delegation_size[i]; 
    }
    printf("\n");

    whir_codewords.resize(delegation_size.size());
    whir_MT.resize(delegation_size.size());
    whir_arrays.resize(delegation_size.size());
    claims.resize(delegation_size.size());
    generate_randomness(Random,N);
}


// This function is invoked by every prover to mask the data
// Here we mask them naively, but in practice they should be masked the same way as described in the paper
void prepare_data(vector<vector<vector<F>>> &Commit_data, vector<vector<F>> &Shares, vector<vector<F>> &Random){
    Commit_data.resize(BATCH_FACTOR);
    int Batch_size = Shares[0].size()/(BATCH_FACTOR/2);
    int data_ctr = 0;
    int rand_ctr = 0;
    int N = Shares.size();
    int last_batch = 0;
    printf("Batch Size: %d\n",Batch_size);
    for(int i = 0; i < BATCH_FACTOR; i++){
        Commit_data[i].resize(N);
        for(int j = 0; j < N; j++){
            Commit_data[i][j].resize(Batch_size,F(0));
        }
        for(int k = 0; k < Batch_size-_queries[0]; k++){
            for(int j = 0; j < N; j++){
                Commit_data[i][j][k] = Shares[j][data_ctr];
            }
            data_ctr++;
            if(data_ctr == Shares[0].size()){
                break;
            }
        }
        for(int k = Batch_size-_queries[0]; k < Batch_size; k++){
            for(int j = 0; j < N; j++){
                Commit_data[i][j][k] = Random[j][rand_ctr];
            }
            rand_ctr++;
        }
        if(data_ctr == Shares[0].size()){
            last_batch = i;
            break;
        }

    }
    if(data_ctr < Shares[0].size()){
        printf("Error more batches are needed\n");
        exit(-1);
    }
    //printf("%d,%d,%d\n",rand_ctr,last_batch,Batch_size);
    last_batch++;
    Commit_data[last_batch].resize(N);
    for(int i = 0; i < N; i++)Commit_data[last_batch][i].resize(Batch_size,F(0));
    
    
    
    if(Shares[0].size()-rand_ctr > Batch_size){
        printf("Case 1\n");
        for(int i = 0; i < Batch_size; i++){
            for(int j = 0; j < N; j++){
                Commit_data[last_batch][j][i] = Random[j][rand_ctr]; 
            }
            rand_ctr++;
        }
    }else{
        printf("Case 2\n");
        for(int i = 0; i < Batch_size- (Shares[0].size()-rand_ctr); i++){
            for(int j = 0; j < N; j++){
                Commit_data[last_batch][j][i] = Random[j][rand_ctr]; 
            }
            rand_ctr++;
        }
    }
    
}


void Encode(vector<vector<vector<F>>> &Codeword, int N, vector<int> rate){
    for(int i = 0; i < BATCH_FACTOR; i++){
        if(Codeword[i].size() == 0) return;
        for(int j = 0; j < N; j++){
            fft(Codeword[i][j],(int)log2(Codeword[i][j].size()),true);
            Codeword[i][j].resize(rate[0]*Codeword[i][j].size(),F(0));
            fft(Codeword[i][j],(int)log2(Codeword[i][j].size()),false);
        }
    }
}   

void Commit(vector<vector<vector<F>>> &Codeword,vector<vector<_hash>> &C, int N){
    int batches = 0;
    for(int i = 0; i < Codeword.size(); i++){
        if(!Codeword[i].size()) break;
        batches++;
    }
    vector<F> batch_vec(Codeword.size(), F(0));
    vector<vector<_hash>> H;
    C.resize((int)log2(Codeword[0].size()*Codeword[0][0].size())+1);
    C[0].resize(Codeword[0].size()*Codeword[0][0].size());
    int ctr = 0;
    for(int i = 0; i < N; i++){

        for(int j = 0; j < Codeword[0][i].size(); j++){
            for(int k  = 0; k < batches; k++){
                batch_vec[k] = Codeword[k][i][j];
            }
            // MT is not required and is overkill. Switch to chain hashing
            merkle_tree::merkle_tree_prover::MT_commit_Blake(batch_vec.data(),H,batch_vec.size());
            C[0][ctr++] = H[H.size()-1][0];
            H.clear();
        }   
    }
    merkle_tree::merkle_tree_prover::create_tree_blake(C[0].size(),C,sizeof(__hhash_digest), true);
}




void new_commit(vector<vector<_hash>> &C, 
                vector<vector<vector<F>>> &Codeword, 
                vector<vector<vector<F>>> &Commit_data, 
                vector<vector<F>> &Shares, vector<vector<F>> &Random, 
                vector<vector<int>> rate){
    prepare_data(Commit_data, Shares, Random);
    Codeword = Commit_data;
    Encode(Codeword, Shares.size(), rate[0]);
    Commit(Codeword,C,Shares.size());
    Shares.clear();
    Random.clear();
}

F evaluate_vector(vector<F> r, vector<vector<vector<F>>> &Commit_data){
    vector<F> r1,r2,r3;
    vector<F> b1,b2,b3;
    int N = Commit_data[0].size();
    int ctr = 0;
    for(int i = 0; i < (int)log2(Commit_data.size()); i++) r1.push_back(r[ctr++]);   
    for(int i = 0; i < (int)log2(Commit_data[0][0].size()); i++) r2.push_back(r[ctr++]);   
    for(int i = 0; i < (int)log2(N/4); i++) r3.push_back(r[ctr++]);   
    precompute_beta(r1,b1);precompute_beta(r2,b2);precompute_beta(r3,b3);
    vector<vector<F>> temp(N);
    vector<F> temp2(N,F(0));
    for(int i = 0; i < temp.size(); i++) temp[i].resize(Commit_data[0][0].size(),F(0));
    
    for(int i = 0; i < Commit_data.size(); i++){
        if(!Commit_data[i].size()) break;
        for(int j = 0; j < Commit_data[i].size(); j++){
            for(int k = 0; k < Commit_data[i][j].size(); k++){
                temp[j][k] += b1[i]*Commit_data[i][j][k];
            }
        }
    }
    for(int i = 0; i < N; i++){
        for(int j = 0; j < temp[i].size(); j++){
            temp2[i] += b2[j]*temp[i][j];
        }
    }
    temp2 = unpack_secret_share(temp2,N/2,N/2);
    F y = 0;
    for(int i = 0; i < temp2.size()/2; i++) y += b3[i]*temp2[i];
    return y;
}


void batch(vector<vector<_hash>> &C, vector<vector<vector<F>>> &Codeword, vector<vector<vector<F>>> &Commit_data, 
           vector<vector<_hash>> &batched_C, vector<vector<F>> &batched_Codeword, vector<vector<F>> &batched_commit_data, 
           vector<F> b1){
    int N = Commit_data[0].size();
    batched_Codeword.resize(N);batched_commit_data.resize(N);
    for(int i = 0; i < N; i++) {
        batched_Codeword[i].resize(Codeword[0][0].size(),F(0));
        batched_commit_data[i].resize(Commit_data[0][0].size(),F(0));
    }
    for(int i = 0; i < Codeword.size(); i++){
        if(Codeword[i].size() == 0) break;
        for(int j = 0; j < N; j++){
            for(int k  = 0; k < batched_Codeword[j].size(); k++){
                batched_Codeword[j][k] += b1[i]*Codeword[i][j][k];
            }
            for(int k  = 0; k < batched_commit_data[j].size(); k++){
                batched_commit_data[j][k] += b1[i]*Commit_data[i][j][k];
            }
        } 
    }

    vector<F> buff(_degrees[1]);
    vector<vector<_hash>> MT_buff;
    vector<_hash> hash_leaves(N*batched_Codeword[0].size()/_degrees[1]);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < batched_Codeword[i].size()/_degrees[1]; j++){
            for(int k = 0; k < buff.size(); k++){
                buff[k] = batched_Codeword[i][j + k*batched_Codeword[i].size()/_degrees[1]];
            }
            merkle_tree::merkle_tree_prover::MT_commit_Blake(buff.data(),MT_buff,buff.size());
            hash_leaves[i*batched_Codeword[i].size()/_degrees[1] + j] = MT_buff[MT_buff.size()-1][0];
        }
    }
    batched_C.resize((int)log2(hash_leaves.size())+1);
    batched_C[0] = hash_leaves;
    merkle_tree::merkle_tree_prover::create_tree_blake(hash_leaves.size(),batched_C, sizeof(__hhash_digest), true);

    
    // TO DO: RUN Batch verification algorithm. Here we do not have to query the oracle of C_new, we will 
    // query it in the next step
    vector<vector<int>> queries(_queries[0]);
    vector<vector<F>> replies(_queries[0]);

    for(int i = 0; i < queries.size(); i++){
        queries[i].push_back(rand()%(N));
        queries[i].push_back(rand()%Codeword[0][0].size());
    }
  
    for(int i = 0; i < queries.size(); i++){
        F y = F(0);
        for(int k = 0; k < Codeword.size(); k++){
            if(Codeword[k].size() == 0) break;
            replies[i].push_back(Codeword[k][queries[i][0]][queries[i][1]]);
            y += b1[k]*replies[i][k];
        }
    }

    pcs_ps += ((double)(replies.size()*replies[0].size()*16))/1024.0;
    
    vector<bool> visited(2*C[0].size(),false);
    for(int i = 0; i<queries.size(); i++){
        vector<vector<_hash>> temp_MT;
        int pos = queries[i][0]*Codeword[0][0].size()+queries[i][1];
        vector<_hash> path;
        replies[i].resize((next_pow2(replies[i].size())),F(0));
        merkle_tree::merkle_tree_prover::MT_commit_Blake(replies[i].data(),temp_MT,replies[i].size());

        path = merkle_tree::merkle_tree_prover::open_tree_blake(C,pos);
        
        merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(C,path.data(),pos,C[0].size(),visited,pcs_ps);
    }

}



vector<F> dummy_sumcheck_round1(vector<vector<F>> &Commit_data, vector<vector<F>> &B, int degree){
    int N = Commit_data.size();
    int M = Commit_data[0].size();
    vector<vector<F>> poly(N);
    for(int i = 0; i < N; i++) poly[i].resize(2*degree,F(0));
    printf("Sumcheck Degree : %d\n",degree);
    
    
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M/degree; j++){
            vector<F> h1(degree),h2(degree);    
            for(int k = 0; k < degree; k++){
                h1[k] = Commit_data[i][degree*j+k];
                h2[k] = B[i][degree*j+k];
            }
            fft(h1,(int)log2(h1.size()),true);
            fft(h2,(int)log2(h1.size()),true);
            h1.resize(2*h1.size(),F(0));h2.resize(2*h1.size(),F(0));
            fft(h1,(int)log2(h1.size()),false);fft(h2,(int)log2(h1.size()),false);
            for(int k = 0; k < 2*degree; k++){
                poly[i][k] += h1[k]*h2[k];
            }
        }
    }
    poly = transpose(poly);
    for(int i = 0; i < poly.size(); i++) poly[i] = unpack_secret_share(poly[i],N,N);
    vector<F> final_poly(2*degree,0);
    for(int i = 0; i < poly.size(); i++){
        for(int j = 0; j < poly[i].size()/2; j++){
            final_poly[i] += poly[i][2*j];
        }
    }
    return final_poly;
}


void dummy_sumcheck_round2(vector<vector<F>> &Commit_data, vector<vector<F>> &B, int degree, F r){
    int N = Commit_data.size();
    int M = Commit_data[0].size();
    
    vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(degree)),r,degree);
    
    for(int i = 0; i < Commit_data.size(); i++){
        for(int j = 0; j < Commit_data[i].size()/degree; j++){
            F s1 = F(0), s2 = F(0);
            for(int k = 0; k < degree; k++){
                s1 += L[k]*Commit_data[i][j*degree+k];
                s2 += L[k]*B[i][j*degree+k];
            }
            Commit_data[i][j] = s1;
            B[i][j] = s2;
        }
        Commit_data[i].resize(Commit_data[i].size()/degree);
        B[i].resize(B[i].size()/degree);   
    }
}

void encode_concatenated(vector<vector<F>> &Commit_Data, vector<vector<F>> &C_prime, vector<vector<vector<F>>> &C, vector<int> rate, int degree){
    int N = Commit_Data.size();
    C_prime.resize(N);C.resize(N);
    printf("New Rate: %d\n",rate[0]);
    for(int i = 0; i < N; i++){
        
        C_prime[i] = Commit_Data[i];
        C_prime[i].resize(rate[0]*C_prime[i].size(),F(0));
        fft(C_prime[i],(int)log2(C_prime[i].size()),false);
        C[i].resize(C_prime[i].size()/degree);
        for(int j = 0; j < C[i].size(); j++){
            C[i][j].resize(degree);
            for(int k = 0; k < degree; k++){
                C[i][j][k] = C_prime[i][j + k*C_prime[i].size()/C[i][j].size()];
            }
            C[i][j].resize(degree*rate[1],F(0));
            fft(C[i][j],(int)log2(C[i][j].size()),false);
        }
    }
}

void commit_tensorcode(vector<vector<F>> &C, vector<vector<_hash>> &MT){
    vector<F> leaves;
    for(int i = 0; i < C.size(); i++){
        leaves.insert(leaves.end(),C[i].begin(),C[i].end());
    }
    merkle_tree::merkle_tree_prover::MT_commit_Blake(leaves.data(),MT,leaves.size());
}

void commit_concatenated(vector<vector<vector<F>>> &C, vector<vector<_hash>> &MT){
    int N = C.size();
    vector<F> leaves;
    vector<F> buff;
    for(int i = 0; i < N; i++){
        buff = convert2vector(C[i]);
        leaves.insert(leaves.end(),buff.begin(),buff.end());        
    } 
    MT.clear();
    merkle_tree::merkle_tree_prover::MT_commit_Blake(leaves.data(),MT,leaves.size());
}


vector<F> delegate_checks(vector<vector<F>> &C_prime, vector<vector<vector<F>>> &C, 
                     vector<vector<_hash>> &MT_C, vector<F> L, 
                     vector<vector<int>> &queries, int iter){
    
    int N = C_prime.size();
    int Segments = C_prime[0].size()/_degrees[iter+1];
    // Sample segments
    int l = _queries[iter+1];
    queries.resize(l);
    for(int i = 0; i < l; i++) queries[i] = {rand()%N,rand()%Segments};
    
    // In the first interation the verifier directly queries the tensorcode, no delegation required
    vector<vector<F>> segments(l);
    vector<F> y(l);
    for(int i = 0; i < l; i++){
        for(int j = 0; j < _degrees[iter+1]; j++) segments[i].push_back(C_prime[queries[i][0]][queries[i][1] + j*Segments]);
        y[i] = RS_multifold_segment(segments[i],L,C_prime[0].size(),queries[i][1]);
    }
    if(iter == 0){
        // The provers simply send the segments and the verifier validates its correctness
        
        vector<vector<_hash>> temp_MT;
        vector<bool> visited(2*MT_C[0].size(),false);
        for(int i = 0; i < segments.size(); i++){
            int pos = queries[i][0]*Segments  + queries[i][1];
            vector<_hash> path = merkle_tree::merkle_tree_prover::open_tree_blake(MT_C,pos);
            pcs_ps += ((double)(segments[i].size()*16))/1024.0;
            merkle_tree::merkle_tree_prover::MT_commit_Blake(segments[i].data(),temp_MT,segments.size());
            merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(MT_C,path.data(),pos,MT_C[0].size(),visited,pcs_ps);
            temp_MT.clear();
        }
    }else{
        whir_arrays[iter-1] = convert2vector(segments);
        whir_arrays[iter-1].resize(max_delegation_size,F(0));
        Whir_Commit(whir_arrays[iter-1],whir_codewords[iter-1],whir_MT[iter-1],4);
        
        // Compute the commitment using WHIR
        printf("Delegation starting\n");
        
        if(next_pow2(segments.size()*segments[0].size()) != delegation_size[iter-1]){
            printf("Inconsistent Delegation size\n");
            exit(-1);
        }
        vector<int> queries2(l);
        for(int i = 0; i < l; i++) queries2[i] = rand()%C[0][0].size();
        //claims[iter-1].push_back(prove_multi_folding(segments, L, y,  queries, C_prime[0].size(),pcs_ps));
        vector<vector<F>> extended_segments;
        for(int i = 0; i < queries.size(); i++) extended_segments.push_back(C[queries[i][0]][queries[i][1]]);
        
        vector<F> o; 
        for(int i = 0; i < extended_segments.size(); i++){
            o.push_back(extended_segments[i][queries2[i]]);
        }
        printf("%d\n",C[0][0].size()/_degrees[iter+1]);
        claims[iter-1].push_back(prove_extension(segments, extended_segments, o, queries2, C[0][0].size()/_degrees[iter+1],pcs_ps));

        vector<bool> visited(2*MT_C[0].size(),false);
        pcs_ps += ((double)(2*o.size())*16)/1024.0;
        // Point query MT
        double temp;
        for(int i = 0; i < segments.size(); i++){
            int pos = queries[i][0]*C[0].size()*C[0][0].size()  + queries[i][1]*C[0][0].size() + queries2[i];
            vector<_hash> path = merkle_tree::merkle_tree_prover::open_tree_blake(MT_C,pos/4);
            merkle_tree::merkle_tree_verifier::verify_claim_opt_blake(MT_C,path.data(),pos/4,MT_C[0].size(),visited,pcs_ps);
        }

        for(int i = 0; i < claims[iter-1].size(); i++){
            if(claims[iter-1][i].second.size() != (int)log2(delegation_size[iter-1])){
                printf("Error in Claims %d,%d\n",i,iter-1);
                exit(-1);
            }
            for(int j = 0; j < (int)log2(max_delegation_size); j++){
                claims[iter-1][i].second.push_back(F(0));
            }
        }

    }
    return y;

}

vector<vector<F>> tensor_eval_vector(vector<int> point, int cols, int rows, int rate){
    int N = 2*cols;
    F omega1 = getRootOfUnity((int)log2(rate*rows));
    F omega2 = getRootOfUnity((int)log2(4*cols));
    omega1 = omega1.fastPow(omega1,point[1]);
    omega2 = omega2.fastPow(omega2,2*point[0]+1);
    vector<F> pow1(rows,F(1)),pow2(cols,F(1));
    vector<vector<F>> B(N);
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
    
    for(int i  = 0; i < B.size(); i++){
        B[i] = pow1;
        for(int j = 0; j < B[i].size(); j++){
            B[i][j] = pow2[i]*B[i][j];
        }
    }
    return B;
}


void dummy_sumcheck_round3(vector<vector<F>> &B, vector<vector<int>> queries, F s, int iter){
    
    if(iter+1 >= _rates.size()) return;
    F p = s;
    
    for(int i = 0; i < queries.size(); i++){
        vector<vector<F>> temp_B = tensor_eval_vector(queries[i],B.size()/2,B[0].size(),_rates[iter+1][0]);
        for(int j = 0; j < temp_B.size(); j++){
            for(int k = 0; k < temp_B[j].size(); k++){
                B[j][k] += p*temp_B[j][k];
            }
        }
        
        p *= s;   
    }

}

F fast_eval_powers_y(F c_r, int point, int N){
    F omega1 = getRootOfUnity(1+(int)log2(N));
    omega1 = omega1.fastPow(omega1,2*point+1);
    
    vector<F> pow(N/2,F(1));
    for(int i = 1; i < N/2; i++) pow[i] = pow[i-1]*omega1;
    fft(pow,(int)log2(N/2),true);
    

    F final_eval = F(0);
    F p = F(1);
    for(int i = 0; i < (N/2); i++){
        final_eval += p*pow[i];
        p *= c_r;
    }
    
    /*
    F final_eval = F(1);
    for(int i = 0; i < (int)log2(N); i++){
        final_eval *= (F(1)+ omega1*c_r);
        omega1 = omega1*omega1;
        c_r = c_r*c_r; 
    }
    */
    
    return final_eval;    
}

F fast_eval_powers_x(vector<vector<F>> &L, int point, int rows, int rate){
    
    F omega1 = getRootOfUnity((int)log2(rate*rows));
    omega1 = omega1.fastPow(omega1,point);
    vector<F> omegas(L.size(),omega1);
    for(int i = 1; i < L.size(); i++) omegas[i] = omegas[i-1].fastPow(omegas[i-1],L[i-1].size());
    
    F final_eval = F(1);
    for(int i = 0; i < L.size(); i++){
        F y = F(0);
        F omega = F(1);
        for(int j = 0; j < L[i].size(); j++){
            y += L[i][j]*omega;
            omega = omega*omegas[i];
        }
        final_eval *= y;
    }
    return final_eval;    
}



void open(vector<vector<_hash>> &MT_C, vector<vector<vector<F>>> &Codeword, vector<vector<vector<F>>> &Commit_data, 
          vector<F> r, F _y, vector<vector<int>> rates){
    vector<F> r1,r2,r3;
    vector<F> b1,b2,b3;
    vector<F> aggr_rand;
    // The individual instance size of the sumcheck
    vector<int> instance_size;
    int N = Commit_data[0].size();
    int ctr = 0;
    for(int i = 0; i < (int)log2(Commit_data.size()); i++) r1.push_back(r[ctr++]);   
    for(int i = 0; i < (int)log2(Commit_data[0][0].size()); i++) r2.push_back(r[ctr++]);   
    for(int i = 0; i < (int)log2(N/4); i++) r3.push_back(r[ctr++]);   
    precompute_beta(r1,b1);precompute_beta(r2,b2);precompute_beta(r3,b3);
    fft(b2,(int)log2(b2.size()),false);
    instance_size.push_back(Commit_data[0].size());

    // First Step: Batching 
    vector<vector<_hash>> MT_new_C;
    vector<vector<F>> C_prime,batched_commit_data;
    batch(MT_C,Codeword,Commit_data,MT_new_C,C_prime,batched_commit_data,b1);
    // The verifier has to do consistency checks 

    // Discard old merkle tree once checks finished
    MT_C = MT_new_C;
    /*************************/
    for(int i = 0; i < batched_commit_data.size(); i++) fft(batched_commit_data[i],(int)log2(batched_commit_data[i].size()),true);
    

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
    vector<vector<F>> B(N);
    for(int i = 0; i < N; i++){
        B[i].resize(b2.size());
        for(int j = 0; j < B[i].size(); j++){
            B[i][j] = b2[j]*b3[i];
        }
    }


    // TODO: SUMCHECK MASKING
    // Start the iterations
    vector<F> y_evals(N,F(0));
    
    for(int i = 0; i < batched_commit_data.size(); i++){
        for(int j = 0; j < batched_commit_data[i].size(); j++) y_evals[i] += B[i][j]*batched_commit_data[i][j];
    }
    y_evals = unpack_secret_share(y_evals,N,N);
    
    printf("%d\n",y_evals.size());
    F K = F(0);
    
    for(int i = 0; i < N/4; i++){
        K += y_evals[2*i];  
    }
    if(K != _y) printf(">> Error 1\n");
    //exit(-1);
    //if(K == 0) printf(">>> OK\n");

    vector<vector<F>> C_prime_new;
    vector<vector<vector<F>>> C,C_new;
    vector<vector<F>> Lagrange_Evals;    

    K = _y;
    vector<F> c;
    vector<vector<vector<int>>> queries(rates.size()-1);
    for(int i = 0; i < rates.size()-1; i++){
        printf("Iter %lf Ps \n",pcs_ps);
        instance_size.push_back(batched_commit_data[0].size());
        // Sumcheck
        vector<F> poly = dummy_sumcheck_round1(batched_commit_data, B, _degrees[i+1]);
        F eval = F(0); 
        // Sumcheck Verification
        for(int i = 0; i < poly.size()/2; i++) eval+= poly[2*i];
        if(eval != K){
            printf("Sumcheck Error Round %d\n",i);
            exit(-1);
        }
        // Sample a random challenge
        c.push_back(F::_random());
        
        K = F(0);
        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(2*_degrees[i+1])),c[i],2*_degrees[i+1]);
	    for(int j = 0; j < L.size(); j++){
            K +=  poly[j]*L[j];
        }
        
        // Folding
        //L = compute_lagrange_coeff((int)log2(_degrees[i+1]),c[i],_degrees[i+1]);
        
        dummy_sumcheck_round2(batched_commit_data, B, _degrees[i+1], c[i]);
        
        
        if(i < rates.size()-2){
            L = compute_lagrange_coeff((int)log2(_degrees[i+1]),c[i],_degrees[i+1]);
            Lagrange_Evals.push_back(L);
            encode_concatenated(batched_commit_data, C_prime_new, C_new, _rates[i+2], _degrees[i+2]);    
            // Commit
            commit_concatenated(C_new,MT_new_C);
            // Check Consistency
        
            vector<F> y = delegate_checks(C_prime,C,MT_C,L,queries[i],i);
            // Update commitment
            C_prime = C_prime_new;
            C = C_new;
            MT_C = MT_new_C ;
            
            for(int j = 0; j < queries[i].size(); j++){
                queries[i][j][1] = _rates[i+2][0]*queries[i][j][1]/_rates[i+1][0];
                if(C_prime[queries[i][j][0]][queries[i][j][1]] != y[j]){
                    printf(">>> Error %d\n",j);
                    exit(-1);
                }
            }
            // Update Sumcheck instance
            // The verifier samples the random challenge s
            F s = F::_random();
            aggr_rand.push_back(s);
            dummy_sumcheck_round3(B, queries[i], s, i+1);
            F p = F(s);
            for(int j = 0; j < y.size(); j++){
                K += p*y[j];
                p*= s;
            }
            
            
        }else{
            L = compute_lagrange_coeff((int)log2(_degrees[i+1]),c[i],_degrees[i+1]);
            Lagrange_Evals.push_back(L);
            
            C_prime_new = C_prime;
            for(int j = 0; j < N; j++) RS_multifold5(C_prime_new[j],L,L.size());
            commit_tensorcode(C_prime_new, MT_new_C);
            // delegate checks
            vector<F> y = delegate_checks(C_prime,C,MT_C,L,queries[i],i);
            
            C_prime = C_prime_new;
            C = C_new;
            MT_C = MT_new_C;
        }
        
        
    }


    // Verification Steps
    vector<vector<F>> Final_Code;
    for(int i = 0; i < C_prime.size(); i++) Final_Code.push_back(C_prime[i]);
    for(int i = 0; i < C_prime.size(); i++){
        fft(Final_Code[i],(int)log2(Final_Code[i].size()),true);
        for(int j = batched_commit_data[i].size(); j < Final_Code[i].size(); j++){
            if(Final_Code[i][j] != F(0)){
                printf("Error in final verification step 1\n");
                exit(-1);   
            }
        }
        for(int j = 0; j < batched_commit_data[i].size(); j++){
            if(Final_Code[i][j] != batched_commit_data[i][j]){
                printf("Error in final verification step 2\n");
                exit(-1);   
            }
        }
    }

    vector<F> final_poly,final_B;
    for(int i = 0; i < batched_commit_data.size(); i++){
        final_poly.push_back(batched_commit_data[i][0]);
        final_B.push_back(B[i][0]);
    }

    final_poly = unpack_secret_share(final_poly,N/2,N);
    final_B = unpack_secret_share(final_B,N/2,N);
    for(int i = 0; i < final_poly.size(); i++) K -= final_poly[i]*final_B[i]; 
    if(K != 0){
        printf("Error in Verification Sumcheck\n");
        exit(-1);
    }else{
        printf("Verification Success\n");
    }
    F r_c = F::_random();
    
    F B_eval_claim = 0;
    F p = F(1);
    for(int i = 0; i < final_B.size(); i++){
        B_eval_claim += p*final_B[i];
        p *= r_c;
    }



    // Verification steps to validate the correctness of B. These parts will be delegated to the provers
    b2.clear(); b3.clear();
    precompute_beta(r2,b2);precompute_beta(r3,b3);
    fft(b2,(int)log2(b2.size()),false);
    int d =1;
    for(int i = 0; i < Lagrange_Evals.size(); i++){
        d *= Lagrange_Evals[i].size();
        for(int j = 0; j < b2.size()/(d); j++){
            b2[j] = Lagrange_Evals[i][0]*b2[Lagrange_Evals[i].size()*j];
            for(int k = 1; k < Lagrange_Evals[i].size(); k++){
                b2[j] += Lagrange_Evals[i][k]*b2[Lagrange_Evals[i].size()*j+k]; 
            }
        }
    }
    p = r_c;
    for(int i = 1; i < b3.size(); i++){
        b3[0] += p*b3[i];
        p *= r_c;
    }
    
    
    vector<F> partial_evals_x,partial_evals_y;
    partial_evals_x.push_back(b2[0]);
    partial_evals_y.push_back(b3[0]);
    for(int i = 2; i < _rates.size(); i++){
        Lagrange_Evals.erase(Lagrange_Evals.begin());
        p = aggr_rand[i-2];
        
        for(int j = 0; j < queries[i-2].size(); j++){
            partial_evals_x.push_back(p*fast_eval_powers_x(Lagrange_Evals,queries[i-2][j][1],instance_size[i],_rates[i][0]));
            p *= aggr_rand[i-2];
        }
        for(int j = 0; j < queries[i-2].size(); j++){
            partial_evals_y.push_back(fast_eval_powers_y(r_c, queries[i-2][j][0], N));
        }
    }
    printf("%lf\n",pcs_ps);
    F final_eval = F(0);
    for(int i = 0; i < partial_evals_y.size(); i++) final_eval += partial_evals_x[i]*partial_evals_y[i];
    if(final_eval != B_eval_claim){
        printf("Error in final verification step\n");
        exit(-1);
    }

}






