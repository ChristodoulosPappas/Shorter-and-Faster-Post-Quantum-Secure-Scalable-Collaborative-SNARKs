#include "delegate_protocol.h"
#include "timer.hpp"
#include "coPCS_utils.h"
extern timer vt;

vector<pair<F,vector<F>>> _quadratic_sumcheck(F y, vector<F> &v1, vector<F> &v2,F previous_r, double &ps, bool Verify = true){
	//vector<F> r = generate_randomness(int(log2(v1.size())));
	int rounds = int(log2(v1.size()));
	F rand = previous_r;
	vector<F> r;
	for(int i = 0; i < rounds; i++){
		quadratic_poly poly = quadratic_poly(F_ZERO,F_ZERO,F_ZERO);
			linear_poly l1,l2;
			
			int L = 1 << (rounds -1- i);
			for(int j = 0; j < L; j++){
				l1 = linear_poly(v1[2*j+1] - v1[2*j],v1[2*j]);
				l2 = linear_poly(v2[2*j+1] - v2[2*j],v2[2*j]);
				poly = poly + (l1*l2);
				
			}

		vector<F> input;
		
		clock_t s,e;
		s = clock();
        
		if(Verify && poly.eval(0)+ poly.eval(1) != y){
            printf("Error in sumcheck round %d\n",i);
            exit(-1);
        }
        rand = F::_random();//mimc_hash(rand,poly.a);
		rand = F::_random();//mimc_hash(rand,poly.b);
		rand = F::_random();///mimc_hash(rand,poly.c);
		e = clock();
		ps += 3*sizeof(F)/1024.0;
		y = poly.eval(rand);
        r.push_back(rand);
		
        //vt += (double)(e-s)/(double)CLOCKS_PER_SEC;

		for(int j = 0; j < L; j++){
            v1[j] = rand*(v1[2*j+1]-v1[2*j]) + v1[2*j];
            v2[j] = rand*(v2[2*j+1]-v2[2*j]) + v2[2*j];
        }
		//vector<vector<F>> temp = mimc_multihash3(input);
		//Pr.w_hashes.push_back(temp);
		
		//rand = temp[temp.size()-1][temp[0].size()-1];
		//rand = mimc_multihash(input);
	}	
	
    return {make_pair(v1[0] ,r),make_pair(v2[0] ,r)};
}


vector<pair<F,vector<F>>> quadruple_sumcheck(F y, vector<F> &v1, vector<F> &v2, vector<F> &v3, vector<F> &v4, F previous_r, double &ps){
	//vector<F> r = generate_randomness(int(log2(v1.size())));
	int rounds = int(log2(v1.size()));
	vector<quadruple_poly> p;
	F rand = previous_r;
	vector<F> r;
	for(int i = 0; i < rounds; i++){
		quadruple_poly poly = quadruple_poly(F_ZERO,F_ZERO,F_ZERO,F_ZERO,F_ZERO);
		linear_poly l1,l2,l3,l4;
			
			int L = 1 << (rounds -1- i);
			for(int j = 0; j < L; j++){
				l1 = linear_poly(v1[2*j+1] - v1[2*j],v1[2*j]);
				l2 = linear_poly(v2[2*j+1] - v2[2*j],v2[2*j]);
				l3 = linear_poly(v3[2*j+1] - v3[2*j],v3[2*j]);
				l4 = linear_poly(v4[2*j+1] - v4[2*j],v4[2*j]);
				poly = poly + (l1*l2*l3*l4);
				
			}

		vector<F> input;
		
		clock_t s,e;
		s = clock();
        //vt.start();
		if(poly.eval(0)+ poly.eval(1) != y){
            printf("Error in sumcheck round %d\n",i);
            //exit(-1);
        }
        rand = F::_random();//mimc_hash(rand,poly.a);
		rand = F::_random();//mimc_hash(rand,poly.b);
		rand = F::_random();///mimc_hash(rand,poly.c);
		rand = F::_random();//mimc_hash(rand,poly.d);
		rand = F::_random();//mimc_hash(rand,poly.d);
		e = clock();
		ps += 5*sizeof(F)/1024.0;
		y = poly.eval(rand);
        //vt.end();
		
        r.push_back(rand);
		
        //vt += (double)(e-s)/(double)CLOCKS_PER_SEC;

		for(int j = 0; j < L; j++){
            v1[j] = rand*(v1[2*j+1]-v1[2*j]) + v1[2*j];
            v2[j] = rand*(v2[2*j+1]-v2[2*j]) + v2[2*j];
            v3[j] = rand*(v3[2*j+1]-v3[2*j]) + v3[2*j];
            v4[j] = rand*(v4[2*j+1]-v4[2*j]) + v4[2*j];
        }
		//vector<vector<F>> temp = mimc_multihash3(input);
		//Pr.w_hashes.push_back(temp);
		
		//rand = temp[temp.size()-1][temp[0].size()-1];
		//rand = mimc_multihash(input);
		p.push_back(poly);
	}
    ps += 2*sizeof(F)/1024.0;
			
    return {make_pair(v1[0],r),make_pair(v2[0],r),make_pair(v3[0],r),make_pair(v4[0],r)};
}


pair<F,vector<F>> prove_fft(vector<F> &v, vector<F> r, F K, int rate,double &ps){
	v.resize(rate*v.size(),F(0));
	vector<F> W_reduced(v.size(),F(0));
    printf("%d,%d\n",v.size(),W_reduced.size());
	
    phiGInit(W_reduced,r.begin(),F(1),r.size(),false);

    vector<pair<F,vector<F>>> claims = _quadratic_sumcheck(K, W_reduced, v, F::_random(),ps);
	
    
    // Verification step
    vector<F> W(v.size(),F(0));
    phiGInit(W,r.begin(),F(1),r.size(),false);
	if(evaluate_vector(W,claims[0].second) != claims[0].first){
        printf("Error V, Delegation 7\n");
        exit(-1);
    }
    for(int i = 0; i < (int)log2(rate); i++){
        F inv = (F(1)-claims[1].second[claims[1].second.size()-1]).inv();
        claims[1].first *= inv*claims[1].first;
    }
    for(int i = 0; i < (int)log2(rate); i++){
        claims[1].second.pop_back();
    }
    return claims[1];
}


pair<F,vector<F>> prove_multi_folding(vector<vector<F>> &segments, vector<F> L, vector<F> y, vector<vector<u32>> queries, int n, double &ps){
    vector<vector<F>> partial_segments(segments.size());
    vector<vector<F>> powers(segments.size());
    for(int i = 0; i < segments.size(); i++){
        partial_segments[i] = get_RS_multifold_segment(segments[i], L, n, queries[i][1]);
    }
    F omega = getRootOfUnity((int)log2(n));
    omega = omega.inv();
    for(int i = 0; i < segments.size(); i++){
        powers[i].resize(L.size(),F(1));
        powers[i][1] = omega.fastPow(omega,queries[i][1]);
        for(int j = 2; j < powers[i].size(); j++){
            powers[i][j] = powers[i][j-1]*powers[i][1];
        }
    }

    // Run the first sumcheck instance to prove the correctness of y
    // the verifier samples some random points
    pad_vector(y);
    vector<F> r1,b1; 
    for(int i = 0; i < (int)log2(y.size()); i++)r1.push_back(F::_random());
    
    // Verifier steps
    F K = F(0);
    precompute_beta(r1,b1);
    for(int i = 0; i < y.size(); i++){
        K += b1[i]*y[i];
    }
    K = F(L.size())*K;
    //*********//
    // Prepare for sumcheck
    vector<F> V1 = convert2vector(partial_segments);
    vector<F> V2 = convert2vector(powers);
    int S = next_pow2(partial_segments.size());
    vector<F> V3,V4(S*L.size());
    for(int i = 0; i < S; i++){
        V3.insert(V3.end(),L.begin(), L.end());
        for(int j = 0; j < L.size(); j++){
            V4[i*L.size() + j] = b1[i];
        }
    }
    pad_vector(V1);pad_vector(V2);pad_vector(V3);pad_vector(V4);
    vector<pair<F,vector<F>>> claims = quadruple_sumcheck(K, V1,V2,V3,V4,F::_random(),ps);
    
    // Verifier steps
    // Locally check the correctness of V2,V3,V4
    vector<F> r11,r12,b12,b11; 
    for(int i = 0; i < (int)log2(L.size()); i++){
        r11.push_back(claims[0].second[i]);
    }
    for(int i = 0; i < (int)log2(next_pow2(segments.size())); i++){
        r12.push_back(claims[0].second[i+(int)log2(L.size())]);
    }
    precompute_beta(r12,b12);precompute_beta(r11,b11);

    F _y = F(0);
    for(int i = 0; i < segments.size(); i++){
        F omega_idx = omega.fastPow(omega,queries[i][1]);
        F c = F(1);
        for(int j = 0; j < (int)log2(L.size()); j++){
            c *= (F(1) - r11[j] + r11[j]*omega_idx);
            omega_idx = omega_idx*omega_idx;
        }
        _y += b12[i]*c;
    }
    if(_y != claims[1].first){
        printf("Error V, Delegation 1\n");
        exit(-1);
    }
    _y = F(0);
    for(int i = 0; i < b11.size(); i++) _y += b11[i]*L[i];
    if(_y != claims[2].first){
        printf("Error V, Delegation 2\n");
        exit(-1);
    }
    _y = F(1);
    
    for(int i = 0; i < r12.size(); i++) _y *= ((F(1)-r12[i])*(F(1)-r1[i]) + r12[i]*r1[i]);
    if(_y != claims[3].first){
        printf("Error V, Delegation 3\n");
        exit(-1);
    }


    // Prepare for the second sumcheck
    V1.clear();
    V1.resize(L.size(),F(0));
    for(int i = 0; i < segments.size(); i++){
        for(int j = 0; j < segments[0].size(); j++) V1[j] += b12[i]*segments[i][j];
    }
    V2 = get_RS_multifold_reduced_matrix(r11, n, L.size());
    claims = _quadratic_sumcheck(claims[0].first,V1,V2,r12[r12.size()-1],ps);

    // Verifier checks
    V2 = get_RS_multifold_reduced_matrix(r11, n, L.size());
    if(evaluate_vector(V2,claims[1].second) != claims[1].first){
        printf("Error V, Delegation 4\n");
        exit(-1);
    }

    for(int i = 0; i < r12.size(); i++) claims[0].second.push_back(r12[i]);

    return claims[0];
}

vector<pair<F,vector<F>>> _quadruple_sumcheck(F y, vector<F> &v1, vector<F> &v2, vector<F> &v3, vector<F> &v4,F previous_r, double &ps){
	//vector<F> r = generate_randomness(int(log2(v1.size())));
	int rounds = int(log2(v1.size()));
	vector<quadruple_poly> p;
	F rand = previous_r;
	vector<F> r;
	for(int i = 0; i < rounds; i++){
		quadruple_poly poly = quadruple_poly(F_ZERO,F_ZERO,F_ZERO,F_ZERO,F_ZERO);
			linear_poly l1,l2,l3,l4;
			
			int L = 1 << (rounds -1- i);
			for(int j = 0; j < L; j++){
				l1 = linear_poly(v1[2*j+1] - v1[2*j],v1[2*j]);
				l2 = linear_poly(v2[2*j+1] - v2[2*j],v2[2*j]);
				l3 = linear_poly(v3[2*j+1] - v3[2*j],v3[2*j]);
				l4 = linear_poly(v4[2*j+1] - v4[2*j],v4[2*j]);
				poly = poly + (l1*l2*l3*l4);
				
			}

		vector<F> input;
		
		clock_t s,e;
		s = clock();
        
        if(poly.eval(0)+ poly.eval(1) != y){
            printf("Error in sumcheck round %d\n",i);
            //exit(-1);
        }
        rand = F::_random();//mimc_hash(rand,poly.a);
		rand = F::_random();//mimc_hash(rand,poly.b);
		rand = F::_random();///mimc_hash(rand,poly.c);
		rand = F::_random();//mimc_hash(rand,poly.d);
		e = clock();
		ps += 5*sizeof(F)/1024.0;
		y = poly.eval(rand);
        
        r.push_back(rand);
		
        //vt += (double)(e-s)/(double)CLOCKS_PER_SEC;

		for(int j = 0; j < L; j++){
            v1[j] = rand*(v1[2*j+1]-v1[2*j]) + v1[2*j];
            v2[j] = rand*(v2[2*j+1]-v2[2*j]) + v2[2*j];
            v3[j] = rand*(v3[2*j+1]-v3[2*j]) + v3[2*j];
            v4[j] = rand*(v4[2*j+1]-v4[2*j]) + v4[2*j];
        }
		//vector<vector<F>> temp = mimc_multihash3(input);
		//Pr.w_hashes.push_back(temp);
		
		//rand = temp[temp.size()-1][temp[0].size()-1];
		//rand = mimc_multihash(input);
		p.push_back(poly);
	}	
    ps += sizeof(F)/1024.0;
		
    return {make_pair(v1[0],r),make_pair(v2[0],r),make_pair(v3[0],r),make_pair(v4[0],r)};
}

vector<pair<F,vector<F>>> _cubic_sumcheck(F y, vector<F> &v1, vector<F> &v2, vector<F> &v3,F previous_r, double &ps, bool verify = true){
	//vector<F> r = generate_randomness(int(log2(v1.size())));
	int rounds = int(log2(v1.size()));
	vector<cubic_poly> p;
	F rand = previous_r;
	vector<F> r;
	for(int i = 0; i < rounds; i++){
		cubic_poly poly = cubic_poly(F_ZERO,F_ZERO,F_ZERO,F_ZERO);
			cubic_poly temp_poly = cubic_poly(F_ZERO,F_ZERO,F_ZERO,F_ZERO);
			linear_poly l1,l2,l3;
			
			int L = 1 << (rounds -1- i);
			for(int j = 0; j < L; j++){
				l1 = linear_poly(v1[2*j+1] - v1[2*j],v1[2*j]);
				l2 = linear_poly(v2[2*j+1] - v2[2*j],v2[2*j]);
				l3 = linear_poly(v3[2*j+1] - v3[2*j],v3[2*j]);
				poly = poly + (l1*l2*l3);
				
			}

		vector<F> input;
		
		clock_t s,e;
		s = clock();
        
        if(verify && poly.eval(0)+ poly.eval(1) != y){
            printf("Error in sumcheck round %d\n",i);
            //exit(-1);
        }
        rand = F::_random();//mimc_hash(rand,poly.a);
		rand = F::_random();//mimc_hash(rand,poly.b);
		rand = F::_random();///mimc_hash(rand,poly.c);
		rand = F::_random();//mimc_hash(rand,poly.d);
		e = clock();
		ps += 4*sizeof(F)/1024.0;
		y = poly.eval(rand);
        
        r.push_back(rand);
		
        //vt += (double)(e-s)/(double)CLOCKS_PER_SEC;

		for(int j = 0; j < L; j++){
            v1[j] = rand*(v1[2*j+1]-v1[2*j]) + v1[2*j];
            v2[j] = rand*(v2[2*j+1]-v2[2*j]) + v2[2*j];
            v3[j] = rand*(v3[2*j+1]-v3[2*j]) + v3[2*j];
        }
		//vector<vector<F>> temp = mimc_multihash3(input);
		//Pr.w_hashes.push_back(temp);
		
		//rand = temp[temp.size()-1][temp[0].size()-1];
		//rand = mimc_multihash(input);
		p.push_back(poly);
	}	
    ps += sizeof(F)/1024.0;
		
    return {make_pair(v1[0],r),make_pair(v2[0],r),make_pair(v3[0],r)};
}

pair<F,vector<F>> prove_extension(vector<vector<F>> &segments, vector<vector<F>> &extended_segments, vector<F> o, vector<int> queries, int rate, double &ps){
    vector<vector<F>> indexes(next_pow2(segments.size()));
    for(int i = 0; i < indexes.size(); i++){
        indexes[i].resize(extended_segments[0].size(),F(0));
        if(i < queries.size()) indexes[i][queries[i]] = F(1);
    }
    
    // Verifier steps
    vector<F> r,b;
    for(int i = 0; i < (int)log2(indexes.size()); i++) r.push_back(F::_random());
    precompute_beta(r,b);
    F K = F(0);
    for(int i = 0; i < segments.size(); i++) K += o[i]*b[i];
    /********** */

    // Prepare sumcheck
    vector<F> V1 = convert2vector(extended_segments);
    pad_vector(V1);
    vector<F> V2 = convert2vector(indexes);
    vector<F> V3(V2.size());
    for(int i = 0; i < indexes.size(); i++){
        for(int j = 0; j < indexes[0].size(); j++) V3[i*indexes[0].size() + j] = b[i];
    }
    vector<pair<F,vector<F>>> claims = _cubic_sumcheck(K,V1,V2,V3,r[r.size()-1],ps);
    
    // Verifier steps
    vector<F> r1,r2,b1,b2;
    for(int i = 0; i < (int)log2(indexes[0].size()); i++) r1.push_back(claims[0].second[i]);
    for(int i = (int)log2(indexes[0].size()); i < (int)log2(indexes.size()*indexes[0].size()); i++) r2.push_back(claims[0].second[i]);
    precompute_beta(r1,b1);precompute_beta(r2,b2);
    F _y = F(0);
    for(int i = 0; i < segments.size(); i++) _y += b1[queries[i]]*b2[i];
    if(_y != claims[1].first){
        printf("Error V, Delegation 5\n");
        exit(-1);   
    }
    _y = F(1);
    for(int i = 0; i < r2.size(); i++) _y *= ((F(1)-r2[i])*(F(1)-r[i]) + r2[i]*r[i]);
    if(_y != claims[2].first){
        printf("Error V, Delegation 6\n");
        exit(-1);
    }
    /**************/
    V1.clear();V1.resize(segments[0].size(),0);
    for(int i = 0; i < segments.size(); i++){
        for(int j = 0; j < segments[i].size(); j++){
            V1[j] += b2[i]*segments[i][j];
        }
    }
    
    

    pair<F,vector<F>> claim =  prove_fft(V1,r1,claims[0].first,rate, ps);
    for(int i = 0; i < r2.size(); i++){
        claim.second.push_back(r2[i]);
    }
    
    return claim;
}



F delegate_beta_eval(vector<F> r, vector<vector<F>> L, F y, int offset, double &ps){
    vector<F> L_vector,b,b_fft; 
    precompute_beta(r,b);
    b_fft = b;
    fft(b_fft,(int)log2(b_fft.size()),false);
    F omega  = getRootOfUnity(offset);
    
    L_vector = L[0];
    for(int i = 1; i < L.size(); i++){
        vector<F> buff;
        buff.resize(L[i].size()*L_vector.size());
        for(int j = 0; j < L[i].size(); j++){
            for(int k = 0; k < L_vector.size(); k++){
                buff[j*L_vector.size() + k] = L[i][j]*L_vector[k];
            }
        }
        L_vector = buff;
    }
    /*
    F y = F(0);
    for(int i = 0; i < L_vector.size(); i++){
        y += L_vector[i]*b_fft[i];
    }
    */
    vector<F> omega_vector(L_vector.size(),F(1));
    F p = omega;
    for(int i = 1; i < omega_vector.size(); i++){
        omega_vector[i] = omega*omega_vector[i-1];
        p = omega*p;
    }
    
    vector<pair<F,vector<F>>> claims = _cubic_sumcheck(y,L_vector,b_fft,omega_vector,F::_random(),ps);
    
    // Validate L_vector eval
    vt.start();
    int ctr = 0;
    vector<vector<F>> _r(L.size()); 
    for(int i = 0; i < L.size(); i++){
        for(int j = 0; j < (int)log2(L[i].size()); j++){
            _r[i].push_back(claims[0].second[ctr++]);
        }
    }
    F eval = F(1);
    for(int i = 0; i < L.size(); i++){
        eval *= evaluate_vector(L[i],_r[i]);
    }
    if(eval != claims[0].first) {
        printf("Error in delegating betas 1\n");
        exit(-1);
    }
    vt.end();
    
    
    
    vector<F> W_reduced(2*b.size(),F(0));
	vector<F> __r = claims[0].second;
    __r.push_back(F(0));
    phiGInit(W_reduced,__r.begin(),F(1),__r.size(),false);

    claims = _quadratic_sumcheck(claims[1].first, W_reduced, b, F::_random(),ps,false);
	
    
    vt.start();
    
    if(beta_identity(claims[0].second,r) != claims[1].first){
        //printf("Error in delegating betas 1\n");
        //exit(-1);        
    }
    vt.end();
    
    return y;
}



F delegate_lagrange_eval(vector<vector<F>> L1, vector<vector<F>> L2, F y, int offset, double &ps){
    vector<F> L1_vector,L2_vector,L1_fft; 
    F omega  = getRootOfUnity(offset);
    L1_vector = L1[0];
    L2_vector = L2[0];
    for(int i = 1; i < L1.size(); i++){
        vector<F> buff1,buff2;
        buff1.resize(L1[i].size()*L1_vector.size());
        for(int j = 0; j < L1[i].size(); j++){
            for(int k = 0; k < L1_vector.size(); k++){
                buff1[j*L1_vector.size() + k] = L1[i][j]*L1_vector[k];
            }
        }
        L1_vector = buff1;
    }
    
        
    for(int i = 1; i < L2.size(); i++){
        vector<F> buff1,buff2;
        buff2.resize(L2[i].size()*L2_vector.size());
        for(int j = 0; j < L2[i].size(); j++){
            for(int k = 0; k < L2_vector.size(); k++){
                buff2[j*L2_vector.size() + k] = L2[i][j]*L2_vector[k];
            }
        }
        L2_vector = buff2;
    }

    L1_fft = L1_vector;
    fft(L1_fft,(int)log2(L1_fft.size()),false);
    vector<F> omega_vector(L1_fft.size(),F(1));
    F p = omega;
    for(int i = 1; i < omega_vector.size(); i++){
        omega_vector[i] = omega*omega_vector[i-1];
        p = omega*p;
    }
    /*
    F y = F(0);
    for(int i = 0; i < L1_vector.size(); i++){
        y += L2_vector[i]*L1_fft[i];
    }
    */
    
    vector<pair<F,vector<F>>> claims = _cubic_sumcheck(y,L2_vector,L1_fft,omega_vector,F::_random(),ps);
    
    // Validate L_vector eval
    int ctr = 0;
    vector<vector<F>> _r(L2.size()); 
    for(int i = 0; i < L2.size(); i++){
        for(int j = 0; j < (int)log2(L2[i].size()); j++){
            _r[i].push_back(claims[0].second[ctr++]);
        }
    }
    F eval = F(1);
    for(int i = 0; i < L2.size(); i++){
        eval *= evaluate_vector(L2[i],_r[i]);
    }
    if(eval != claims[0].first) {
        printf("Error in delegating betas 1\n");
        exit(-1);
    }
    eval = F(1);
    for(int i = 0; i < (int)log2(omega_vector.size()); i++){
        eval *= (F(1)-claims[0].second[i]+claims[0].second[i]*omega);
        omega = omega*omega;
    }
    if(eval != claims[2].first){
        printf("Error in delegating betas 2\n");
        exit(-1);
    }

    
    vector<F> W_reduced(L1_fft.size(),F(0));
	
    phiGInit(W_reduced,claims[0].second.begin(),F(L1_fft.size()).inv(),claims[0].second.size(),false);
    // there is a bug in zkCNN code
    claims = _quadratic_sumcheck(claims[1].first, W_reduced, L1_vector, F::_random(),ps,false);
	ctr = 0;
    _r.clear();_r.resize(L1.size());
    for(int i = 0; i < L1.size(); i++){
        for(int j = 0; j < (int)log2(L1[i].size()); j++){
            _r[i].push_back(claims[0].second[ctr++]);
        }
    }
    eval = F(1);
    for(int i = 0; i < L1.size(); i++){
        eval *= evaluate_vector(L1[i],_r[i]);
    }
    if(eval != claims[1].first){
        printf("Error in delegating betas 2\n");
        exit(-1);
    }
    
    return y;  
}


F delegate_lagrange_eval(vector<vector<F>> L1, vector<vector<F>> L2, F y, int offset, int queries, double &ps){
    vector<F> L1_vector,L2_vector,L1_fft; 
    F omega  = getRootOfUnity(offset);
    L1_vector = L1[0];
    L2_vector = L2[0];
    for(int i = 1; i < L1.size(); i++){
        vector<F> buff1,buff2;
        buff1.resize(L1[i].size()*L1_vector.size());
        for(int j = 0; j < L1[i].size(); j++){
            for(int k = 0; k < L1_vector.size(); k++){
                buff1[j*L1_vector.size() + k] = L1[i][j]*L1_vector[k];
            }
        }
        L1_vector = buff1;
    }
    
        
    for(int i = 1; i < L2.size(); i++){
        vector<F> buff1,buff2;
        buff2.resize(L2[i].size()*L2_vector.size());
        for(int j = 0; j < L2[i].size(); j++){
            for(int k = 0; k < L2_vector.size(); k++){
                buff2[j*L2_vector.size() + k] = L2[i][j]*L2_vector[k];
            }
        }
        L2_vector = buff2;
    }

    L1_fft = L1_vector;
    for(int i = L1_fft.size()-queries; i < L1_fft.size(); i++) L1_fft[i] = 0;
    
    fft(L1_fft,(int)log2(L1_fft.size()),false);
    vector<F> omega_vector(L1_fft.size(),F(1));
    F p = omega;
    for(int i = 1; i < omega_vector.size(); i++){
        omega_vector[i] = omega*omega_vector[i-1];
        p = omega*p;
    }
    /*
    F y = F(0);
    for(int i = 0; i < L1_vector.size(); i++){
        y += L2_vector[i]*L1_fft[i];
    }
    */
    vector<F> ones(L2_vector.size(),F(1));
    for(int i = ones.size()-queries; i < ones.size(); i++) ones[i] = F(0);
    
    
    vector<pair<F,vector<F>>> claims = _cubic_sumcheck(y,L2_vector,L1_fft,omega_vector,F::_random(),ps);
    printf("OK\n");
    // Validate L_vector eval
    int ctr = 0;
    vector<vector<F>> _r(L2.size()); 
    for(int i = 0; i < L2.size(); i++){
        for(int j = 0; j < (int)log2(L2[i].size()); j++){
            _r[i].push_back(claims[0].second[ctr++]);
        }
    }
    F eval = F(1);
    for(int i = 0; i < L2.size(); i++){
        eval *= evaluate_vector(L2[i],_r[i]);
    }
    if(eval != claims[0].first) {
        printf("Error in delegating betas 1\n");
        exit(-1);
    }
    eval = F(1);
    for(int i = 0; i < (int)log2(omega_vector.size()); i++){
        eval *= (F(1)-claims[0].second[i]+claims[0].second[i]*omega);
        omega = omega*omega;
    }
    if(eval != claims[2].first){
        printf("Error in delegating betas 2\n");
        exit(-1);
    }

    
    vector<F> W_reduced(L1_fft.size(),F(0));
	
    phiGInit(W_reduced,claims[0].second.begin(),F(L1_fft.size()).inv(),claims[0].second.size(),false);
    // there is a bug in zkCNN code
    
    claims = _cubic_sumcheck(claims[1].first, W_reduced, L1_vector,ones, F::_random(),ps,false);
	ctr = 0;
    _r.clear();_r.resize(L1.size());
    for(int i = 0; i < L1.size(); i++){
        for(int j = 0; j < (int)log2(L1[i].size()); j++){
            _r[i].push_back(claims[0].second[ctr++]);
        }
    }
    eval = F(1);
    for(int i = 0; i < L1.size(); i++){
        eval *= evaluate_vector(L1[i],_r[i]);
    }
    if(eval != claims[1].first){
        printf("Error in delegating betas 2\n");
        exit(-1);
    }
    for(int i = ones.size()-queries; i < ones.size(); i++){
        int val = i;
        F b = F(1);
        for(int j = 0; j < claims[2].second.size(); j++){
            b *= (F(1)-F(val&1))*(F(1)-claims[2].second[j]) + F(val&1)*claims[2].second[j];
            val = val >> 1;
        }
        claims[2].first += b;
    }
    
    if(claims[2].first != 1){
        printf("Error in delegating betas 3\n");
    }
    return y;  
}
