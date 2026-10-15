
#include "coPCS_utils.h"

vector<vector<F>> global_poly;
int rate = 4;
//int l = 500;

void setup(vector<vector<F>> &R_shares, int N, int M, int l, int k, int _k){
    vector<vector<F>> R(l);
    R_shares.resize(l);
    for(int i = 0; i < l; i++){
        R[i].resize(_k);
        R_shares[i].resize(N,0);
        for(int j = 0; j < _k; j++){
            R[i][j] = F(j*3122 + i);
        }
    }
    if(global_poly.size() != 0){
        for(int i = 0; i < l; i++){
            for(int j = 0; j < _k; j++){
                global_poly[j][i+M/k] = R[i][j];
            }
        }
    }
    
    for(int i = 0; i < l; i++){
        fft(R[i],(int)log2(_k),true);
        R[i].resize(2*N);
        fft(R[i],(int)log2(2*N),false);
        for(int j = 0; j < N; j++){
            R_shares[i][j] = R[i][2*j+1];
        }
    }
    
}


void encode_protocol_step2(vector<F> &row, vector<F> &rand, vector<F> &codeword){
    vector<F> buff(next_pow2(row.size() + rand.size()),0);
    for(int i = 0; i < row.size(); i++){
        buff[i] = row[i];
    }
    for(int i = 0; i < rand.size(); i++){
        buff[i+row.size()] = rand[i];
    }
    fft(buff,(int)log2(buff.size()),true);
    F omega = getRootOfUnity(1+(int)log2(rate)+(int)log2(buff.size())).inv();
    F mul = F(1);
    for(int j = 0; j < buff.size(); j++){
        buff[j] = mul*buff[j];
        mul = mul*omega;
    }        
                
    codeword.resize(rate*buff.size(),0);
    for(int i = 0; i < buff.size(); i++){
        codeword[i] = buff[i];
    }
    fft(codeword,(int)log2(codeword.size()),false);
}

void RS_fold(vector<F> &codeword, F a){
    int n = codeword.size();
    F two_inv = F(2).inv();
    vector<F> new_codeword(codeword.size()/2);
    F omega = getRootOfUnity((int)log2(codeword.size()));
    omega = omega.inv();
    F inv_omegas = F(1);
    for(int i = 0; i < new_codeword.size(); i++){
        new_codeword[i] = two_inv*((F(1)-a)*(codeword[i] + codeword[i+n/2])+ a*inv_omegas*(codeword[i] - codeword[i+n/2])); 
        inv_omegas = inv_omegas*omega;
    }
    codeword = new_codeword;
}


void RS_multifold(vector<F> &codeword, vector<F> a){
    F two_inv = F(2).inv();
    
    for(int j = 0; j < a.size(); j++){
        int n = codeword.size();
        vector<F> new_codeword(codeword.size()/2);
        F omega = getRootOfUnity((int)log2(codeword.size()));
        omega = omega.inv();
        F inv_omegas = F(1);
        for(int i = 0; i < new_codeword.size(); i++){
            new_codeword[i] = two_inv*((F(1)-a[j])*(codeword[i] + codeword[i+n/2])+ a[j]*inv_omegas*(codeword[i] - codeword[i+n/2])); 
            inv_omegas = inv_omegas*omega;
        }
        codeword = new_codeword;
    }   
}

void RS_multifold2(vector<F> &codeword, F a, int fold_degree){
    F two_inv = F(2).inv();
    int log_fold_degree = (int)log2(fold_degree);
    
    for(int j = 0; j < log_fold_degree; j++){
        int n = codeword.size();
        vector<F> new_codeword(codeword.size()/2);
        F omega = getRootOfUnity((int)log2(codeword.size()));
        omega = omega.inv();
        F inv_omegas = F(1);
        for(int i = 0; i < new_codeword.size(); i++){
            new_codeword[i] = two_inv*((codeword[i] + codeword[i+n/2])+ a*inv_omegas*(codeword[i] - codeword[i+n/2])); 
            inv_omegas = inv_omegas*omega;
        }
        a = a*a;
        codeword = new_codeword;
    }      
}

void RS_multifold3(vector<F> &codeword, vector<F> a, int fold_degree){
    F two_inv = F(2).inv();
    int log_fold_degree = (int)log2(fold_degree);
    int ctr = 0;
    for(int j = 0; j < log_fold_degree; j++){
         int n = codeword.size();
        vector<F> new_codeword(codeword.size()/2);
        F omega = getRootOfUnity((int)log2(codeword.size()));
        omega = omega.inv();
        F inv_omegas = F(1);
        for(int i = 0; i < new_codeword.size(); i++){
            new_codeword[i] = two_inv*((a[ctr]*(codeword[i] + codeword[i+n/2])+ a[ctr+1]*inv_omegas*(codeword[i] - codeword[i+n/2]))); 
            inv_omegas = inv_omegas*omega;
            if(i != 0 && (i%(new_codeword.size()/a.size()) == 0)) {
                ctr+=2;
                //printf("OK %d, %d\n", i , new_codeword.size());
            }
        }
        for(int i = 0; i < a.size(); i++) a[i] = F(1);
        codeword = new_codeword;
    }      
}

void RS_multifold4(vector<F> &codeword, vector<F> a, int fold_degree){
    F four_inv = F(4).inv();
    int log_fold_degree = (int)log2(fold_degree);
    int ctr = 0;
    int n = codeword.size();
    vector<F> new_codeword(codeword.size()/4);
    F omega = getRootOfUnity((int)log2(codeword.size()));
    omega = omega.inv();
    F omega3 = omega*omega;
    F omega4 = omega3*omega;
    F omega2 = getRootOfUnity((int)log2(codeword.size()));
    
    omega2 = omega2.fastPow(omega2,new_codeword.size());
    
    omega2 = omega2.inv();
         
    F inv_omegas = F(1);
    F inv_omegas3 = F(1);
    F inv_omegas4 = F(1);
    for(int i = 0; i < new_codeword.size(); i++){
        new_codeword[i] = four_inv*(
                    a[0]*(codeword[i] + codeword[i+n/4] + codeword[i+n/2] + codeword[i+3*n/4]) + 
                    a[2]*inv_omegas3*(codeword[i] + codeword[i+n/2] - (codeword[i+n/4] + codeword[i+3*n/4])) +
                    a[1]*inv_omegas*(codeword[i] - codeword[i+n/2] + omega2*(codeword[i+n/4] - codeword[i+3*n/4])) +
                    a[3]*inv_omegas4*(codeword[i] - codeword[i+n/2] - omega2*(codeword[i+n/4] - codeword[i+3*n/4])) 
                    ); 
        inv_omegas = inv_omegas*omega;
        inv_omegas3 = inv_omegas3*omega3;
        inv_omegas4 = inv_omegas4*omega4;

    }
    codeword = new_codeword; 
}




F RS_multifold_segment(vector<F> segment, vector<F> L, int n, int idx){
    F len_inv = F(L.size()).inv();
    int fold_degree = L.size();
    int seg_n = segment.size();
    int log_fold_degree = (int)log2(fold_degree);
    F omega = getRootOfUnity((int)log2(n));
    omega = omega.inv();
    omega = omega.fastPow(omega,n/seg_n);
    
    
    F inv_omegas = getRootOfUnity((int)log2(n));
    inv_omegas = inv_omegas.inv();
    inv_omegas = inv_omegas.fastPow(inv_omegas,idx);

    vector<vector<F>> partial_codewords(segment.size()/2);
    for(int j = 0; j < seg_n/2; j++){
        partial_codewords[j].resize(2);
        partial_codewords[j][0] = (segment[j] + segment[j+seg_n/2]);
        partial_codewords[j][1] = inv_omegas*(segment[j] - segment[j+seg_n/2]);
        inv_omegas = inv_omegas*omega;
    }
    if(L.size() > 2){

        
        vector<F> pow((int)log2(fold_degree),F(1));
        omega = getRootOfUnity((int)log2(n));
        omega = omega.inv();
        pow[1] = omega*omega;
        for(int i = 2; i < pow.size(); i++) pow[i] = pow[i-1]*pow[i-1];
        
        vector<F> pow_omega(pow.size());
        for(int i = 1; i < pow.size(); i++) pow_omega[i] = pow[i].fastPow(pow[i],idx);
        
        
        for(int i = 1; i < pow.size(); i++) pow[i] = pow[i].fastPow(pow[i],n/(seg_n));
        
        
        
        for(int i = 1; i < log_fold_degree; i++){
            seg_n = seg_n/2;
            //F pow_omega = F(1);
            for(int j = 0; j < seg_n/2; j++){
                partial_codewords[j].resize(1<<(i+1));
                for(int k = 0; k < partial_codewords[j].size()/2; k++){
                    F v1 = partial_codewords[j][k] + partial_codewords[j+seg_n/2][k];
                    F v2 = pow_omega[i]*(partial_codewords[j][k] - partial_codewords[j+seg_n/2][k]);
                    
                    partial_codewords[j][k] = v1;
                    partial_codewords[j][k+partial_codewords[j].size()/2] = v2;
                }
                partial_codewords[j+seg_n/2].clear();
                pow_omega[i] = pow_omega[i]*pow[i];
            }
        }
    }
    
    F y = F(0);
    for(int i = 0; i < L.size(); i++) y += L[i]*partial_codewords[0][i];
    
    return len_inv*y;
}


vector<F> get_RS_multifold_reduced_matrix(vector<F> r, int n, int seg_n){
    vector<F> reduced_matrix(seg_n,F(0));
    
    
    F omega = getRootOfUnity((int)log2(n));
    omega = omega.inv();
    omega = omega.fastPow(omega,n/seg_n);
    
    F inv_omegas = F(1);
    for(int j = 0; j < seg_n/2; j++){
        reduced_matrix[j] = (F(1)-r[0] + inv_omegas*r[0]);
        reduced_matrix[j+seg_n/2] = (F(1)-r[0] - inv_omegas*r[0]);
        inv_omegas = inv_omegas*omega;
    }
    for(int i = 1; i < r.size(); i++){
        F omega = getRootOfUnity((int)log2(n));
        omega = omega.inv();
        omega = omega.fastPow(omega,(1<<(i))*n/(seg_n));
        //F pow_omega = F(1);
        F inv_omegas = F(1);
        for(int j = 0; j < seg_n/2; j++){
            
            reduced_matrix[j] = (F(1)-r[i] + inv_omegas*r[i])*reduced_matrix[j];
            reduced_matrix[j+seg_n/2] = (F(1)-r[i] + inv_omegas*r[i])*reduced_matrix[j+seg_n/2];
            inv_omegas = inv_omegas*omega;
        }
    }


    
    return reduced_matrix;
}

vector<F> get_RS_multifold_segment(vector<F> segment, vector<F> L, int n, int idx){
    F len_inv = F(L.size()).inv();
    int fold_degree = L.size();
    int seg_n = segment.size();
    int log_fold_degree = (int)log2(fold_degree);
    F omega = getRootOfUnity((int)log2(n));
    omega = omega.inv();
    omega = omega.fastPow(omega,n/seg_n);
    
    
    /*
    F inv_omegas = getRootOfUnity((int)log2(n));
    inv_omegas = inv_omegas.inv();
    inv_omegas = inv_omegas.fastPow(inv_omegas,idx);
    */
    F inv_omegas = F(1);
    vector<vector<F>> partial_codewords(segment.size()/2);
    for(int j = 0; j < seg_n/2; j++){
        partial_codewords[j].resize(2);
        partial_codewords[j][0] = (segment[j] + segment[j+seg_n/2]);
        partial_codewords[j][1] = inv_omegas*(segment[j] - segment[j+seg_n/2]);
        inv_omegas = inv_omegas*omega;
    }
    if(L.size() > 2){

        
        vector<F> pow((int)log2(fold_degree),F(1));
        omega = getRootOfUnity((int)log2(n));
        omega = omega.inv();
        pow[1] = omega*omega;
        for(int i = 2; i < pow.size(); i++) pow[i] = pow[i-1]*pow[i-1];
        
        vector<F> pow_omega(pow.size());
        for(int i = 1; i < pow.size(); i++) pow_omega[i] = F(1);//pow[i].fastPow(pow[i],idx);
        
        
        for(int i = 1; i < pow.size(); i++) pow[i] = pow[i].fastPow(pow[i],n/(seg_n));
        
        
        
        for(int i = 1; i < log_fold_degree; i++){
            seg_n = seg_n/2;
            //F pow_omega = F(1);
            for(int j = 0; j < seg_n/2; j++){
                partial_codewords[j].resize(1<<(i+1));
                for(int k = 0; k < partial_codewords[j].size()/2; k++){
                    F v1 = partial_codewords[j][k] + partial_codewords[j+seg_n/2][k];
                    F v2 = pow_omega[i]*(partial_codewords[j][k] - partial_codewords[j+seg_n/2][k]);
                    
                    partial_codewords[j][k] = v1;
                    partial_codewords[j][k+partial_codewords[j].size()/2] = v2;
                }
                partial_codewords[j+seg_n/2].clear();
                pow_omega[i] = pow_omega[i]*pow[i];
            }
        }
    }
    
    //F y = F(0);
    //for(int i = 0; i < L.size(); i++) y += L[i]*partial_codewords[0][i];
    
    return partial_codewords[0];
}


F get_RS_multifold_segment_tensor(vector<F> segment, vector<F> r, int n, int idx){
    F len_inv = F(segment.size()).inv();
    int fold_degree = segment.size();
    int seg_n = segment.size();
    int log_fold_degree = (int)log2(fold_degree);
    F omega = getRootOfUnity((int)log2(n));
    omega = omega.inv();
    omega = omega.fastPow(omega,n/seg_n);
    
    
    F inv_omegas = getRootOfUnity((int)log2(n));
    inv_omegas = inv_omegas.inv();
    inv_omegas = inv_omegas.fastPow(inv_omegas,idx);
    vector<F> partial_codewords(segment.size()/2);
    for(int j = 0; j < seg_n/2; j++){
        partial_codewords[j] = (F(1)-r[0])*(segment[j] + segment[j+seg_n/2]) + r[0]*inv_omegas*(segment[j] - segment[j+seg_n/2]);
        inv_omegas = inv_omegas*omega;
    }
    if(r.size() > 1){

        
        vector<F> pow((int)log2(fold_degree),F(1));
        omega = getRootOfUnity((int)log2(n));
        omega = omega.inv();
        pow[1] = omega*omega;
        for(int i = 2; i < pow.size(); i++) pow[i] = pow[i-1]*pow[i-1];
        
        //for(int i = 1; i < pow.size(); i++) pow[i].fastPow(pow[i],idx);
        
        
        for(int i = 1; i < pow.size(); i++) pow[i] = pow[i].fastPow(pow[i],n/(seg_n));
        
        
        
        for(int i = 1; i < log_fold_degree; i++){
            seg_n = seg_n/2;
            omega = getRootOfUnity((int)log2(n/(1<<i)));
            omega = omega.inv();
            F pow_omega = omega.fastPow(omega,idx);
            for(int j = 0; j < seg_n/2; j++){
                partial_codewords[j] = (F(1)-r[i])*(partial_codewords[j] + partial_codewords[j+seg_n/2]) + r[i]*(pow_omega*(partial_codewords[j] - partial_codewords[j+seg_n/2]));
                pow_omega = pow_omega*pow[i];
            }
        }
    }
    
    //F y = F(0);
    //for(int i = 0; i < L.size(); i++) y += L[i]*partial_codewords[0][i];
    
    return len_inv*partial_codewords[0];
}


void RS_multifold5(vector<F> &codeword, vector<F> a, int fold_degree){
    F len_inv = F(a.size()).inv();
    int log_fold_degree = (int)log2(fold_degree);
    int n = codeword.size();
    vector<F> new_codeword(codeword.size()/a.size(),F(0));
    
    vector<vector<F>> partial_codewords(codeword.size()/2);
    for(int i = 0; i < partial_codewords.size(); i++){
        partial_codewords[i].resize(2,F(0));
    } 
    
    F omega = getRootOfUnity((int)log2(n));
    omega = omega.inv();
    F inv_omegas = F(1);
    for(int j = 0; j < codeword.size()/(2); j++){
        partial_codewords[j][0] = (codeword[j] + codeword[j+n/2]);
        partial_codewords[j][1] = inv_omegas*(codeword[j] - codeword[j+n/2]);
        inv_omegas = inv_omegas*omega;
    }
    if(fold_degree > 2){
    vector<F> pow((int)log2(fold_degree),F(1));
    pow[1] = omega*omega;
    for(int i = 2; i < pow.size(); i++) pow[i] = pow[i-1]*pow[i-1];
    

    for(int i = 1; i < (int)log2(fold_degree); i++){
        n = n/2;
        F pow_omega = F(1);
        for(int j = 0; j < codeword.size()/(1<<(i+1)); j++){
            partial_codewords[j].resize(1<<(i+1));
            for(int k = 0; k < partial_codewords[j].size()/2; k++){
                F v1 = partial_codewords[j][k] + partial_codewords[j+n/2][k];
                F v2 = pow_omega*(partial_codewords[j][k] - partial_codewords[j+n/2][k]);
                
                partial_codewords[j][k] = v1;
                partial_codewords[j][k+partial_codewords[j].size()/2] = v2;
            }
            partial_codewords[j+n/2].clear();
            pow_omega = pow_omega*pow[i];
        }
    }
    }
    for(int i = 0; i < codeword.size()/fold_degree; i++){
        for(int j = 0; j < fold_degree; j++){
            new_codeword[i] += a[j]*partial_codewords[i][j]; 
        }
        new_codeword[i] *= len_inv;
    }
    

    codeword = new_codeword; 
}


void RS_distributed_fold(vector<F> &codeword, vector<F> a, int fold_degree, int seg_size, int N, int id){
    F len_inv = F(a.size()).inv();
    int log_fold_degree = (int)log2(fold_degree);
    int n = codeword.size();
    vector<F> new_codeword(codeword.size()/a.size(),F(0));
    
    vector<vector<F>> partial_codewords(codeword.size()/2);
    for(int i = 0; i < partial_codewords.size(); i++){
        partial_codewords[i].resize(2,F(0));
    } 
    
    F omega = getRootOfUnity((int)log2(N*n));
    F omega_offset = omega.inv();
    omega_offset = omega_offset.fastPow(omega_offset,(N-1)*seg_size);

    omega = omega.inv();
    F inv_omegas = omega.fastPow(omega,id*seg_size);
    for(int j = 0; j < codeword.size()/(2); j++){
        if(j % seg_size == 0 && j != 0){
            inv_omegas = omega_offset*inv_omegas;
        }
        partial_codewords[j][0] = (codeword[j] + codeword[j+n/2]);
        partial_codewords[j][1] = inv_omegas*(codeword[j] - codeword[j+n/2]);
        inv_omegas = inv_omegas*omega;
        
    }
    if(fold_degree > 2){
        vector<F> pow((int)log2(fold_degree),F(1));
        pow[1] = omega*omega;
        for(int i = 2; i < pow.size(); i++) pow[i] = pow[i-1]*pow[i-1];
        

        for(int i = 1; i < (int)log2(fold_degree); i++){
            n = n/2;
            F pow_omega = omega.fastPow(omega,(1<<i)*id*seg_size);
            omega_offset = omega_offset*omega_offset;
                    
            for(int j = 0; j < codeword.size()/(1<<(i+1)); j++){
                if(j % seg_size == 0 && j!=0){
                    //printf("%d\n",j);
                    pow_omega *= omega_offset;
                }
                partial_codewords[j].resize(1<<(i+1));
                for(int k = 0; k < partial_codewords[j].size()/2; k++){
                    F v1 = partial_codewords[j][k] + partial_codewords[j+n/2][k];
                    F v2 = pow_omega*(partial_codewords[j][k] - partial_codewords[j+n/2][k]);
                    
                    partial_codewords[j][k] = v1;
                    partial_codewords[j][k+partial_codewords[j].size()/2] = v2;
                }
                partial_codewords[j+n/2].clear();
                pow_omega = pow_omega*pow[i];
                
            }
        }
    }
    for(int i = 0; i < codeword.size()/fold_degree; i++){
        for(int j = 0; j < fold_degree; j++){
            new_codeword[i] += a[j]*partial_codewords[i][j]; 
        }
        new_codeword[i] *= len_inv;
    }
    

    codeword = new_codeword; 
}



F Fold_point(vector<F> &codeword, int idx, F a, int degree){
    int log_degree = (int)log2(degree);
    vector<F> buff(1);
    vector<F> points(1,F(1));
    int n = codeword.size();
    F two_inv = F(2).inv();
    F omega = getRootOfUnity((int)log2(codeword.size()));
    omega = omega.inv();
    if(idx != 0) omega.fastPow(omega,idx);
    for(int i = 0; i < log_degree; i++){
        buff.resize(1<<(i+1));
        for(int j = 0; j < points.size(); j++){
            buff[j] = points[j]*two_inv*((codeword[idx] + codeword[idx+n/2]));
            buff[j+buff.size()/2] = points[j]*two_inv*(omega*(codeword[idx] - codeword[idx+n/2]));
        }
        points.resize(2*points.size());
        for(int j = 0; j < points.size(); j++) points[j] = buff[j];
        omega = omega*omega;     
        n = n/2;
    }
    F point  = F(0);
    F pow = F(1);
    for(int i = 0; i < points.size(); i++){
        point += pow*points[i];
        pow = a*pow;
    }
    return point;
}

