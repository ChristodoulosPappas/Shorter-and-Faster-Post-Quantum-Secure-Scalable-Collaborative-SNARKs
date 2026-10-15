#include "coSumcheck.h"
#include "timer.hpp"
extern timer pt_cp;
extern timer vt;




vector<F> cubic_coSumcheck_phase1(vector<F> &p1, vector<F> &p2, vector<F> &p3, vector<F> &b1, vector<F> &b2, vector<vector<F>> R, F r, int d ,int iter, int N){
    pt_cp.start();
    vector<F> H(4*d,F(0));
    bool zeroflag;
    for(int j = 0; j < p1.size()/d; j++){
        zeroflag = true;
        vector<F> h1(d),h2(d),h3(d),h4(d);
        for(int k = 0; k < d; k++){
            h1[k] = p1[j*d + k];
            h2[k] = p2[j*d + k];
            h3[k] = b1[j*d + k];
            h4[k] = p3[j*d + k];
        }
        
        for(int k = 0; k < d; k++){
            if(h1[k] != 0){
                zeroflag = false;
                break;
            }
        }
        if(zeroflag) continue;
        fft(h1,(int)log2(d),true);
        fft(h2,(int)log2(d),true);
        fft(h3,(int)log2(d),true);
        fft(h4,(int)log2(d),true);
        h1.resize(4*d,F(0));
        h2.resize(4*d,F(0));
        h3.resize(4*d,F(0));
        h4.resize(4*d,F(0));
        fft(h1,(int)log2(4*d),false);
        fft(h2,(int)log2(4*d),false);
        fft(h3,(int)log2(4*d),false);
        fft(h4,(int)log2(4*d),false);
        for(int k = 0; k < 4*d; k++){
            H[k] += (h1[k]*h2[k]-h4[k])*h3[k];
        }

    }
    
    
    vector<F> H_r = R[iter];//(2*d,F(0));
    
    for(int j = iter+1; j < R.size(); j++){
        for(int k = 0; k < R[j].size()/4; k++){
            H_r[0] += R[j][4*k];
        }
    }
    

    for(int i = 0; i < H.size(); i++){
        H[i] += r*H_r[i];
    }
    pt_cp.end();
    
    return sumcheck_aggr(H, b2, N);    
}

void cubic_coSumcheck_phase2(vector<F> &p1, vector<F> &p2, vector<F> &p3, vector<F> &b1, vector<F> L, int d){
    bool zeroflag;
    
    for(int j = 0; j < p1.size()/d; j++){
        zeroflag = true;
        for(int k = 0; k < d; k++){
            if(p1[j*d + k] != 0){
                zeroflag = false;
                break;
            }
        }
        if(!zeroflag){
            p1[j] = L[0]*p1[j*d];
            for(int k = 1; k < d; k++){
                p1[j] += L[k]*p1[j*d + k];
            }
        }else{
            p1[j] = 0;
        }
        zeroflag = true;
        for(int k = 0; k < d; k++){
            if(p2[j*d + k] != 0){
                zeroflag = false;
                break;
            }
        }
        if(!zeroflag){
            p2[j] = L[0]*p2[j*d];
            for(int k = 1; k < d; k++){
                p2[j] += L[k]*p2[j*d + k];
            }
        }else{
            p2[j] = 0;
        }
        zeroflag = true;
        for(int k = 0; k < d; k++){
            if(p3[j*d + k] != 0){
                zeroflag = false;
                break;
            }
        }
        
        if(!zeroflag){
            p3[j] = L[0]*p3[j*d];
            for(int k = 1; k < d; k++){
                p3[j] += L[k]*p3[j*d + k];
            }
        }else{
            p3[j] = 0;
        }
        zeroflag = true;
        for(int k = 0; k < d; k++){
            if(b1[j*d + k] != 0){
                zeroflag = false;
                break;
            }
        }
        if(!zeroflag){
            b1[j] = L[0]*b1[j*d];
            for(int k = 1; k < d; k++){
                b1[j] += L[k]*b1[j*d + k];
            }
        }else{
            b1[j] = 0;
        }
        /*
        p1[j] = L[0]*p1[j*d];
        p2[j] = L[0]*p2[j*d];
        p3[j] = L[0]*p3[j*d];
        b1[j] = L[0]*b1[j*d];
        for(int k = 1; k < d; k++){
            p1[j] += L[k]*p1[j*d + k];
            p2[j] += L[k]*p2[j*d + k];
            p3[j] += L[k]*p3[j*d + k];
            b1[j] += L[k]*b1[j*d + k];
        }*/
        
    }
    p1.resize(p1.size()/d);
    p2.resize(p2.size()/d);
    p3.resize(p3.size()/d);
    b1.resize(b1.size()/d);
}

vector<F> quadratic_coSumcheck_phase1(vector<F> &p1, vector<F> &p2, vector<vector<F>> R, F r, int d ,int iter, int N){
    pt_cp.start();
    bool zeroflag;
    vector<F> H(2*d,F(0));
    for(int j = 0; j < p1.size()/d; j++){
        zeroflag = true;
        vector<F> h1(d),h2(d);
        for(int k = 0; k < d; k++){
            h1[k] = p1[j*d + k];
            h2[k] = p2[j*d + k];
        }
        for(int k = 0; k < d; k++){
            if(h1[k] != 0){
                zeroflag = false;
                break;
            }
        }
        if(zeroflag) continue;
        

        fft(h1,(int)log2(d),true);
        fft(h2,(int)log2(d),true);
        h1.resize(2*d,F(0));
        h2.resize(2*d,F(0));
        fft(h1,(int)log2(2*d),false);
        fft(h2,(int)log2(2*d),false);
        for(int k = 0; k < 2*d; k++){
            H[k] += h1[k]*h2[k];
        }
    }
    
    
    vector<F> H_r = R[iter];//(2*d,F(0));
    
    for(int j = iter+1; j < R.size(); j++){
        for(int k = 0; k < R[j].size()/2; k++){
            H_r[0] += R[j][2*k];
        }
    }
    

    for(int i = 0; i < H.size(); i++){
        H[i] += r*H_r[i];
    }
    vector<F> ones(N/4,F(1));
    pt_cp.end();
    
    return sumcheck_aggr(H,ones, N);    
}


void quadratic_coSumcheck_phase2(vector<F> &p1, vector<F> &p2, vector<F> L, int d){
    for(int j = 0; j < p1.size()/d; j++){
        p1[j] = L[0]*p1[j*d];
        p2[j] = L[0]*p2[j*d];
        for(int k = 1; k < d; k++){
            p1[j] += L[k]*p1[j*d + k];
            p2[j] += L[k]*p2[j*d + k];
        }
    }
    p1.resize(p1.size()/d);
    p2.resize(p2.size()/d);

}

F quadratic_coSumcheck_phase2_sparse(vector<vector<F>> &R, vector<F> L, int iter, int d){
    F s = F(0);
    for(int j = 0; j < R[iter].size(); j++) s += L[j]*R[iter][j];

    for(int j = iter+1; j < R.size(); j++){
        for(int k = 0; k < R[j].size(); k++){
            R[j][k] *= L[0];
        }
    }
    if(iter+1 < R.size()){
        R[iter+1][0] += s;
    }
    return s;
}

vector<vector<F>> ip_vec;


vector<pair<F,vector<F>>> quadratic_coSumcheck(vector<F> &p1, vector<F> &p2, vector<vector<F>> R, F K, F K_r, vector<int> degrees, int N, double &ps){
    F c = hash_to_field({K,K_r});
    K = K + c*K_r;
    F final_eval_r;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get my process id
    vector<F> sumcheck_r;
    ip_vec.clear();
    ip_vec.resize(degrees.size());
    for(int k = 0; k < degrees.size(); k++){
        ip_vec[k].resize(2*degrees[k],F(1));
    }
    
    for(int i = 0; i < degrees.size(); i++){
        
        vector<F> H = quadratic_coSumcheck_phase1(p1,p2,R,c,degrees[i],i,N);
        if(rank == 0)ps += ((double)H.size()*16)/1024.0;
        if(rank == 0) vt.start();
        
        F eval = F(0);
        for(int j = 0; j < H.size()/2; j++){
            eval += H[2*j];
        }
        //printf("%lld,%lld,%lld,%lld\n",eval.real,K.real,eval.img,K.img);
        if(eval != K){
            if(rank == 0) printf("ERROR %d\n",i);
            //return;    
        }
        K = 0;
        pt_cp.start();
        
        F r = hash_to_field(H);
        sumcheck_r.push_back(r);
        
        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees[i]*2)),r,degrees[i]*2);
        for(int j = 0; j < L.size(); j++){
            ip_vec[i][j] *= L[j];
        }
        for(int j = 0; j < ip_vec.size(); j++){
            if(j != i){
                for(int i = 0; i < ip_vec[j].size(); i++){
                    ip_vec[j][i] *= L[0];
                }
            }
        }
        
        if(rank == 0) vt.end();
        
        for(int j = 0; j < H.size(); j++) K += L[j]*H[j];
        //if(rank == 0) printf("OK %d,%d\n",i,degrees.size());
        //if(i == 2) break;
        final_eval_r = quadratic_coSumcheck_phase2_sparse(R,L,i,2*degrees[i]);
        
        if(rank == 0) vt.start();
        L = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees[i])),r,degrees[i]);
        if(rank == 0) vt.end();
        quadratic_coSumcheck_phase2(p1, p2, L , degrees[i]);
        pt_cp.end();
        
    }
    vector<vector<F>> H = send_final_sumcheck_poly({p1[0],p2[0],p1[1],p2[1],final_eval_r},N);
    if(rank == 0) ps+= ((double)H.size()*H[0].size()*16)/1024.0;
    if(rank == 0)vt.start();

    //printf("%lld,%lld\n",H[0].real,H[0].img);
    if(rank == 0){
        for(int i = 0; i < H[0].size()/2; i++){
            K -= (H[0][i]*H[1][i] + H[2][i]*H[3][i] + c*H[4][i]);
        }
        if(K != 0){
            printf("Error in final sumcheck check 1\n");
        }
    }
    
    F r = hash_to_field(H[0]);
    for(int i = 1; i < H.size(); i++) r = hash_to_field(H[i]);
    
    sumcheck_r.insert(sumcheck_r.begin(),r);
        
    vector<F> L;
    if(N/4 == 1){
        L.push_back(F(1));
    }else{
        L = compute_lagrange_coeff(getRootOfUnity((int)log2(N/4)),r,N/4);
    }
    
    vector<pair<F,vector<F>>> claims;
    for(int i = 0; i < H.size(); i++){
        F eval = 0;
        for(int j = 0; j < H[i].size()/2; j++){
            eval += L[j]*H[i][j];
        }
        claims.push_back(make_pair(eval, sumcheck_r));
    }
    if(rank == 0)vt.end();

    return claims;
    
}



vector<pair<F,vector<F>>> cubic_coSumcheck(vector<F> &p1, vector<F> &p2, vector<F> &p3, vector<F> &r1, vector<F> &r2, vector<vector<F>> R, F K, F K_r, vector<int> degrees, int N, double &ps){
    F c = hash_to_field({K,K_r});
    K = K + c*K_r;
    F final_eval_r;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get my process id
    vector<F> sumcheck_r,b1,b2;
    precompute_beta(r1,b1);
    precompute_beta(r2,b2);
    
    ip_vec.clear();
    ip_vec.resize(degrees.size());
    for(int k = 0; k < degrees.size(); k++){
        ip_vec[k].resize(4*degrees[k],F(1));
    }
    vector<vector<F>> Lagrange_Vectors; 
    
    for(int i = 0; i < degrees.size(); i++){
        vector<F> H = cubic_coSumcheck_phase1(p1,p2,p3,b1,b2,R,c,degrees[i],i,N);
        
        if(rank == 0)ps += ((double)H.size()*16)/1024.0;
        if(rank == 0) vt.start();
        //printf("%d,%d,%d\n",p1.size(),p2.size(),p3.size());
        F eval = F(0);
        for(int j = 0; j < H.size()/4; j++){
            eval += H[4*j];
        }
        //printf("%lld,%lld,%lld,%lld\n",eval.real,K.real,eval.img,K.img);
        if(eval != K){
            if(rank == 0) printf("ERROR %d\n",i);
            //return;    
        }
        K = 0;
        F r = hash_to_field(H);
        sumcheck_r.push_back(r);
        //pt_cp.start();
        
        vector<F> L = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees[i]*4)),r,degrees[i]*4);
        
        for(int j = 0; j < L.size(); j++){
            ip_vec[i][j] *= L[j];
        }
        for(int j = 0; j < ip_vec.size(); j++){
            if(j != i){
                for(int k = 0; k < ip_vec[j].size(); k++){
                    ip_vec[j][k] *= L[0];
                }
            }
        }
        if(rank == 0) vt.end();
        
        if(rank == 0) vt.start();
        for(int j = 0; j < H.size(); j++) K += L[j]*H[j];
        final_eval_r = quadratic_coSumcheck_phase2_sparse(R,L,i,4*degrees[i]);
        
        L = compute_lagrange_coeff(getRootOfUnity((int)log2(degrees[i])),r,degrees[i]);
        if(rank == 0) vt.end();
        Lagrange_Vectors.push_back(L);
        cubic_coSumcheck_phase2(p1, p2, p3, b1, L , degrees[i]);
        //pt_cp.end();
        
    }
    
    vector<vector<F>> H = send_final_sumcheck_poly({p1[0],p2[0], p3[0] ,final_eval_r},N);
    if(rank == 0) ps+= ((double)H.size()*H[0].size()*16)/1024.0;
    if(rank == 0)vt.start();
    //printf("%lld,%lld\n",H[0].real,H[0].img);
    if(rank == 0){
        for(int i = 0; i < H[0].size()/2; i++){
            K -= b2[i]*(b1[0]*(H[0][i]*H[1][i]-H[2][i]) + c*H[3][i]);
        }
        if(K != 0){
            printf("Error in final sumcheck check\n");
        }
    }
    F r = hash_to_field(H[0]);r = hash_to_field(H[1]);r = hash_to_field(H[2]);
    sumcheck_r.insert(sumcheck_r.begin(),r);
    
    vector<F> L;
    if(N/4 == 1){
        L.push_back(F(1));
    }else{
        L = compute_lagrange_coeff(getRootOfUnity((int)log2(N/4)),r,N/4);
    }
    
    vector<pair<F,vector<F>>> claims;
    for(int i = 0; i < H.size(); i++){
        F eval = 0;
        for(int j = 0; j < H[i].size()/2; j++){
            eval += L[j]*H[i][j];
        }
        claims.push_back(make_pair(eval, sumcheck_r));
    }

    int ctr = 0;
    vector<F> _r = r1;//_r.insert(_r.end(),r1.begin(),r1.end());
    F y = F(1);
    // Validate the correctness of b1[0]
    for(int i = 0; i < Lagrange_Vectors.size(); i++){
        vector<F> temp_r,temp_b;
        for(int j = 0; j < (int)log2(Lagrange_Vectors[i].size()); j++) temp_r.push_back(_r[ctr++]);
        precompute_beta(temp_r,temp_b);
        F _y = F(0);

        for(int j = 0; j < temp_b.size(); j++) _y += temp_b[j]*Lagrange_Vectors[i][j];
        y *= _y;
    }
    if(y != b1[0]){
        printf("Error in final sumcheck verification 2\n");
    }
    if(rank == 0)vt.end();
    
    return claims;
}
