/* PREPROCESSOR INCLUSION */
#include <cmath>
#include <ctime>
#include <cstring>
#include <cstdio> 
#include <cstdlib>
#include <complex>

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE_STRICT
//#include "Path/to/Eigen/eigen-3.4.0/Eigen/Dense"
#include "/home/kingofbroccoli/Programmazione/My_C++/Libraries/eigen-3.4.0/Eigen/Dense"

/* PREPROCESSOR DEFINITION */
#define MY_SUCCESS  1
#define MY_FAIL   0
#define MY_TRUE  1
#define MY_FALSE   0
#define MY_TROUBLE   -7
#define MY_MEMORY_FAIL   -8
#define MY_VALUE_ERROR   -9
#define OPEN_FILE_ERROR   -10
#define CHAR_LENGHT     200

/* TYPE DEFINITION */
using namespace std::complex_literals; // For imaginary units, imag and real
using Real = double; // This is equivalent to: typedef double real;
using Complex = std::complex<double>;
using RealVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using RealMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>; // This is equivalent to: typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> RealMatrix;
using ComplexVector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using ComplexMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

/* PROTOTYPES */

// Generate graph from degree sequence
void signed_adjacency_extraction(double* a_ij, double* a_ji, double p1, double p2, double epsilon);
int build_crossing_index_table(int *crossindex, int *deg_seq, int fhs, int N);
bool erdos_gallai_test(int *deg_seq, int N, int *crossindex, int kstar);
void build_dprime(int *rdsp, int *cip, int *ksp, int *fnpc, int *rds, int hs, int fhs, int *crossindex, int kstar);
void update_rds_connection(int *rds, int *crossindex, int *kstar, int *anc, int *fnc, int *fingerprint, int q, int hs);
void update_rdsp_connection(int *rdsp, int *cip, int *ksp, int *fnpc, int srd, int dqm1, int hs);
void update_rdsp_last_connection(int *rdsp, int *cip, int *ksp, int *fnpc, int dqm1, int hs);
void update_rds_newhub(int *rds, int *crossindex, int *kstar, int *anc, int *fnc, int *hs, int fhs);
void update_rdsp_newhub(int *rdsp, int *cip, int *ksp, int *fnpc, int hs, int fhs);
void fail_degree_k(int L, int R, int *fdk, int *k, int ridx, int *rdsp, int *cip, int ciphs, int ksp, int *fnpc);
int find_maximum_fail_degree(int *rdsp, int *cip, int ciphs, int ksp, int *fnpc);
void build_allowed_nodes_counter(int *anc, int *crossindex, int *fnc, int hs);
void build_allowed_nodes_counter_at_beginning(int *anc, int *crossindex, int hs);
void extract_allowed_node(int *eni, int *tnan, int *crossindex, int hs, int *anc, int *fnc, int mfd);
int find_smallest_reduced_degree(int *fnc, int *fnpc);
void graph_from_degree_sequence(RealMatrix& graph, int N, int *rds, int *rdsp, int *fingerprint, int *crossindex, int kstar, void (*weights_generator)(double*, double*, double, double, double), double p1, double p2, double epsilon);
// Erdos-Renyi
void erdos_renyi(RealMatrix& graph, int N, double p_c, void (*weights_generator)(double*, double*, double, double, double), double p1, double p2, double epsilon);
// Eigenvalues and Eigenvectors
void get_eigenvalues_eigenvectors_and_IPR(ComplexVector& eigvals, ComplexMatrix &eigvects, RealVector& IPR, const RealMatrix &M);
ComplexVector get_eigenvalues(const RealMatrix &M);
ComplexMatrix get_eigenvectors(const RealMatrix &M);
RealVector get_IPR(const ComplexMatrix &eigvects);
double get_single_IPR(const ComplexVector &eigvect);
// Random Number Generator
double RNG(); // RNG [0, 1]
double RNG_0(); // RNG (0, 1]
double RNG_1(); // RNG [0, 1)
int RandIntegers(int nmin, int nmax);
void Init_PowerLaw(double *PLTable, double gamma, int k_min, int tsize);
void powerlaw_even_degree_seq(int *k, double *PLTable, int N, int k_min, int tsize);
void BubbleSort(int *x, int n, char order='D');
// Basic usage
char* my_char_malloc(int size);
double* my_double_malloc(int size);
int* my_int_malloc(int size);
int* my_int_calloc(int size);
int* my_int_realloc(int *pint, int new_size);
char* my_char_malloc(int size);
int* my_int_malloc(int size);
double* my_double_malloc(int size);
FILE* my_open_writing_binary_file(const char* file_name);
int my_max(int a, int b);
void swap(int *x_ptr, int *y_ptr);
void fill_array(int *x, int m, int N);
int sum_array(int *n, int N);
int linear_search_interval(unsigned int p, unsigned int *A, int N);
int linear_search_interval(double p, double *A, int N);

/* MAIN BEGINNING */

int main(int argc, char **argv){
    clock_t begin = clock(); // Starting Clock
    float time_spent;
    int N = 1000;
    char gamma_label[] = "4.0";
    int k_min = 2;
    char epsilon_label[] = "0.90";
    int N_ext = 1;
    int where_to_start = 0;
    double gamma = atof(gamma_label);
    double epsilon = atof(epsilon_label);
    int k_max = 2 * ceil(k_min * pow(N, (1.0/(gamma-1.0)))); // Natural cutoff
    int tsize;
    double *PLTable;
    int *rds, *rdsp, *fingerprint, *crossindex;
    int kstar, fhs;
    int i, ext;
    char graph_label[] = "PowerLaw";
    char weights_label[] = "Signed_Adjacency";
    double od_p1 = 1.0;
    double od_p2 = 1.0;
    void (*graph_generator)(RealMatrix& graph, int, int*, int*, int*, int*, int, void (*)(double*, double*, double, double, double), double, double, double) = &graph_from_degree_sequence;
    void (*weights_generator)(double*, double*, double, double, double) = &signed_adjacency_extraction;
    bool graphical;
    RealMatrix graph(N, N);
    ComplexVector eigvals;
    ComplexMatrix eigvects;
    RealVector IPR(N);
    char *name_buffer, *directory;
    FILE *fp;

    // Preliminaries
    name_buffer = my_char_malloc(CHAR_LENGHT);
    directory = my_char_malloc(CHAR_LENGHT);
    snprintf(directory, CHAR_LENGHT, "Spectra/%s_gamma%s_%s_%.1f_%.1f_epsilon%s_N%d_Cpp", graph_label, gamma_label, weights_label, od_p1, od_p2, epsilon_label, N);
    snprintf(name_buffer, CHAR_LENGHT, "mkdir Spectra %s", directory);
    system(name_buffer);
    // Allocate Memory
    rds = my_int_malloc(N);
    rdsp = my_int_calloc(N);
    fingerprint = my_int_malloc(N);
    tsize = k_max - k_min + 1;
    PLTable = my_double_malloc(tsize);
    Init_PowerLaw(PLTable, gamma, k_min, tsize);
    crossindex = NULL; // It is mandatory to have a NULL or a valid pointer to use realloc, not an uninitialized one

    // Let's start!
    for(ext=1; ext<=N_ext; ext++){
        srand48(time(0)); // The random seed is given
        // Generate the sequence
        graphical = MY_FALSE;
        while(not graphical){
            //fill_array(rds, 0, N);
            powerlaw_even_degree_seq(rds, PLTable, N, k_min, tsize);
            // It is important to remove zeros and use a different N (but save with original N)
            BubbleSort(rds, N); // Sort the degrees
            fhs = rds[0]; // First Hub Size (global variable)
            if(N > fhs){
                crossindex = my_int_realloc(crossindex, fhs+1);
                kstar = build_crossing_index_table(crossindex, rds, fhs, N);
                graphical = erdos_gallai_test(rds, N, crossindex, kstar);
            }
        }   // We get a graphical degree sequence
        // Generate and diagonalise graph
        //graph_generator(graph, rds, rdsp, fingerprint, N, p_c, weights_generator, od_p1, od_p2, epsilon);
        graph_generator(graph, N, rds, rdsp, fingerprint, crossindex, kstar, weights_generator, od_p1, od_p2, epsilon);
        get_eigenvalues_eigenvectors_and_IPR(eigvals, eigvects, IPR, graph);
        // Saving
        snprintf(name_buffer, CHAR_LENGHT, "%s/Extraction_%d.bin", directory, where_to_start+ext);
        fp = my_open_writing_binary_file(name_buffer);
        for(i=0; i<eigvals.size(); i++){
            fwrite(&eigvals(i), sizeof(Complex), 1, fp);
            fwrite(&IPR(i), sizeof(Real), 1, fp);
        }
        fclose(fp);
    }
    //delete &M; // Directly calling the destructor is not needed since it is an auto variable, the destructor is called automatically when the variable goes out of scope
    // Free allocated memory
    free(rds);
    free(rdsp);
    free(fingerprint);
    free(crossindex);
    free(name_buffer);
    free(directory);
    free(PLTable);
    // Computation of time spent
    clock_t end = clock(); //Clock conclusivo
    time_spent = ((float)(end - begin)) / CLOCKS_PER_SEC; //Calcolo tempo di esecuzione
    printf("\n # The program took %f seconds.\n \n", time_spent);

    return MY_SUCCESS;
}

/* MAIN - THE END *

/* FUNCTIONS */

// +/-1 weights extraction where the signs are symmetric with probability epsilon
void signed_adjacency_extraction(double* a_ij, double* a_ji, double p1, double p2, double epsilon){
    *a_ij = +1;
    *a_ji = +1;
    if(RNG() < epsilon){ // Sign-Symmetric
        if(RNG() < 0.5){ // Competitive // if(RNG() > 0.5) --> Mutualistic and there is nothing to do
            *a_ij *= -1;
            *a_ji *= -1;
        }
    }
    else{ // Sign-Antisymmetric
        if(RNG() < 0.5){
            *a_ji *= -1; // Predator-Prey
        }
        else{
            *a_ij *= -1; // Prey-Predator
        }
    }
    return;
}

int build_crossing_index_table(int *crossindex, int *deg_seq, int fhs, int N){
    int i, k, kp1, kstar;
    bool kstar_notfound = MY_TRUE;
    if(fhs > N){
        printf("Not possible to build crossindex table of non-graphical sequence");
        exit(MY_TROUBLE);
    }
    crossindex[0] = N;
    kstar = fhs;
    k = 1;
    //while(k < fhs){
    while(kstar_notfound){
        kp1 = k+1;
        i = crossindex[k-1]-1;
        while(kp1 > deg_seq[i]){
            i = i-1;
        }
        crossindex[k] = i+1;
        if(crossindex[k] < kp1){
            kstar = k;
            kstar_notfound = MY_FALSE; // kstar should always exist
            //k = fhs; // for clarity I prefer to use a bool instead of modifying k
        }
        k = k+1;
    }
    // After finding kstar we keep building the crossing index table
    for(k=kstar+1; k<fhs; k++){
        kp1 = k+1;
        i = crossindex[k-1]-1;
        while(kp1 > deg_seq[i]){
            i = i-1;
        }
        crossindex[k] = i+1;
    }
    crossindex[fhs] = 0; // In principle we have already used calloc but better be safe
    return kstar;
}

bool erdos_gallai_test(int *deg_seq, int N, int *crossindex, int kstar){
    int k;
    double L, R;
    bool graphical = MY_TRUE;
    if((sum_array(deg_seq, N) % 2) == 1){ // The degrees sum is odd
        graphical = MY_FALSE;
    }
    L = deg_seq[0];
    R = N-1;
    if(R < L){
        graphical = MY_FALSE;
    }
    k = 1; //  The first step k=0 is in the initialisation of R and L
    while((k < kstar) && (graphical)){ // graphical==MY_TRUE is redundant if MY_TRUE is 1 (True)
        R += crossindex[k] - 1;
        L += deg_seq[k];
        if(R < L){
            graphical = MY_FALSE;
        }
        k++;
    }
    return graphical;
}

void build_dprime(int *rdsp, int *cip, int *ksp, int *fnpc, int *rds, int hs, int fhs, int *crossindex, int kstar){
    //Notice that cip[hs] always point at the beginning of the sequence, even if there is no node with degree hs left. In principle cip[hs] is never going to be required as crossindex.
    int kbar = rds[hs-1]; // One-to-Smallest degree in the Leftmost Adj Set 
    int nkbnc = crossindex[kbar-1] - crossindex[hs] - hs; // Nodes with degree kbar not connected to the hub = (crossindex[kbar-1] - 1) - (hubsize-1) = #(number of nodes within kbar except the hub) - #(new connections
    int i, start_new_kbar_pals, ksp1;
    fill_array(fnpc, 0, fhs+1);
    fnpc[1] = 1; // The hub itself
    // Fill reduced degree sequence prime
    if(kbar != hs)
        start_new_kbar_pals = crossindex[kbar]-1; // We have to shift rdsp to the left by 1 but we cannot start by -1 (in any case the first and the last step start from 0 and goes to crossindex[0]=N)
    else
        start_new_kbar_pals = crossindex[hs];
    for(i=crossindex[hs]; i<start_new_kbar_pals; i++){
        rdsp[i] = rds[i+1] - 1;
        fnpc[rdsp[i]] += 1;
    }
    for(i=start_new_kbar_pals; i<start_new_kbar_pals+nkbnc; i++){
        rdsp[i] = kbar;
    }
    for(i=start_new_kbar_pals+nkbnc; i<crossindex[kbar-1]-1; i++){
        rdsp[i] = kbar - 1;
    }
    fnpc[kbar-1] += crossindex[hs] + hs - 1 - start_new_kbar_pals; // crossindex[kbar-1]-1 - (start_new_kbar_pals+nkbnc) 
    for(i=crossindex[kbar-1]-1; i<crossindex[1]-1; i++){
        rdsp[i] = rds[i+1];
    }
    if(kbar > 1)
        for(i=crossindex[1]-1; i<crossindex[0]; i++)
            rdsp[i] = 1;
    else{
        //cip[0] = crossindex[0] - fnpc[0];
        rdsp[cip[0]-1] = 1;
        rdsp[crossindex[0]-1] = 0;
    }
    for(i=1; i<kbar-1; i++)
        cip[i] = crossindex[i] - 1;
    for(i=kbar; i<hs-1; i++)
        cip[i] = crossindex[i+1] - 1; // # This is true also if there are no kbar+1 nodes and all the kbar are connected to the hub (there will start kbar-1 and before there will be kbar+1)
    for(i=hs-1; i<hs+1; i++) //
        cip[i] = crossindex[hs];
    cip[kbar-1] = start_new_kbar_pals + nkbnc; // crossindex[kbar] + nkbnc - 1
    cip[0] = crossindex[0] - fnpc[0];
    for(i=kbar+1; i<hs+1; i++){
        if(crossindex[i]==crossindex[hs])
            cip[i-1] = cip[hs];
    }
    // Let's find kstar prime (ksp)
    *ksp = kstar;
    ksp1 = *ksp-1;
    while(cip[ksp1]-cip[hs]<=ksp1){
        *ksp = ksp1;
        ksp1 -= 1;
    }
    return;
}

void update_rds_connection(int *rds, int *crossindex, int *kstar, int *anc, int *fnc, int *fingerprint, int q, int hs){
    int dq = rds[q];
    int dqm1 = dq-1;
    int ksm1, cdqm1;
    // Reduce hub degree
    rds[crossindex[hs]] -= 1; // rds[crossindex[hs]] is the hub
    // Reduce chosen node degree
    crossindex[dqm1] -= 1; // We shift crossindex of degree dq-1
    cdqm1 = crossindex[dqm1];
    rds[cdqm1]--; // We reduce the last dq making it the first dq-1
    anc[dq]--;
    fnc[dqm1]++;
    swap((fingerprint+q), (fingerprint+cdqm1)); // We swap the fingerprints
    // Let's update kstar
    ksm1 = *kstar - 1;
    while(crossindex[ksm1]-crossindex[hs]<=ksm1){
        *kstar = ksm1;
        ksm1--;
    }
    return;
}

void update_rdsp_connection(int *rdsp, int *cip, int *ksp, int *fnpc, int srd, int dqm1, int hs){
    if(dqm1 < srd+1){ // q is not within the hs-1 connected nodes, excluding forbidden nodes
        // Remove last prime-connection
        rdsp[cip[srd]]++;
        cip[srd]++;
        fnpc[srd]--;
        // Create new connection
        cip[dqm1]--;
        rdsp[cip[dqm1]]--;
        fnpc[dqm1]++;
        // Let's update kstar prime (ksp): in this case cip is both incremented and reduced so we need to check both directions
        int ksp1 = *ksp - 1; // I don't like the aesthetic but it is pointless to define the variable outside the if statement
        while(cip[ksp1]-cip[hs]<=ksp1){ // This implies cip[ksp-1]-cip[hs] > ksp-1
            *ksp = ksp1;
            ksp1--;
        }
        while(cip[*ksp]-cip[hs]>*ksp){ // This implies cip[ksp]-cip[hs] <= ksp ---> This together with the previous one means that ksp is the first one
            *ksp = *ksp + 1;
        }
    }
    return;
}

void update_rdsp_last_connection(int *rdsp, int *cip, int *ksp, int *fnpc, int dqm1, int hs){
    int ksp1;
    // Create new connection
    cip[dqm1]--;
    rdsp[cip[dqm1]]--;
    fnpc[dqm1]++;
    // Let's update kstar prime (ksp)
    ksp1 = *ksp-1;
    while(cip[ksp1]-cip[hs]<=ksp1){ // This implies cip[ksp-1]-cip[hs] > ksp-1
        *ksp = ksp1;
        ksp1--;
    }
    return;
}

void update_rds_newhub(int *rds, int *crossindex, int *kstar, int *anc, int *fnc, int *hs, int fhs){
    crossindex[*hs]++; // We shift crossindex of 1 to leave the exausted hub 
    *hs = rds[crossindex[*hs]]; // New hub size
    if(*hs > 0){
        // Let's update kstar
        int i, ksm1 = *kstar-1; // I don't like the aesthetic but it is pointless to define the variable outside the if statement
        while(crossindex[ksm1]-crossindex[*hs]<=ksm1){ // If we do not need kstar for rds after the EG test the update can be commented
            *kstar = ksm1;
            ksm1--;
        }
        for(i=1; i<*hs+1; i++){
            anc[i] = anc[i] + fnc[i];
        }
        fill_array(fnc, 0, fhs+1);
        anc[*hs]--;
    }
    return;
}

void update_rdsp_newhub(int *rdsp, int *cip, int *ksp, int *fnpc, int hs, int fhs){
    int i, kbar, nkbc, nkbnc, ksp1; // nkb
    // It is important to update kbar before removing the hub
    kbar = rdsp[cip[hs]+hs-1]; // One-to-Smallest degree in the Leftmost Adj Set
    rdsp[cip[hs]] = 0;
    cip[hs]++; // We shift cip of 1 to leave the exausted hub
    fill_array(fnpc, 0, fhs+1);
    fnpc[1] = 1; // The hub itself
    // Let's compute nkb
    nkbc = hs - 1 + cip[hs] - cip[kbar]; // Nodes with degree kbar connected to the hub = (hub_size-1) - (cip[kbar]-cip[hs]) = #(new connections) - #(nodes before kbar) # Notice that the hub is already at the end of the sequence
    // We should call them nodes kb that has remained kbar
    nkbnc = cip[kbar-1] - cip[hs] - hs + 1; // Nodes with degree kbar not connected to the hub = (cip[kbar-1]-cip[hs]) - (hubsize-1) = #(number of nodes within kbar except the hub) - #(new connections) # Notice that the hub is already at the end of the sequence
    for(i=cip[hs]; i<cip[kbar]; i++){
        rdsp[i]--;
        fnpc[rdsp[i]]++;
    }
    for(i=cip[kbar]+nkbnc; i<cip[kbar-1]; i++){ // It would be equivalent from i=cip[kbar-1]-nkbc to i=cip[kbar-1])
        rdsp[i] = kbar - 1;
    }
    // Update crossindex prime
    cip[kbar-1] = cip[kbar] + nkbnc; // It is important to use cip[kbar] before the update
    fnpc[kbar-1] += nkbc;
    for(i=kbar; i<hs; i++){
        cip[i] = cip[i+1];
    }
    for(i=kbar+1; i<hs; i++){
        cip[i] = cip[i-1] - fnpc[i];
    }
    // Let's update kstar prime (ksp)
    ksp1 = *ksp-1;
    while(cip[ksp1]-cip[hs]<=ksp1){
        *ksp = ksp1;
        ksp1--;
    }
    return;
}

void fail_degree_k(int L, int R, int *fdk, int *k, int ridx, int *rdsp, int *cip, int ciphs, int ksp, int *fnpc){ // In principle it would be enough to give R-L to the function 
    if(L < R-1) // Case 3
        *fdk = 0;
    else if(L == R-1){ // Case 2
        *fdk = rdsp[my_max(ciphs+*k+1, cip[*k+1])]; // First node whose index is greater than k and whose degree is smaller than k+2
        while((cip[*fdk-1] - cip[*fdk] - fnpc[*fdk]) == 0) // Check that there are any non-forbidden nodes with degree fdk
            *fdk = *fdk - 1;
    }
    else{ // Case 1 (it would be L==R and since L<=R that's the only case left)
        *fdk = rdsp[ciphs+*k+1]; // First node whose index is greater than k
        while((cip[*fdk-1] - cip[*fdk] - fnpc[*fdk]) == 0) // Check that there are any non-forbidden nodes with degree fdk
            *fdk = *fdk - 1;
        *k = ridx; // After Case (1) we don't have to check further values of k
    }
    return;
}

int find_maximum_fail_degree(int *rdsp, int *cip, int ciphs, int ksp, int *fnpc){
    int mfd, fdk, L, R, ridx, k;
    if(rdsp[ciphs+ksp] == ksp) // ridx = r index, to distinguish from R
        ridx = ksp+1;
    else
        ridx = ksp;
    L = rdsp[ciphs];
    R = cip[0]-ciphs - 1;
    k = 0;
    fail_degree_k(L, R, &mfd, &k, ridx, rdsp, cip, ciphs, ksp, fnpc); // No need to find max at this stage
    k = 1; // The first step k=0 is in the initialisation
    while(k < ksp){
        L += rdsp[ciphs+k];
        R += cip[k]-ciphs - 1;
        fail_degree_k(L, R, &fdk, &k, ridx, rdsp, cip, ciphs, ksp, fnpc);
        mfd = my_max(mfd, fdk);
        k++;
    }
    // After the previous loop either k=ksp if no (1) and therefore we need to keep checking (one or two times depending on ridx) or k=ridx+1 if (1) and therefore we are done
    while(k < ridx+1){ 
        L += rdsp[ciphs+k];
        R += 2*k - rdsp[ciphs+k];
        fail_degree_k(L, R, &fdk, &k, ridx, rdsp, cip, ciphs, ksp, fnpc);
        mfd = my_max(mfd, fdk);
        k++;
    }
    return mfd;
}

void build_allowed_nodes_counter(int *anc, int *crossindex, int *fnc, int hs){
    int i;
    // In principle we could use cumulative instead of counter: that would make the extraction easier (and we could use binary search) but the update would involve all the degrees after the extracted one
    for(i=1; i<hs; i++)
        anc[i] = crossindex[i-1] - crossindex[i] - fnc[i];
    anc[hs] = crossindex[hs-1] - crossindex[hs] - 1; // The -1 is the hub while we don't have - fnc[hs] since by construction we don't have any forbidden node of degree hs (a part from the hub)
    return;
}

void build_allowed_nodes_counter_at_beginning(int *anc, int *crossindex, int hs){
    int i;
    // In principle we could use cumulative instead of counter: that would make the extraction easier (and we could use binary search) but the update would involve all the degrees after the extracted one
    for(i=1; i<hs; i++)
        anc[i] = crossindex[i-1] - crossindex[i];
    anc[hs] = crossindex[hs-1] - crossindex[hs] - 1; // The -1 is the hub
    return;
}

void extract_allowed_node(int *eni, int *tnan, int *crossindex, int hs, int *anc, int *fnc, int mfd){
    int rfn, si, ps, q, i;
    si = my_max(mfd+1,1); // Starting index - &fnc[si] is equivalent to fnc+si
    rfn = sum_array(&fnc[si], hs+1-si); // We don't want to substract the nodes with zero degree, that's why the max # It would be enough to sum up to hs-1 since there are no forbidden nodes of degree hs, by construction
    *tnan = crossindex[mfd] - crossindex[hs] -  1 - rfn; // Total number of allowed nodes (the -1 is from the hub) # This is the size of the allowed set needed for the sample weights
    q = RandIntegers(0, *tnan-1); // We can also use q = rng.integers(1:tnan+1), while (ps < q) and eni = crossindex[hs] + q + fnc[i:hs+1].sum() (without the +1 from the hub)
    // Start from large degree
    ps = anc[hs];
    i = hs;
    while(ps <= q){ // ps < q+1
        i--;
        ps += anc[i]; // Extracted degree is i
    }
    *eni = crossindex[hs] + 1 + q + sum_array(&fnc[i], hs+1-i); // Extracted Node Index: the +1 is the hub // It would be enough to sum up to hs-1 since there are no forbidden nodes of degree hs, by construction
    // It is probably faster to start from low degree (they are more numerous) and then compute crossindex[0]-q but the following code is not correct
    return;
}

int find_smallest_reduced_degree(int *fnc, int *fnpc){
    int srd;
    if(fnc[0] != fnpc[0])
        srd = 0;
    else if(fnc[1] != (fnpc[1]-1)) // -1 because of the hub
        srd = 1;
    else{
        srd = 2;
        while(fnc[srd]==fnpc[srd]) // If we use a for we can stop when we don't want to explode
            srd += 1;
    }
    return srd;
}

//void graph_from_degree_sequence(RealMatrix& graph, int N, int *rds, int *rdsp, int *fingerprint, void (*weights_generator)(double*, double*, double, double, double), double p1, double p2, double epsilon){
void graph_from_degree_sequence(RealMatrix& graph, int N, int *rds, int *rdsp, int *fingerprint, int *crossindex, int kstar, void (*weights_generator)(double*, double*, double, double, double), double p1, double p2, double epsilon){
    int *cip, *anc, *fnc, *fnpc;
    int fhs, hs, ksp, q, kij, dqm1, srd, hs_updated, ps, sdlas, mfd, i;
    double a_ij, a_ji; 

    //double logweight;
    fhs = rds[0];
    // Clean and reset everything
    graph.setZero();
    fill_array(rdsp, 0, N);
    for(i=0; i<N; i++){ // In principle we can also keep them disordered but I prefer to reset them each time
        fingerprint[i] = i; // Fill in indexes
    }
    //logweight = 0;
    cip = my_int_calloc(fhs+1);
    anc = my_int_calloc(fhs+1); // # Allowed Nodes Counter: anc[i] = allowed nodes of degree i
    fnc = my_int_calloc(fhs+1); // Forbidden nodes counter (it includes forbidden zeros so that fnc[n] = number forbidden nodes of degree n)
    fnpc = my_int_calloc(fhs+1);
    // Let's start
    hs = fhs;
    build_allowed_nodes_counter_at_beginning(anc, crossindex, fhs);
    build_dprime(rdsp, cip, &ksp, fnpc, rds, hs, fhs, crossindex, kstar);
    while(hs>1){
        kij = crossindex[0] - crossindex[hs] - 1; // The first connection of an hub is always possible (except with itself, thus -1)
        q = crossindex[hs] + 1 + RandIntegers(0, kij-1); // The +1 is the hub
        //weight = weight / hs * kij;
        //logweight += log(kij) - log(hs);
        //weights_generator(&graph(fingerprint[crossindex[hs]], fingerprint[q]), &graph(fingerprint[q], fingerprint[crossindex[hs]]), p1, p2, epsilon);
        weights_generator(&a_ij, &a_ji, p1, p2, epsilon);
        graph(fingerprint[crossindex[hs]], fingerprint[q]) = a_ij;
        graph(fingerprint[q], fingerprint[crossindex[hs]]) = a_ji;
        dqm1 = rds[q]-1;
        srd = find_smallest_reduced_degree(fnc, fnpc); // Find Smallest Reduced Degree
        update_rds_connection(rds, crossindex, &kstar, anc, fnc, fingerprint, q, hs);
        update_rdsp_connection(rdsp, cip, &ksp, fnpc, srd, dqm1, hs);
        for(i=1; i<hs; i++){
            hs_updated = hs-i; // Updated Hub Size
            ps = anc[hs];
            sdlas = hs;
            while(ps < hs_updated){
                sdlas--;
                ps += anc[sdlas]; // Extracted degree is sdlas (Smallest degree in the Leftmost Adj Set)
            }
            //sdlas = rds[crossindex[hs] + hs_updated + fnc[sdlas:hs+1].sum()] # Smallest degree in the Leftmost Adj Set 
            mfd = 0;
            if(sdlas != rds[crossindex[0]-1])  // if sdlas==rds[-1] then mfd=0 
                mfd = find_maximum_fail_degree(rdsp, cip, cip[hs], ksp, fnpc);
            extract_allowed_node(&q, &kij, crossindex, hs, anc, fnc, mfd);
            dqm1 = rds[q]-1;
            //weight = weight / hs_updated * kij;
            //logweight += log(kij) - log(hs);
            //weights_generator(&graph(fingerprint[crossindex[hs]], fingerprint[q]), &graph(fingerprint[q], fingerprint[crossindex[hs]]), p1, p2, epsilon);
            weights_generator(&a_ij, &a_ji, p1, p2, epsilon);
            graph(fingerprint[crossindex[hs]], fingerprint[q]) = a_ij;
            graph(fingerprint[q], fingerprint[crossindex[hs]]) = a_ji;
            if(i<(hs-1)){ // "Bulk" of connections
                srd = find_smallest_reduced_degree(fnc, fnpc); // Find new Smallest Reduced Degree
                update_rdsp_connection(rdsp, cip, &ksp, fnpc, srd, dqm1, hs);
            }
            else // Last connection
                update_rdsp_last_connection(rdsp, cip, &ksp, fnpc, dqm1, hs);
            // We update rds after rdsp in order to easily compute srd
            update_rds_connection(rds, crossindex, &kstar, anc, fnc, fingerprint, q, hs);
        }
        // New Hub: we update hs
        update_rds_newhub(rds, crossindex, &kstar, anc, fnc, &hs, fhs);
        //kbar = rds[crossindex[hs]+rds[crossindex[hs]]-1]; // Here we need kbar_new
        if(hs > 1)
            update_rdsp_newhub(rdsp, cip, &ksp, fnpc, hs, fhs);
    }
    // When there are only ones left there is no need to compute prime quantities
    while(hs != 0){
        kij = crossindex[0] - crossindex[1] - 1; // The first connection of an hub is always possible (except with itself, thus -1)
        q = crossindex[hs] + 1 + RandIntegers(0, kij-1); // The +1 is the hub
        //weight = weight * kij;
        //logweight += log(kij);
        //weights_generator(&graph(fingerprint[crossindex[hs]], fingerprint[q]), &graph(fingerprint[q], fingerprint[crossindex[hs]]), p1, p2, epsilon);
        weights_generator(&a_ij, &a_ji, p1, p2, epsilon);
        graph(fingerprint[crossindex[hs]], fingerprint[q]) = a_ij;
        graph(fingerprint[q], fingerprint[crossindex[hs]]) = a_ji;
        update_rds_connection(rds, crossindex, &kstar, anc, fnc, fingerprint, q, hs);
        // We update hs
        update_rds_newhub(rds, crossindex, &kstar, anc, fnc, &hs, fhs);
    }
    return;
}

// Erdos-Renyi Graph extraction
void erdos_renyi(RealMatrix& graph, int N, double p_c, void (*weights_generator)(double*, double*, double, double, double), double p1, double p2, double epsilon){
    int i, j;
    double a_ij, a_ji;
    graph.setZero();
    for(i=0; i<graph.rows(); i++){
        for(j=i+1; j<graph.cols(); j++){
            if(RNG() < p_c){
                weights_generator(&a_ij, &a_ji, p1, p2, epsilon);
                graph(i, j) = a_ij;
                graph(j, i) = a_ji;
            }
        }
    }
    return;
}

void get_eigenvalues_eigenvectors_and_IPR(ComplexVector& eigvals, ComplexMatrix &eigvects, RealVector& IPR, const RealMatrix &M){
    double tmp;
    int i, j;
    Eigen::EigenSolver<RealMatrix> eigenS(M);
    eigvals = eigenS.eigenvalues();
    eigvects = eigenS.eigenvectors();
    IPR.setZero();
    for(j=0; j<eigvects.cols(); j++){
        for(i=0; i<eigvects.rows(); i++){
            tmp = std::norm(eigvects(i, j));
            IPR(j) += tmp*tmp;
        }
    }
    return;
}

ComplexVector get_eigenvalues(const RealMatrix &M){
    Eigen::EigenSolver<RealMatrix> eigenS(M);
    return eigenS.eigenvalues();
}

ComplexMatrix get_eigenvectors(const RealMatrix &M){
    Eigen::EigenSolver<RealMatrix> eigenS(M);
    return eigenS.eigenvectors();
}

RealVector get_IPR(const ComplexMatrix &eigvects){
    static double tmp; // den
    RealVector IPR = RealVector::Zero(eigvects.cols());
    int i, j;
    for(j=0; j<eigvects.cols(); j++){
        for(i=0; i<eigvects.rows(); i++){
            tmp = std::norm(eigvects(i, j));
            IPR(j) += tmp*tmp;
        }
    }
    return IPR;
}

double get_single_IPR(const ComplexVector &eigvect){
    static double IPR = 0, tmp; // den = 0
    int i;
    for(i=0; i<eigvect.rows(); i++){
        tmp = std::norm(eigvect(i));
        IPR += tmp*tmp;
    }
    return IPR;
}

void BubbleSort(int *x, int n, char order){
    int i, j;
    if(order=='D'){
    for(i = 0; i < n-1; i++)    
        // Last i elements are already in place
        for(j = 0; j < n-i-1; j++)
            if(x[j] < x[j+1])
                swap(&x[j], &x[j+1]);
    }
    else if(order=='A'){
    for(i = 0; i < n-1; i++)    
        // Last i elements are already in place
        for(j = 0; j < n-i-1; j++)
            if(x[j] > x[j+1])
                swap(&x[j], &x[j+1]);
    }
    else{
        fprintf(stderr, "\n ERROR: UNKNOWN SORTING ORDER \n");
        exit(MY_VALUE_ERROR);
    }
    return;
}

char* my_char_malloc(int size){
    char *buffer;
    buffer = (char*) malloc(size * sizeof(char));
    if (buffer==NULL) {
        printf ("\nERROR: Failed malloc of file buffer.\n");
        exit (MY_MEMORY_FAIL);
    }
    return buffer;
}

double* my_double_malloc(int size){
    double *dp;
    dp = (double*) malloc(size * sizeof(double));
    if (dp == NULL){
        printf("\n ERROR: DOUBLE MALLOC HAS FAILED \n");
        exit(MY_MEMORY_FAIL);
    }
    return dp;
}

int* my_int_malloc(int size){
    int *pint;
    pint = (int*) malloc(size * sizeof(int));
    if (pint==NULL) {
        printf ("\nERROR: Failed allocayion of int array.\n");
        exit (MY_MEMORY_FAIL);
    }
    return pint;
}

int* my_int_calloc(int size){
    int *pint;
    pint = (int*) calloc(size, sizeof(int));
    if (pint==NULL) {
        printf ("\nERROR: Failed calloc of int array.\n");
        exit (MY_MEMORY_FAIL);
    }
    return pint;
}

int* my_int_realloc(int *pint, int new_size){
    int *pint_appoggio;
    pint_appoggio = (int*) realloc(pint, new_size * sizeof(int));
    if (pint_appoggio == NULL){ 
        printf("\n Error: INT REALLOC HAS FAILED \n");
        free(pint);
        exit(MY_MEMORY_FAIL);
    }
    return pint_appoggio;
}

double RNG(){
    double x;
    x = ( (double) lrand48() ) / ((double) RAND_MAX);
    return x;
}

double RNG_0(){
    double x;
    x = ( (double) lrand48() + 1.) / ((double) RAND_MAX + 1. );
    return x;
}

double RNG_1(){
    double x;
    x = ( (double) lrand48() ) / ((double) RAND_MAX + 1. );
    return x;
}

int RandIntegers(int nmin, int nmax){ // Inspired by https://www.cs.yale.edu/homes/aspnes/pinewiki/C(2f)Randomization.html
    long int n;
    int range = nmax - nmin + 1;
    long int rejection_lim = RAND_MAX - (RAND_MAX % range);
    while((n = lrand48()) >= rejection_lim);
    n = nmin + (n % range); 
    return (int) n;
}

void Init_PowerLaw(double *PLTable, double gamma, int k_min, int tsize){
    int i;
    double s = 0;
    // Create the table
    for(i=0; i<tsize; i++){
        PLTable[i] = pow((k_min + i), -gamma);
        s += PLTable[i];
    }
    // Normalise the table
    for(i=0; i<tsize; i++)
        PLTable[i] /= s;
    // Create the cumulative
    for(i=1; i<tsize; i++)
        PLTable[i] += PLTable[i-1];
    return;
}

void powerlaw_even_degree_seq(int *k, double *PLTable, int N, int k_min, int tsize){
    int ksum, i;
    double xi;
    bool odd = MY_TRUE;
    while(odd){
        ksum = 0; // Sum of all the degrees
        for(i=0; i<N; i++){
            xi = RNG_1(); // Uniform in [0, 1)
            k[i] = k_min + linear_search_interval(xi, PLTable, tsize);
            ksum  += k[i];
        }
        if((ksum%2) == 0) // This is 0 if the degree sequence is even, or 1 otherwise
            odd = MY_FALSE;
    }
    return;
}

FILE* my_open_writing_binary_file(const char* file_name){
    FILE* file_pointer;
    file_pointer = fopen(file_name, "wb");
    if (file_pointer==NULL) {
        printf(" Errore nell'apertura di %s \n", file_name);
        exit(OPEN_FILE_ERROR);
    }
    return file_pointer;
}

int my_max(int a, int b){
    int max = a;
    if(b>a){
        max = b;
    }
    return max;
}

void swap(int *x_ptr, int *y_ptr){
    int tmp = *x_ptr;
    *x_ptr = *y_ptr;
    *y_ptr = tmp;
    return;
}

void fill_array(int *x, int m, int N){
    int i;
    for(i=0; i<N; i++){
        x[i] = m;
    }
    return;
}

int sum_array(int *n, int N){
    int i;
    int sum=0;
    for(i=0; i<N;i++){
        sum += n[i];
    }
    return sum;
}

int linear_search_interval(unsigned int p, unsigned int *A, int N){
    int i=0;
    while (p > A[i])
        i++;
    if(i>N){
        fprintf(stderr,"\n ERROR: VALUE OUTSIDE ARRAY \n");
        exit(MY_VALUE_ERROR);
    }
    return i;
}

int linear_search_interval(double p, double *A, int N){
    int i=0;
    while (p > A[i])
        i++;
    if(i>N){
        fprintf(stderr,"\n ERROR: VALUE OUTSIDE ARRAY \n");
        exit(MY_VALUE_ERROR);
    }
    return i;
}