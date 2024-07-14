#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include <cmath>
#include <cassert>
#include <tapa.h>
#include <unordered_map>
#include <queue>
#include <gflags/gflags.h>
#include <chrono>

#include "mmio.h"
#include "lu.h"
#include "lu_common.h"

using float_v16 = tapa::vec_t<float, 16>;
using int_v16 = tapa::vec_t<int, 16>;
using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::vector;
using std::min;
using std::max;

constexpr int NUM_CH = 16;
constexpr int WINDOW_SIZE = 8192;
constexpr int WINDOW_SIZE_div_2 = 8192; //??????

int main(int argc, char **argv){
    cout <<"Run LU ... "<< endl;
    
    int iteration_num = 1;

     if(argc == 4) {
        iteration_num = atoi(argv[3]);
    }
    
    else if(argc != 3) {
        cout << "Message: " << argv[0] << " [Sparse Matrix Path] [N] [iteration_num] " << std::endl;
        return EXIT_FAILURE;
    }

    char *filename = argv[1];

    int N = tapa::round_up<8>(atoi(argv[2]));

    std::string bitstream;
    if(const auto bitstream_ptr = getenv("BITFILE")) {
        bitstream = bitstream_ptr;
    }

    cout << "\nConfiguration : \n";
    cout << "iteration_num = " << iteration_num <<  "\n";
    cout << "N = " << N <<  "\n";
    cout << "HBM_CHANNEL_A_NUM = " << HBM_CHANNEL_A_NUM << endl;

    Read_matrix_size(filename, &M, &k, &nnzR, &isSymmetric); //no definition

    cout << "\n Matrix Size: \n";
    cout << "Sparse matrix A: #Rows = " << M << ", #Cols = " << K << ", #nnzR = " << nnzR << "\n";
    cout << "\n Create Data Struct: \n";

    vector<INDEX_TYPE> RowPtr_CSR(M+1, 0); //no definition
    vector<INDEX_TYPE> ColIdx_CSR(nnzR, 0);
    vector<VALUE_TYPE> Val_CSR(nnzR, 0.0);

    vector<INDEX_TYPE> ColPtr_CSC(K + 1, 0);
    vector<INDEX_TYPE> RowIdx_CSC(nnzR, 0);
    vector<VALUE_TYPE> Val_CSC(nnzR, 0.0);

    vector<INDEX_TYPE> RowIdx_COO(nnzR, 0);
    vector<INDEX_TYPE> ColIdx_COO(nnzR, 0);
    vector<VALUE_TYPE> Val_COO(nnzR, 0.0);

    cout << "Reading Sparse Matrix A... \n";

    // Read_matrix_2_CSR(filename, 
    //                   M, 
    //                   K, 
    //                   nnzR, 
    //                   RowPtr_CSR, 
    //                   ColIdx_CSR, 
    //                   Val_CSR
    //                  );
    
    Read_matrix_2_CSC(filename, 
                      M, 
                      K, 
                      nnzR, 
                      ColPtr_CSC, 
                      RowIdx_CSC, 
                      Val_CSC
                     );

    cout << "Reading Sparse Matrix A done\n ";

    
    // CSR_2_COO(M, 
    //           K, 
    //           nnzR,
    //           RowPtr_CSR, 
    //           ColIdx_CSR, 
    //           Val_CSR,
    //           RowIdx_COO,
    //           ColIdx_COO,
    //           Val_COO
    //          ); 

    // CSR_2_CSC(M, 
    //           K, 
    //           nnzR,
    //           RowPtr_CSR, 
    //           ColIdx_CSR, 
    //           Val_CSR,
    //           ColPtr_CSC,
    //           RowIdx_CSC,
    //           Val_CSC
    //          );
    
    CSC_2_COO(M, 
              K, 
              nnzR,
              ColPtr_CSC, 
              RowIdx_CSC, 
              Val_CSC,
              RowIdx_COO,
              ColIdx_COO,
              Val_COO
             );


    // 稀疏矩阵以PE数量维度切分,每块以COO存储
    cout << "Create Matrix Band... ";
    vector<Matrix_COO> Matrix_Band_COO(PE_NUM * HBM_CHANNEL_A_NUM);
    Matrix_Scatter(M,
                   K,
                   nnzR,
                   RowIdx_COO,
                   ColIdx_COO,
                   Val_COO,
                   PE_NUM * HBM_CHANNEL_A_NUM,
                   Matrix_Band_COO
                  );
    Display_Matrix_Band_COO(Matrix_Band_COO);
    cout << "done\n";

    vector<SparseSlice> Matrix_Band_Slice(PE_NUM * HBM_CHANNEL_A_NUM);
    Create_Matrix_Band_SparseSlice_ex(Matrix_Band_COO,
                                      Matrix_Band_Slice
                                     );
    cout << "done\n";

    cout << "Create SpElement_list... ";

    vector<vector<SpElement> > SpElement_list_pes;
    vector<INDEX_TYPE> SpElement_list_ptr;
    
    
    Create_SpElement_list_for_all_PEs(HBM_CHANNEL_A_NUM * PE_NUM, 
                                      M, 
                                      K, 
                                      Slice_SIZE, 
                                      BATCH_SIZE, 
                                      Matrix_Band_Slice, 
                                      SpElement_list_pes, 
                                      SpElement_list_ptr,
                                      WINDOWS
                                     );   
    cout << "done\n";


    // 打印每个PE的内容
    Display_PE(SpElement_list_pes);


    cout << "\nCreate Date for FPGA: \n";
    cout << "Create SpElement_list data for FPGA... ";

    aligned_vector<INDEX_TYPE> SpElement_list_ptr_fpga;
    Create_SpElement_list_data_FPGA(SpElement_list_ptr, SpElement_list_ptr_fpga);

    cout << "done\n";

    cout << "Create Sparse Matrix A data for FPGA... ";

    vector<aligned_vector<unsigned long> > Matrix_A_fpga_data(HBM_CHANNEL_A_NUM);
    Create_SpElement_list_for_all_channels(SpElement_list_pes,
                                           SpElement_list_ptr,
                                           Matrix_A_fpga_data,
                                           HBM_CHANNEL_A_NUM
                                          );

    cout << "done\n";

    cout << "\n Run kernel: \n";
    cout << " Run LU on CPU ... \n";
    auto CPU_start = std::chrono::steady_clock::now();

    LU_CPU_Slive(); // need definition

    auto CPU_end = std::chrono::steady_clock::now();

    cout << " LU on CPU done \n";

    double CPU_time = std::chrono::duration_cast<std::chrono::nanoseconds>(CPU_end - CPU_start).count();
    CPU_time *= 1e-9;
    printf("CPU time is %f ms\n", CPU_time * 1000);
    cout << "CPU GFLOPS: " << (2.0 * N * (nnzR + M) / 1e9 / CPU_time) << endl << endl;

    INDEX_TYPE Batch_num = SpElement_list_ptr.size() - 1;
    INDEX_TYPE Sparse_Matrix_len = SpElement_list_ptr[Batch_num];

    cout << "Run LU on FPGA ... \n ";

    double FPGA_time = tapa::invoke(LU,
                                    bitstream,
                                    tapa::read_only_mmap<INDEX_TYPE>(SpElement_list_ptr_fpga),


    );

    cout << " FPGA done \n";

    FPGA_time *= (1e-9 / iteration_num);
    printf("FPGA time is %f ms\n", FPGA_time * 1000);

    float GFLOPS = 2.0 * N * (nnzR + M) / 1e9 / FPGA_time;
    printf("FPGA GFLOPS: %f \n", GFLOPS);

    INDEX_TYPE error_num = 0;

    //todo
    cout << "Verify the correctness of result... ";
    




    return EXIT_SUCCESS;
}



