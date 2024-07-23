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
constexpr int WINDOW_SIZE_div_2 = 4096;
constexpr int WINDOW_LARGE_SIZE = WINDOW_SIZE * NUM_CH;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

template <typename data_t>
struct edge{
    int col;
    int row;
    data_t attr;
    
    edge(int d = -1, int s = -1, data_t v = 0): col(d), row(s), attr(v) {}
    
    edge& operator=(const edge& rhs) {
        col = rhs.col;
        row = rhs.row;
        attr = rhs.attr;
        return *this;
    }
};

void generate_dependency_graph_for_pes_cyclic(
    int N,
    const aligned_vector<int>& csc_col_ptr,
    const aligned_vector<int>& csc_row_idx,
    const aligned_vector<int>& csc_val,
    vector<aligned_vector<ap_uint<64>>>& dep_graph_ch,
    aligned_vector<int>& dep_graph_ptr
){
    int bound = (N % WINDOW_LARGE_SIZE == 0) ? N / WINDOW_LARGE_SIZE : N / WINDOW_LARGE_SIZE + 1; // 需要处理的窗口数量
    int total_iter_count = 0; //总迭代次数
	int total_effect_iter_count = 0; //有效迭代次数
    for(int i = 0; i < bound; i++){
        vector<int> cscColPtr;
        std::unordered_map<int, vector<edge<float>>> dep_map;

        //extract csc
        int col_ptr = 0;
        for(int j = 0; j < WINDOW_LARGE_SIZE && j < N - i * WINDOW_LARGE_SIZE; j++){ // 当前窗口
            int start = (i * WINDOW_LARGE_SIZE + j == 0) ? 0 : csc_col_ptr[ i * WINDOW_LARGE_SIZE + j - 1];
			for(int k = start; k < csc_col_ptr[ i * WINDOW_LARGE_SIZE + j ]; k++){
				if(csc_row_idx[k] >= i * WINDOW_LARGE_SIZE){
					int c = csc_row_idx[k] - i * WINDOW_LARGE_SIZE;//计算行索引的相对位置
					float v = csc_val[k];
					if(c == j) v = 1.0/v;
					edge<float> e(c, j, v);
					if(dep_map.find(c) == dep_map.end()){
						vector<edge<float>> vec;
						dep_map[c] = vec;
					}
					dep_map[c].push_back(e);
					col_ptr++;
				}
			}
			cscColPtr.push_back(col_ptr);
        }

        //generate sflu
        vector<int> parents;  //创建依赖关系
		std::queue<int> roots;

        int prev = 0; //记录上一个处理的节点在稀疏矩阵中的位置
        for(int j = 0; j < WINDOW_LARGE_SIZE && j < N - i * WINDOW_LARGE_SIZE; j++){
            parents.push_back(cscColPtr[j] - prev - 1);
            if(cscColPtr[j] - prev - 1 == 0){
                roots.push(j);
            }
            prev = cscColPtr[j];
        }

        vector<int> inst;
		int layer_count = 0;

        while(!roots.empty()){
            int size = roots.size();
            aligned_vector<vector<ap_uint<64>>> nodes_pe(NUM_CH);
            aligned_vector<vector<ap_uint<64>>> edges_pe(NUM_CH * NUM_CH);
            vector<int> node_count_pe(NUM_CH);
			aligned_vector<vector<int>> edge_count_pe(NUM_CH);

            for(int j = 0; j < size; j++){
                int root = roots.front();
                for(auto e : dep_map[root]){ //处理这个节点的依赖关系
                    ap_uint<64> a;
                    int ch = e.row % NUM_CH;
                    a(63,48) = (ap_uint<16>)((e.row/NUM_CH) & 0xFFFF);
					a(47,32) = (ap_uint<16>)((e.col/NUM_CH) & 0xFFFF);
					a(31,0) = tapa::bit_cast<ap_uint<32>>(e.attr);
					if(e.row == e.col){
						nodes_pe[ch].push_back(a);
					}else{
						edges_pe[ch*NUM_CH+(e.col%NUM_CH)].push_back(a);
						parents[e.row]--;
						if(parents[e.row] == 0) {
							roots.push(e.row);
						}
                    }
                }
                roots.pop();
            }

            int maxNode = 0;
			int maxEdge = 0;
            vector<vector<ap_uint<64>>> dep_graph_tmp(NUM_CH);
            int effect_edge = 0; //记录有效边的数量
            for(int pe_i = 0; pe_i < NUM_CH; pe_i++){
                int max_effect_edge = 0;
				int node_count = 0;
                vector<ap_uint<64>> nodes = nodes_pe[pe_i];

                int rem_node_num = nodes.size();//剩余未处理的节点数量
                int pushed_node_count = 0;  //已经分配到数据包中的节点的数量
                vector<bool> used_node(rem_node_num, false);
                while(rem_node_num > pushed_node_count){
                    std::set<int> row;
                    vector<ap_uint<64>> packet(8);
                    for(int n = 0; n < 8; n++){
                        ap_uint<64> a = 0;
                        a(63,48) = 0xFFFF;
                        a(31,0) = tapa::bit_cast<ap_uint<32>>((float)(1.0));
                        packet[n] = a;
                    }
                    for(int n = 0; n < rem_node_num; n++){
                        if(!used_node[n]){
                            auto nd = nodes[n];
                            int row_i = (nd(63,48) | (int) 0);
                            if(row.find(row_i % 8) == row.end()){
                                row.insert(row_i % 8);
                                packet[row_i % 8] = nd;
                                used_node[n] = true;
                                pushed_node_count++;
                            }
                        }
                        if(row.size() == 8) break;
                    }
                    for(int n = 0; n < 8; n++){
                        dep_graph_tmp[pe_i].push_back(packet[n]);
                    }
                    node_count++;
                } 
                    
                node_count_pe[pe_i] = node_count;
                if(node_count > maxNode) maxNode = node_count;
                for(int block_id = 0; block_id < NUM_CH; block_id++){
                    int edge_count = 0;
                    vector<ap_uint<64>> edge_list = edges_pe[pe_i*NUM_CH+block_id];
                    int rem_edge_num = edge_list.size();
                    int pushed_edge_count = 0;
                    vector<bool> used_edge(rem_edge_num, false);
                    int pack_chunk_count = 0;
                    vector<std::set<int>> row_raw(8); // last 8 elements
                    int next_slot = 0;
                    // std::set<int> row_raw;
                    while(pushed_edge_count < rem_edge_num){
                        std::set<int> row;
                        // std::set<int> col;
                        vector<ap_uint<64>> packet(8);
                        row_raw[next_slot].clear();

                        for(int n = 0; n < 8; n++){
                            ap_uint<64> a = 0;
                            a(63,48) = 0xFFFF;
                            packet[n] = a;
                        }
                        
                        for(int n = 0; n < rem_edge_num; n++){
                            if(!used_edge[n]){
                                auto e = edge_list[n];
                                int row_i = (e(63,48) | (int) 0);
                                bool found = false;
                                for(int m = 0; m < 8; m++){
                                    if(row_raw[m].find(row_i) != row_raw[m].end()){
                                        found = true;
                                        break;
                                    }
                                }
                                if(row.find(row_i%8) == row.end() && !found){
                                    row.insert(row_i%8);
                                    row_raw[next_slot].insert(row_i);
                                    packet[row_i % 8] = e;
                                    used_edge[n] = true;
                                }
                            }
                            if(row.size() == 8) break;
                        }
                        for(int n = 0; n < 8; n++){
                            dep_graph_tmp[pe_i].push_back(packet[n]);
                        }
                        pack_chunk_count++;
                        edge_count++;
                        next_slot = (next_slot+1)%8;
                        pushed_edge_count += row.size();
                    }
                    edge_count_pe[pe_i].push_back(edge_count);
                    if(edge_count > maxEdge) maxEdge = edge_count;
                    if(edge_count > max_effect_edge) max_effect_edge = edge_count;
                }
                effect_edge += max_effect_edge;
                inst.push_back(maxNode);
                inst.push_back(maxEdge);

                total_iter_count += (maxNode + maxEdge)*NUM_CH;
                total_effect_iter_count += maxNode * NUM_CH + effect_edge;

                //process dep graph ptr
                for(int pe_i = 0; pe_i < NUM_CH; pe_i++){
                    // std::clog << "pe: " << pe_i << std::endl;
                    int offset = 0;
                    int prev_size = dep_graph_ch[pe_i].size();
                    vector<ap_uint<64>> node_tmp_cache;
                    for(int b = offset; b < offset + node_count_pe[pe_i]*8; b++){
                        node_tmp_cache.push_back(dep_graph_tmp[pe_i][b]);
                    }
                    for(int b = 0; b < maxNode - node_count_pe[pe_i]; b++){
                        for(int n = 0; n < 8; n++){
                            ap_uint<64> a = 0;
                            a(63,48) = 0xFFFF;
                            a(31,0) = tapa::bit_cast<ap_uint<32>>((float)(1.0));
                            node_tmp_cache.push_back(a);
                        }
                    }
                    offset += node_count_pe[pe_i]*8;
                    for(int l = 0; l < maxNode*8; l++){
                        dep_graph_ch[pe_i].push_back(node_tmp_cache[l]);
                    }
                    for(int b = 0; b < NUM_CH; b++){
                        for(int l = offset; l < offset + edge_count_pe[pe_i][b]*8; l++){
                            dep_graph_ch[pe_i].push_back(dep_graph_tmp[pe_i][l]);
                        }
                        for(int l = 0; l < maxEdge - edge_count_pe[pe_i][b]; l++){
                            for(int n = 0; n < 8; n++){
                                ap_uint<64> a = 0;
                                a(63,48) = 0xFFFF;
                                dep_graph_ch[pe_i].push_back(a);
                            }
                        }
                        offset += edge_count_pe[pe_i][b]*8;
                    }
                }
                layer_count++;             
            }
            dep_graph_ptr.push_back(layer_count);
            for(auto num : inst){
                dep_graph_ptr.push_back(num);
            }
        }
    }
}

void merge_ptr(int N,
	aligned_vector<int>& dep_graph_ptr,
	aligned_vector<int>& edge_list_ptr,
	aligned_vector<int>& merge_inst_ptr){
	
	int bound = (N % WINDOW_LARGE_SIZE == 0) ? N / WINDOW_LARGE_SIZE : N / WINDOW_LARGE_SIZE + 1;
	int edge_list_offset = 0;
	int dep_graph_offset = 0;
	for(int round = 0; round < bound; round++){
		merge_inst_ptr.push_back((NUM_CH*round)*MULT_SIZE);
		int N_level = dep_graph_ptr[dep_graph_offset++];
		merge_inst_ptr.push_back(N_level);
		int sum = 0;
		for(int i = 0; i < (NUM_CH*round)*MULT_SIZE; i++){
			sum +=edge_list_ptr[i+edge_list_offset];
		}
		merge_inst_ptr.push_back((sum + 7)/8);
		for(int i = 0; i < (NUM_CH*round)*MULT_SIZE; i++){
			merge_inst_ptr.push_back(edge_list_ptr[i+edge_list_offset]);
		}
		edge_list_offset+=(NUM_CH*round)*MULT_SIZE;
		for(int i = 0; i < N_level*2; i++){
			merge_inst_ptr.push_back(dep_graph_ptr[i+dep_graph_offset]);
		}
		dep_graph_offset+= N_level*2;
	}
	int size = merge_inst_ptr.size();
	merge_inst_ptr.insert(merge_inst_ptr.begin(), size);
}

void merge_data(
	int N,
	aligned_vector<int>& dep_graph_ptr,
	aligned_vector<int>& edge_list_ptr,
	vector<aligned_vector<ap_uint<64>>>& dep_graph_ch,
	vector<aligned_vector<ap_uint<64>>>& edge_list_ch,
	vector<aligned_vector<ap_uint<64>>>& comp_packet_ch
){
	int bound = (N%WINDOW_LARGE_SIZE == 0)?N/WINDOW_LARGE_SIZE:N/WINDOW_LARGE_SIZE+1;
	for(int ch = 0; ch < NUM_CH; ch++){
		int edge_list_offset = 0;
		int dep_graph_offset = 0;
		int edge_list_ch_offset = 0;
		int dep_graph_ch_offset = 0;

		for(int round = 0; round < bound; round++){
			for(int i = 0; i < (NUM_CH*round)*MULT_SIZE; i++){
				int len = edge_list_ptr[i+edge_list_offset];
				for(int j = 0; j < len; j++){
					comp_packet_ch[ch].push_back(edge_list_ch[ch][j+edge_list_ch_offset]);
				}
				edge_list_ch_offset+=len;
			}
			edge_list_offset += (NUM_CH*round)*MULT_SIZE;
			int N_level = dep_graph_ptr[dep_graph_offset++];
			for(int i = 0; i < N_level; i++){
				int N_node = dep_graph_ptr[i*2 + dep_graph_offset];
				int N_edge = dep_graph_ptr[i*2 + 1 + dep_graph_offset];
				for(int j = 0; j < N_node; j++){
					for(int k = 0; k < 8; k++){
						comp_packet_ch[ch].push_back(dep_graph_ch[ch][(j*8 + k) + dep_graph_ch_offset]);
					}
				}
				dep_graph_ch_offset+=N_node*8;
				for(int j = 0; j < N_edge * NUM_CH; j++){
					for(int k = 0; k < 8; k++){
						comp_packet_ch[ch].push_back(dep_graph_ch[ch][(j*8 + k) + dep_graph_ch_offset]);
					}
				}
				dep_graph_ch_offset+=(N_edge)*NUM_CH * 8;
			}
			dep_graph_offset+=N_level*2;
		}
	}
}




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



