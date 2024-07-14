#include <ap_int.h>
#include <cstudio>
#include <cstring>
#include <cstdint>
#include <tapa.h>
#include "lu.h"

constexpr int WINDOW_SIZE = 8192;
constexpr int WINDOW_SIZE_div_8 = 1024;
constexpr int WINDOW_SIZE_MAX = 8192;
constexpr int WINDOW_SIZE_div_16 = 512;
constexpr int NUM_CH = 16;
constexpr int FIFO_DEPTH = 2;
constexpr int WINDOW_LARGE_SIZE = WINDOW_SIZE * NUM_CH;

using float_v16 = tapa::vec_t<float, 16>;
using float_v8 = tapa::vec_t<float, 8>;
using float_v2 = tapa::vec_t<float, 2>;

using int_v16 = tapa::vec_t<int, 16>;

struct MultXVec {
	tapa::vec_t<ap_uint<16>, 8> row;
	float_v8 axv;//value
};

//读取输入流
template <typename data_t>
inline void bh(tapa::istream<data_t> & q) {
#pragma HLS inline
    for (;;) {
#pragma HLS pipeline II=1
        data_t tmp; 
		q.try_read(tmp);
    }
}

void black_hole_int(tapa::istream<int> & fifo_in) {
    bh(fifo_in);
}

void black_hole_float(tapa::istream<float>& fifo_in) {
	bh(fifo_in);
}

void black_hole_float_vec(tapa::istream<float_v16>& fifo_in) {
	bh(fifo_in);
}

void black_hole_ap_uint(tapa::istream<ap_uint<512>>& fifo_in){
	bh(fifo_in);
}

void black_hole_xvec(tapa::istream<MultXVec>& fifo_in){
	bh(fifo_in);
}

// 异步读取模块
template <typename T1, typename T2>
inline void Async_Read(tapa::async_mmap<T1> &mmap_in,    // 异步内存映射对象
                       tapa::ostream<T1> &Stream_out,    // 输出流
                       const T2 mmap_in_len,             // 数据长度
                       T2 &i_request,                    // 请求计数器
                       T2 &i_response                    // 响应计数器
                      ) {

#pragma HLS inline
    if((i_request < mmap_in_len) & !mmap_in.read_addr.full()) {  // 检查是否仍然有未满的读地址通道和读取请求尚未达到总长度
        mmap_in.read_addr.try_write(i_request);                  // 向读地址通道写入当前的读请求地址
        ++i_request;
    }
    if(!Stream_out.full() & !mmap_in.read_data.empty()) {        // 检查输出流是否未满且读数据通道中是否有数据可用
        T1 temp;
        mmap_in.read_data.try_read(temp);                        // 从读数据通道中读取数据
        Stream_out.try_write(temp);                              // 写入输出流
        ++i_response;
    }
}

void SpElement_list_ptr_Loader(const INDEX_TYPE Batch_num,                        // 数据批处理批次数
                               const INDEX_TYPE M,                                // 行数
                               const INDEX_TYPE N,
                               const INDEX_TYPE K, 
                               const INDEX_TYPE Iteration_num,                    // 迭代次数
                               tapa::async_mmap<INDEX_TYPE> &SpElement_list_ptr,  // 批处理指针
                               tapa::ostream<INDEX_TYPE> &PE_Param                // PE指令流
                              ) {
    
    // 将参数写入PE指令流
    PE_Param.write(Batch_num);
    PE_Param.write(M);
    PE_Param.write(N);
    PE_Param.write(K);
    PE_Param.write(Iteration_num);                           

    const INDEX_TYPE Iteration_time = (Iteration_num == 0) ? 1 : Iteration_num;
    
    const INDEX_TYPE Batch_num_plus_1 = Batch_num + 1;

    const INDEX_TYPE Iteration_time_N = Iteration_time * ((N + 7) >> 3);
    // 循环迭代
iter:
    for(INDEX_TYPE iter = 0; iter < Iteration_time_N; ++iter) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    // 读取批处理指针内容写入到PE指令流中
    Load_ptr:
        for(INDEX_TYPE i_request = 0, i_response = 0; i_response < Batch_num_plus_1;) {
#pragma HLS loop_tripcount min=1 max=800
#pragma HLS pipeline II=1
            Async_Read(SpElement_list_ptr,
                       PE_Param,
                       Batch_num_plus_1,
                       i_request, 
                       i_response
                      );
        }
    }
}

// 稀疏矩阵加载模块 read matrix A
void Sparse_Matrix_Loader(const INDEX_TYPE Matrix_len,                       // 稀疏矩阵数据长度
                          const INDEX_TYPE N, 
                          const INDEX_TYPE Iteration_num,                    // 迭代次数
                          tapa::async_mmap<ap_uint<512>> &Matrix_A_data,       // 稀疏矩阵数据
                          tapa::ostream<ap_uint<512>> &Matrix_A_Stream       // 稀疏矩阵数据流
                          ) {

    const INDEX_TYPE Iteration_time = (Iteration_num == 0) ? 1 : Iteration_num;
    const INDEX_TYPE Iteration_time_N = Iteration_time * ((N + 7) >> 3);
    // 循环迭代
iter:
    for(INDEX_TYPE iter = 0; iter < Iteration_time_N; ++iter) {
#pragma HLS loop_flatten off
#pragma HLS loop_tripcount min=1 max=16
    // 读取稀疏矩阵内容写入到稀疏矩阵数据流中
    Load_A:
        for(INDEX_TYPE i_request = 0, i_response = 0; i_response < Matrix_len;) {
#pragma HLS loop_tripcount min=1 max=10000
#pragma HLS pipeline II=1
            Async_Read(Matrix_A_data,
                       Matrix_A_Stream,
                       Matrix_len,
                       i_request, 
                       i_response
                      );
        }
    }
}

//把512位的数据分割成两个256位
void Segment(tapa::istream<ap_uint<512>> & Matrix_A_Stream,
             tapa::ostreams<ap_uint<256>, 2> & Matrix_A_Stream_256
            ) {
Seg:
    for(;;) {
#pragma HLS pipeline II=1
        bool flag_nop = Matrix_A_Stream.empty();
        for(INDEX_TYPE i = 0; i < 2; ++i) {
            flag_nop |= Matrix_A_Stream_256[i].full();
        }
        if(!flag_nop) {
            ap_uint<512> tmp; Matrix_A_Stream.try_read(tmp);
            for(INDEX_TYPE i = 0; i < 2; ++i) {
                Matrix_A_Stream_256[i].try_write(tmp(255 + i * 256, i * 256));
            }
        }
    }
}


void PEG_Xvec(
	tapa::istream<int>& fifo_inst_in,
	tapa::istream<MultXVec>& fifo_node_in,
	tapa::istream<ap_uint<512>>& sub_mul_A, 	//-x算法
	tapa::istream<float_v16>& fifo_req_x,
	tapa::ostream<MultXVec>& fifo_aXVec
){
	for(;;){
		float local_A[4][8][WINDOW_SIZE_div_8];
#pragma HLS bind_storage variable=local_A letency=2
#pragma HLS array_partition variable=local_A complete dim=1
#pragma HLS array_partition variable=local_A complete dim=2
		const int phase = fifo_inst_in.read(); // phase 0: -x , phase 1: lu
		const int N_round = fifo_inst_in.read();

		for(int i = 0; i < N_round; i++){
			const int num_read = fifo_inst_in.read();
			const int num_write = fifo_inst_in.read();

			if(phase == 0){
				// step 1 : - x loop
				for(int j = 0; j < num_read; ){
					if(!fifo_node_in.empty()){
						float_v16 A_val; 
						fifo_req_A.try_read(A_val);
						for(int k = 0; k < 16; k++){
							for(int l = 0; l < 4; l++){
								local_A[l][k%8][(j<<1)+(k>>3)] = A_val[l];
							}
						}
						j++
					}
				}
			}else{
				for(int j = 0; j < num_read;){
					#pragma HLS pipeline II=1
					#pragma HLS dependence variable=local_x false
					if(!fifo_node_in.empty()){
						MultXVec node_block; fifo_node_in.try_read(node_block);

						for(int k = 0; k < 8; k++){
							auto row = node_block.row[k];
							auto val = node_block.axv[k];
							if(row[15] == 0){
								for(int l = 0; l < 4; l++){
									local_x[l][k][(row >> 3)] = val;
								}
							}
						}
						j++;
					}
				}
			}

			for(int j = 0; j < num_write;){
				#pragma HLS pipeline II=1

				if(!sub_mul_A.empty()){
					ap_uint<512> dep_block; 
					sub_mul_A.try_read(dep_block);
					MultXVec xvec;
					for(int k = 0; k < 8; k++){
						#pragma HLS unroll
						ap_uint<64> a = dep_block(64*k + 63, 64*k);
						ap_uint<16> row = a(63,48);
						ap_uint<16> col = a(47,32);
						ap_uint<32> val = a(31,0);

						xvec.row[k] = row;
						if(row[15] == 0){
							float val_f = tapa::bit_cast<float>(val);
							xvec.axv[k] = (local_A[k/2][col%8][(col >> 3)] * val_f);
						}
					}
					fifo_aXVec.write(xvec);
					j++;
				}
			}
		}
	}
}

//Matrix multiplication and sub unit
void Mul_sub_core(){

}

void dispatch_inst_A(

){

}

void cache_result_and_write(
    tapa::istream<MultXVec>& fifo_A_in,
	tapa::ostream<MultXVec>& fifo_A_to_fwd,
	tapa::ostream<float>& fifo_A_out,
	tapa::istream<int>& fifo_inst_in,
	tapa::ostream<int>& fifo_inst_out
){
    for(;;){
#pragma HLS loop_flatten off

		float local_A[8][WINDOW_SIZE_div_8];

#pragma HLS bind_storage variable=local_A type=RAM_2P impl=URAM
#pragma HLS array_partition complete variable=local_x dim=1

        const int num_ite = fifo_inst_in.read();
        const int num_layer = fifo_inst_in.read();
        fifo_inst_out.write(num_ite);
        fifo_inst_out.write(num_layer);
        for(int i = 0; i < num_layer; i++){
            const int N_node = fifo_inst_in.read();
			fifo_inst_out.write(N_node);

			for(int j = 0; j < N_node;){
			#pragma HLS pipeline II=1

				if(!fifo_x_in.empty()){
					MultXVec ravx; fifo_A_in.try_read(ravx);
					fifo_A_to_fwd.write(ravx);
					for(int k = 0; k < 8; k++){
						auto a_row = ravx.row[k];
						auto a_val = ravx.axv[k];
						if(a_row[15] == 0){
							local_A[k][a_row >> 3] = a_val;
						}
					}
					j++;
				}
			}
		}
    }
}

void cache_result_and_fwd(
	tapa::istream<int>& fifo_inst_in,
	tapa::istream<MultXVec>& fifo_A_in,
	tapa::istream<int>& block_fwd,
	tapa::ostream<int>& block_fwd_out,
	tapa::ostream<float>& fifo_A_fwd
){
	for(;;){
#pragma HLS loop_flatten off

		float local_A[8][WINDOW_SIZE_div_8];

#pragma HLS bind_storage variable=local_A type=RAM_2P impl=URAM
#pragma HLS array_partition complete variable=local_A dim=1

		const int num_ite = fifo_inst_in.read();
		const int num_layer = fifo_inst_in.read();

		for(int i = 0; i < num_layer; i++){
			const int N_node = fifo_inst_in.read();
			for(int j = 0; j < N_node;){
			#pragma HLS pipeline II=1
				if(!fifo_A_in.empty()){
					MultXVec ravx; 
                    fifo_A_in.try_read(ravx);
					for(int k = 0; k < 8; k++){
						auto a_row = ravx.row[k];
						auto a_val = ravx.axv[k];
						if(a_row[15] == 0){
							local_A[k][a_row >> 3] = a_val;
						}
					}
					j++;
				}
			}
		}
		
		for(int i = 0 ; i < NUM_CH; i++){
			//forwarding to next batch
			const int fwd_block = block_fwd.read();
			block_fwd_out.write(fwd_block);

			if(fwd_block == 1){
				for(int j = 0; j < WINDOW_SIZE_div_16; j++){
				#pragma HLS loop_tripcount min=1 max=64
				#pragma HLS pipeline II=1
					fifo_A_fwd.write(local_A[j%8][i*64 + (j >> 3)]);
				}
			}
		}
	}
}

void cache_result_and_feed(
	const int pe_i,
	tapa::istream<MultXVec>& fifo_A_in,
	tapa::istream<MultXVec>& fifo_prev,
	tapa::ostream<MultXVec>& fifo_A_out,
	tapa::istream<int>& fifo_inst
) {
	for(;;){

		if(pe_i == 0 && !fifo_prev.empty()){
			fifo_prev.read(nullptr);
		}

		MultXVec cache_A[WINDOW_SIZE_div_8];

		#pragma HLS aggregate variable=cache_A
		#pragma HLS bind_storage variable=cache_A type=RAM_2P impl=URAM

		const int N = fifo_inst.read();
		for(int i = 0; i < N;){
			#pragma HLS pipeline II=1
			if(!fifo_A_in.empty()){
				fifo_A_in.try_read(cache_A[i]);
				i++;
			}
		}

		for(int i = 0; i < pe_i+1; i++){
			for(int j = 0; j < N;){
				MultXVec tmp;
				if(pe_i == i){
					fifo_A_out.write(cache_A[j]);
					j++;
				} else if(!fifo_prev.empty()){
					fifo_prev.try_read(tmp);
					fifo_A_out.write(tmp);
					j++;
				}
			}
		}
	}
}

void read_comp_packet(
	const int A_LEN,
	tapa::async_mmap<ap_uint<512>>& comp_packet_ch,
){
	for(int i_req = 0, i_resp = 0; i_resp < A_LEN;){
#pragma HLS_pipeline II=1
		if(i_req < A_LEN && !comp_packet_ch.read_addr.full()){
			comp_packet_ch.read_addr.try_write(i_req);
			++i_req;
		}
		if(!comp_packet_ch.read_addr.empty()){
			ap_uint<512> tmp;
			comp_packet_ch.read_data.try_read(tmp);
			fifo_comp_packet_out.write(tmp);
			++i_resp;
		}
	}
}

void sub_mul_kernel(){

}


//分割数据，发送给不同的PE单元，处理图的依赖关系
void split_comp_packet(
	const int pe_i,
	const int NUM_ITE,
	tapa::istream<ap_uint<512>>& comp_packet_ch,
	tapa::istream<int>& comp_packet_ptr,
	tapa::ostream<ap_uint<512>>& fifo_solve_node,
	tapa::ostream<int>& fifo_inst_solve_node,
	tapa::ostream<int>& comp_packet_ptr_out,
){
	for(int round = 0; round < NUM_ITE; round++){
		const int dep_graph_N = comp_packet_ptr.read();
		comp_packet_ptr_out.write(dep_graph_N);
		fifo_inst_solve_node.write(dep_graph_N);

		for(int i = 0; i<dep_graph - 1; i++){
			const int N_node = comp_packet_ptr.read();
			const int N_edge = comp_packet_ptr.read();
			comp_packet_ptr_out.write(N_node);
			comp_packet_ptr_out.write(N_edge);
			fifo_inst_solve_node.write(N_node);

			for(int j= 0; j < N_node;){
#pragma HLS pipeline II=1
				if(!comp_packet_ch.empty()){
					ap_uint<512> packet;
					comp_packet_ch.try_read(packet);
					fifo_solve_node.write(packet);
					j++
				}
			}

			fifo_inst_solve_node.write(N_edge * NUM_CH);

			for(int k = 0; k < N_edge * NUM_CH; ){
				if(!comp_packet_ch.empty()){
					ap_uint<512> packet;
					comp_packet_ch.try_read(packet);
					fifo_sub_mul.write(packet);  // no definition and datapath
					k++;
				}
			}


		}
	}


}




// 销毁整型流中数据
void Destroy_int(tapa::istream<INDEX_TYPE> &Stream_in) {
    for(;;) {
#pragma HLS pipeline II=1
        INDEX_TYPE tmp; 
        Stream_in.try_read(tmp);
    }
}




void LU(tapa::mmap<INDEX_TYPE> SpElement_list_ptr,
             
        tapa::mmaps<ap_uint<512>, HBM_CHANNEL_A_NUM> Matrix_A_data,
        
        ){

    tapa::streams<INDEX_TYPE, HBM_CHANNEL_A_NUM * UNIT_NUM + 1, FIFO_DEPTH> PE_Param("PE_Param");

    tapa::streams<ap_uint<512>, HBM_CHANNEL_A_NUM, FIFO_DEPTH> Matrix_A_Stream("Matrix_A_Stream");

    

}