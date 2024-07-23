#include <ap_INDEX_TYPE.h>
#include <cstdio>
#include <cstring>
#include <cstdINDEX_TYPE>
#include <tapa.h>
#include "lu.h"

constexpr INDEX_TYPE WINDOW_SIZE = 8192;
constexpr INDEX_TYPE WINDOW_SIZE_MAX = 8192;
constexpr INDEX_TYPE WINDOW_SIZE_div_8 = 1024;
constexpr INDEX_TYPE WINDOW_SIZE_div_16 = 512;

constexpr INDEX_TYPE NUM_CH = 16;
constexpr INDEX_TYPE FIFO_DEPTH = 2;
constexpr INDEX_TYPE WINDOW_LARGE_SIZE = WINDOW_SIZE * NUM_CH;

using float_v16 = tapa::vec_t<float, 16>;
using float_v8 = tapa::vec_t<float, 8>;
using float_v2 = tapa::vec_t<float, 2>;
using int_v16 = tapa::vec_t<int, 16>;

struct MultXVec {
	tapa::vec_t<ap_uint<16>, 8> row;
	float_v8 axv;//value
};

template <typename data_t>
inline void bh(tapa::istream<data_t> & q) {
#pragma HLS inline
    for (;;) {
#pragma HLS pipeline II=1
        data_t tmp; 
		q.try_read(tmp);
    }
}

void black_hole_int(tapa::istream<INDEX_TYPE> & fifo_in) {
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

void read_len( const INDEX_TYPE N, 
			   const INDEX_TYPE NUM_ITE,
			   tapa::ostreams<INDEX_TYPE, NUM_CH>& N_value){
	for(INDEX_TYPE i = 0; i < NUM_ITE; i++){
		INDEX_TYPE remain = N - i * WINDOW_LARGE_SIZE;
		INDEX_TYPE len = WINDOW_SIZE;
		if(remain < WINDOW_LARGE_SIZE){
			len = (remain + 15) / 16;
		}
		for(INDEX_TYPE j = 0; j < NUM_CH; j++){
			N_value[j].write(len);
		}
	}
}

//监控任务的完成情况
void write_progress(tapa::istream<bool>& finish_write, 
					tapa::istream<int>& block_id, 
					tapa::ostream<int>& block_finish
){
	int block_done = 0;

	for(;;){
#pragma HLS pipeline II=1
		if(!finish_write.empty()){
			bool process_done = finish_write.read(nullptr);
			if(process_done){
				block_done += 1;
			}else{
				block_done = 0;
			}
		}

		if(!block_id.empty() & !block_finish.full()){
			int block = block_id.peek(nullptr);
			if(block_done > block){
				block_id.read(nullptr);
				block_finish.try_write(block);
			} 
		}
	}
}

//从存储器中读取数据，发送到流中
void read_all_ptr(
	tapa::async_mmap<int>& merge_inst_ptr,
	tapa::ostream<int>& merge_inst_q
){
	int N = 0;
	for (int i_req = 0, i_resp = 0; i_resp < 1;) {
		if((i_req < 1) & !merge_inst_ptr.read_addr.full()){
			merge_inst_ptr.read_addr.try_write(i_req);
			++i_req;
		}
		if(!merge_inst_ptr.read_data.empty()){
			merge_inst_ptr.read_data.try_read(N);
			++i_resp;
		}
	}
	for (int i_req = 0, i_resp = 0; i_resp < N;) {
		if((i_req < N) & !merge_inst_ptr.read_addr.full()){
				merge_inst_ptr.read_addr.try_write(i_req+1);
				++i_req;
		}
		if(!merge_inst_ptr.read_data.empty()){
			int tmp;
			merge_inst_ptr.read_data.try_read(tmp);
			merge_inst_q.write(tmp);
			++i_resp;
		}
	}
}

//读取矩阵数据
void read_comp_packet(
	const int A_LEN,
	tapa::async_mmap<ap_uint<512>>& comp_packet_ch,
	tapa::ostream<ap_uint<512>>& fifo_comp_packet_out
){
	for (int i_req = 0, i_resp = 0; i_resp < A_LEN;) {
	#pragma HLS pipeline II=1
		if(i_req < A_LEN && !comp_packet_ch.read_addr.full()){
				comp_packet_ch.read_addr.try_write(i_req);
				++i_req;
		}
		if(!comp_packet_ch.read_data.empty()){
			ap_uint<512> tmp; comp_packet_ch.read_data.try_read(tmp);
			fifo_comp_packet_out.write(tmp);
			++i_resp;
		}
	}
}


void split_comp_packet(
	const int pe_i,
	const int NUM_ITE,
	tapa::istream<ap_uint<512>>& comp_packet_ch,
	tapa::istream<int>& comp_packet_ptr,
	tapa::ostream<int>& comp_packet_ptr_out,
	tapa::ostream<ap_uint<512>>& fifo_solve_node,
	tapa::ostream<ap_uint<512>>& fifo_spmv,
	tapa::ostream<int>& fifo_inst_solve_node,
	tapa::ostream<int>& fifo_inst_spmv
){
	for(int round = 0; round < NUM_ITE; round++){
		const int spmv_N = comp_packet_ptr.read();
		const int dep_graph_N = comp_packet_ptr.read();
		const int spmv_total_len = comp_packet_ptr.read();
		comp_packet_ptr_out.write(spmv_N);
		comp_packet_ptr_out.write(dep_graph_N);
		comp_packet_ptr_out.write(spmv_total_len);

		fifo_inst_spmv.write(spmv_N);
		fifo_inst_spmv.write(dep_graph_N);
		fifo_inst_solve_node.write(dep_graph_N);

		fifo_inst_solve_node.write(spmv_total_len);

		//处理spmv数据，这部分可以不用
		for(int i = 0; i < spmv_N; i++){
			const int N = comp_packet_ptr.read();
			const int num_ite = (N + 7) / 8;
			comp_packet_ptr_out.write(N);
			fifo_inst_spmv.write(N);

			for (int j = 0; j < num_ite;) {
			#pragma HLS pipeline II=1
				if(!comp_packet_ch.empty()){
					ap_uint<512> packet; comp_packet_ch.try_read(packet);
					fifo_spmv.write(packet);
					j++;
				}
			}
		}

		//这部分很重要，处理依赖图的数据
		for(int i = 0; i < dep_graph_N-1; i++){
			const int N_node = comp_packet_ptr.read();
			const int N_edge = comp_packet_ptr.read();

			comp_packet_ptr_out.write(N_node);
			comp_packet_ptr_out.write(N_edge);
			fifo_inst_solve_node.write(N_node);
			fifo_inst_spmv.write(N_node);
			fifo_inst_spmv.write(N_edge);

			for (int j = 0; j < N_node;) {
			#pragma HLS pipeline II=1
				if(!comp_packet_ch.empty()){
					ap_uint<512> packet; comp_packet_ch.try_read(packet);
					fifo_solve_node.write(packet);
					j++;
				}
			}

			fifo_inst_solve_node.write(N_edge * NUM_CH);

			for (int j = 0; j < N_edge * NUM_CH;) {
			#pragma HLS pipeline II=1
				if(!comp_packet_ch.empty()){
					ap_uint<512> packet; comp_packet_ch.try_read(packet);
					fifo_spmv.write(packet);
					j++;
				}
			}
		}

		const int N_node = comp_packet_ptr.read();
		const int N_edge = comp_packet_ptr.read();
		comp_packet_ptr_out.write(N_node);
		comp_packet_ptr_out.write(N_edge);
		fifo_inst_solve_node.write(N_node);

		for (int j = 0; j < N_node;) {
		#pragma HLS pipeline II=1
			if(!comp_packet_ch.empty()){
				ap_uint<512> packet; comp_packet_ch.try_read(packet);
				fifo_solve_node.write(packet);
				j++;
			}
		}
	}
}


//Matrix - x unit
void sub_mul_unit(
	tapa::istream<INDEX_TYPE> &PE_Param_in,
	tapa::istream<ap_uINDEX_TYPE<512>> &Matrix_A_Stream,
	tapa::ostream<INDEX_TYPE> &PE_Param_out,
	tapa::ostream<MultXVec> &fifo_aXVec
){
	const INDEX_TYPE M = PE_Param_in.read();//hang
	const INDEX_TYPE N = PE_Param_in.read();//lie
	const INDEX_TYPE Iteration_num = PE_Param_in.read();

	PE_Param_out.write(M);
	PE_Param_out.write(N);
	PE_Param_out.write(Iteration_num);

	const INDEX_TYPE Iteration_time = (Iteration_num == 0) ? 1 : Iteration_num;
    
    const INDEX_TYPE Iteration_time_N = Iteration_time * ((N + 7) >> 3);

iter:
	for(INDEX_TYPE rp = 0; rp < Iteration_time_N; rp++){
		VALUE_TYPE Matrix_A_onchip[4][8][WINDOW_SIZE_div_8];
	}


}

void dispatch_inst_x(
	const int pe_i,
	const int NUM_ITE,
	tapa::istream<INDEX_TYPE> &fifo_inst_in,
	tapa::ostream<INDEX_TYPE> &fifo_inst_down,
	tapa::istream<INDEX_TYPE> &fifo_need_in,
	tapa::ostream<INDEX_TYPE> &fifo_need_out,
	tapa::istream<MultXVec> &fifo_prev_pe,
	tapa::ostream<MultXVec> &fifo_next_pe,
){
	for(int round = 0; round < NUM_ITE; round++){
		const int N_round = fifo_inst_in.read();
		const int N_layer = fifo_inst_in.read();
		fifo_inst_down.write(0); //phase 0: lu step 1
		fifo_inst_down.write(N_round);

		for(int ite = 0; ite < N_round; ite++){
			const int N = fifo_inst_in.read();
			const int need = fifo_need_in.read();
			const int num_ite = (N + 7) / 8;

			fifo_need_out.write(need);
			fifo_inst_down.write(num_ite);

			// dispatch step 1
			if(need == 1){
				fifo_inst_down.write(WINDOW_SIZE_div_16);
				for(int i = 0; i < WINDOW_SIZE_div_16;){
					#pragma HLS pipeline II=1
					if(!req_x_in.empty()){
						float_v16 x_val; req_x_in.try_read(x_val);
						req_x_down.write(x_val);
						req_x_out.write(x_val);
						i++;
					}
				}
			} else {
				fifo_inst_down.write(0);
			}
		}

		for(int i = 0; i < N_layer - 1; i++){
			const int N_node = fifo_inst_in.read();
			const int N_edge = fifo_inst_in.read();
			fifo_inst_down.write(1); //phase 1: lu step 2
			fifo_inst_down.write(NUM_CH);

			//dispatch step 2
			for(int level = 0; level < NUM_CH; level++){
				fifo_inst_down.write(N_edge);
				fifo_inst_down.write(N_node);

				for(int j = 0; j < N_node;){
					#pragma HLS pipeline II=1
					if(!fifo_prev_pe.empty()){
						MultXVec node_block; fifo_prev_pe.try_read(node_block);
						fifo_next_pe.write(node_block);
						fifo_node_down.write(node_block);
						j++;
					}
				}

			}
		}

	}
}












































void fill_zero(tapa::ostream<float_v16>& fifo_out){
	float_v16 tmp_x;
	fifo_out.try_write(tmp_x);
}

void fill_zero_xvec(tapa::ostream<MultXVec>& fifo_out){
	MultXVec tmp;
	fifo_out.try_write(tmp);
}

void LU(tapa::mmaps<ap_uint<512>, NUM_CH> comp_packet_ch,
        tapa::mmap<int> merge_inst_ptr,
		const INDEX_TYPE N,
		const INDEX_TYPE NUM_ITE,
		const INDEX_TYPE A_LEN
        ){

	tapa::streams<int, NUM_CH, FIFO_DEPTH> N_value("n_value");
	tapa::stream<bool, FIFO_DEPTH> finish_write("finish_write");
	tapa::stream<int, FIFO_DEPTH> block_id("block_id");
	tapa::stream<int, FIFO_DEPTH> block_finish("block_finish");
	tapa::streams<ap_uint<512>, NUM_CH, FIFO_DEPTH> fifo_comp_packet_out("fifo_comp_packet_out");
	tapa::streams<int, NUM_CH+1, FIFO_DEPTH> merge_inst_q("merge_inst_q");
	



	tapa::task()
		.invoke<tapa::join>(read_len, N, NUM_ITE, N_value)//读取矩阵，并分成16块，给16个channel

		.invoke<tapa::detach>(write_progress, finish_write, block_id, block_finish)//后台监控block的完成情况

		.invoke<tapa::join>(read_all_ptr, merge_inst_ptr, merge_inst_q)
		
		.invoke<tapa::join,NUM_CH>(read_comp_packet, A_LEN, comp_packet_ch, fifo_comp_packet_out)//读取矩阵数据
		
		.invoke<tapa::join, NUM_CH>(split_comp_packet, tapa::seq(), NUM_ITE, fifo_comp_packet_out, merge_inst_q, merge_inst_q,
																													fifo_solve_node,
																													fifo_spmv,
																													fifo_inst_solve_node,
																													fifo_inst_spmv)
		
		.invoke<tapa::detach>(black_hole_int, merge_inst_q)
		.invoke<tapa::detach, NUM_CH>(cache_x_and_feed, tapa::seq(), fifo_node_to_edge, fifo_broadcast, fifo_broadcast, fifo_forward_node)

    

}