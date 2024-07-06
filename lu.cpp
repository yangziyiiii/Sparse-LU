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

//把512位的数据分割成
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


void PEG_Xvec(){

}

//Matrix multiplication and add unit
void Mul_add_core(){

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