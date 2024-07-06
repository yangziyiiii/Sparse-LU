#ifndef LU_COMMON_H
#define LU_COMMON_H

#include <vector>
#include <iostream>
#include <bitset>
#include <omp.h>
#include "mmio_highlevel.h"
#include "lu.h"

using std::cout;
using std::endl;
using std::vector;
using std::min;
using std::max;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T> >;

struct SpElement{
    INDEX_TYPE colIdx;
    INDEX_TYPE rowIdx;
    VALUE_TYPE val;
    
    SpElement(INDEX_TYPE colidx = -1, INDEX_TYPE rowidx = -1, VALUE_TYPE value = 0.0): colIdx(colidx), rowIdx(rowidx), val(value) {}
    
    SpElement& operator=(const SpElement& sp) {
        colIdx = sp.colIdx;
        rowIdx = sp.rowIdx;
        val    = sp.val;
        return *this;
    }
};

;

// COO格式存储的稀疏元素
struct Matrix_COO {
    INDEX_TYPE         M;
    INDEX_TYPE         K;
    INDEX_TYPE         nnzR;

    vector<INDEX_TYPE> ColIdx;
    vector<INDEX_TYPE> RowIdx;
    vector<INDEX_TYPE> RowIdx_copy; // new(3.28) 保存原始行索引副本
    vector<VALUE_TYPE> Val;

    // 16 bit mask vector
    vector<unsigned short> mask;
    // mask map val index
    vector<vector<INDEX_TYPE> > map;

    Matrix_COO() : M(0), K(0), nnzR(0), ColIdx() , RowIdx(), Val(), mask(), map() {}
};

// SparseSlice格式存储矩阵块元素
struct SparseSlice {
    INDEX_TYPE         sliceSize;        // 块大小
    INDEX_TYPE         numColSlices;     // 列块数
    INDEX_TYPE         numRowSlices;     // 行块数
    INDEX_TYPE         numSlices;        // 非空块数

    // 块以CSC格式存储
    vector<INDEX_TYPE> sliceColPtr;      // 块列指针
    vector<INDEX_TYPE> sliceRowIdx;      // 块行索引
    vector<Matrix_COO> sliceVal;         // 块元素内部以COO格式存储稀疏元素

    SparseSlice() : sliceSize(0), numColSlices(0), numRowSlices(0), sliceColPtr(), sliceRowIdx(), sliceVal() {}
};

// 读取矩阵规模
void Read_matrix_size(char       *filename,
                      INDEX_TYPE *M, 
                      INDEX_TYPE *K, 
                      INDEX_TYPE *nnzR,
                      INDEX_TYPE *isSymmetric
                     ) {

    mmio_info(M, K, nnzR, isSymmetric, filename);
}

// 读取矩阵数据以CSR格式存储
void Read_matrix_2_CSR(char       *filename, 
                       const INDEX_TYPE M, 
                       const INDEX_TYPE K, 
                       const INDEX_TYPE nnzR,

                       vector<INDEX_TYPE> &RowPtr, 
                       vector<INDEX_TYPE> &ColIdx, 
                       vector<VALUE_TYPE> &Val
                      ) {

    INDEX_TYPE *RowPtr_d = (INDEX_TYPE *)malloc(sizeof(INDEX_TYPE) * (M + 1));
    INDEX_TYPE *ColIdx_d = (INDEX_TYPE *)malloc(sizeof(INDEX_TYPE) * nnzR);
    VALUE_TYPE *Val_d    = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * nnzR);

    mmio_data_csr(RowPtr_d, ColIdx_d, Val_d, filename);

    for(INDEX_TYPE i = 0; i < M + 1; ++i)
        RowPtr[i] = RowPtr_d[i];
    
    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        ColIdx[i] = ColIdx_d[i];
        Val[i]    = Val_d[i];
    }

    free(Val_d);
    free(ColIdx_d);
    free(RowPtr_d);
}

// 读取矩阵数据以CSC格式存储
void Read_matrix_2_CSC(char       *filename, 
                       const INDEX_TYPE M, 
                       const INDEX_TYPE K, 
                       const INDEX_TYPE nnzR,

                       vector<INDEX_TYPE> &ColPtr, 
                       vector<INDEX_TYPE> &RowIdx, 
                       vector<VALUE_TYPE> &Val
                      ) {

    INDEX_TYPE *ColPtr_d = (INDEX_TYPE *)malloc(sizeof(INDEX_TYPE) * (K + 1));
    INDEX_TYPE *RowIdx_d = (INDEX_TYPE *)malloc(sizeof(INDEX_TYPE) * nnzR);
    VALUE_TYPE *Val_d    = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * nnzR);

    mmio_data_csc(ColPtr_d, RowIdx_d, Val_d, filename);

    for(INDEX_TYPE i = 0; i < K + 1; ++i)
        ColPtr[i] = ColPtr_d[i];
    
    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        RowIdx[i] = RowIdx_d[i];
        Val[i]    = Val_d[i];
    }

    free(Val_d);
    free(RowIdx_d);
    free(ColPtr_d);
}

// CSC转CSR
void CSC_2_CSR(const INDEX_TYPE M,
               const INDEX_TYPE K,
               const INDEX_TYPE nnzR,

               const vector<INDEX_TYPE> &ColPtr_CSC,
               const vector<INDEX_TYPE> &RowIdx_CSC,
               const vector<VALUE_TYPE> &Val_CSC,
               
               vector<INDEX_TYPE> &RowPtr_CSR,
               vector<INDEX_TYPE> &ColIdx_CSR,
               vector<VALUE_TYPE> &Val_CSR
              ) {

    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        RowPtr_CSR[RowIdx_CSC[i] + 1]++;
    }
    
    for(INDEX_TYPE i = 0; i < M; ++i) {
        RowPtr_CSR[i + 1] += RowPtr_CSR[i];
    }
    
    vector<INDEX_TYPE> row_nnzR(M, 0);
    for(INDEX_TYPE i = 0; i < K; ++i) {
        for(INDEX_TYPE j = ColPtr_CSC[i]; j < ColPtr_CSC[i + 1]; ++j) {
            INDEX_TYPE row = RowIdx_CSC[j];
            INDEX_TYPE col = i;
            VALUE_TYPE val = Val_CSC[j];
            
            INDEX_TYPE pos = RowPtr_CSR[row] + row_nnzR[row];
            Val_CSR[pos] = val;
            ColIdx_CSR[pos] = col;
            row_nnzR[row]++;
        }
    }
}

// CSR转CSC
void CSR_2_CSC(const INDEX_TYPE M, 
               const INDEX_TYPE K, 
               const INDEX_TYPE nnzR,

               const vector<INDEX_TYPE> &RowPtr_CSR, 
               const vector<INDEX_TYPE> &ColIdx_CSR, 
               const vector<VALUE_TYPE> &Val_CSR,

               vector<INDEX_TYPE> &ColPtr_CSC,
               vector<INDEX_TYPE> &RowIdx_CSC,
               vector<VALUE_TYPE> &Val_CSC
              ) {

    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        ColPtr_CSC[ColIdx_CSR[i] + 1]++;
    }

    for(INDEX_TYPE i = 0; i < K; ++i) {
        ColPtr_CSC[i + 1] += ColPtr_CSC[i];
    }

    vector<INDEX_TYPE> col_nnzR(K, 0);
    for(INDEX_TYPE i = 0; i < M; ++i) {
        for(INDEX_TYPE j = RowPtr_CSR[i]; j < RowPtr_CSR[i + 1]; ++j) {
            INDEX_TYPE row = i;
            INDEX_TYPE col = ColIdx_CSR[j];
            VALUE_TYPE val = Val_CSR[j];
            
            INDEX_TYPE pos = ColPtr_CSC[col] + col_nnzR[col];
            Val_CSC[pos] = val;
            RowIdx_CSC[pos] = row;
            col_nnzR[col]++;
        }
    }
}

// CSR转COO
void CSR_2_COO(const INDEX_TYPE M, 
               const INDEX_TYPE K, 
               const INDEX_TYPE nnzR,

               const vector<INDEX_TYPE> &RowPtr_CSR, 
               const vector<INDEX_TYPE> &ColIdx_CSR, 
               const vector<VALUE_TYPE> &Val_CSR,

               vector<INDEX_TYPE> &RowIdx_COO,
               vector<INDEX_TYPE> &ColIdx_COO,
               vector<VALUE_TYPE> &Val_COO
             ) {

    INDEX_TYPE row = 0;
    for(INDEX_TYPE i = 0; i < M; ++i) {
        for(INDEX_TYPE j = RowPtr_CSR[i]; j < RowPtr_CSR[i + 1]; ++j) {
            RowIdx_COO[j] = row;
            ColIdx_COO[j] = ColIdx_CSR[j];
            Val_COO[j]    = Val_CSR[j];
        }

        row++;
    }
}


// CSC转COO
void CSC_2_COO(const INDEX_TYPE M, 
               const INDEX_TYPE K, 
               const INDEX_TYPE nnzR,

               const vector<INDEX_TYPE> &ColPtr_CSC, 
               const vector<INDEX_TYPE> &RowIdx_CSC, 
               const vector<VALUE_TYPE> &Val_CSC,

               vector<INDEX_TYPE> &RowIdx_COO,
               vector<INDEX_TYPE> &ColIdx_COO,
               vector<VALUE_TYPE> &Val_COO
             ) {

    INDEX_TYPE col = 0;
    for(INDEX_TYPE i = 0; i < K; ++i) {
        for(INDEX_TYPE j = ColPtr_CSC[i]; j < ColPtr_CSC[i + 1]; ++j) {
            RowIdx_COO[j] = RowIdx_CSC[j];
            ColIdx_COO[j] = col;
            Val_COO[j]    = Val_CSC[j];
        }

        col++;
    }
}


// 打印稀疏矩阵
void Display_Sparse_Matrix(const INDEX_TYPE M, 
                           const INDEX_TYPE K, 
                           const INDEX_TYPE nnzR,

                           const vector<INDEX_TYPE> &RowIdx_COO,
                           const vector<INDEX_TYPE> &ColIdx_COO,
                           const vector<VALUE_TYPE> &Val_COO
                          ) {
    cout <<  "Sparse Matrix :\n";

    INDEX_TYPE k = 0, zero = 0;
    for(INDEX_TYPE i = 0; i < M; ++i) {
        for(INDEX_TYPE j = 0; j < K; ++j) {
            if(i == RowIdx_COO[k] && j == ColIdx_COO[k]) {
                cout << std::setw(3) << std::setfill(' ') << Val_COO[k] << ' ';
                ++k;
            }
            else
                cout << std::setw(3) << std::setfill(' ') << zero << ' ';
        }
        cout << endl;
    }
    cout << "--- Sparse Matrix Display END ---\n" << endl;
}








#endif