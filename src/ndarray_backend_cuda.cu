#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <float.h>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
#define BLOCK_V 4
#define BLOCK_L 8
#define BLOCK_S 4

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

__device__ size_t TranslateToNonCompactIndex(size_t gid, CudaVec shape, CudaVec strides) {
  size_t dim = shape.size;
  size_t base = 1;
  size_t idx = 0;
  for (int i = dim - 1; i >= 0; i--) {
    int idx_dim_i = (gid / base) % shape.data[i];
    idx += strides.data[i] * idx_dim_i;
    base *= shape.data[i];
  }
  return idx;
}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  // 根基gid 去从 a 的某个位置找元素
  size_t idx = TranslateToNonCompactIndex(gid, shape, strides);

  if (gid < size)
    out[gid] = a[offset + idx];
  /// END YOUR SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}




__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                  CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  size_t idx = TranslateToNonCompactIndex(gid, shape, strides);

  if (gid < size)
    out[offset + idx] = a[gid];
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                            VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}


__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec shape,
                                  CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idx = TranslateToNonCompactIndex(gid, shape, strides);

  if (gid < size)
    out[offset + idx] = val;
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(out->size, val, out->ptr, VecToCuda(shape),
                                            VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION
#define KERNEL_LEFT size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid]

#define DIM_INIT CudaDims dim = CudaOneDim(out->size)

// 二元参数调用 宏定义
#define EWISE_BINARY_KERNEL_INVOKE <<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size)

#define SCALAR_BINARY_KERNEL_INVOKE <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size)

#define EWISE_UNARY_KERNEL_INVOKE <<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size)

__global__ void EwiseMulKernel (const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  KERNEL_LEFT = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  DIM_INIT;
  EwiseMulKernel EWISE_BINARY_KERNEL_INVOKE;
}

__global__ void ScalarMulKernel (const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  KERNEL_LEFT = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  DIM_INIT;
  ScalarMulKernel SCALAR_BINARY_KERNEL_INVOKE;
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  KERNEL_LEFT = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  DIM_INIT;
  EwiseDivKernel EWISE_BINARY_KERNEL_INVOKE;
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  KERNEL_LEFT = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  DIM_INIT;
  ScalarDivKernel SCALAR_BINARY_KERNEL_INVOKE;
}

__global__ void EwisePowerKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  KERNEL_LEFT = pow(a[gid], b[gid]);
}

void EwisePower(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  DIM_INIT;
  EwisePowerKernel EWISE_BINARY_KERNEL_INVOKE;
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  KERNEL_LEFT = pow(a[gid], val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  DIM_INIT;
  ScalarPowerKernel SCALAR_BINARY_KERNEL_INVOKE;
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  KERNEL_LEFT = a[gid] > b[gid] ? a[gid] : b[gid];
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  DIM_INIT;
  EwiseMaximumKernel EWISE_BINARY_KERNEL_INVOKE;
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  KERNEL_LEFT = a[gid] > val ? a[gid] : val;
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  DIM_INIT;
  ScalarMaximumKernel SCALAR_BINARY_KERNEL_INVOKE;
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  KERNEL_LEFT = (a[gid] == b[gid]);
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  DIM_INIT;
  EwiseEqKernel EWISE_BINARY_KERNEL_INVOKE;
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  KERNEL_LEFT = (a[gid] == val);
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  DIM_INIT;
  ScalarEqKernel SCALAR_BINARY_KERNEL_INVOKE;
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  KERNEL_LEFT = (a[gid] >= b[gid]);
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  DIM_INIT;
  EwiseGeKernel EWISE_BINARY_KERNEL_INVOKE;
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  KERNEL_LEFT = (a[gid] >= val);
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  DIM_INIT;
  ScalarGeKernel SCALAR_BINARY_KERNEL_INVOKE;
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  KERNEL_LEFT = log(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  DIM_INIT;
  EwiseLogKernel EWISE_UNARY_KERNEL_INVOKE;
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  KERNEL_LEFT = exp(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  DIM_INIT;
  EwiseExpKernel EWISE_UNARY_KERNEL_INVOKE;
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  KERNEL_LEFT = tanh(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  DIM_INIT;
  EwiseTanhKernel EWISE_UNARY_KERNEL_INVOKE;
}
/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulKernel(const scalar_t *a, scalar_t *b, scalar_t *out, int m, int n, int p) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < m * p) {
    int i = gid / p;
    int j = gid % p;
    out[gid] = 0.0f;
    for(int k = 0; k < n; k++) {
      out[gid] += a[i * n + k] * b[k * p + j];
    }
  }
}

__global__ void MatmulParallelKernel(const scalar_t* A, const scalar_t* B, scalar_t* out, int m, int n, int p) {
  /**
   * 并行化加速计算矩阵乘法。每个线程计算一个V*V的子矩阵，每个block计算一个L*L的大矩阵
   * "We should implement a single function that works across all size metrices"
   * 注意方法需要能处理 size 不是 L S V 倍数的情况。灵活运用 m, n, p 边界。
  */
  __shared__ scalar_t a_shared[BLOCK_L][BLOCK_S];
  __shared__ scalar_t b_shared[BLOCK_S][BLOCK_L];

  float c[BLOCK_V][BLOCK_V] = {0};
  float a[BLOCK_V]={0}, b[BLOCK_V]={0};
  
  int yblock = blockIdx.y;
  int xblock = blockIdx.x;
  int nthreads = blockDim.y * blockDim.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  // 每个 block 负责处理一个 L*L 的子矩阵
  for (int k0 = 0; k0 < n; k0 += BLOCK_S) {
    __syncthreads();
    // thread cooperative fetching shared memory
    // sA[:, :] = A[k : k + S, yblock * L : yblock * L + L];
    // sB[:, :] = B[k : k + S, xblock * L : xblock * L + L];
    for (int j = 0; j < BLOCK_L * BLOCK_S / nthreads; ++j) {
      int y_a = (j * nthreads + tid) / BLOCK_S;
      int x_a = (j * nthreads + tid) % BLOCK_S;
      int y_A = yblock * BLOCK_L + y_a;
      int x_A = k0 + x_a;

      int y_b = (j * nthreads + tid) / BLOCK_L;
      int x_b = (j * nthreads + tid) % BLOCK_L;
      int y_B = k0 + y_b;
      int x_B = xblock * BLOCK_L + x_b;

      // 这里可能需要判断边界条件，特别是在size不是 BLOCK_L_S_V的倍数时
      // 
      if (y_A < m && x_A < n) {
        a_shared[y_a][x_a] = A[y_A * n + x_A];
      } else {
        a_shared[y_a][x_a] = 0;
      }
      

      // if (y_B < n && x_B < p)
      // 这里可能需要判断边界条件，特别是在size不是 BLOCK_L_S_V的倍数时
      if (y_B < n && x_B < p) {
        b_shared[y_b][x_b] = B[y_B * p + x_B];
      } else {
        b_shared[y_b][x_b] = 0;
      }
    }
    __syncthreads();
    // 每个 thread 负责处理一个 V*V 的子矩阵
    for (int k = 0; k < BLOCK_S; ++k) {
      // copy
      for(int v = 0; v < BLOCK_V; v++) {
        a[v] = a_shared[threadIdx.y * BLOCK_V + v][k];
        b[v] = b_shared[k][threadIdx.x * BLOCK_V + v];
      }
      // compute
      for(int i = 0; i < BLOCK_V; i++) {
        for(int j = 0; j < BLOCK_V; j++) {
          c[i][j] += a[i] * b[j];
        }
      }
    }
  }
  // 每个 thread 计算好的 V*V 结果放回global memory
  int ybase = blockIdx.y * blockDim.y + threadIdx.y;
  int xbase = blockIdx.x * blockDim.x + threadIdx.x;

  for(int i = 0; i < BLOCK_V; i++) {
    for(int j = 0; j < BLOCK_V; j++) {
      int y = ybase * BLOCK_V + i;
      int x = xbase * BLOCK_V + j;
      if (y < m && x < p) {
        out[y * p + x] = c[i][j];
      }
    }
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  // TODO cooperative fetching and the block shared memory register tiling covered in class.
  // CudaDims dim = CudaOneDim(M * P);
  // MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);

  // printf("GRID SIZE:(%d, %d)\n",(M + BLOCK_L - 1) / BLOCK_L, (P + BLOCK_L - 1) / BLOCK_L);
  // 并行参数, grid和block都是2维
  dim3 grid((P + BLOCK_L - 1) / BLOCK_L, (M + BLOCK_L - 1) / BLOCK_L);
  // 每个 block 有 (N/V * N/V)
  dim3 block(BLOCK_L / BLOCK_V, BLOCK_L / BLOCK_V);
  MatmulParallelKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END YOUR SOLUTION
}

void MatmulVanilla(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  
  CudaDims dim = CudaOneDim(M * P);
  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

// 地址a开始的N个元素，每组256个连续数据并行地找出最大值，放在output里。
__global__ void ReduceMaxParallelKernel(const scalar_t* a, scalar_t* output, size_t N) {
  // 创建同一个block的共享内存
  __shared__ scalar_t block_cache[BASE_THREAD_NUM];

  // 每个block内部的线程id
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // // Load data from global memory to shared memory
  block_cache[tid] = (gid < N) ? a[gid] : -FLT_MAX;  // Assuming a minimum value for comparison

  while (gid < N) {
    block_cache[tid] = fmaxf(a[gid], block_cache[tid]);
    gid += blockDim.x * gridDim.x;
  }
  // 等待同一个block里面的所有线程都执行到此处
  __syncthreads();

  // 同一个block里的元素进行max reduce 
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      block_cache[tid] = fmaxf(block_cache[tid], block_cache[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = block_cache[tid];
  }
}

void ReduceMaxParallel(const scalar_t* a, scalar_t* out, size_t reduce_size) {
  // 自己计算 并行参数
  int blocks_per_grid = (reduce_size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  int threads_per_block = BASE_THREAD_NUM;

  // allocate global memory
  scalar_t* reduction_result;
  
  cudaError_t err = cudaMalloc(&reduction_result, blocks_per_grid * ELEM_SIZE);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

  // Reduction Phase
  ReduceMaxParallelKernel<<<blocks_per_grid, threads_per_block>>>(a, reduction_result, reduce_size);
  
  while(blocks_per_grid > 1) {
    int blocks_next = (blocks_per_grid + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
    ReduceMaxParallelKernel<<<blocks_next, threads_per_block>>>(reduction_result, reduction_result, blocks_per_grid);
    blocks_per_grid = blocks_next;
  }

  // 代码这个位置在cpu里，所有找不到cuda里的内存地址。会报segment fault。
  // out->ptr = reduction_result[0];
  // 只能用这个指令
  cudaMemcpy(out, reduction_result, ELEM_SIZE, cudaMemcpyDeviceToDevice);
  cudaFree(reduction_result);
}

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_len) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_len) return;
  scalar_t max_val = a[gid * reduce_size];
  for (size_t i = gid * reduce_size + 1; i < (gid + 1) * reduce_size; i++) {
    max_val = max_val > a[i] ? max_val : a[i];
  }
  out[gid] = max_val;
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  // TODO a more industrial-grade implementation, use a hierarchical mechanism that first aggregated across some smaller span,
  // then had a secondary function that aggregated across these reduced arrays

  // reduce_max如果没指定axis, 简单版的实现非常低效。需要利用Cuda并行优化
  for(int i = 0; i < out->size; i++) {
    ReduceMaxParallel(&a.ptr[i * reduce_size], &out->ptr[i], reduce_size);
  }
  // CudaDims dim = CudaOneDim(out->size);
  // ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  
  /// END YOUR SOLUTION
}

__global__ void ReduceSumParallelKernel(const scalar_t* a, scalar_t* output, size_t N) {
  // 创建同一个block的共享内存
  __shared__ scalar_t block_cache[BASE_THREAD_NUM];

  // 每个block内部的线程id
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // // Load data from global memory to shared memory
  block_cache[tid] = 0.0f;  // Assuming a minimum value for comparison

  while (gid < N) {
    block_cache[tid] += a[gid];
    gid += blockDim.x * gridDim.x;
  }
  // 等待同一个block里面的所有线程都执行到此处
  __syncthreads();

  // 同一个block里的元素进行max reduce 
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      block_cache[tid] += block_cache[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = block_cache[tid];
  }
}

void ReduceSumParallel(const scalar_t* a, scalar_t* out, size_t reduce_size) {
  // 自己计算 并行参数
  int blocks_per_grid = (reduce_size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  int threads_per_block = BASE_THREAD_NUM;

  // allocate global memory
  scalar_t* reduction_result;
  
  cudaError_t err = cudaMalloc(&reduction_result, blocks_per_grid * ELEM_SIZE);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

  // Reduction Phase
  ReduceSumParallelKernel<<<blocks_per_grid, threads_per_block>>>(a, reduction_result, reduce_size);
  
  while(blocks_per_grid > 1) {
    int blocks_next = (blocks_per_grid + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
    ReduceSumParallelKernel<<<blocks_next, threads_per_block>>>(reduction_result, reduction_result, blocks_per_grid);
    blocks_per_grid = blocks_next;
  }

  // 代码这个位置在cpu里，所有找不到cuda里的内存地址。会报segment fault。
  // out->ptr = reduction_result[0];
  // 只能用这个指令
  cudaMemcpy(out, reduction_result, ELEM_SIZE, cudaMemcpyDeviceToDevice);
  cudaFree(reduction_result);
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_len) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_len) return;
  scalar_t total = a[gid * reduce_size];
  for (size_t i = gid * reduce_size + 1; i < (gid + 1) * reduce_size; i++) {
    total += a[i];
  }
  out[gid] = total;
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  // DIM_INIT;
  // ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  for(int i = 0; i < out->size; i++) {
    ReduceSumParallel(&a.ptr[i * reduce_size], &out->ptr[i], reduce_size);
  }
  /// END YOUR SOLUTION
}

void VanillaReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  DIM_INIT;
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
}

void VanillaReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  DIM_INIT;
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("ewise_power", EwisePower);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_vanilla", MatmulVanilla);

  m.def("reduce_max", ReduceMax);
  m.def("vanilla_max", VanillaReduceMax);
  m.def("reduce_sum", ReduceSum);
  m.def("vanilla_sum", VanillaReduceSum);
}
