#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__
void scan_kernel(int* input, int N, int* result)
{
    __shared__ float temp[THREADS_PER_BLOCK];
    int index = threadIdx.x;
    temp[index] = input[index];
    __syncthreads();

    // upsweep phase
    for (int i = 1; i < N/2; i *= 2) {
        int span = i * 2;
        if (index % span == span - 1)
            temp[index] = temp[index] + temp[index-i];
        __syncthreads();
    }

    if (index == N - 1)
        temp[index] = 0;
    __syncthreads();

    // downsweep phase
    for (int i = N/2; i >= 1; i /= 2) {
        int span = i * 2;
        if (index % span == span - 1) {
            int t = temp[index-i];
            temp[index-i] = temp[index];
            temp[index] += t;
        }
        __syncthreads();
    }

    result[index] = temp[index];
    __syncthreads();
}

__global__
void upsweep(int* arr, int twod)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int twod1 = twod*2;
    int index = threadId * twod1;
    
    arr[index+twod1-1] += arr[index+twod-1];
}

__global__
void downsweep(int* arr, int twod)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int twod1 = twod*2;
    int index = threadId * twod1;

    int temp = arr[index+twod-1];
    arr[index+twod-1] = arr[index+twod1-1];
    arr[index+twod1-1] += temp;
}

inline std::pair<int, int> getParams(int N, int twod) {
    const int threadsPerBlock = THREADS_PER_BLOCK;
    int twod1 = twod * 2;
    int num_threads = N / twod1;
    if (num_threads < threadsPerBlock)
        return std::make_pair(1, num_threads);
    int blocks = num_threads / threadsPerBlock;
    return std::make_pair(blocks, threadsPerBlock);
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.

    N = nextPow2(N);

    // upsweep phase
    for (int twod = 1; twod <= N/2; twod*=2) {
        auto [blocks, threadsPerBlock] = getParams(N, twod);
        upsweep<<<blocks, threadsPerBlock>>>(result, twod);
        cudaDeviceSynchronize();
    }

    // set last element to 0 on device
    cudaMemset(result + (N-1), 0, sizeof(int));

    // downsweep phase
    for (int twod = N/2; twod >= 1; twod /= 2) {
        auto [blocks, threadsPerBlock] = getParams(N, twod);
        downsweep<<<blocks, threadsPerBlock>>>(result, twod);
        cudaDeviceSynchronize();
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


__global__
void mark_repeats_kernel(int* input, int length, int* mark) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < length - 1) {
        if (input[index] == input[index+1])
            mark[index] = 1;
    }
}

__global__
void get_repeat_sequence_kernel(int* input, int length, int* result) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("input[%d]: %d, input[%d]: %d\n", index, input[index], index+1, input[index+1]);
    if (index < length - 1) {
        int count = input[index];
        if (count != input[index+1]) {
            result[count] = index;
        }
    }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    int* mark_arr;
    int rounded_len = nextPow2(length);
    cudaMemset(device_output, 0, rounded_len * sizeof(int));
    cudaMalloc(&mark_arr, rounded_len * sizeof(int));
    cudaMemset(mark_arr, 0, rounded_len * sizeof(int));

    const int threadsPerBlock = THREADS_PER_BLOCK;
    int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    mark_repeats_kernel<<<blocks, threadsPerBlock>>>(device_input, length, mark_arr);
    cudaDeviceSynchronize();

    exclusive_scan(0, rounded_len, mark_arr);
    
    get_repeat_sequence_kernel<<<blocks, threadsPerBlock>>>(mark_arr, length, device_output);
    cudaDeviceSynchronize();

    int ret_host;
    cudaMemcpy(&ret_host, mark_arr + length, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(mark_arr);
    
    return ret_host;
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
