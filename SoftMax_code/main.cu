#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
//check error
cudaError ErrorCheck(cudaError_t status,const char *filename,int linenNumber)
{
    if(status != cudaSuccess)
    {
        printf("CUDA API error:\ncode=%d\nname=%s\ndescription=%s\nfile=%s\nline=%d\n",status,cudaGetErrorName(status),cudaGetErrorString(status),filename,linenNumber);
        return status;
    }
    return status;
}


//核函数，让每个线程处理一行数据
__global__ void SoftMax_Kernal(const float *input_data,float *output_data,int colom,int dim)
{
    int idx = blockDim.x * blockIdx.x +threadIdx.x;
    //设定线程执行范围
    if(idx < colom)
    {
        //先找出每行数据的最大值，再让此行的数据减去最大，防止再进行e^x计算时数据溢出
        float max_val=-1e20f;
        //先让线程指向每行的首个位置
        const float *data=input_data + idx * dim;
        //使用for循环取每行的最大值
        for (int i = 0; i < dim; i++)
        {
            max_val=max(max_val,data[i]);
        }
        //各行数据减去对应最大值,再进行e^x计算,再求和
        float temp_sum=0.0;
        for (int i = 0; i < dim; i++)
        {
            temp_sum+=expf(data[i]-max_val);
        }
        //归一化
        for (int i = 0; i < dim; i++)
        {
            output_data[idx*dim+i]=expf(data[i]-max_val)/temp_sum;
        }
    }
}




// 功能：用 for 循环初始化
void initData(float* data, int rows, int dim) {
    for (int i = 0; i < rows; i++) {      // 遍历每一行
        for (int j = 0; j < dim; j++) {   // 遍历每一列
           data[i*dim+j]=j;
        }
    }
}




int main(void)
{
    int row=3;
    int dim=4;
    //host内存分配
    float *host_data,*host_result;
    host_data=new float[row * dim];
    host_result=new float[row * dim];
    initData(host_data,row,dim);

    //device分配内存
    float *device_data,*device_result;
    cudaMalloc(&device_data,row * dim *sizeof(float));
    cudaMalloc(&device_result,row * dim *sizeof(float));

    //拷贝数据到device
    cudaMemcpy(device_data,host_data,sizeof(float)*row*dim,cudaMemcpyHostToDevice);
    //网格参数
    int blockSize = 256;
    int gridSize = (row + blockSize - 1) / blockSize;
    //启用核函数
    SoftMax_Kernal<<<gridSize,blockSize>>>(device_data,device_result,row,dim);
    cudaDeviceSynchronize();
    //从device拷贝回host
    cudaMemcpy(host_result,device_result,sizeof(float)*row*dim,cudaMemcpyDeviceToHost);
     for (int i = 0; i < row; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%.6f ", host_result[i * dim + j]);
        }
        printf("\n");
    }

    //释放内存
    delete [] host_data;
    delete [] host_result;

    cudaFree(device_data);
    cudaFree(device_result);
    cudaDeviceReset();
    return 0;
}