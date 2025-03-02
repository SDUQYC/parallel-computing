//修改后的
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<omp.h>
#include <immintrin.h>
#include <emmintrin.h>
double inline SumDistance(const int k, const int n, const int dim, double* coord, int* pivots,double* allDistance){
    double* rebuiltCoord = (double*)malloc(sizeof(double) * n * k);
    int i;
    for(i=0; i<n; i++){
        int ki;
        for(ki=0; ki<k; ki++){
            rebuiltCoord[i*k + ki]=allDistance[pivots[ki]*n+i];
        }
    }
    double chebyshevSum = 0;
    for(i=0; i<n; i++){
        int j;
        for(j=i+1; j<n; j++){
            double chebyshev = 0;
            int ki;
            for(ki=0; ki<k; ki++){
                double dis = fabs(rebuiltCoord[i*k + ki] - rebuiltCoord[j*k+ ki]);
                chebyshev = dis>chebyshev ? dis : chebyshev;
            }
            chebyshevSum += chebyshev;
        }
    }
    free(rebuiltCoord);
    return chebyshevSum*2;
}
void inline Combination(int ki, const int k, const int n, const int dim, const int M, double* coord, int* pivots,double* maxDistanceSum, int* maxDisSumPivots, double* minDistanceSum, int* minDisSumPivots,double* allDistance){
    if(ki==k-1){
        int i;
        for(i=pivots[ki-1]+1; i<n; i++){
            pivots[ki] = i;
            // Calculate sum of distance while combining different pivots.
            //计算组合不同枢轴时的距离之和。
            double distanceSum = SumDistance(k, n, dim, coord, pivots,allDistance);
            // put data at the end of array
            //将数据放在数组末尾
#pragma omp critical
            {
            maxDistanceSum[M] = distanceSum;
            minDistanceSum[M] = distanceSum;
            int kj;
            for(kj=0; kj<k; kj++){
                maxDisSumPivots[M*k + kj] = pivots[kj];
            }
            for(kj=0; kj<k; kj++){
                minDisSumPivots[M*k + kj] = pivots[kj];
            }
            // sort
            int a;
            for(a=M; a>0; a--){
                if(maxDistanceSum[a] > maxDistanceSum[a-1]){
                    double temp = maxDistanceSum[a];
                    maxDistanceSum[a] = maxDistanceSum[a-1];
                    maxDistanceSum[a-1] = temp;
                    int kj;
                    for(kj=0; kj<k; kj++){
                        int temp = maxDisSumPivots[a*k + kj];
                        maxDisSumPivots[a*k + kj] = maxDisSumPivots[(a-1)*k + kj];
                        maxDisSumPivots[(a-1)*k + kj] = temp;
                    }
                }else
                {
                    break;
                }
            }
            for(a=M; a>0; a--){
                if(minDistanceSum[a] < minDistanceSum[a-1]){
                    double temp = minDistanceSum[a];
                    minDistanceSum[a] = minDistanceSum[a-1];
                    minDistanceSum[a-1] = temp;
                    int kj;
                    for(kj=0; kj<k; kj++){
                        int temp = minDisSumPivots[a*k + kj];
                        minDisSumPivots[a*k + kj] = minDisSumPivots[(a-1)*k + kj];
                        minDisSumPivots[(a-1)*k + kj] = temp;
                    }
                }else
                {
                    break;
                }
            }
        }
        }
        return;
    }

    // Recursively call Combination() to combine pivots
    //递归调用Combination（）以组合枢轴
    int i;
    for(i=pivots[ki-1]+1; i<n; i++) {
        pivots[ki] = i;
        Combination(ki+1, k, n, dim, M, coord, pivots, maxDistanceSum, maxDisSumPivots, minDistanceSum, minDisSumPivots,allDistance);
    }
}
// }
int main(int argc, char* argv[]){
    // filename : input file namespace
    char* filename = (char*)"uniformvector-2dim-5h.txt";
    if( argc==2 ) {
        filename = argv[1];
    }  else if(argc != 1) {
        printf("Usage: ./pivot <filename>\n");
        return -1;
    }
    // M : number of combinations to store
    //M：要存储的组合数
    const int M = 1000;
    // dim : dimension of metric space
    //dim：公制空间的尺寸
    int dim;
    // n : number of points
    int n;
    // k : number of pivots
    int k;

    // Read parameter
    //读取参数
    FILE* file = fopen(filename, "r");
    if( file == NULL ) {
        printf("%s file not found.\n", filename);
        return -1;
    }
    fscanf(file, "%d", &dim);
    fscanf(file, "%d", &n);
    fscanf(file, "%d", &k);
    printf("dim = %d, n = %d, k = %d\n", dim, n, k);
    // printf("The output may be slow, please wait patiently");
    // Start timing
    //开始计时
    struct timeval start;

    // Read Data
    //读取数据
    double* coord = (double*)malloc(sizeof(double) * dim * n);
    int i;
    for(i=0; i<n; i++){
        int j;
        for(j=0; j<dim; j++){
            fscanf(file, "%lf", &coord[i*dim + j]);
        }
    }
    fclose(file);
    gettimeofday(&start, NULL);
    // double* allDistance=(double*)malloc(sizeof(double) * n * n);
    // AllDistanceconst(k,n,dim,coord,allDistance);
    // for (i = 0; i <n*n; i++)
    // {
    //     /* code */
    //     printf("\n%d ",i);
    //     printf("\n%d ",allDistance[i]);
    // }
    // int * depth=(int*)malloc(sizeof(int));
    // *depth=0;
    //为所有距离分配空间 相当于有一个n*n的矩阵来存放任意两点间的距离
    double* allDistance = (double*)malloc(sizeof(double)* n * n );
    //这里负载均衡，我认为不需要使用schedule但使用stastic，1可能性能会更好一丢丢
#pragma omp parallel for num_threads(64) schedule(static,1)
    for(i=0; i<n; i++){
        int j=0;
        for ( j = 0; j < n; j++)
        {
            double distance=0;
            int k;
            for(k=0; k<dim; k++){
                distance += pow(coord[i*dim + k] - coord[j*dim + k] ,2);
            }
            allDistance[i*n+j]=sqrt(distance);
            
        }   
    }
    // maxDistanceSum : the largest M distance sum
    //maxDistanceSum：最大M距离总和
    double* maxDistanceSum = (double*)malloc(sizeof(double) * (M+1));
    for(i=0; i<M; i++){
        maxDistanceSum[i] = 0;
    }
    // maxDisSumPivots : the top M pivots combinations
    //maxDisSumPivots：前M个枢轴组合
    int* maxDisSumPivots = (int*)malloc(sizeof(int) * k * (M+1));
    for(i=0; i<M; i++){
        int ki;
        for(ki=0; ki<k; ki++){
            maxDisSumPivots[i*k + ki] = 0;
        }
    }
    // minDistanceSum : the smallest M distance sum
    //minDistanceSum：最小的M距离和
    double* minDistanceSum = (double*)malloc(sizeof(double) * (M+1));
    for(i=0; i<M; i++){
        minDistanceSum[i] = __DBL_MAX__;
    }
    // minDisSumPivots : the bottom M pivots combinations
    //minDisSumPivots：底部M个枢轴组合
    int* minDisSumPivots = (int*)malloc(sizeof(int) * k * (M+1));
    for(i=0; i<M; i++){
        int ki;
        for(ki=0; ki<k; ki++){
            minDisSumPivots[i*k + ki] = 0;
        }
    }

    // temp : indexes of pivots with dummy array head
    //temp:带有虚拟阵列头的枢轴索引
    int* temp = (int*)malloc(sizeof(int) * (k+1));
    temp[0] = -1;

    // Main loop. Combine different pivots with recursive function and evaluate them. Complexity : O( n^(k+2) )
    //主回路。将不同的枢轴与递归函数相结合，并对其进行求值。复杂性：O（n^（k+2））
    #pragma omp parallel for num_threads(64) schedule(dynamic)
    for ( i = 0; i < n; i++)
    {
        // printf("%d\n",i);
        int* pivots = (int*)malloc(sizeof(int) * k);
        pivots[0]=i;
        Combination(1, k, n, dim, M, coord, pivots, maxDistanceSum, maxDisSumPivots, minDistanceSum, minDisSumPivots,allDistance);
        free(pivots);
    }

    // End timing
    //结束计时
    struct timeval end;
    gettimeofday (&end, NULL);
    printf("Using time : %f ms\n", (end.tv_sec-start.tv_sec)*1000.0+(end.tv_usec-start.tv_usec)/1000.0);

    // Store the result
    //存储结果
    FILE* out = fopen("result.txt", "w");
    for(i=0; i<M; i++){
        int ki;
        for(ki=0; ki<k-1; ki++){
            fprintf(out, "%d ", maxDisSumPivots[i*k + ki]);
        }
        fprintf(out, "%d\n", maxDisSumPivots[i*k + k-1]);
    }
    for(i=0; i<M; i++){
        int ki;
        for(ki=0; ki<k-1; ki++){
            fprintf(out, "%d ", minDisSumPivots[i*k + ki]);
        }
        fprintf(out, "%d\n", minDisSumPivots[i*k + k-1]);
    }
    fclose(out);

    // Log
    int ki;
    printf("max : ");
    for(ki=0; ki<k; ki++){
        printf("%d ", maxDisSumPivots[ki]);
    }
    printf("%lf\n", maxDistanceSum[0]);
    printf("min : ");
    for(ki=0; ki<k; ki++){
        printf("%d ", minDisSumPivots[ki]);
    }
    printf("%lf\n", minDistanceSum[0]);

}
