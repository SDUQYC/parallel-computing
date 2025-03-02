# 编译的方法
gcc pivot.c -O3 -lm -fopenmp -o pivot

# 执行方法
./pivot

# 如何设置线程数
在源代码的163和215行有如下
#pragma omp parallel for num_threads(64) schedule(dynamic)
修改num_threads()中的参数可以改变