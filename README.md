# parallel-computing
### 问题介绍:
<img width="800" alt="image" src="https://github.com/user-attachments/assets/580112f0-3405-4825-90fa-d0bdf960368d" />

### pre文件:
源代码包括以下文件

a)	pivot.c 源代码文件

b)	uniformvector-2dim-5h.txt 为第一组输入数据文件；uniformvector-4dim-1h.txt为第二组输入文件（运行时间较长）。 

c)	refer-2dim-5h.txt 基准输出文件 

### new文件:通过并行优化计算后的文件

### 优化思路：

#### 算法优化

1. **距离计算的优化**  
   在计算重构距离时，我发现每次调用 `SubDistance` 方法都需要重复计算两点间的实际距离。为了减少重复计算，我引入了一个大小为 \( n \times n \) 的二维表来存储任意两点间的距离。虽然实际只需要一半的空间，但为了编程方便，我使用了完整的表。通过查表的方式，显著减少了计算开销。

2. **切比雪夫距离计算的优化**  
   在计算切比雪夫距离时，我发现距离重复计算结果是相等的。因此，我只需计算一次，最后将结果乘以 2 即可。这一优化将循环次数减少了一半，具体体现在 `j = j + 1` 和 `return chebyshevSum * 2` 的实现中。

3. **排序的提前终止**  
   在排序过程中，由于数据总是被放置在数组末尾，我发现一旦有更优的方案，即可提前终止比较，而不需要继续遍历到` ( a = = 0 )`。这一优化减少了不必要的比较操作。

#### 并行计算

4. **距离计算的并行化**  
   在计算任意两点间的距离时，我发现任务分配是均匀的，每个进程的计算量相同，负载均衡。因此，我通过简单的进程分配实现了并行计算，显著提升了计算效率。

5. **切比雪夫距离和的并行化**  
   由于源程序使用递归计算切比雪夫距离和，而递归本身难以并行化，我通过剥离递归的最外层，使其并行执行内部的递归逻辑。结合 `schedule(dynamic)` 动态分配任务，进一步优化了性能。动态分配特别适合本算法，因为任务的复杂度逐渐降低，静态分配无法很好地平衡负载。

#### 访存优化

6. **访存优化**  
   我对关键函数进行了 `inline` 处理，使得计算结果在访存中的顺序更加连续，从而提高了缓存命中率。虽然我尝试了其他访存优化策略，但效果不如 `inline` 显著，因此最终仅采用了这一优化。

#### 编译器优化

7. **编译器优化**  
   现代编译器已经具备强大的优化能力。我通过启用 `-O3` 优化选项，充分利用了编译器对代码的性能提升，进一步提高了程序的运行效率。
