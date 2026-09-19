# 中文

*感谢Umar Jamil的开源课程。*

*tips：这里对一些和主题无关的部分采取了简化实现，例如：QKV三者的维度是相同的；把张量计算视为矩阵计算；头部仅仅是把原先的矩阵拆分了*

## PART 1：算法

1. 在现代GPU中，矩阵乘法已经得到了充分的优化；而Attention(Q, K, V)却是一个I/O受限的操作，原因是在当时的PyTorch实现中，Attention需要频繁访问HBM；故，要想办法分块（tiling）地去计算Attention，使得其能被放入共享内存中。

2. （注意矩阵的softmax是行归一化操作，**顺便问一句，为什么是行归一化不是列归一化，难道只是随机选择的吗？**）为了不让softmax的指数项数值溢出，我们给每个指数都减去一个行向量最大分量，或者说减去无穷维范数。这样的朴素softmax的时间和空间复杂度都为O(N)，并且必须串行地遍历输入向量三次（一次为了找最大分量，一次计算分母，再一次计算分子）。朴素softmax的伪代码形式如下：

$$\begin{aligned} & m_0 = -\infty \\ & \text{\textbf{for }} i = 1 \text{ \textbf{to} } N \\ & \quad m_i = \max(m_{i-1}, x_i) \\ & l_0 = 0 \\ & \text{\textbf{for }} j = 1 \text{ \textbf{to} } N \\ & \quad l_j = l_{j-1} + e^{x_j - m_N} \\ & \text{\textbf{for }} k = 1 \text{ \textbf{to} } N \\ & \quad x_k \leftarrow \frac{e^{x_k - m_N}}{l_N} \end{aligned}$$

3. 我们尝试把前两次for循环运用贪心算法整合起来，把全局最大值用当前搜索到的局部最大值代替，然后再乘以修正因子，这样就是在线softmax算法。在线softmax的伪代码形式如下：

$$\begin{aligned} & m_0 = -\infty \\ & l_0 = 0 \\ & \text{\textbf{for }} i = 1 \text{ \textbf{to} } N \\ & \quad m_i = \max(m_{i-1}, x_i) \\ & \quad l_i = l_{i-1} \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i} \\ & \text{\textbf{for }} k = 1 \text{ \textbf{to} } N \\ & \quad x_k \leftarrow \frac{e^{x_k - m_N}}{l_N} \end{aligned}$$

4. 考虑并行性，我们可以引入分块矩阵乘法；根据分块矩阵乘法的性质，我们可以把前面讨论中的标量也视作分块矩阵。**然后我们会发现一个问题！** 在分块之前，softmax指数减去的最大值是每一行的最大值；但是分块之后，却变成了减每一个“块行”的最大值！因此我们需要进一步优化。为了方便讨论，定义 softmax<sup>*</sup> 为 softmax 的分子，也就是不做归一化处理的softmax。优化的思路基本就是，把上述的在线softmax算法改写成矩阵的形式。伪代码如下：(和上面的在线softmax的逻辑完全相同，只是改用了便于并行化的分块矩阵形式)

**初始化**

（略）

**STEP 1**

$$S_1 = Q_1 K_1^T$$

$$m_1 = \max\left(\text{rowmax}(Q_1 K_1^T), m_0\right)$$

$$l_1 = \text{rowsum}\left[\exp(S_1 - m_1)\right] + l_0 \cdot \exp(m_0 - m_1)$$

$$P_{11} = \exp(S_1 - m_1)$$

$$O_1 = \text{diag}(\exp(m_0 - m_1)) O_0 + P_{11} V_1$$

$$(O_1 = \begin{bmatrix} \exp(m_0 - m_1)_1 & 0 \\ 0 & \exp(m_0 - m_1)_2 \end{bmatrix} \times \begin{bmatrix} O_{11} & O_{12} & \dots & O_{1,128} \\ O_{21} & O_{22} & \dots & O_{2,128} \end{bmatrix} )$$

---

**STEP 2**

$$S_{12} = Q_1 K_2^T$$

$$m_2 = \max\left(\text{rowmax}(Q_1 K_2^T), m_1\right)$$

$$S_2 = Q_1 K_2^T$$

$$l_2 = \text{rowsum}\left[\exp(S_2 - m_2)\right] + l_1 \cdot \exp(m_1 - m_2)$$

$$P_{12} = \exp(S_2 - m_2)$$

$$O_2 = \text{diag}(\exp(m_1 - m_2)) O_1 + P_{12} V_2$$

---

**STEP 3**

**......（同上）**

---

**STEP FINAL**

$$O_{final} = \text{diag}(l_N)^{-1} O_N$$


5. 以上内容就是FlashAttention-2的前向传播算法。

## PART 2：系统

1. CUDA: GPU擅长计算而不擅长控制。在CUDA C中，程序员应该手动指定每个线程应该做什么。CUDA会为每个线程分配一个索引，并且每次启动的线程数量一定是32的倍数（其实就是一个warp有32个线程）。并且这些线程共享一个控制单元。（**是一个warp共享一个还是每次启动的线程共享一个？**）如果64个线程进入一个if，但是只有40个判定为真，那么剩下24个线程将会进入一个空的for循环。由于GPU的计算单元虽然很多但是也是有限的，所以当它要处理一个非常大的矩阵张量时，这个张量会被划分成一个一个块（block），块ID和每个块里线程ID也都需要手动指定。有一些繁琐的规则，略。（我现在差不多知道为什么Percy Liang说写CUDA真的很麻烦了，hh）

2. 张量布局（Tensor Logouts）：GPU矩阵的处理方式与C相同，也就是只提供那个指向第一个元素的指针，也就是说不能直接进行广播等高级操作。定义shape为形状，stride为从一个元素到这个维度上的下一个元素的距离（具体是什么，读者可以在numpy里自己试一试numpy.ndarray.shape和numpy.ndarray.strides是什么样子的）。只需要shape和stride信息（默认采用行主序布局）就可以很方便地实现reshape或者转置transpose，因为只需要改变shape和stride就可以了，不用改变内存中的元素的位置。

3. Triton: 在Triton中，每次启动只需要指定线程块，而不需要指定线程块中的具体线程。也就是说，CUDA以线程为基本单位，而Triton以块为基本单位。