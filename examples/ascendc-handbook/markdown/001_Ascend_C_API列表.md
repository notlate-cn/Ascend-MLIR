<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0003.html -->
<!-- 下载时间: 2026-03-04 11:39:01 -->

# Ascend C API列表

# Ascend C API列表

Ascend C提供一组类库API，开发者使用标准C++语法和类库API进行编程。Ascend C编程类库API示意图如下所示，分为： 

  * **基础数据结构** ：kernel API中使用到的基础数据结构，比如GlobalTensor和LocalTensor。
  * **语言扩展层 C API** ：开放芯片完备编程能力，支持数组分配内存，一般基于指针编程，提供与业界一致的C语言编程体验。
  * **基础API** ：实现对硬件能力的抽象，开放芯片的能力，保证完备性和兼容性。标注为ISASI（Instruction Set Architecture Special Interface，硬件体系结构相关的接口）类别的API，不能保证跨硬件版本兼容。
  * **高阶API** ：实现一些常用的计算算法，用于提高编程开发效率，通常会调用多种基础API实现。高阶API包括数学库、Matmul、Softmax等API。高阶API可以保证兼容性。
  * **Utils API（公共辅助函数）** ：丰富的通用工具类，涵盖标准库、平台信息获取、运行时编译及日志输出等功能，支持开发者高效实现算子开发与性能优化。



![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552082647.png)

#### 基础数据结构

表1 基础数据结构列表

展开

接口名 |  功能描述  
---|---  
[LocalTensor](atlasascendc_api_07_0006.html) |  LocalTensor用于存放AI Core中Local Memory（内部存储）的数据，支持逻辑位置[TPosition](atlasascendc_api_07_0174.html)为VECIN、VECOUT、VECCALC、A1、A2、B1、B2、CO1、CO2。  
[GlobalTensor](atlasascendc_api_07_0007.html) |  GlobalTensor用来存放Global Memory（外部存储）的全局数据。  
[Coordinate](atlasascendc_api_07_00143.html) |  Coordinate本质上是一个元组（tuple），用于表示张量在不同维度的位置信息，即坐标值。  
[Layout](atlasascendc_api_07_00076.html) |  Layout<Shape, Stride>数据结构是描述多维张量内存布局的基础模板类，通过编译时的形状（Shape）和步长（Stride）信息，实现逻辑坐标空间到一维内存地址空间的映射，为复杂张量操作和硬件优化提供基础支持。  
[TensorTrait](atlasascendc_api_07_0011.html) |  TensorTrait数据结构是描述Tensor相关信息的基础模板类，包含Tensor的数据类型、逻辑位置和Layout内存布局。  
  
#### 基础API

表2 Memory数据搬运API列表

展开

接口名 |  功能描述  
---|---  
[DataCopy](atlasascendc_api_07_0101.html) |  数据搬运接口，包括普通数据搬运、增强数据搬运、切片数据搬运、随路格式转换。  
[Copy](atlasascendc_api_07_0106.html) |  VECIN、VECCALC、VECOUT之间的搬运指令，支持mask操作和DataBlock间隔操作。  
  
表3 Memory矢量计算API列表

展开

分类 |  接口名 |  功能描述  
---|---|---  
基础算术 |  [Exp](atlasascendc_api_07_0025.html) |  按元素取自然指数。  
[Ln](atlasascendc_api_07_0026.html) |  按元素取自然对数。  
[Abs](atlasascendc_api_07_0027.html) |  按元素取绝对值。  
[Reciprocal](atlasascendc_api_07_0028.html) |  按元素取倒数。  
[Sqrt](atlasascendc_api_07_0029.html) |  按元素做开方。  
[Rsqrt](atlasascendc_api_07_0030.html) |  按元素做开方后取倒数。  
[Relu](atlasascendc_api_07_0032.html) |  按元素做线性整流Relu。  
[Add](atlasascendc_api_07_0035.html) |  按元素求和。  
[Sub](atlasascendc_api_07_0036.html) |  按元素求差。  
[Mul](atlasascendc_api_07_0037.html) |  按元素求积。  
[Div](atlasascendc_api_07_0038.html) |  按元素求商。  
[Max](atlasascendc_api_07_0039.html) |  按元素求最大值。  
[Min](atlasascendc_api_07_0040.html) |  按元素求最小值。  
[Adds](atlasascendc_api_07_0054.html) |  矢量内每个元素与标量求和。  
[Muls](atlasascendc_api_07_0055.html) |  矢量内每个元素与标量求积。  
[Maxs](atlasascendc_api_07_0056.html) |  源操作数矢量内每个元素与标量相比，如果比标量大，则取源操作数值，比标量的值小，则取标量值。  
[Mins](atlasascendc_api_07_0057.html) |  源操作数矢量内每个元素与标量相比，如果比标量大，则取标量值，比标量的值小，则取源操作数值。  
[LeakyRelu](atlasascendc_api_07_0060.html) |  按元素做带泄露线性整流Leaky ReLU。  
逻辑计算 |  [Not](atlasascendc_api_07_0031.html) |  按元素做按位取反。  
[And](atlasascendc_api_07_0041.html) |  针对每对元素执行按位与运算。  
[Or](atlasascendc_api_07_0042.html) |  针对每对元素执行按位或运算。  
[ShiftLeft](atlasascendc_api_07_0058.html) |  对源操作数中的每个元素进行左移操作，左移的位数由输入参数scalarValue决定。  
[ShiftRight](atlasascendc_api_07_0059.html) |  对源操作数中的每个元素进行右移操作，右移的位数由输入参数scalarValue决定。  
复合计算 |  [Axpy](atlasascendc_api_07_0063.html) |  源操作数中每个元素与标量求积后和目的操作数中的对应元素相加。  
[CastDequant](atlasascendc_api_07_0074.html) |  对输入做量化并进行精度转换。  
[AddRelu](atlasascendc_api_07_0043.html) |  按元素求和，结果和0对比取较大值。  
[AddReluCast](atlasascendc_api_07_0044.html) |  按元素求和，结果和0对比取较大值，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。  
[AddDeqRelu](atlasascendc_api_07_0045.html) |  依次计算按元素求和、结果进行deq量化后再进行relu计算（结果和0对比取较大值）。  
[SubRelu](atlasascendc_api_07_0046.html) |  按元素求差，结果和0对比取较大值。  
[SubReluCast](atlasascendc_api_07_0047.html) |  按元素求差，结果和0对比取较大值，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。  
[MulAddDst](atlasascendc_api_07_0048.html) |  按元素将src0Local和src1Local相乘并和dstLocal相加，将最终结果存放进dstLocal中。  
[MulCast](atlasascendc_api_07_0049.html) |  按元素求积，并根据源操作数和目的操作数Tensor的数据类型进行精度转换。  
[FusedMulAdd](atlasascendc_api_07_0050.html) |  按元素将src0Local和dstLocal相乘并加上src1Local，最终结果存放入dstLocal。  
[MulAddRelu](atlasascendc_api_07_0051.html) |  按元素将src0Local和dstLocal相乘并加上src1Local，将结果和0作比较，取较大值，最终结果存放进dstLocal中。  
比较与选择 |  [Compare](atlasascendc_api_07_0066.html) |  逐元素比较两个tensor大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。  
[Compare（结果存放入寄存器）](atlasascendc_api_07_0067.html) |  逐元素比较两个tensor大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。Compare接口需要mask参数时，可以使用此接口。计算结果存放入寄存器中。  
[Compares](atlasascendc_api_07_0068.html) |  逐元素比较一个tensor中的元素和另一个Scalar的大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。  
[Select](atlasascendc_api_07_0070.html) |  给定两个源操作数src0和src1，根据selMask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst。选择的规则为：当selMask的比特位是1时，从src0中选取，比特位是0时从src1选取。  
[GatherMask](atlasascendc_api_07_0071.html) |  以内置固定模式对应的二进制或者用户自定义输入的Tensor数值对应的二进制为gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中。  
精度转换指令 |  [Cast](atlasascendc_api_07_0073.html) |  根据源操作数和目的操作数Tensor的数据类型进行精度转换。  
归约计算 |  [ReduceMax](atlasascendc_api_07_0076.html) |  在所有的输入数据中找出最大值及最大值对应的索引位置。  
[ReduceMin](atlasascendc_api_07_0077.html) |  在所有的输入数据中找出最小值及最小值对应的索引位置。  
[ReduceSum](atlasascendc_api_07_0078.html) |  对所有的输入数据求和。  
[WholeReduceMax](atlasascendc_api_07_0079.html) |  每个repeat内所有数据求最大值以及其索引index。  
[WholeReduceMin](atlasascendc_api_07_0080.html) |  每个repeat内所有数据求最小值以及其索引index。  
[WholeReduceSum](atlasascendc_api_07_0081.html) |  每个repeat内所有数据求和。  
[BlockReduceMax](atlasascendc_api_07_0082.html) |  对每个repeat内所有元素求最大值。  
[BlockReduceMin](atlasascendc_api_07_0083.html) |  对每个repeat内所有元素求最小值。  
[BlockReduceSum](atlasascendc_api_07_0084.html) |  对每个repeat内所有元素求和。源操作数相加采用二叉树方式，两两相加。  
[PairReduceSum](atlasascendc_api_07_0085.html) |  PairReduceSum：相邻两个（奇偶）元素求和。  
[RepeatReduceSum](atlasascendc_api_07_0086.html) |  每个repeat内所有数据求和。和[WholeReduceSum](atlasascendc_api_07_0081.html)接口相比，不支持mask逐bit模式。建议使用功能更全面的[WholeReduceSum](atlasascendc_api_07_0081.html)接口。  
数据转换 |  [Transpose](atlasascendc_api_07_0199.html) |  可实现16*16的二维矩阵数据块的转置和[N,C,H,W]与[N,H,W,C]互相转换。  
[TransDataTo5HD](atlasascendc_api_07_0200.html) |  数据格式转换，一般用于将NCHW格式转换成NC1HWC0格式。特别的，也可以用于二维矩阵数据块的转置。  
数据填充 |  [Duplicate](atlasascendc_api_07_0088.html) |  将一个变量或一个立即数，复制多次并填充到向量。  
[Brcb](atlasascendc_api_07_0089.html) |  给定一个输入张量，每一次取输入张量中的8个数填充到结果张量的8个datablock（32Bytes）中去，每个数对应一个datablock。  
[CreateVecIndex](atlasascendc_api_07_0090.html) |  以firstValue为起始值创建向量索引。  
数据分散/数据收集 |  [Gather](atlasascendc_api_07_0092.html) |  给定输入的张量和一个地址偏移张量，Gather指令根据偏移地址将输入张量按元素收集到结果张量中。  
掩码操作 |  [SetMaskCount](atlasascendc_api_07_0094.html) |  设置mask模式为Counter模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入计算数据量，实际迭代次数由Vector计算单元自动推断。  
[SetMaskNorm](atlasascendc_api_07_0095.html) |  设置mask模式为Normal模式。该模式为系统默认模式，支持开发者配置迭代次数。  
[SetVectorMask](atlasascendc_api_07_0096.html) |  用于在矢量计算时设置mask。  
[ResetMask](atlasascendc_api_07_0097.html) |  恢复mask的值为默认值（全1），表示矢量计算中每次迭代内的所有元素都将参与运算。  
量化设置 |  [SetDeqScale](atlasascendc_api_07_0099.html) |  设置DEQSCALE寄存器的值。  
  
表4 标量计算API列表

展开

接口名 |  功能描述  
---|---  
[GetBitCount](atlasascendc_api_07_0016.html) |  获取一个uint64_t类型数字的二进制中0或者1的个数。  
[CountLeadingZero](atlasascendc_api_07_0017.html) |  计算一个uint64_t类型数字前导0的个数（二进制从最高位到第一个1一共有多少个0）。  
[Cast（float转half、int32_t）](atlasascendc_api_07_0018.html) |  将一个scalar的类型转换为指定的类型。  
[CountBitsCntSameAsSignBit](atlasascendc_api_07_0019.html) |  计算一个uint64_t类型数字的二进制中，从最高数值位开始与符号位相同的连续比特位的个数。  
[GetSFFValue](atlasascendc_api_07_0020.html) |  获取一个uint64_t类型数字的二进制中第一个0或1出现的位置。  
[Cast（float转bfloat16_t）](atlasascendc_api_07_0021.html) |  float类型标量数据转换成bfloat16_t类型标量数据。  
[Cast（多类型转float）](atlasascendc_api_07_0022.html) |  bfloat16_t类型标量数据转换成float类型标量数据。  
  
表5 资源管理API列表

展开

接口名 |  功能描述  
---|---  
[TPipe](atlasascendc_api_07_0108.html) |  TPipe是用来管理全局内存等资源的框架。通过TPipe类提供的接口可以完成内存等资源的分配管理操作。  
[GetTPipePtr](atlasascendc_api_07_0120.html) |  获取框架当前管理全局内存的TPipe指针，用户获取指针后，可进行TPipe相关的操作。  
[TBufPool](atlasascendc_api_07_0121.html) |  TPipe可以管理全局内存资源，而TBufPool可以手动管理或复用Unified Buffer/L1 Buffer物理内存，主要用于多个stage计算中Unified Buffer/L1 Buffer物理内存不足的场景。  
[TQue](atlasascendc_api_07_0136.html) |  提供入队出队等接口，通过队列（Queue）完成任务间同步。  
[TQueBind](atlasascendc_api_07_0146.html) |  TQueBind绑定源逻辑位置和目的逻辑位置，根据源位置和目的位置，来确定内存分配的位置 、插入对应的同步事件，帮助开发者解决内存分配和管理、同步等问题。  
[TBuf](atlasascendc_api_07_0160.html) |  使用Ascend C编程的过程中，可能会用到一些临时变量。这些临时变量占用的内存可以使用TBuf数据结构来管理。  
[InitSpmBuffer](atlasascendc_api_07_0166.html) |  初始化SPM Buffer。  
[WriteSpmBuffer](atlasascendc_api_07_0167.html) |  将需要溢出暂存的数据拷贝到SPM Buffer中。  
[ReadSpmBuffer](atlasascendc_api_07_0168.html) |  从SPM Buffer读回到local数据中。  
[GetUserWorkspace](atlasascendc_api_07_0172.html) |  获取用户使用的workspace指针。  
[SetSysWorkSpace](atlasascendc_api_07_0171.html) |  在进行融合算子编程时，由于框架通信机制需要使用到workspace，也就是系统workspace，所以在该场景下，开发者要调用该接口，设置系统workspace的指针。  
[GetSysWorkSpacePtr](atlasascendc_api_07_0170.html) |  获取系统workspace指针。  
  
表6 同步控制API列表

展开

接口名 |  功能描述  
---|---  
[TQueSync](atlasascendc_api_07_0178.html) |  TQueSync类提供同步控制接口，开发者可以使用这类API来自行完成同步控制。  
[IBSet](atlasascendc_api_07_0202.html) |  当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。调用IBSet设置某一个核的标志位，与IBWait成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。  
[IBWait](atlasascendc_api_07_0203.html) |  当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。IBWait与IBSet成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。  
[SyncAll](atlasascendc_api_07_0204.html) |  当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。目前多核同步分为硬同步和软同步，硬件同步是利用硬件自带的全核同步指令由硬件保证多核同步，软件同步是使用软件算法模拟实现。  
[InitDetermineComputeWorkspace](atlasascendc_api_07_0206.html) |  初始化GM共享内存的值，完成初始化后才可以调用[WaitPreBlock](atlasascendc_api_07_0207.html)和[NotifyNextBlock](atlasascendc_api_07_0208.html)。  
[WaitPreBlock](atlasascendc_api_07_0207.html) |  通过读GM地址中的值，确认是否需要继续等待，当GM的值满足当前核的等待条件时，该核即可往下执行，进行下一步操作。  
[NotifyNextBlock](atlasascendc_api_07_0208.html) |  通过写GM地址，通知下一个核当前核的操作已完成，下一个核可以进行操作。  
[SetNextTaskStart](atlasascendc_api_07_00087.html) |  在SuperKernel的子Kernel中调用，调用后的指令可以和后续其他的子Kernel实现并行，提升整体性能。  
[WaitPreTaskEnd](atlasascendc_api_07_00088.html) |  在SuperKernel的子Kernel中调用，调用前的指令可以和前序其他的子Kernel实现并行，提升整体性能。  
  
表7 缓存处理API列表

展开

接口名 |  功能描述  
---|---  
[DataCachePreload](atlasascendc_api_07_0176.html) |  从源地址所在的特定DDR地址预加载数据到data cache中。  
[DataCacheCleanAndInvalid](atlasascendc_api_07_0177.html) |  该接口用来刷新Cache，保证Cache的一致性。  
  
表8 系统变量访问API列表

展开

接口名 |  功能描述  
---|---  
[GetBlockNum](atlasascendc_api_07_0184.html) |  获取当前任务配置的Block数，用于代码内部的多核逻辑控制等。  
[GetBlockIdx](atlasascendc_api_07_0185.html) |  获取当前core的index，用于代码内部的多核逻辑控制及多核偏移量计算等。  
[GetDataBlockSizeInBytes](atlasascendc_api_07_0186.html) |  获取当前芯片版本一个datablock的大小，单位为byte。开发者根据datablock的大小来计算API指令中待传入的repeatTime 、DataBlock Stride、Repeat Stride等参数值。  
[GetArchVersion](atlasascendc_api_07_0187.html) |  获取当前AI处理器架构版本号。  
[InitSocState](atlasascendc_api_07_00094.html) |  由于AI Core上存在一些全局状态，如原子累加状态、Mask模式等，在实际运行中，这些值可以被前序执行的算子修改而导致计算出现不符合预期的行为，在[静态Tensor编程](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_00019.html)的场景中用户必须在Kernel入口处调用此函数来初始化AI Core状态 。  
  
表9 原子操作接口列表

展开

接口名 |  功能描述  
---|---  
[SetAtomicAdd](atlasascendc_api_07_0210.html) |  设置接下来从VECOUT到GM，L0C到GM，L1到GM的数据传输是否进行原子累加，可根据参数不同设定不同的累加数据类型。  
[SetAtomicType](atlasascendc_api_07_0211.html) |  通过设置模板参数来设定原子操作不同的数据类型。  
[DisableDmaAtomic](atlasascendc_api_07_0212.html) |  原子操作函数，清空原子操作的状态。  
  
表10 调试接口列表

展开

接口名 |  功能描述  
---|---  
[DumpTensor](atlasascendc_api_07_0192.html) |  基于算子工程开发的算子，可以使用该接口Dump指定Tensor的内容。  
[printf](atlasascendc_api_07_0193.html) |  基于算子工程开发的算子，可以使用该接口实现CPU侧/NPU侧调试场景下的格式化输出功能。  
[ascendc_assert](atlasascendc_api_07_00168.html) |  ascendc_assert提供了一种在CPU/NPU域实现断言功能的接口。当断言条件不满足时，系统会输出断言信息并格式化打印在屏幕上。  
[assert](atlasascendc_api_07_0194.html) |  基于算子工程开发的算子，可以使用该接口实现CPU/NPU域assert断言功能。  
[DumpAccChkPoint](atlasascendc_api_07_0195.html) |  基于算子工程开发的算子，可以使用该接口Dump指定Tensor的内容。该接口可以支持指定偏移位置的Tensor打印。  
[PrintTimeStamp](atlasascendc_api_07_00002.html) |  提供时间戳打点功能，用于在算子Kernel代码中标记关键执行点。  
[Trap](atlasascendc_api_07_0196.html) |  当软件产生异常后，使用该指令使kernel中止运行。  
[GmAlloc](atlasascendc_api_07_1207.html) |  进行核函数的CPU侧运行验证时，用于创建共享内存：在/tmp目录下创建一个共享文件，并返回该文件的映射指针。  
[ICPU_RUN_KF](atlasascendc_api_07_1208.html) |  进行核函数的CPU侧运行验证时，CPU调测总入口，完成CPU侧的算子程序调用。  
[ICPU_SET_TILING_KEY](atlasascendc_api_07_1209.html) |  用于指定本次CPU调测使用的tilingKey。调测执行时，将只执行算子核函数中该tilingKey对应的分支。  
[GmFree](atlasascendc_api_07_1210.html) |  进行核函数的CPU侧运行验证时，用于释放通过GmAlloc申请的共享内存。  
[SetKernelMode](atlasascendc_api_07_1211.html) |  CPU调测时，设置内核模式为单AIV模式，单AIC模式或者MIX模式，以分别支持单AIV矢量算子，单AIC矩阵算子，MIX混合算子的CPU调试。  
[TRACE_START](atlasascendc_api_07_1212.html) |  通过CAModel进行算子性能仿真时，可对算子任意运行阶段打点，从而分析不同指令的流水图，以便进一步性能调优。 用于表示起始位置打点，一般与[TRACE_STOP](atlasascendc_api_07_1213.html)配套使用。  
[TRACE_STOP](atlasascendc_api_07_1213.html) |  通过CAModel进行算子性能仿真时，可对算子任意运行阶段打点，从而分析不同指令的流水图，以便进一步性能调优。 用于表示终止位置打点，一般与[TRACE_START](atlasascendc_api_07_1212.html)配套使用。  
[MetricsProfStart](atlasascendc_api_07_1214.html) |  用于设置性能数据采集信号启动，和MetricsProfStop配合使用。使用[msProf](/document/detail/zh/CANNCommunityEdition/900beta1/devaids/optool/atlasopdev_16_0081.html)工具进行算子上板调优时，可在kernel侧代码段前后分别调用MetricsProfStart和MetricsProfStop来指定需要调优的代码段范围。  
[MetricsProfStop](atlasascendc_api_07_1215.html) |  设置性能数据采集信号停止，和MetricsProfStart配合使用。使用[msProf](/document/detail/zh/CANNCommunityEdition/900beta1/devaids/optool/atlasopdev_16_0081.html)工具进行算子上板调优时，可在kernel侧代码段前后分别调用MetricsProfStart和MetricsProfStop来指定需要调优的代码段范围。  
  
表11 工具函数接口列表

展开

接口名 |  功能描述  
---|---  
[Async](atlasascendc_api_07_00163.html) |  Async提供了一个统一的接口，用于在不同模式下（AIC或AIV）执行特定函数，从而避免代码中直接的硬件条件判断（如使用ASCEND_IS_AIV或ASCEND_IS_AIC）。  
[GetTaskRatio](atlasascendc_api_07_0188.html) |  适用于Cube/Vector分离模式，用来获取Cube/Vector的配比。  
  
表12 Kernel Tiling接口列表

展开

接口名 |  功能描述  
---|---  
[GET_TILING_DATA](atlasascendc_api_07_0214.html) |  用于获取算子kernel入口函数传入的tiling信息，并填入注册的Tiling结构体中，此函数会以宏展开的方式进行编译。如果用户注册了多个TilingData结构体，使用该接口返回默认注册的结构体。  
[GET_TILING_DATA_WITH_STRUCT](atlasascendc_api_07_0215.html) |  使用该接口指定结构体名称，可获取指定的tiling信息，并填入对应的Tiling结构体中，此函数会以宏展开的方式进行编译。  
[GET_TILING_DATA_MEMBER](atlasascendc_api_07_0216.html) |  用于获取tiling结构体的成员变量。  
[TILING_KEY_IS](atlasascendc_api_07_0217.html) |  在核函数中判断本次执行时的tiling_key是否等于某个key，从而标识tiling_key==key的一条kernel分支。  
[REGISTER_TILING_DEFAULT](atlasascendc_api_07_00003.html) |  用于在kernel侧注册用户使用标准C++语法自定义的默认TilingData结构体。  
[REGISTER_TILING_FOR_TILINGKEY](atlasascendc_api_07_00004.html) |  用于在kernel侧注册与TilingKey相匹配的TilingData自定义结构体；该接口需提供一个逻辑表达式，逻辑表达式以字符串“TILING_KEY_VAR”代指实际TilingKey，表达TIlingKey所满足的范围。  
[REGISTER_NONE_TILING](atlasascendc_api_07_00164.html) |  在Kernel侧使用标准C++语法自定义的TilingData结构体时，若用户不确定需要注册哪些结构体，可使用该接口告知框架侧需使用未注册的标准C++语法来定义TilingData，并配套[GET_TILING_DATA_WITH_STRUCT](atlasascendc_api_07_0215.html)，[GET_TILING_DATA_MEMBER](atlasascendc_api_07_0216.html)，[GET_TILING_DATA_PTR_WITH_STRUCT](atlasascendc_api_07_00140.html)来获取对应的TilingData。  
[KERNEL_TASK_TYPE_DEFAULT](atlasascendc_api_07_0218.html) |  设置全局默认的kernel type，对所有的tiling key生效。  
[KERNEL_TASK_TYPE](atlasascendc_api_07_0218.html) |  设置某一个具体的tiling key对应的kernel type。  
  
表13 ISASI接口列表

展开

分类 |  接口名 |  功能描述  
---|---|---  
矢量计算 |  [VectorPadding](atlasascendc_api_07_0221.html) |  根据padMode（pad模式）与padSide（pad方向）对源操作数按照datablock进行填充操作。  
[BilinearInterpolation](atlasascendc_api_07_0222.html) |  双线性插值操作，分为垂直迭代和水平迭代。  
[GetCmpMask](atlasascendc_api_07_0223.html) |  获取[Compare（结果存入寄存器）](atlasascendc_api_07_0067.html)指令的比较结果。  
[SetCmpMask](atlasascendc_api_07_0224.html) |  为[Select](atlasascendc_api_07_0070.html)不传入mask参数的接口设置比较寄存器。  
[GetReduceRepeatSumSpr](atlasascendc_api_07_0225.html) |  获取[ReduceSum](atlasascendc_api_07_0078.html)（针对tensor前n个数据计算）接口的计算结果。  
[GetReduceRepeatMaxMinSpr](atlasascendc_api_07_0226.html) |  获取[ReduceMax](atlasascendc_api_07_0076.html)、[ReduceMin](atlasascendc_api_07_0077.html)连续场景下的最大/最小值以及相应的索引值。  
[ProposalConcat](atlasascendc_api_07_0227.html) |  将连续元素合入Region Proposal内对应位置，每次迭代会将16个连续元素合入到16个Region Proposals的对应位置里。  
[ProposalExtract](atlasascendc_api_07_0228.html) |  与ProposalConcat功能相反，从Region Proposals内将相应位置的单个元素抽取后重排，每次迭代处理16个Region Proposals，抽取16个元素后连续排列。  
[RpSort16](atlasascendc_api_07_0229.html) |  根据Region Proposals中的score域对其进行排序（score大的排前面），每次排16个Region Proposals。  
[MrgSort4](atlasascendc_api_07_0230.html) |  将已经排好序的最多4 条region proposals队列，排列并合并成1条队列，结果按照score域由大到小排序。  
[Sort32](atlasascendc_api_07_0231.html) |  排序函数，一次迭代可以完成32个数的排序。  
[MrgSort](atlasascendc_api_07_0232.html) |  将已经排好序的最多4 条队列，合并排列成 1 条队列，结果按照score域由大到小排序。  
[GetMrgSortResult](atlasascendc_api_07_0233.html) |  获取[MrgSort](atlasascendc_api_07_0232.html)或[MrgSort4](atlasascendc_api_07_0230.html)已经处理过的队列里的Region Proposal个数，并依次存储在四个List入参中。  
[Gatherb](atlasascendc_api_07_0234.html) |  给定一个输入的张量和一个地址偏移张量，Gatherb指令根据偏移地址将输入张量收集到结果张量中。  
[Scatter](atlasascendc_api_07_0235.html) |  给定一个连续的输入张量和一个目的地址偏移张量，Scatter指令根据偏移地址生成新的结果张量后将输入张量分散到结果张量中。  
数据搬运 |  [DataCopyPad](atlasascendc_api_07_0265.html) |  该接口提供数据非对齐搬运的功能。  
[SetPadValue](atlasascendc_api_07_0266.html) |  设置DataCopyPad接口填充的数值。  
矩阵计算 |  [Mmad](atlasascendc_api_07_0249.html) |  完成矩阵乘加操作。  
[MmadWithSparse](atlasascendc_api_07_0250.html) |  完成矩阵乘加操作，传入的左矩阵A为稀疏矩阵， 右矩阵B为稠密矩阵 。  
[SetHF32Mode](atlasascendc_api_07_0258.html) |  此接口同[SetHF32TransMode](atlasascendc_api_07_0259.html)、[SetMMRowMajor](atlasascendc_api_07_0260.html)以及[SetMMColumnMajor](atlasascendc_api_07_00197.html)一样，都用于设置寄存器的值。SetHF32Mode接口用于设置MMAD的HF32模式。  
[SetHF32TransMode](atlasascendc_api_07_0259.html) |  此接口同[SetHF32Mode](atlasascendc_api_07_0258.html)、[SetMMRowMajor](atlasascendc_api_07_0260.html)以及[SetMMColumnMajor](atlasascendc_api_07_00197.html)一样，都用于设置寄存器的值。SetHF32TransMode用于设置MMAD的HF32取整模式，仅在MMAD的HF32模式生效时有效。  
[SetMMRowMajor](atlasascendc_api_07_0260.html) |  此接口同[SetHF32Mode](atlasascendc_api_07_0258.html)、[SetHF32TransMode](atlasascendc_api_07_0259.html)一样，都用于设置寄存器的值，本接口用于设置MMAD计算时优先通过N方向。  
[SetMMColumnMajor](atlasascendc_api_07_00197.html) |  此接口同[SetHF32Mode](atlasascendc_api_07_0258.html)、[SetHF32TransMode](atlasascendc_api_07_0259.html)一样，都用于设置寄存器的值，本接口用于设置MMAD计算时优先通过M方向。  
[Conv2D](atlasascendc_api_07_0262.html) |  计算给定输入张量和权重张量的2-D卷积，输出结果张量。Conv2d卷积层多用于图像识别，使用过滤器提取图像中的特征。  
[Gemm](atlasascendc_api_07_0263.html) |  根据输入的切分规则，将给定的两个输入张量做矩阵乘，输出至结果张量。将A和B两个输入矩阵乘法在一起，得到一个输出矩阵C。  
[SetFixPipeConfig](atlasascendc_api_07_0252.html) |  [DataCopy](atlasascendc_api_07_00130.html)（CO1->GM、CO1->A1）过程中进行随路量化时，通过调用该接口设置量化流程中tensor量化参数。  
[SetFixpipeNz2ndFlag](atlasascendc_api_07_0253.html) |  [DataCopy](atlasascendc_api_07_00130.html)（CO1->GM、CO1->A1）过程中进行随路格式转换（NZ2ND）时，通过调用该接口设置NZ2ND相关配置。  
[SetFixpipePreQuantFlag](atlasascendc_api_07_0254.html) |  [DataCopy](atlasascendc_api_07_00130.html)（CO1->GM、CO1->A1）过程中进行随路量化时，通过调用该接口设置量化流程中scalar量化参数。  
[SetFixPipeClipRelu](atlasascendc_api_07_0255.html) |  [DataCopy](atlasascendc_api_07_00130.html)（CO1->GM）过程中进行随路量化后，通过调用该接口设置ClipRelu操作的最大值。  
[SetFixPipeAddr](atlasascendc_api_07_0256.html) |  [DataCopy](atlasascendc_api_07_00130.html)（CO1->GM）过程中进行随路量化后，通过调用该接口设置element-wise操作时LocalTensor的地址。  
[Fill](atlasascendc_api_07_0237.html) |  初始化LocalTensor（TPosition为A1/A2/B1/B2）为某一个具体的数值。  
[LoadData](atlasascendc_api_07_0238.html) |  LoadData包括Load2D和Load3D数据加载功能。  
[LoadDataWithTranspose](atlasascendc_api_07_0239.html) |  该接口实现带转置的2D格式数据从A1/B1到A2/B2的加载。  
[SetAippFunctions](atlasascendc_api_07_0240.html) |  设置图片预处理（AIPP，AI core pre-process）相关参数。  
[LoadImageToLocal](atlasascendc_api_07_0241.html) |  将图像数据从GM搬运到A1/B1。 搬运过程中可以完成图像预处理操作：包括图像翻转，改变图像尺寸（抠图，裁边，缩放，伸展），以及色域转换，类型转换等。  
[LoadUnZipIndex](atlasascendc_api_07_0242.html) |  加载GM上的压缩索引表到内部寄存器。  
[LoadDataUnzip](atlasascendc_api_07_0243.html) |  将GM上的数据解压并搬运到A1/B1/B2上。  
[LoadDataWithSparse](atlasascendc_api_07_0244.html) |  用于搬运存放在B1里的512B的稠密权重矩阵到B2里，同时读取128B的索引矩阵用于稠密矩阵的稀疏化。  
[SetFmatrix](atlasascendc_api_07_0245.html) |  用于调用[Load3Dv1/Load3Dv2](atlasascendc_api_07_0238.html)时设置FeatureMap的属性描述。  
[SetLoadDataBoundary](atlasascendc_api_07_0246.html) |  设置[Load3D](atlasascendc_api_07_0238.html)时A1/B1边界值。  
[SetLoadDataRepeat](atlasascendc_api_07_0247.html) |  用于设置Load3Dv2接口的repeat参数。设置repeat参数后，可以通过调用一次Load3Dv2接口完成多个迭代的数据搬运。  
[SetLoadDataPaddingValue](atlasascendc_api_07_0248.html) |  设置padValue，用于Load3Dv1/Load3Dv2。  
[Fixpipe](atlasascendc_api_07_0251.html) |  矩阵计算完成后，对结果进行处理，例如对计算结果进行量化操作，并把数据从CO1搬迁到Global Memory中。  
同步控制 |  [SetFlag/WaitFlag](atlasascendc_api_07_0270.html) |  同一核内不同流水线之间的同步指令。具有数据依赖的不同流水指令之间需要插此同步。  
[PipeBarrier](atlasascendc_api_07_0271.html) |  阻塞相同流水，具有数据依赖的相同流水之间需要插此同步。  
[DataSyncBarrier](atlasascendc_api_07_0272.html) |  用于阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可通过参数控制）执行结束。  
[CrossCoreSetFlag](atlasascendc_api_07_0273.html) |  针对分离模式，AI Core上的Cube核（AIC）与Vector核（AIV）之间的同步设置指令。  
[CrossCoreWaitFlag](atlasascendc_api_07_0274.html) |  针对分离模式，AI Core上的Cube核（AIC）与Vector核（AIV）之间的同步等待指令。  
缓存处理 |  [ICachePreLoad](atlasascendc_api_07_0276.html) |  从指令所在DDR地址预加载指令到ICache中。  
[GetICachePreloadStatus](atlasascendc_api_07_0277.html) |  获取ICACHE的PreLoad的状态。  
系统变量访问 |  [GetProgramCounter](atlasascendc_api_07_0279.html) |  获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。  
[GetSubBlockNum](atlasascendc_api_07_0280.html) |  获取AI Core上Vector核的数量。  
[GetSubBlockIdx](atlasascendc_api_07_0281.html) |  获取AI Core上Vector核的ID。  
[GetSystemCycle](atlasascendc_api_07_0282.html) |  获取当前系统cycle数，若换算成时间需要按照50MHz的频率，时间单位为us，换算公式为：time = (cycle数/50) us 。  
原子操作 |  [SetAtomicMax](atlasascendc_api_07_0284.html) |  原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较，将待拷贝的内容和GM已有内容进行比较，将最大值写入GM。  
[SetAtomicMin](atlasascendc_api_07_0285.html) |  原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较，将待拷贝的内容和GM已有内容进行比较，将最小值写入GM。  
[SetStoreAtomicConfig](atlasascendc_api_07_0286.html) |  设置原子操作使能位与原子操作类型。  
[GetStoreAtomicConfig](atlasascendc_api_07_0287.html) |  获取原子操作使能位与原子操作类型的值。  
调试接口 |  [CheckLocalMemoryIA](atlasascendc_api_07_0261.html) |  监视设定范围内的UB读写行为，如果监视到有设定范围的读写行为则会出现EXCEPTION报错，未监视到设定范围的读写行为则不会报错。  
Cube分组管理 |  [CubeResGroupHandle](atlasascendc_api_07_0289.html) |  CubeResGroupHandle用于在分离模式下通过软同步控制AIC和AIV之间进行通讯，实现AI Core计算资源分组。  
[GroupBarrier](atlasascendc_api_07_0301.html) |  当同一个[CubeResGroupHandle](atlasascendc_api_07_0289.html)中的两个AIV任务之间存在依赖关系时，可以使用GroupBarrier控制同步。  
[KfcWorkspace](atlasascendc_api_07_0307.html) |  KfcWorkspace为通信空间描述符，管理不同[CubeResGroupHandle](atlasascendc_api_07_0289.html)的消息通信区划分，与CubeResGroupHandle配合使用。KfcWorkspace的构造函数用于创建KfcWorkspace对象。  
  
#### 高阶API

表14 数学计算API列表

展开

接口名 |  功能描述  
---|---  
[Acos](atlasascendc_api_07_0504.html) |  按元素做反余弦函数计算。  
[Acosh](atlasascendc_api_07_0564.html) |  按元素做双曲反余弦函数计算。  
[Asin](atlasascendc_api_07_0496.html) |  按元素做反正弦函数计算。  
[Asinh](atlasascendc_api_07_0560.html) |  按元素做反双曲正弦函数计算。  
[Atan](atlasascendc_api_07_0516.html) |  按元素做三角函数反正切运算。  
[Atanh](atlasascendc_api_07_0556.html) |  按元素做反双曲正切余弦函数计算。  
[Axpy](atlasascendc_api_07_0585.html) |  源操作数中每个元素与标量求积后和目的操作数中的对应元素相加。  
[Ceil](atlasascendc_api_07_0572.html) |  获取大于或等于x的最小的整数值，即向正无穷取整操作。  
[ClampMax](atlasascendc_api_07_0576.html) |  将srcTensor中大于scalar的数替换为scalar，小于等于scalar的数保持不变，作为dstTensor输出。  
[ClampMin](atlasascendc_api_07_0577.html) |  将srcTensor中小于scalar的数替换为scalar，大于等于scalar的数保持不变，作为dstTensor输出。  
[Cos](atlasascendc_api_07_0508.html) |  按元素做三角函数余弦运算。  
[Cosh](atlasascendc_api_07_0528.html) |  按元素做双曲余弦函数计算。  
[CumSum](atlasascendc_api_07_0605.html) |  对数据按行依次累加或按列依次累加。  
[Digamma](atlasascendc_api_07_0597.html) |  按元素计算x的gamma函数的对数导数。  
[Erf](atlasascendc_api_07_0544.html) |  按元素做误差函数计算，也称为高斯误差函数。  
[Erfc](atlasascendc_api_07_0548.html) |  返回输入x的互补误差函数结果，积分区间为x到无穷大。  
[Exp](atlasascendc_api_07_0589.html) |  按元素取自然指数。  
[Floor](atlasascendc_api_07_0568.html) |  获取小于或等于x的最小的整数值，即向负无穷取整操作。  
[Fmod](atlasascendc_api_07_0608.html) |  按元素计算两个浮点数相除后的余数。  
[Frac](atlasascendc_api_07_0540.html) |  按元素做取小数计算。  
[Lgamma](atlasascendc_api_07_0593.html) |  按元素计算x的gamma函数的绝对值并求自然对数。  
[Log](atlasascendc_api_07_0512.html) |  按元素以e、2、10为底做对数运算。  
[Power](atlasascendc_api_07_0520.html) |  实现按元素做幂运算功能。  
[Round](atlasascendc_api_07_0581.html) |  将输入的元素四舍五入到最接近的整数。  
[Sign](atlasascendc_api_07_0552.html) |  按元素执行Sign操作，Sign是指返回输入数据的符号。  
[Sin](atlasascendc_api_07_0500.html) |  按元素做正弦函数计算。  
[Sinh](atlasascendc_api_07_0524.html) |  按元素做双曲正弦函数计算。  
[Tan](atlasascendc_api_07_0532.html) |  按元素做正切函数计算。  
[Tanh](atlasascendc_api_07_0492.html) |  按元素做逻辑回归Tanh。  
[Trunc](atlasascendc_api_07_0536.html) |  按元素做浮点数截断操作，即向零取整操作。  
[Xor](atlasascendc_api_07_0601.html) |  按元素执行Xor（异或）运算。  
  
表15 量化操作API列表

展开

接口名 |  功能描述  
---|---  
[AscendAntiQuant](atlasascendc_api_07_0822.html) |  按元素做伪量化计算，比如将int8_t数据类型伪量化为half数据类型。  
[AscendDequant](atlasascendc_api_07_0820.html) |  按元素做反量化计算，比如将int32_t数据类型反量化为half/float等数据类型。  
[AscendQuant](atlasascendc_api_07_0818.html) |  按元素做量化计算，比如将half/float数据类型量化为int8_t数据类型。  
  
表16 归一化操作API列表

展开

接口名 |  功能描述  
---|---  
[BatchNorm](atlasascendc_api_07_0806.html) |  对于每个batch中的样本，对其输入的每个特征在batch的维度上进行归一化。  
[DeepNorm](atlasascendc_api_07_0808.html) |  在深层神经网络训练过程中，可以替代LayerNorm的一种归一化方法。  
[GroupNorm](atlasascendc_api_07_0816.html) |  将输入的C维度分为groupNum组，对每一组数据进行标准化。  
[LayerNorm](atlasascendc_api_07_0797.html) |  将输入数据收敛到[0, 1]之间，可以规范网络层输入输出数据分布的一种归一化方法。  
[LayerNormGrad](atlasascendc_api_07_0797.html) |  用于计算LayerNorm的反向传播梯度。  
[LayerNormGradBeta](atlasascendc_api_07_0799.html) |  用于获取反向beta/gmma的数值，和LayerNormGrad共同输出pdx, gmma和beta。  
[Normalize](atlasascendc_api_07_0810.html) |  [LayerNorm](atlasascendc_api_07_0797.html)中，已知均值和方差，计算shape为[A，R]的输入数据的标准差的倒数rstd和归一化输出y。  
[RmsNorm](atlasascendc_api_07_0804.html) |  实现对shape大小为[B，S，H]的输入数据的RmsNorm归一化。  
[WelfordUpdate](atlasascendc_api_07_0812.html) |  实现Welford算法的前处理。  
[WelfordFinalize](atlasascendc_api_07_0814.html) |  实现Welford算法的后处理。  
  
表17 激活函数API列表

展开

接口名 |  功能描述  
---|---  
[AdjustSoftMaxRes](atlasascendc_api_07_0760.html) |  用于对SoftMax相关计算结果做后处理，调整SoftMax的计算结果为指定的值。  
[FasterGelu](atlasascendc_api_07_0772.html) |  FastGelu化简版本的一种激活函数。  
[FasterGeluV2](atlasascendc_api_07_0773.html) |  实现FastGeluV2版本的一种激活函数。  
[GeGLU](atlasascendc_api_07_0786.html) |  采用GeLU作为激活函数的GLU变体。  
[Gelu](atlasascendc_api_07_0771.html) |  GELU是一个重要的激活函数，其灵感来源于relu和dropout，在激活中引入了随机正则的思想。  
[LogSoftMax](atlasascendc_api_07_0768.html) |  对输入tensor做LogSoftmax计算。  
[ReGlu](atlasascendc_api_07_0790.html) |  一种GLU变体，使用Relu作为激活函数。  
[Sigmoid](atlasascendc_api_07_0793.html) |  按元素做逻辑回归Sigmoid。  
[Silu](atlasascendc_api_07_0780.html) |  按元素做Silu运算。  
[SimpleSoftMax](atlasascendc_api_07_0755.html) |  使用计算好的sum和max数据对输入tensor做softmax计算。  
[SoftMax](atlasascendc_api_07_0754.html) |  对输入tensor按行做Softmax计算。  
[SoftmaxFlash](atlasascendc_api_07_0756.html) |  SoftMax增强版本，除了可以对输入tensor做softmaxflash计算，还可以根据上一次softmax计算的sum和max来更新本次的softmax计算结果。  
[SoftmaxFlashV2](atlasascendc_api_07_0758.html) |  SoftmaxFlash增强版本，对应FlashAttention-2算法。  
[SoftmaxFlashV3](atlasascendc_api_07_10001.html) |  SoftmaxFlash增强版本，对应Softmax PASA算法。  
[SoftmaxGrad](atlasascendc_api_07_0757.html) |  对输入tensor做grad反向计算的一种方法。  
[SoftmaxGradFront](atlasascendc_api_07_0759.html) |  对输入tensor做grad反向计算的一种方法。  
[SwiGLU](atlasascendc_api_07_0776.html) |  采用Swish作为激活函数的GLU变体。  
[Swish](atlasascendc_api_07_0783.html) |  神经网络中的Swish激活函数。  
  
表18 归约操作API列表

展开

接口名 |  功能描述  
---|---  
[Sum](atlasascendc_api_07_0826.html) |  获取最后一个维度的元素总和。  
[Mean](atlasascendc_api_07_0829.html) |  根据最后一轴的方向对各元素求平均值。  
[ReduceXorSum](atlasascendc_api_07_0833.html) |  按照元素执行Xor（按位异或）运算，并将计算结果ReduceSum求和。  
[ReduceSum](atlasascendc_api_07_10018.html) |  对一个多维向量按照指定的维度进行数据累加。  
[ReduceMean](atlasascendc_api_07_10157.html) |  对一个多维向量按照指定的维度求平均值。  
[ReduceMax](atlasascendc_api_07_10055.html) |  对一个多维向量在指定的维度求最大值。  
[ReduceMin](atlasascendc_api_07_10056.html) |  对一个多维向量在指定的维度求最小值。  
[ReduceAny](atlasascendc_api_07_10142.html) |  对一个多维向量在指定的维度求逻辑或。  
[ReduceAll](atlasascendc_api_07_10145.html) |  对一个多维向量在指定的维度求逻辑与。  
[ReduceProd](atlasascendc_api_07_10150.html) |  对一个多维向量在指定的维度求积。  
  
表19 排序操作API列表

展开

接口名 |  功能描述  
---|---  
[TopK](atlasascendc_api_07_0836.html) |  获取最后一个维度的前k个最大值或最小值及其对应的索引。  
[Concat](atlasascendc_api_07_0839.html) |  对数据进行预处理，将要排序的源操作数srcLocal一一对应的合入目标数据concatLocal中，数据预处理完后，可以进行Sort。  
[Extract](atlasascendc_api_07_0841.html) |  处理Sort的结果数据，输出排序后的value和index。  
[Sort](atlasascendc_api_07_0842.html) |  排序函数，按照数值大小进行降序排序。  
[MrgSort](atlasascendc_api_07_0847.html) |  将已经排好序的最多4条队列，合并排列成1条队列，结果按照score域由大到小排序。  
  
表20 数据过滤API列表

展开

接口名 |  功能描述  
---|---  
[Select](atlasascendc_api_07_0859.html) |  给定两个源操作数src0和src1，根据maskTensor相应位置的值（非bit位）选取元素，得到目的操作数dst。  
[DropOut](atlasascendc_api_07_0862.html) |  提供根据MaskTensor对源操作数进行过滤的功能，得到目的操作数。  
  
表21 张量变换API列表

展开

接口名 |  功能描述  
---|---  
[Transpose](atlasascendc_api_07_0865.html) |  对输入数据进行数据排布及Reshape操作。  
[TransData](atlasascendc_api_07_10181.html) |  将输入数据的排布格式转换为目标排布格式。  
[Broadcast](atlasascendc_api_07_0853.html) |  将输入按照输出shape进行广播。  
[Pad](atlasascendc_api_07_0849.html) |  对height * width的二维Tensor在width方向上pad到32B对齐。  
[UnPad](atlasascendc_api_07_0851.html) |  对height * width的二维Tensor在width方向上进行unpad。  
[Fill](atlasascendc_api_07_0891.html) |  将Global Memory上的数据初始化为指定值。  
  
表22 索引计算API列表

展开

接口名 |  功能描述  
---|---  
[Arange](atlasascendc_api_07_0856.html) |  给定起始值，等差值和长度，返回一个等差数列。  
  
表23 矩阵计算API列表

展开

接口名 |  功能描述  
---|---  
[Matmul](atlasascendc_api_07_0613.html) |  Matmul矩阵乘法的运算。  
  
表24 HCCL通信类API列表

展开

接口名 |  功能描述  
---|---  
[HCCL通信类](atlasascendc_api_07_0867.html) |  在AI Core侧编排集合通信任务。  
  
表25 卷积计算API列表

展开

接口名 |  功能描述  
---|---  
[Conv3D](atlasascendc_api_07_10005.html) |  3维卷积正向矩阵运算。  
[Conv3DBackpropInput](atlasascendc_api_07_0918.html) |  卷积的反向运算，求解特征矩阵的反向传播误差。  
[Conv3DBackpropFilter](atlasascendc_api_07_0893.html) |  卷积的反向运算，求解权重的反向传播误差。  
  
#### Utils API

表26 C++标准库API列表

展开

接口名 |  功能描述  
---|---  
[max](atlasascendc_api_07_10053.html) |  比较相同数据类型的两个数中的最大值。  
[min](atlasascendc_api_07_10054.html) |  比较相同数据类型的两个数中的最小值。  
[integer_sequence](atlasascendc_api_07_10106.html) |  用于生成一个整数序列。  
[tuple](atlasascendc_api_07_10108.html) |  允许存储多个不同类型元素的容器。  
[get](atlasascendc_api_07_10109.html) |  从tuple容器中提取指定位置的元素。  
[make_tuple](atlasascendc_api_07_10110.html) |  用于便捷地创建tuple对象。  
[is_convertible](atlasascendc_api_07_10114.html) |  在程序编译时判断两个类型之间是否可以进行隐式转换。  
[is_base_of](atlasascendc_api_07_10115.html) |  在程序编译时判断一个类型是否为另一个类型的基类。  
[is_same](atlasascendc_api_07_10116.html) |  在程序编译时判断两个类型是否完全相同。  
[is_void](atlasascendc_api_07_10430.html) |  在程序编译时，检测一个类型是否为void类型。  
[is_integral](atlasascendc_api_07_10431.html) |  在程序编译时，检测一个类型是否为整数类型。  
[is_floating_point](atlasascendc_api_07_10432.html) |  在程序编译时，检测一个类型是否为浮点类型。  
[is_array](atlasascendc_api_07_10433.html) |  在程序编译时，检测一个类型是否为数组类型。  
[is_pointer](atlasascendc_api_07_10434.html) |  在程序编译时，判断一个类型是否为指针类型。  
[is_reference](atlasascendc_api_07_10435.html) |  在程序编译时，检测一个类型是否为引用类型。  
[is_const](atlasascendc_api_07_10436.html) |  在程序编译时，检测一个类型是否为const限定的类型。  
[remove_const](atlasascendc_api_07_10437.html) |  在程序编译时，对传入的模板参数类型移除const限定符。  
[remove_volatile](atlasascendc_api_07_10438.html) |  在程序编译时，对传入的模板参数类型移除volatile限定符。  
[remove_cv](atlasascendc_api_07_10439.html) |  在程序编译时，对传入的模板参数类型移除const限定符或volatile限定符，或同时移除这两种限定符。  
[remove_reference](atlasascendc_api_07_10440.html) |  在程序编译时，从给定类型中移除引用限定符。  
[remove_pointer](atlasascendc_api_07_10441.html) |  在程序编译时，从给定类型中移除指针限定符。  
[add_const](atlasascendc_api_07_10442.html) |  在程序编译时，为指定类型添加const限定符。  
[add_volatile](atlasascendc_api_07_10443.html) |  在程序编译时，为指定类型添加volatile限定符。  
[add_cv](atlasascendc_api_07_10444.html) |  在程序编译时，为指定类型添加const和volatile限定符。  
[add_pointer](atlasascendc_api_07_10445.html) |  在程序编译时，为指定类型添加指针限定符。  
[add_lvalue_reference](atlasascendc_api_07_10446.html) |  在程序编译时，为指定类型添加左值引用限定符。  
[add_rvalue_reference](atlasascendc_api_07_10447.html) |  在程序编译时，为指定类型添加右值引用限定符。  
[enable_if](atlasascendc_api_07_10117.html) |  在程序编译时根据某个条件启用或禁用特定的函数模板、类模板或模板特化。  
[conditional](atlasascendc_api_07_10118.html) |  在程序编译时根据一个布尔条件从两个类型中选择一个类型。  
[integral_constant](atlasascendc_api_07_10198.html) |  用于封装一个编译时常量整数值，是标准库中许多类型特性和编译时计算的基础组件。  
  
表27 平台信息获取API列表

展开

接口名 |  功能描述  
---|---  
[PlatformAscendC](atlasascendc_api_07_1026.html) |  在实现Host侧的Tiling函数时，可能需要获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。PlatformAscendC类提供获取这些平台信息的功能。  
[PlatformAscendCManager](atlasascendc_api_07_1039.html) |  基于Kernel Launch算子工程，通过基础调用（Kernel Launch）方式调用算子的场景下，可能需要获取硬件平台相关信息，比如获取硬件平台的核数。PlatformAscendCManager类提供获取平台信息的功能。  
  
表28 原型注册与管理API列表

展开

接口名 |  功能描述  
---|---  
[原型注册接口（OP_ADD）](atlasascendc_api_07_0945.html) |  注册算子的原型定义。  
[OpDef](atlasascendc_api_07_0946.html) |  用于算子原型定义。  
[OpParamDef](atlasascendc_api_07_0957.html) |  用于算子参数定义。  
[OpAttrDef](atlasascendc_api_07_0976.html) |  用于算子属性定义。  
[OpAICoreDef](atlasascendc_api_07_0978.html) |  用于定义AI处理器上相关实现信息，并关联Tiling实现、Shape推导等函数。  
[OpAICoreConfig](atlasascendc_api_07_0988.html) |  用于配置AI Core配置信息。  
[OpMC2Def](atlasascendc_api_07_0999.html) |  该类用于在host侧配置通算融合算子的通信域名称。配置后在kernel侧可以获取通信域对应的context地址。  
  
表29 Tiling数据结构注册API列表

展开

接口名 |  功能描述  
---|---  
[TilingData结构定义](atlasascendc_api_07_1005.html) |  定义一个TilingData的类，添加所需的成员变量（TilingData字段），用于保存所需TilingData参数。完成该TilingData类的定义后，该类通过继承TilingDef类（用来存放、处理用户自定义Tiling结构体成员变量的基类）提供TilingData字段设置、序列化和保存等接口。  
[TilingData结构注册](atlasascendc_api_07_1006.html) |  注册定义的TilingData结构体并和自定义算子绑定。  
  
表30 Tiling调测API列表

展开

接口名 |  功能描述  
---|---  
[OpTilingRegistry](atlasascendc_api_07_00071.html) |  OpTilingRegistry类属于context_ascendc命名空间，主要用于加载Tiling实现的动态库，并获取算子的Tiling函数指针以进行调试和验证。  
[ContextBuilder](atlasascendc_api_07_1007.html) |  ContextBuilder类提供一系列的API接口，支持手动构造[TilingContext](/document/detail/zh/CANNCommunityEdition/900beta1/API/basicdataapi/atlasopapi_07_00223.html)类来验证Tiling函数以及KernelContext类用于TilingParse函数的验证。  
  
表31 Tiling模板编程API列表

展开

接口名 |  功能描述  
---|---  
[模板参数定义](atlasascendc_api_07_00011.html) |  通过该类接口进行模板参数ASCENDC_TPL_ARGS_DECL和模板参数组合ASCENDC_TPL_ARGS_SEL（即可使用的模板）的定义。  
[GET_TPL_TILING_KEY](atlas_ascendc_10_00008.html) |  Tiling模板编程时，开发者通过调用此接口自动生成TilingKey。该接口将传入的模板参数通过定义的位宽，转成二进制，按照顺序组合后转成uint64数值，即TilingKey。  
[ASCENDC_TPL_SEL_PARAM](atlasascendc_api_07_00057.html) |  Tiling模板编程时，开发者通过调用此接口自动生成并配置TilingKey。  
  
表32 Tiling下沉API列表

展开

接口名 |  功能描述  
---|---  
[DEVICE_IMPL_OP_OPTILING](atlasascendc_api_07_00060.html) |  在[Tiling下沉](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_00014.html)场景中，该宏定义用于生成Tiling下沉的注册类，再通过调用注册类的成员函数来注册需要下沉的Tiling函数。  
  
表33 RTC API列表

展开

接口名 |  功能描述  
---|---  
[aclrtcCompileProg](atlasascendc_api_07_00154.html) |  编译接口，编译指定的程序。  
[aclrtcCreateProg](atlasascendc_api_07_00155.html) |  通过给定的参数，创建编译程序的实例。  
[aclrtcDestroyProg](atlasascendc_api_07_00156.html) |  销毁编译程序的实例。  
[aclrtcGetBinData](atlasascendc_api_07_00157.html) |  获取编译后的二进制数据。  
[aclrtcGetBinDataSize](atlasascendc_api_07_00158.html) |  获取编译的二进制数据大小。用于在[aclrtcGetBinData](atlasascendc_api_07_00157.html)获取二进制数据时分配对应大小的内存空间。  
[aclrtcGetCompileLogSize](atlasascendc_api_07_00159.html) |  获取编译日志的大小。用于在[aclrtcGetCompileLog](atlasascendc_api_07_00160.html)获取日志内容时分配对应大小的内存空间。  
[aclrtcGetCompileLog](atlasascendc_api_07_00160.html) |  获取编译日志的内容，以字符串形式保存。  
  
表34 log API列表

展开

接口名 |  功能描述  
---|---  
[ASC_CPU_LOG](atlasascendc_api_07_00152.html) |  提供Host侧打印Log的功能。开发者可以在算子的TilingFunc代码中使用ASC_CPU_LOG_XXX接口来输出相关内容。  
  
#### AI CPU API

表35 AI CPU API列表

展开

接口名 |  功能描述  
---|---  
[printf](atlasascendc_api_07_00166.html) |  该接口提供AI CPU算子Kernel调试场景下的格式化输出功能，默认将输出内容解析并打印在屏幕上。  
[assert](atlasascendc_api_07_00167.html) |  该接口实现AI CPU算子Kernel调试场景下的assert断言功能。
