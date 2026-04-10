<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_00167.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# assert

# assert

#### 产品支持情况

展开

产品 |  是否支持  
---|---  
Atlas A3 训练系列产品 / Atlas A3 推理系列产品  |  √  
Atlas A2 训练系列产品 / Atlas A2 推理系列产品  |  √  
Atlas 200I/500 A2 推理产品  |  x  
Atlas 推理系列产品 AI Core |  x  
Atlas 推理系列产品 Vector Core |  x  
Atlas 训练系列产品  |  x  
  
#### 功能说明

该接口实现AI CPU算子Kernel调试场景下的assert断言功能。

算子执行中，如果assert内部条件判断不为真，则输出assert条件、触发文件名、行号等信息。

#### 需要包含的头文件
    
    
    #include "aicpu_api.h"
    

#### 函数原型
    
    
    assert(expr)
    

#### 参数说明

展开

参数名 |  输入/输出 |  描述  
---|---|---  
expr |  输入 |  assert断言是否终止程序的条件。为true则程序继续执行，为false则终止程序。  
  
#### 返回值说明

无

#### 约束说明

  * 该接口仅支持通过<<<...>>>调用，并在[异构编译](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_00042.html)场景使用。
  * kernel开发不要包含系统的assert.h，会导致宏定义冲突。
  * assert接口调用形式与C语言一致，不需要使用AscendC命名空间。


  * 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
  * 使用该接口时，若采用bisheng命令行编译，开发者需要手动链接相关的静态库；而使用CMake编译时，框架会自动处理链接问题，无需开发者额外关注。具体编译命令如下：通过--cce-aicpu-laicpu_api为Device链接libaicpu_api.a，通过--cce-aicpu-L指定libaicpu_api.a的库路径。 
        
        $bisheng -O2 foo.aicpu --cce-aicpu-L${INSTALL_DIR}/lib64/device/lib64 --cce-aicpu-laicpu_api -I${INSTALL_DIR}/include/ascendc/aicpu_api -c -o foo.aicpu.o

${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。请先设置 ASCEND_HOME_PATH 或 ASCEND_TOOLKIT_HOME，并将 ${INSTALL_DIR} 替换为对应环境变量的值，例如 /path/to/Ascend/ascend-toolkit/latest。




#### 调用示例

在算子kernel侧实现代码中需要增加断言的地方使用assert检查代码示例如下：
    
    
    int assertFlag = 10;
    // 断言条件
    assert(assertFlag == 12);
    

程序运行时会触发assert，打印效果如下：
    
    
    [ASSERT]` assertFlag == 12 ' at /home/.../test.cpp:36
    

**父主题：** [AI CPU API](atlasascendc_api_07_00165.html)
