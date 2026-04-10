<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_00060.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# DEVICE_IMPL_OP_OPTILING

# DEVICE_IMPL_OP_OPTILING

#### 功能说明

在[Tiling下沉](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_00014.html)场景中，该宏定义用于生成Tiling下沉的注册类，再通过调用注册类的成员函数来注册需要下沉的Tiling函数。

#### 函数原型
    
    
    namespace optiling {
    using SinkTilingFunc = std::function<ge::graphStatus(gert::TilingContext *context)>;
    
    class DeviceOpImplRegisterImpl;
    // 开发者仅关注Tiling成员函数
    class DeviceOpImplRegister {
    public:
      DeviceOpImplRegister(const char *opType);
      ~DeviceOpImplRegister();
      DeviceOpImplRegister(DeviceOpImplRegister &&other) noexcept;
      DeviceOpImplRegister(const DeviceOpImplRegister &other);
      DeviceOpImplRegister &operator=(const DeviceOpImplRegister &) = delete;
      DeviceOpImplRegister &operator=(DeviceOpImplRegister &&) = delete;
      DeviceOpImplRegister &Tiling(SinkTilingFunc func);
    
    // ...
    };
    }  // namespace optiling
    
    #define DEVICE_IMPL_OP_OPTILING(optype)                                                                      \
      static optiling::DeviceOpImplRegister VAR_UNUSED g_deviceOpImplRegister##optype =                                    \
          optiling::DeviceOpImplRegister(#optype)
    #endif
    

#### 参数说明

表1 DEVICE_IMPL_OP_OPTILING参数说明

展开

参数 |  输入/输出 |  说明  
---|---|---  
optype |  输入 |  需要注册Tiling函数的OpType（算子类型）。  
  
表2 Tiling成员函数参数说明

展开

参数 |  输入/输出 |  说明  
---|---|---  
func |  输入 |  需要注册的Tiling函数，该函数接受一个[TilingContext](/document/detail/zh/CANNCommunityEdition/900beta1/API/basicdataapi/atlasopapi_07_00223.html)作为输入，以[ge::graphStatus](/document/detail/zh/CANNCommunityEdition/900beta1/API/basicdataapi/atlasopapi_07_00513.html)为返回值。  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    DEVICE_IMPL_OP_OPTILING(TestOptype).Tiling(TestTilingFunc); // 将Tiling函数以及其OpType注册到Tiling下沉
    

**父主题：** [Tiling下沉](atlasascendc_api_07_00185.html)
