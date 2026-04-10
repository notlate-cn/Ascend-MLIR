#!/usr/bin/env python3
"""
AscendC 文档重组工具
将下载的文档按章节结构重新组织
"""

import os
import re
import json
from collections import defaultdict

INPUT_DIR = "/Volumes/GM9/code/Ascend-MLIR/examples/ascendc-handbook"
OUTPUT_DIR = "/Volumes/GM9/code/Ascend-MLIR/examples/ascendc-handbook/organized"

CHAPTER_STRUCTURE = {
    "01_基础数据结构": {
        "title": "基础数据结构",
        "apis": [
            ("LocalTensor", ["LocalTensor简介", "LocalTensor"]),
            ("GlobalTensor", ["GlobalTensor简介", "GlobalTensor"]),
            ("Coordinate", ["Coordinate简介", "Coordinate"]),
            ("Layout", ["Layout简介", "Layout"]),
            ("TensorTrait", ["TensorTrait简介", "TensorTrait"]),
            ("TPosition", ["TPosition"]),
        ]
    },
    "02_基础API": {
        "title": "基础API",
        "subchapters": {
            "02_01_数据搬运": {
                "title": "数据搬运",
                "apis": [
                    ("DataCopy", ["DataCopy简介", "DataCopy"]),
                    ("Copy", ["Copy"]),
                ]
            },
            "02_02_矢量计算": {
                "title": "矢量计算",
                "subsections": {
                    "基础算术": ["Exp", "Ln", "Abs", "Reciprocal", "Sqrt", "Rsqrt", "Relu", "Add", "Sub", "Mul", "Div", "Max", "Min", "Adds", "Muls", "Maxs", "Mins", "LeakyRelu"],
                    "逻辑计算": ["Not", "And", "Or", "ShiftLeft", "ShiftRight"],
                    "复合计算": ["Axpy", "CastDequant", "AddRelu", "AddReluCast", "AddDeqRelu", "SubRelu", "SubReluCast", "MulAddDst", "MulCast", "FusedMulAdd", "MulAddRelu"],
                    "比较与选择": ["Compare", "Compares", "Select", "GatherMask"],
                    "精度转换": ["Cast"],
                    "归约计算": ["ReduceMax", "ReduceMin", "ReduceSum", "WholeReduceMax", "WholeReduceMin", "WholeReduceSum", "BlockReduceMax", "BlockReduceMin", "BlockReduceSum", "PairReduceSum", "RepeatReduceSum"],
                    "数据转换": ["Transpose", "TransDataTo5HD"],
                    "数据填充": ["Duplicate", "Brcb", "CreateVecIndex"],
                    "数据收集": ["Gather"],
                    "掩码操作": ["SetMaskCount", "SetMaskNorm", "SetVectorMask", "ResetMask"],
                    "量化设置": ["SetDeqScale"],
                }
            },
            "02_03_标量计算": {
                "title": "标量计算",
                "apis": [
                    ("GetBitCount", ["GetBitCount"]),
                    ("CountLeadingZero", ["CountLeadingZero"]),
                    ("Cast标量", ["Cast（float转half、int32_t）", "Cast（float转bfloat16_t）", "Cast（多类型转float）"]),
                    ("CountBitsCntSameAsSignBit", ["CountBitsCntSameAsSignBit"]),
                    ("GetSFFValue", ["GetSFFValue"]),
                ]
            },
            "02_04_资源管理": {
                "title": "资源管理",
                "apis": [
                    ("TPipe", ["TPipe简介", "TPipe"]),
                    ("GetTPipePtr", ["GetTPipePtr"]),
                    ("TBufPool", ["TBufPool简介", "TBufPool"]),
                    ("TQue", ["TQue简介", "TQue"]),
                    ("TQueBind", ["TQueBind简介", "TQueBind"]),
                    ("TBuf", ["TBuf简介", "TBuf"]),
                    ("SPM_Buffer", ["InitSpmBuffer", "WriteSpmBuffer", "ReadSpmBuffer"]),
                    ("Workspace", ["GetUserWorkspace", "SetSysWorkSpace", "GetSysWorkSpacePtr"]),
                ]
            },
            "02_05_同步控制": {
                "title": "同步控制",
                "apis": [
                    ("TQueSync", ["TQueSync"]),
                    ("IBSet", ["IBSet"]),
                    ("IBWait", ["IBWait"]),
                    ("SyncAll", ["SyncAll"]),
                    ("核间同步", ["InitDetermineComputeWorkspace", "WaitPreBlock", "NotifyNextBlock"]),
                    ("任务同步", ["SetNextTaskStart", "WaitPreTaskEnd"]),
                ]
            },
            "02_06_缓存处理": {
                "title": "缓存处理",
                "apis": [
                    ("DataCachePreload", ["DataCachePreload"]),
                    ("DataCacheCleanAndInvalid", ["DataCacheCleanAndInvalid"]),
                ]
            },
            "02_07_系统变量": {
                "title": "系统变量访问",
                "apis": [
                    ("GetBlockNum", ["GetBlockNum"]),
                    ("GetBlockIdx", ["GetBlockIdx"]),
                    ("GetDataBlockSizeInBytes", ["GetDataBlockSizeInBytes"]),
                    ("GetArchVersion", ["GetArchVersion"]),
                    ("InitSocState", ["InitSocState"]),
                ]
            },
            "02_08_原子操作": {
                "title": "原子操作",
                "apis": [
                    ("SetAtomicAdd", ["SetAtomicAdd"]),
                    ("SetAtomicType", ["SetAtomicType"]),
                    ("DisableDmaAtomic", ["DisableDmaAtomic"]),
                ]
            },
            "02_09_调试接口": {
                "title": "调试接口",
                "apis": [
                    ("DumpTensor", ["DumpTensor"]),
                    ("printf", ["printf"]),
                    ("assert", ["assert", "ascendc_assert"]),
                    ("DumpAccChkPoint", ["DumpAccChkPoint"]),
                    ("PrintTimeStamp", ["PrintTimeStamp"]),
                    ("Trap", ["Trap"]),
                    ("CPU调测", ["GmAlloc", "GmFree", "ICPU_RUN_KF", "ICPU_SET_TILING_KEY", "SetKernelMode"]),
                    ("性能仿真", ["TRACE_START", "TRACE_STOP", "MetricsProfStart", "MetricsProfStop"]),
                ]
            },
            "02_10_Kernel_Tiling": {
                "title": "Kernel Tiling接口",
                "apis": [
                    ("GET_TILING_DATA", ["GET_TILING_DATA"]),
                    ("GET_TILING_DATA_WITH_STRUCT", ["GET_TILING_DATA_WITH_STRUCT"]),
                    ("GET_TILING_DATA_MEMBER", ["GET_TILING_DATA_MEMBER"]),
                    ("TILING_KEY_IS", ["TILING_KEY_IS"]),
                    ("Tiling注册", ["REGISTER_TILING_DEFAULT", "REGISTER_TILING_FOR_TILINGKEY", "REGISTER_NONE_TILING"]),
                    ("Kernel类型", ["设置Kernel类型"]),
                ]
            },
            "02_11_ISASI接口": {
                "title": "ISASI接口（硬件相关）",
                "apis": [
                    ("矢量计算ISASI", ["VectorPadding(ISASI)", "BilinearInterpolation(ISASI)", "GetCmpMask(ISASI)", "SetCmpMask(ISASI)", "GetReduceRepeatSumSpr(ISASI)", "GetReduceRepeatMaxMinSpr(ISASI)"]),
                    ("排序ISASI", ["ProposalConcat", "ProposalExtract", "RpSort16", "MrgSort4", "Sort32", "MrgSort", "GetMrgSortResult"]),
                    ("数据搬运ISASI", ["Gatherb(ISASI)", "Scatter(ISASI)", "DataCopyPad(ISASI)", "SetPadValue(ISASI)"]),
                    ("矩阵计算ISASI", ["Mmad", "MmadWithSparse", "SetHF32Mode", "SetHF32TransMode", "SetMMRowMajor", "SetMMColumnMajor"]),
                    ("Conv2D_Gemm", ["Conv2D（废弃）", "Gemm（废弃）"]),
                    ("FixPipe", ["Fixpipe", "SetFixPipeConfig", "SetFixpipeNz2ndFlag", "SetFixpipePreQuantFlag", "SetFixPipeClipRelu", "SetFixPipeAddr"]),
                    ("LoadData", ["Fill", "Load2D", "LoadDataWithTranspose", "SetAippFunctions", "LoadImageToLocal", "LoadUnzipIndex", "LoadDataUnzip", "LoadDataWithSparse", "SetFmatrix", "SetLoadDataBoundary", "SetLoadDataRepeat", "SetLoadDataPaddingValue"]),
                    ("同步控制ISASI", ["SetFlag/WaitFlag(ISASI)", "PipeBarrier(ISASI)", "DataSyncBarrier(ISASI)", "CrossCoreSetFlag(ISASI)", "CrossCoreWaitFlag(ISASI)"]),
                    ("缓存ISASI", ["ICachePreLoad(ISASI)", "GetICachePreloadStatus(ISASI)"]),
                    ("系统变量ISASI", ["GetProgramCounter(ISASI)", "GetSubBlockNum(ISASI)", "GetSubBlockIdx(ISASI)", "GetSystemCycle(ISASI)"]),
                    ("原子操作ISASI", ["SetAtomicMax(ISASI)", "SetAtomicMin(ISASI)", "SetStoreAtomicConfig(ISASI)", "GetStoreAtomicConfig(ISASI)"]),
                    ("调试ISASI", ["CheckLocalMemoryIA(ISASI)"]),
                    ("Cube分组", ["CubeResGroupHandle使用说明", "GroupBarrier使用说明"]),
                ]
            },
        }
    },
    "03_高阶API": {
        "title": "高阶API",
        "subchapters": {
            "03_01_数学计算": {
                "title": "数学计算",
                "apis": [
                    ("三角函数", ["Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh", "Tanh", "Asinh", "Acosh", "Atanh"]),
                    ("指数对数", ["Exp", "Ln", "Log", "Power", "Lgamma", "Digamma"]),
                    ("取整函数", ["Floor", "Ceil", "Round", "Trunc", "Frac"]),
                    ("其他数学", ["Erf", "Erfc", "Sign", "ClampMax", "ClampMin", "Axpy", "Fmod", "CumSum", "Xor"]),
                ]
            },
            "03_02_量化操作": {
                "title": "量化操作",
                "apis": [
                    ("AscendQuant", ["AscendQuant"]),
                    ("AscendDequant", ["AscendDequant"]),
                    ("AscendAntiQuant", ["AscendAntiQuant"]),
                ]
            },
            "03_03_归一化": {
                "title": "归一化操作",
                "apis": [
                    ("LayerNorm", ["LayerNorm", "LayerNormGradBeta"]),
                    ("RmsNorm", ["RmsNorm"]),
                    ("BatchNorm", ["BatchNorm"]),
                    ("DeepNorm", ["DeepNorm"]),
                    ("GroupNorm", ["GroupNorm"]),
                    ("Normalize", ["Normalize"]),
                    ("Welford", ["WelfordUpdate", "WelfordFinalize"]),
                ]
            },
            "03_04_激活函数": {
                "title": "激活函数",
                "apis": [
                    ("SoftMax", ["SoftMax", "SimpleSoftMax", "SoftmaxFlash", "SoftmaxFlashV2", "SoftmaxFlashV3", "SoftmaxGrad", "SoftmaxGradFront", "AdjustSoftMaxRes", "LogSoftMax"]),
                    ("Gelu", ["Gelu", "FasterGelu", "FasterGeluV2"]),
                    ("GLU系列", ["SwiGLU", "GeGLU", "ReGlu"]),
                    ("其他激活", ["Silu", "Swish", "Sigmoid"]),
                ]
            },
            "03_05_归约操作": {
                "title": "归约操作",
                "apis": [
                    ("Sum", ["Sum"]),
                    ("Mean", ["Mean"]),
                    ("ReduceXorSum", ["ReduceXorSum"]),
                    ("高阶归约", ["ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceAny", "ReduceAll", "ReduceProd"]),
                ]
            },
            "03_06_排序操作": {
                "title": "排序操作",
                "apis": [
                    ("TopK", ["TopK"]),
                    ("Sort", ["Concat", "Extract", "Sort", "MrgSort"]),
                ]
            },
            "03_07_数据过滤": {
                "title": "数据过滤",
                "apis": [
                    ("Select", ["Select"]),
                    ("DropOut", ["DropOut"]),
                ]
            },
            "03_08_张量变换": {
                "title": "张量变换",
                "apis": [
                    ("Transpose", ["Transpose"]),
                    ("TransData", ["TransData"]),
                    ("Broadcast", ["Broadcast"]),
                    ("Pad", ["Pad", "UnPad"]),
                    ("Fill", ["Fill"]),
                    ("Arange", ["Arange"]),
                ]
            },
            "03_09_矩阵计算": {
                "title": "矩阵计算",
                "apis": [
                    ("Matmul", ["Matmul使用说明"]),
                ]
            },
            "03_10_HCCL通信": {
                "title": "HCCL通信",
                "apis": [
                    ("HCCL", ["HCCL使用说明"]),
                ]
            },
            "03_11_卷积计算": {
                "title": "卷积计算",
                "apis": [
                    ("Conv3D", ["Conv3D使用说明", "Conv3DBackpropFilter使用说明", "Conv3DBackpropInput使用说明"]),
                ]
            },
        }
    },
    "04_Utils_API": {
        "title": "Utils API（公共辅助函数）",
        "subchapters": {
            "04_01_标准库": {
                "title": "C++标准库",
                "apis": [
                    ("比较函数", ["max", "min"]),
                    ("序列容器", ["integer_sequence", "tuple", "get", "make_tuple"]),
                    ("类型判断", ["is_convertible", "is_base_of", "is_same", "is_void", "is_integral", "is_floating_point", "is_array", "is_pointer", "is_reference", "is_const"]),
                    ("类型修改", ["remove_const", "remove_volatile", "remove_cv", "remove_reference", "remove_pointer", "add_const", "add_volatile", "add_cv", "add_pointer", "add_lvalue_reference", "add_rvalue_reference"]),
                    ("条件编译", ["enable_if", "conditional", "integral_constant"]),
                ]
            },
            "04_02_运行时编译": {
                "title": "运行时编译",
                "apis": [
                    ("aclrtc", ["aclrtcCreateProg", "aclrtcDestroyProg", "aclrtcCompileProg", "aclrtcGetBinData", "aclrtcGetBinDataSize", "aclrtcGetCompileLog", "aclrtcGetCompileLogSize"]),
                ]
            },
            "04_03_平台信息": {
                "title": "平台信息",
                "apis": [
                    ("PlatformAscendC", ["PlatformAscendC简介", "PlatformAscendCManager"]),
                ]
            },
            "04_04_日志输出": {
                "title": "日志输出",
                "apis": [
                    ("ASC_CPU_LOG", ["ASC_CPU_LOG"]),
                ]
            },
        }
    },
    "05_语言扩展层C_API": {
        "title": "语言扩展层 C API",
        "apis": [
            ("简介", ["简介"]),
            ("模板参数", ["模板参数", "模板参数定义"]),
            ("构造函数", ["构造函数与析构函数"]),
            ("随路量化", ["随路量化激活搬运"]),
            ("Async", ["Async"]),
            ("DEVICE_IMPL_OP_OPTILING", ["DEVICE_IMPL_OP_OPTILING"]),
            ("ASCENDC_TPL_SEL_PARAM", ["ASCENDC_TPL_SEL_PARAM"]),
        ]
    },
    "06_算子原型注册": {
        "title": "算子原型注册",
        "apis": [
            ("原型注册", ["原型注册接口（OP_ADD）"]),
            ("Input", ["Input"]),
            ("ParamType", ["ParamType"]),
            ("OpAttrDef", ["OpAttrDef"]),
            ("SetTiling", ["SetTiling"]),
            ("OpAICoreConfig", ["OpAICoreConfig构造函数"]),
            ("OpMC2Def", ["OpMC2Def简介"]),
            ("TilingData", ["TilingData结构定义", "TilingData结构注册"]),
        ]
    },
}


def read_md_file(filepath):
    """读取Markdown文件内容"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return None


def find_md_file_by_title(title, md_dir):
    """根据标题查找对应的Markdown文件"""
    files = os.listdir(md_dir)
    for f in files:
        if f.endswith('.md'):
            if title in f or f.replace('.md', '').endswith(title):
                return os.path.join(md_dir, f)
            name_without_num = re.sub(r'^\d+_', '', f.replace('.md', ''))
            if name_without_num == title:
                return os.path.join(md_dir, f)
    return None


def find_all_md_files_for_titles(titles, md_dir):
    """查找多个标题对应的文件"""
    files = []
    for title in titles:
        filepath = find_md_file_by_title(title, md_dir)
        if filepath:
            files.append(filepath)
    return files


def create_chapter_doc(chapter_path, chapter_title, api_files, md_dir):
    """创建章节文档"""
    os.makedirs(os.path.dirname(chapter_path), exist_ok=True)
    
    content = f"# {chapter_title}\n\n"
    content += f"> 来源: 昇腾社区官网 AscendC算子开发文档\n\n"
    content += "---\n\n"
    content += "## 目录\n\n"
    
    for api_name, titles in api_files:
        content += f"- [{api_name}](#{api_name.lower().replace('/', '_')})\n"
    
    content += "\n---\n\n"
    
    for api_name, titles in api_files:
        content += f"\n\n---\n\n"
        content += f"## {api_name}\n\n"
        
        for title in titles:
            filepath = find_md_file_by_title(title, md_dir)
            if filepath:
                file_content = read_md_file(filepath)
                if file_content:
                    lines = file_content.split('\n')
                    api_content_lines = []
                    in_content = False
                    for line in lines:
                        if line.startswith('# ') and not in_content:
                            in_content = True
                            continue
                        if in_content:
                            if line.startswith('<!-- '):
                                continue
                            api_content_lines.append(line)
                    content += '\n'.join(api_content_lines) + '\n'
            else:
                content += f"\n> 文档 {title} 未找到\n\n"
    
    with open(chapter_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  创建: {chapter_path}")


def organize_documents():
    """重组文档"""
    print("=" * 60)
    print("AscendC 文档重组工具")
    print("=" * 60)
    
    md_dir = os.path.join(INPUT_DIR, "markdown")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_chapters = 0
    
    for chapter_key, chapter_info in CHAPTER_STRUCTURE.items():
        print(f"\n处理章节: {chapter_info['title']}")
        
        if 'apis' in chapter_info:
            chapter_path = os.path.join(OUTPUT_DIR, f"{chapter_key}_{chapter_info['title']}.md")
            create_chapter_doc(chapter_path, chapter_info['title'], chapter_info['apis'], md_dir)
            total_chapters += 1
        
        if 'subchapters' in chapter_info:
            for sub_key, sub_info in chapter_info['subchapters'].items():
                sub_path = os.path.join(OUTPUT_DIR, chapter_key, f"{sub_key}_{sub_info['title']}.md")
                if 'apis' in sub_info:
                    create_chapter_doc(sub_path, sub_info['title'], sub_info['apis'], md_dir)
                    total_chapters += 1
    
    print(f"\n共创建 {total_chapters} 个章节文档")
    print(f"输出目录: {OUTPUT_DIR}")


def create_index():
    """创建索引文档"""
    index_path = os.path.join(OUTPUT_DIR, "README.md")
    
    content = """# AscendC 算子开发文档

> 来源: 昇腾社区官网
> 文档版本: CANN Community Edition 900beta1

## 文档结构

"""
    
    for chapter_key, chapter_info in CHAPTER_STRUCTURE.items():
        content += f"### [{chapter_info['title']}](./{chapter_key}_{chapter_info['title']}.md)\n\n"
        
        if 'apis' in chapter_info:
            for api_name, titles in chapter_info['apis']:
                content += f"- {api_name}\n"
            content += "\n"
        
        if 'subchapters' in chapter_info:
            for sub_key, sub_info in chapter_info['subchapters'].items():
                content += f"#### [{sub_info['title']}](./{chapter_key}/{sub_key}_{sub_info['title']}.md)\n\n"
                if 'apis' in sub_info:
                    for api_name, titles in sub_info['apis']:
                        content += f"- {api_name}\n"
                    content += "\n"
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n创建索引: {index_path}")


if __name__ == "__main__":
    organize_documents()
    create_index()
