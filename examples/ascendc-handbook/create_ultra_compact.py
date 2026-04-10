#!/usr/bin/env python3
"""
AscendC 文档超精简工具 - 生成LLM超精简版本
只保留函数签名和简要说明，去除冗余内容
"""

import os
import re
from pathlib import Path

INPUT_DIR = "/Volumes/GM9/code/Ascend-MLIR/examples/ascendc-handbook/llm-friendly"
OUTPUT_DIR = "/Volumes/GM9/code/Ascend-MLIR/examples/ascendc-handbook/ultra-compact"


def is_product_table_line(line):
    """判断是否是产品支持表的行"""
    stripped = line.strip()
    
    if stripped.startswith('产品 |') or stripped.startswith('产品|'):
        return True
    if stripped.startswith('| 产品') or stripped.startswith('|产品'):
        return True
    if '是否支持' in stripped and stripped.startswith('|'):
        return True
    if re.match(r'^[\|\-\s:]+$', stripped) and '|' in stripped:
        return True
    
    patterns = [
        r'^\|\s*Atlas',
        r'^\|Atlas',
        r'^Atlas.*\|',
        r'^\|\s*产品',
        r'^\|\s*是否支持',
        r'^\|\s*[√x]',
    ]
    for p in patterns:
        if re.match(p, stripped):
            return True
    
    if stripped.startswith('|') and ('Atlas' in stripped or '产品' in stripped or '是否支持' in stripped or '√' in stripped or 'x' in stripped):
        return True
    
    return False


def should_skip_line(line):
    """判断是否应该跳过该行"""
    stripped = line.strip()
    skip_patterns = [
        '产品支持情况',
        '产品支持',
        '支持产品',
        'Atlas 200',
        'Atlas 300',
        'Atlas 推理',
        'Atlas 训练',
        'Atlas A2',
        'Atlas A3',
        '**父主题',
        '约束说明',
        '注意事项',
        '使用场景',
        '调用示例',
        '功能图示',
        '**图',
        '完整样例链接',
        '展开', '收起',
        '如下样例',
        '本示例',
        '以下为',
    ]
    for pattern in skip_patterns:
        if pattern in stripped:
            return True
    return False


def simplify_table_row(line):
    """精简表格行，只保留参数名和简要描述"""
    stripped = line.strip()
    if not stripped.startswith('|'):
        return line
    
    cells = [c.strip() for c in stripped.split('|')]
    cells = [c for c in cells if c]
    
    if len(cells) < 2:
        return line
    
    if cells[0] in ['参数名', '参数', '字段', '成员名', '成员', '名称']:
        return f"| {cells[0]} | 描述 |"
    
    if cells[0] in ['---', '---:', ':---']:
        return "| --- | --- |"
    
    if len(cells) >= 2:
        desc = cells[1] if len(cells) > 1 else ""
        desc = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', desc)
        desc = desc[:100]
        return f"| {cells[0]} | {desc} |"
    
    return line


def simplify_code_block(code_lines):
    """精简代码块"""
    result = []
    skip_until_brace_close = False
    brace_count = 0
    
    for line in code_lines:
        stripped = line.strip()
        
        if stripped.startswith('//') and len(stripped) > 40:
            continue
        if stripped in ['{', '}', '};', '};']:
            continue
        if stripped.startswith('private:') or stripped.startswith('public:'):
            continue
        if '...' in stripped:
            continue
            
        result.append(line)
    
    return result[:15]


def ultra_compact_content(content):
    """超精简文档内容"""
    
    lines = content.split('\n')
    result = []
    seen_sections = set()
    in_code_block = False
    code_block = []
    in_table = False
    skip_until_next_section = False
    consecutive_empty = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if stripped.startswith('```'):
            if in_code_block:
                code_block = simplify_code_block(code_block)
                if code_block:
                    result.append('```cpp')
                    result.extend(code_block[:15])
                    if len(code_block) > 15:
                        result.append('// ...')
                    result.append('```')
                    result.append('')
                code_block = []
            in_code_block = not in_code_block
            i += 1
            continue
        
        if in_code_block:
            code_block.append(line)
            i += 1
            continue
        
        if stripped.startswith('# '):
            section_key = re.sub(r'#\s*', '', stripped).lower()
            section_key = re.sub(r'简介$', '', section_key).strip()
            
            if section_key in seen_sections:
                skip_until_next_section = True
                i += 1
                continue
            else:
                seen_sections.add(section_key)
                skip_until_next_section = False
        
        if skip_until_next_section:
            if stripped.startswith('## '):
                skip_until_next_section = False
                seen_sections.clear()
            else:
                i += 1
                continue
        
        if should_skip_line(line):
            i += 1
            continue
        
        if is_product_table_line(line):
            i += 1
            continue
        
        if stripped.startswith('|'):
            if stripped.startswith('| ---') or stripped.startswith('|---'):
                in_table = True
                result.append(simplify_table_row(line))
            elif stripped.startswith('| 参数') or stripped.startswith('| 字段') or stripped.startswith('| 成员') or stripped.startswith('| 名称'):
                in_table = True
                result.append(simplify_table_row(line))
            elif in_table:
                result.append(simplify_table_row(line))
            else:
                in_table = False
            i += 1
            continue
        else:
            in_table = False
        
        skip_patterns = [
            '展开', '收起',
            '**父主题',
            '**约束说明**',
            '**注意事项**',
            '约束说明',
            '注意事项',
            '使用场景',
            '调用示例',
            '产品支持',
            '支持的产品',
        ]
        
        if any(p in stripped for p in skip_patterns):
            i += 1
            continue
        
        if stripped.startswith('<!-- ') and stripped.endswith(' -->'):
            i += 1
            continue
        
        if stripped.startswith('![') and '](' in stripped:
            i += 1
            continue
        
        if re.match(r'^\*\*父主题', stripped):
            i += 1
            continue
        
        if stripped == '':
            consecutive_empty += 1
            if consecutive_empty <= 1:
                result.append(line)
            i += 1
            continue
        else:
            consecutive_empty = 0
        
        line = re.sub(r'\[([^\]]+)\]\(atlasascendc_api_[^)]+\.html\)', r'\1', line)
        line = re.sub(r'\[([^\]]+)\]\(/document/detail/[^)]+\)', r'\1', line)
        
        if stripped.startswith('#### ') and len(stripped) > 20:
            keywords = ['原型', '参数', '返回值', '模板', '约束', '注意', '支持']
            if not any(k in stripped for k in keywords):
                i += 1
                continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


def create_ultra_compact_reference():
    """创建超精简API参考"""
    content = """# AscendC API 参考 (超精简版)

## 1. 基础数据结构

### LocalTensor
```cpp
template <typename T> class LocalTensor {
    __aicore__ inline LocalTensor<T>() {};
    __aicore__ inline uint64_t GetPhyAddr() const;
    __aicore__ inline uint32_t GetSize() const;
    __aicore__ inline void SetSize(const uint32_t size);
    __aicore__ inline PrimType GetValue(const uint32_t index) const;
    __aicore__ inline void SetValue(const uint32_t index, const T1 value) const;
    template <typename CAST_T> __aicore__ inline LocalTensor<CAST_T> ReinterpretCast() const;
};
```
支持位置: VECIN, VECOUT, VECCALC, A1, A2, B1, B2, CO1, CO2

### GlobalTensor
```cpp
template <typename T> class GlobalTensor {
    __aicore__ inline GlobalTensor<T>() {}
    __aicore__ inline void SetGlobalBuffer(T* buffer, uint64_t bufferSize);
    __aicore__ inline T GetValue(const uint64_t offset) const;
    __aicore__ inline void SetValue(const uint64_t offset, const T value);
};
```

### TPosition
| 值 | 位置 | 说明 |
|---|---|---|
| 0 | GM | Global Memory |
| 1 | A1 | L1 input A |
| 2 | A2 | L0A input A |
| 3 | B1 | L1 input B |
| 4 | B2 | L0B input B |
| 7 | CO1 | L0C output |
| 9 | VECIN | UB input |
| 10 | VECOUT | UB output |
| 11 | VECCALC | UB compute |

## 2. 基础API

### 数据搬运
```cpp
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, const DataCopyParams& params);
void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, const DataCopyParams& params);
```

### 矢量计算
```cpp
void Exp(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Ln(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Abs(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Sqrt(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Relu(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Add(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);
void Sub(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);
void Mul(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);
void Div(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);
void Max(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);
void Min(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);
void Cast(LocalTensor<dst_T> dst, LocalTensor<src_T> src, const uint32_t count);
```

### 资源管理
```cpp
class TPipe { void InitBuffer(TQue& que, uint8_t num, uint32_t len); };
class TQue { LocalTensor<T> AllocTensor(); void FreeTensor(LocalTensor<T>& t); };
class TBuf { LocalTensor<T> Get(); };
```

### 同步控制
```cpp
void SyncAll();
void PipeBarrier<PIPE_ALL>();
void WaitPreBlock();
void NotifyNextBlock();
```

### 系统接口
```cpp
uint32_t GetBlockNum();
uint32_t GetBlockIdx();
uint8_t GetArchVersion();
```

## 3. 高阶API

### 归一化
```cpp
void LayerNorm(LocalTensor<T> dst, LocalTensor<T> mean, LocalTensor<T> variance,
               LocalTensor<T> src, LocalTensor<T> gamma, LocalTensor<T> beta, const LayerNormParams& params);
void RmsNorm(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> gamma, LocalTensor<T> beta, const RmsNormParams& params);
```

### 激活函数
```cpp
void SoftMax(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> work, const SoftMaxParams& params);
void Gelu(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Sigmoid(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
```

### 矩阵计算
```cpp
class Matmul {
    void Init(TBuf<> aBuf, TBuf<> bBuf, TBuf<> cBuf);
    void SetTensorA(LocalTensor<A_TYPE> a);
    void SetTensorB(LocalTensor<B_TYPE> b);
    void IterateAll(LocalTensor<C_TYPE> c);
    void End();
};
```

## 4. 代码模板

```cpp
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void kernel(GM_ADDR x, GM_ADDR y) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> que;
    pipe.InitBuffer(que, 2, 1024 * sizeof(float));
    
    GlobalTensor<float> gmX, gmY;
    gmX.SetGlobalBuffer((__gm__ float*)x, len);
    gmY.SetGlobalBuffer((__gm__ float*)y, len);
    
    for (uint32_t i = 0; i < len; i += 1024) {
        LocalTensor<float> local = que.AllocTensor<float>();
        DataCopy(local, gmX[i], 1024);
        // 计算...
        DataCopy(gmY[i], local, 1024);
        que.FreeTensor(local);
    }
}
```
"""
    return content


def process_directory(input_dir, output_dir):
    """处理整个目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    total_original = 0
    total_compressed = 0
    
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        out_path = os.path.join(output_dir, rel_path) if rel_path != '.' else output_dir
        os.makedirs(out_path, exist_ok=True)
        
        for file in files:
            if file.endswith('.md'):
                input_file = os.path.join(root, file)
                output_file = os.path.join(out_path, file)
                
                with open(input_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_size = len(content)
                cleaned = ultra_compact_content(content)
                compressed_size = len(cleaned)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                
                total_original += original_size
                total_compressed += compressed_size
                
                ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
                print(f"  {file}: {original_size//1024}KB -> {compressed_size//1024}KB (-{ratio:.1f}%)")
    
    ref_file = os.path.join(output_dir, "API_REFERENCE.md")
    with open(ref_file, 'w', encoding='utf-8') as f:
        f.write(create_ultra_compact_reference())
    print(f"  创建: API_REFERENCE.md")
    
    return total_original, total_compressed


def main():
    print("=" * 60)
    print("AscendC 文档超精简工具")
    print("=" * 60)
    
    print("\n处理文件...")
    original, compressed = process_directory(INPUT_DIR, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"原始大小: {original // 1024} KB")
    print(f"超精简后: {compressed // 1024} KB")
    print(f"压缩率: {(1 - compressed / original) * 100:.1f}%")
    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
