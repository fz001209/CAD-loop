# Multi-AI-Agenten-System (MAAS) - CAD生成与验证流水线

本仓库实现一个"主通路"多Agent CAD生成与验证流水线（MAAS）。  
**人类用户只需输入 Anforderungsliste.yaml（需求文件）**，系统自动完成：规划 → IR编译 → 确定性执行建模 → 渲染证据 → MLLM评审 → 迭代优化 → 归档打包。

> **核心设计理念**：采用 **路线（IR/DSL + 确定性执行器）**，将LLM的"代码生成"职责转变为"结构描述"职责，显著提升复杂模型的稳定性与可调试性。

---

## 📋 目录结构详解

### 1. 核心代码层 (`src/`)
代码与逻辑实现的核心区域。

- **`main.py`** - **程序入口点**
  - **作用**: 命令行接口、配置解析、运行环境初始化
  - **功能**: 解析用户输入参数，读取配置文件，创建工作空间，启动流水线

- **`agents/`** - **智能体实现**
  - **`create_agents.py`** - **智能体工厂**
    - **作用**: 实例化各个AI Agent，绑定特定角色和能力
    - **功能**: 为每个Agent分配LLM模型、配置系统提示词、定义可用工具
  - **`pipeline.py`** - **工作流编排器**
    - **作用**: 定义Agent之间的协作流程和数据流向
    - **功能**: 严格执行 `1 → 2 → 3 → 4A → 5 → 6` 的主通路顺序，控制状态转移
  - **`ir_executor.py`** - **确定性执行器** ⭐ Route B关键
    - **作用**: 将IR（中间表示）转换为具体的CAD几何
    - **功能**: 实现白名单操作集的确定性执行，生成可调试的执行轨迹

- **`utils/`** - **工具库**
  - **`fs.py`** - 文件系统操作
    - 安全读写文件，管理工作空间目录结构
  - **`events.py`** - 事件处理
    - 记录Agent执行日志，触发回调通知
  - **`schemas.py`** - 数据结构定义
    - 使用Pydantic定义所有JSON Schema，确保数据格式一致性
  - **`render_stub.py`** - 渲染模板
    - 从STL生成多视角PNG证据图

### 2. 配置层 (`config/`)
将配置与代码分离，便于调整AI行为。

- **`system_message.yaml`** - **提示词工程核心**
  - 定义每个Agent的"人设"和系统指令
  - 例如："你是一个CAD IR编译器，请输出严格的JSON..."
- **`models_config.yaml`** - **模型配置**
  - API密钥管理（引用环境变量）
  - 模型选择（GPT-4o、Claude等）
  - 温度（Temperature）等参数设置

### 3. 工作空间 (`workspace/`) ⭐ 运行时动态生成
每次运行的工作目录，自动创建。
    workspace/runs/
    ├── <run_id>/ # 单次运行实例（基于时间戳）
    │ ├── input/ # 用户输入文件副本
    │ ├── artifacts/ # 各Agent的生成产物
    │ │ └── attempt_01/ # 每次尝试的完整记录
    │ │ ├── plan.json # Agent1输出
    │ │ ├── ir.json # Agent2输出（关键IR文件）
    │ │ ├── model.step # Agent3生成的CAD模型
    │ │ ├── render/*.png # 多视角渲染证据图
    │ │ ├── output_manifest.json # 执行器详细报告
    │ │ └── ... # 其他中间文件
    │ └── memory/ # 归档存储
    │ ├── events/ # 所有事件记录
    │ ├── events_merged.json # 合并的事件日志
    │ └── final_model.zip # 最终打包成果

### 4. 项目根目录文件
- **`requirements.txt`** - Python依赖库列表
- **`README.md`** - 项目说明文档

---

## 说明 为什么采用（IR/DSL + 确定性执行）而不采用LLM直接生成CadQuery脚本

**问题背景**: 当模型稍复杂时，直接让LLM生成CadQuery脚本往往出现：
- 语法/API不一致导致运行失败
- 几何布尔运算稳定性差
- 迭代修复困难（脚本大、修改点难定位）

**解决方案**: 将"生成代码"改为"生成结构化IR"
- **LLM负责**: 需求编译 → 结构化、可验证、可修补的IR（JSON）
- **执行器负责**: 确定性解释IR → 生成CAD几何（固定CadQuery后端）
- **优势**: 将失败原因从"模型写错脚本"转变为"第N步op参数不合理"，实现可收敛的自动迭代

---

## 🔄 主通路流程（严格执行）
1 Planner → 2 CADWriter(IR Compiler) → 3 Executor(IR Runtime) → 4A Verifier → 5 Optimizer → 6 Memory


### 智能体职责详解

| Agent | 角色 | 输入 | 输出 | 关键职责 |
|-------|------|------|------|----------|
| **1 Planner** | 规划师 | `Anforderungsliste.yaml` | `plan.json` | 解析YAML需求，输出结构化计划草案 |
| **2 CADWriter** | IR编译器 | `plan.json` | `ir.json` | 规范化、补齐为**可执行IR** |
| **3 Executor** | 确定性执行器 | `ir.json` | `output_manifest.json`<br>+模型文件+渲染图 | 执行IR，生成CAD模型，收集几何指标 |
| **4A Verifier** | MLLM验证器 | `plan.json` + `output_manifest.json` | `verify_report.json` | 基于渲染图像严格验证特征 |
| **5 Optimizer** | 优化器 | 验证报告+IR+执行轨迹 | `opt_patch.json` | 生成unified diff补丁，决定迭代路径 |
| **6 Memory** | 归档器 | 所有输入输出 | 合并事件+最终打包 | 完整归档，支持复现 |

---

## 📊 IR（中间表示）规范

### 核心结构
```json
{
  "object": "描述",
  "required_features": [
    {"id": "RF01", "name": "特征名", "must": true, "notes": "说明"}
  ],
  "params": {"参数": "值"},
  "operations": [
    {
      "id": "OP01",                          // 必填：稳定唯一，便于追踪
      "op": "primitive_box",                // 白名单操作
      "args": {
        "x_mm": 10, "y_mm": 10, "z_mm": 10,
        "out_id": "BODY"                    // 推荐：命名中间结果
      }
    }
  ]
}
```

### 强制规则
1. **操作顺序**: 先草图(`sketch_*`) → 再拉伸(`extrude`/`cut_extrude`)
2. **布尔运算**: `union`/`cut`/`intersect`必须提供`args.a_id`和`args.b_id`
3. **特征操作**: 必须有可操作的实体引用
4. **唯一标识**: 每个操作必须有`id`，便于调试和修补

### 白名单操作集
| 类别 | 操作 |
|------|------|
| **基本体** | `primitive_box`, `primitive_cylinder`, `primitive_sphere` |
| **草图** | `sketch_rect`, `sketch_circle`, `sketch_polygon` |
| **实体操作** | `extrude`, `cut_extrude` |
| **变换** | `translate`, `rotate` |
| **特征** | `hole`, `fillet`, `chamfer`, `shell` |
扩展建议: 在保持确定性的前提下，可逐步扩展revolve、sweep、loft等高级操作。

## ⚙️ 运行指南

### 1. 安装依赖
```bash
# 创建虚拟环境
python -m venv .venv

# 激活（Linux/macOS）
source .venv/bin/activate

# 激活（Windows PowerShell）
.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行主通路
```bash
# 基础运行
python main.py --input test_1.yaml
```
# 完整参数示例
python main.py \
  --input test_1.yaml \
  --workspace workspace \
  --model gpt-4o-mini \
  --model_4a gpt-4o

### 3. 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 需求YAML文件路径 | 必填 |
| `--workspace` | 工作空间根目录 | `"workspace"` |
| `--model` | Agent1/2/5/6使用模型 | `"gpt-4o-mini"` |
| `--model_4a` | Agent4A多模态模型 | `"gpt-4o"` |
| `--max_attempts` | 最大迭代尝试次数 | `3` |

## 🔍 调试与问题排查
常见失败类型及排查
执行失败 → 查看exec.log.txt

op_trace字段定位具体失败操作

常见错误：extrude requires an active sketch（操作顺序错误）

几何异常 → 查看output_manifest.json

geometry_metrics.bbox检查尺寸比例

triangle_count评估模型复杂度

验证失败 → 查看verify_report.json

feature_evidence_map检查特征验证情况

issues数组查看具体问题描述

### 调试建议
1. **优先查看执行轨迹**: `exec.log.txt`中的`op_trace`是最直接的调试信息
2. **检查几何指标**: 异常的bbox尺寸通常意味着参数单位混乱
3. **查看渲染证据**: 验证器依赖PNG图像，确保渲染成功
4. **利用patch机制**: Agent5生成的unified diff可以精确修复IR问题

## 🎯 设计约束与保证
✅ 本系统保证
最小用户干预: 只需提供YAML需求文件

全自动流程: 6个Agent协同完成完整CAD工作流

完整可追溯: 每次运行生成完整的归档记录

可调试迭代: 失败可定位到具体操作层面

确定性执行: IR执行器保证结果可复现

⚠️ 当前限制（Minimal v1）
不保证一次成功: 复杂模型可能需要多次迭代

有限操作集: 仅支持基础CAD操作（可扩展）

依赖MLLM质量: 验证准确性受多模态模型能力影响

## 📈 Roadmap与演进方向
短期改进
扩展操作集: 支持revolve、sweep、loft等高级建模

增强装配语义: 添加attach_face等装配级操作

优化验证精度: 改进MLLM提示词，减少误判

中期目标
参数化设计: 支持参数约束和优化

制造可行性: 添加可制造性检查（壁厚、倒角等）

性能优化: 并行执行、缓存机制

长期愿景
多学科优化: 结合结构分析、流体分析

知识积累: 构建设计模式库

交互式设计: 支持人类专家中途干预

## Zusammenfassung: 数据流向示例
text
用户输入
    ↓
main.py解析参数
    ↓
创建工作空间(<run_id>)
    ↓
pipeline.py启动流水线
    ↓
Agent1读取YAML → plan.json
    ↓
Agent2编译IR → ir.json
    ↓
Agent3执行IR → CAD模型+渲染图
    ↓
Agent4A验证 → pass/fail报告
    ↓
如果需要优化 → Agent5生成补丁
    ↓
循环或继续 → Agent6归档所有结果
    ↓
用户获得final_model.zip

## 🏗️ 架构哲学
本系统遵循以下设计原则：

关注点分离: LLM负责"描述"，执行器负责"实现"

确定性优先: 关键路径（IR执行）必须确定、可复现

可调试性: 每个操作都有完整轨迹记录

渐进增强: 从最小可行产品开始，逐步扩展能力

完整归档: 每次运行都是完整的实验记录

通过这种架构，我们既能利用LLM的理解和生成能力，又能避免其在复杂几何建模中的不稳定表现，实现工业级CAD生成的可靠性和可扩展性。
