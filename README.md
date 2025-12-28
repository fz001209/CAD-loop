# Multi-AI-Agenten-System (MAAS)
本仓库实现一个“主通路”多 Agent CAD 生成与验证流水线（MAAS），流程严格按以下顺序执行：
**1 Planner → 2 CADWriter → 3 Executor → 4A Verifer → 5 Optimizer → 6 Memory**
> 说明：各 Agent 仅返回 JSON/代码文本；所有文件写入（plan.json、event*.json、cad_script.py 等）均由 pipeline.py 完成。
## 1. 功能概览
### 流程图
[Flussdiagramm](docs/Flussdiagramm.drawio.png)

### 主通路 Agent（本仓库已实现）
- **Agent 1（Planner）**  
  Imput：'Anforderungsliste.yaml'
  Output：'plan.json'+'event1.json'

- **Agent 2（CADWriter）**  
  Imput：'plan.json' 
  Output：'cad_script.py'+'event2.json'

- **Agent 3（Executor）**  
  Imput：'cad_script.py'  
  Output：'output_manifest.json'+'event3.json'

- **Agent 4A（Verifier）**  
  Imput：'plan.json'+'output_manifest.json'
  Output：'verify_report.json'+'event4A.json'  
  说明：“若 png_paths[] 非空且文件可读，则进行 MLLM 评审；若渲染失败导致无可用图片，则仅执行输出一致性与约束字段完整性检查，并在 issues[] 中标记                           missing_render_evidence。”

- **Agent 5（Optimizer）**  
  Imput：'verify_report.json'+'plan.json'+（可选）'cad_script.py'  
  Output：'opt_patch.json'+'event5.json' 
  说明：失败时决定回路:'next_step = 1'（回到规划）或'next_step = 2'（只修脚本）。通过时输出'noop'patch 以保持事件闭环。

- **Agent 6（Memory）**  
  Imput：本轮全部输入输出与事件记录  
  Output：  
  - 'events_merged.json'（合并所有 event*.json）  
  - 'final_model.zip'（包含最终模型文件与证据图）  
  同时把**所有实体文件**归档到 'memory'（含用户输入、每个 agent 输出、每个事件文件）。


## 2. 仓库目录结构

requirements.txt
README.md
MAAS/
  src/
    main.py
    agents/
      create_agents.py
      pipeline.py
    utils/
      fs.py
      events.py
      schemas.py
      render_stub.py
  config/
    system_message.yaml
    models_config.yaml
  workspace/                  # 运行时自动创建（加入 .gitignore）
    runs/
      <run_id>/
        input/
        artifacts/
        memory/

### 注释
-**1.核心代码层 (src/)**

main.py:
作用: 程序的入口点（Entry Point）。
功能: 负责解析命令行参数，读取配置，初始化环境，并启动pipeline。

agents/:
  create_agents.py:
  作用: 智能体工厂。
  功能: 负责实例化AI Agent。这里会定义Agent的角色（例如：Coder, Reviewer, Planner），并为它们绑定特定的LLM模型和工具。
  pipeline.py:
  作用: 任务编排/工作流。
  功能: 定义 Agent 之间如何协作。例如：用户输入 -> 规划 Agent -> 编码 Agent -> 审查 Agent -> 输出结果。它控制着数据的流向。

utils/ (工具库):
  fs.py: (File System) 处理文件读写操作，确保Agent能安全地保存代码或读取文档。
  events.py: 事件处理。用于日志记录（Logging）或在 Agent 完成某一步骤时触发回调（比如在终端打印进度条）。
  schemas.py: 数据结构定义。通常使用Pydantic定义输入输出的格式（JSON Schema），确保 Agent之间传递的数据格式正确。
  render_stub.py: 模板渲染。可能用于将动态变量填入提示词（Prompt）模板中，或者格式化输出结果。

-**2.配置层 (config/)将配置与代码分离，方便调整AI的行为而无需修改代码。**

system_message.yaml:
作用: 提示词工程（Prompt Engineering）的核心。
内容: 存放 Agent 的“人设”和系统指令。例如：“你是一个 Python 专家，请只输出代码...”。

models_config.yaml:
作用: 模型配置。
内容: 存放 API Key（通常引用环境变量）、模型名称（GPT-4, Claude-3 等）、温度（Temperature）等参数。

-**3.工作目录，空间 (workspace/)这是项目运行时的“工作台”。添加到.gitignore 中，因为它包含的是动态生成的数据，避免无意义保存。**

runs/:
作用: 历史记录容器。
设计模式: 每次运行程序，系统会生成一个唯一的 <run_id>（如时间戳），保证不同任务的数据互不干扰。

  <run_id>/ (单次运行实例):
  input/: 存放用户本次上传的文件或原始需求。
  artifacts/: (产物) Agent 生成的最终结果。比如生成的代码文件、Markdown 报告、图表等。
  memory/: (记忆) 存放对话历史或向量数据库索引。如果 Agent 需要“记住”之前的几轮对话，数据会存在这里。

-**4.项目根目录文件**
requirements.txt:列出了项目依赖的 Python 库（如 openai, langchain, pydantic, pyyaml 等）。

README.md:说明文档，介绍如何安装和使用。

## Zusammenfassung：数据流向示例在这个结构下，当你运行程序时，流程通常是这样的：
启动: main.py 读取 config/ 中的配置。
初始化: create_agents.py 根据配置创建 Agent 实例。
运行: workspace/runs/<new_id> 文件夹被创建。
执行: pipeline.py 接收用户输入，指挥 Agent 工作。
Agent 读取 system_message.yaml 获取指令。
Agent 使用 utils/fs.py 将生成的代码写入 workspace/.../artifacts/。
Agent 将对话存入 workspace/.../memory/。
结束: 用户在 artifacts 文件夹中查看最终成果。
