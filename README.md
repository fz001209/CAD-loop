# Multi-AI-Agenten-System (MAAS)
本仓库实现一个“主通路”多 Agent CAD 生成与验证流水线（MAAS），流程严格按以下顺序执行：
**1 Planner → 2 CADWriter → 3 Executor → 4A Verifer → 5 Optimizer → 6 Memory**

## 1. 功能概览
## 流程图
[Flussdiagramm](docs/Flussdiagramm.png)

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

README.md
CAD-loop/
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
  workspace/                  # 运行时自动创建（建议加入 .gitignore）
    runs/
      <run_id>/
        input/
        artifacts/
        memory/
  requirements.txt
