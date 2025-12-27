from __future__ import annotations

import argparse
from pathlib import Path

from autogen import GroupChat, GroupChatManager

from src.agents.create_agents import create_mainpath_agents
from src.agents.pipeline import (
    stage_paths,
    agent1_planner,
    agent2_cad_writer,
    agent3_executor,
    agent4a_verifier,
    agent5_optimizer,
    agent6_memory,
)
from src.utils.events import new_run_id
from src.utils.fs import ensure_dir, copy_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Anforderungsliste.yaml")
    parser.add_argument("--workspace", default="workspace", help="Workspace root")
    parser.add_argument("--model", default="gpt-4o", help="Model name in your llm config")
    args = parser.parse_args()

    run_id = new_run_id()
    ws = Path(args.workspace)
    run_dir = ensure_dir(ws / "runs" / run_id)
    paths = stage_paths(run_dir)

    # copy user input into run/input
    user_input_src = Path(args.input).resolve()
    user_input_dst = paths["input"] / user_input_src.name
    copy_file(user_input_src, user_input_dst)

    # --------- LLM CONFIG（这里先给最小可改版）---------
    # 在环境变量里设置 OPENAI_API_KEY 。
    llm_config = {
        "temperature": 0.2,
        "config_list": [
            {
                "model": args.model,  # e.g. "gpt-4o"
                "api_type": "openai",
            }
        ],
    }

    agents = create_mainpath_agents(llm_config)

    # 按流程逐步调用对应 agent。
    user, a1, a2, a3, a4, a5, a6 = agents

    # 1
    plan_path = agent1_planner(run_id, paths, user_input_dst, a1)

    # 2
    cad_script_path = agent2_cad_writer(run_id, paths, plan_path, a2)

    # 3
    manifest_path = agent3_executor(run_id, paths, cad_script_path, a3)

    # 4A
    verify_path = agent4a_verifier(run_id, paths, plan_path, manifest_path, a4)

    # 5 (only if fail, but仍然产出 opt_patch.json，便于事件闭环)
    opt_path = agent5_optimizer(run_id, paths, plan_path, cad_script_path, verify_path, a5)

    # 6 (最终输出包 + 总事件记录给 user)
    # final_files_to_package：按主通路，至少 step/stl + verify + plan + script + manifest
    artifacts = paths["artifacts"]
    final_candidates = [
        artifacts / "plan.json",
        artifacts / "cad_script.py",
        artifacts / "output_manifest.json",
        artifacts / "verify_report.json",
        artifacts / "opt_patch.json",
    ]
    # step/stl
    final_candidates += list(artifacts.glob("*.step")) + list(artifacts.glob("*.stp")) + list(artifacts.glob("*.stl"))

    final_zip, merged_events = agent6_memory(
        run_id,
        paths,
        user_input_dst,
        final_candidates,
        a6,
    )

    print("\n=== DONE ===")
    print("run_id:", run_id)
    print("final_model.zip:", final_zip)
    print("events_merged.json:", merged_events)
    print("memory_dir:", paths["memory"])


if __name__ == "__main__":
    main()
