from __future__ import annotations

import argparse
import json
from pathlib import Path

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

    # base paths (for memory/input root)
    base_paths = stage_paths(run_dir)

    # copy user input into run/input
    user_input_src = Path(args.input).resolve()
    user_input_dst = base_paths["input"] / user_input_src.name
    copy_file(user_input_src, user_input_dst)

    # --------- LLM CONFIG（最小可跑版）---------
    # 在环境变量里设置 OPENAI_API_KEY。
    llm_config = {
        "temperature": 0.2,
        "config_list": [
            {
                "model": args.model,  # e.g. "gpt-4o-mini"
                "api_type": "openai",
            }
        ],
    }

    agents = create_mainpath_agents(llm_config)
    user, a1, a2, a3, a4, a5, a6 = agents

    MAX_ATTEMPTS = 3

    # 控制流：下一轮从 1 或 2 开始
    next_step = "1"  # "1" or "2"
    plan_path: Path | None = None

    success = False
    all_final_candidates: list[Path] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        paths = stage_paths(run_dir, attempt=attempt)

        # 1) Planner：若 next_step=1 或 plan 不存在则重新规划
        if next_step == "1" or plan_path is None or not plan_path.exists():
            plan_path = agent1_planner(run_id, paths, user_input_dst, a1, attempt=attempt)

        # 2) CADWriter：生成（若上一轮有 opt_patch，会自动注入 prompt）
        cad_script_path = agent2_cad_writer(run_id, paths, plan_path, a2, attempt=attempt)

        # 3) Executor：执行脚本并生成 output_manifest + exec.log
        manifest_path = agent3_executor(run_id, paths, cad_script_path, a3, attempt=attempt)

        # 收集本轮固定产物（无论成功失败都保留）
        artifacts = paths["artifacts"]
        all_final_candidates += [
            artifacts / "plan.json",
            artifacts / "cad_script.py",
            artifacts / "output_manifest.json",
            artifacts / "exec.log.txt",
        ]
        all_final_candidates += list(artifacts.glob("*.step")) + list(artifacts.glob("*.stp")) + list(artifacts.glob("*.stl"))
        all_final_candidates += list((artifacts / "render").glob("*.png"))

        # 读 manifest 判定 3 是否成功
        try:
            manifest_data = json.loads((artifacts / "output_manifest.json").read_text(encoding="utf-8"))
        except Exception:
            manifest_data = {"status": "fail"}

        # 3 fail → 直接 5（不进 4A）
        if manifest_data.get("status") != "success":
            opt_path = agent5_optimizer(
                run_id=run_id,
                paths=paths,
                plan_path=plan_path,
                cad_script_path=cad_script_path,
                manifest_path=manifest_path,
                verify_path=None,     # 关键：3 fail 没有 verify
                agent=a5,
                attempt=attempt,
            )
            all_final_candidates.append(opt_path)

            decision = json.loads(opt_path.read_text(encoding="utf-8"))
            next_step = str(decision.get("next_step", "2"))
            if next_step not in ("1", "2"):
                next_step = "2"
            continue

        # 3 success → 4A
        verify_path = agent4a_verifier(run_id, paths, plan_path, manifest_path, a4, attempt=attempt)
        all_final_candidates.append(verify_path)

        report = json.loads(verify_path.read_text(encoding="utf-8"))
        if report.get("status") == "pass":
            success = True
            break

        # 4A fail → 5
        opt_path = agent5_optimizer(
            run_id=run_id,
            paths=paths,
            plan_path=plan_path,
            cad_script_path=cad_script_path,
            manifest_path=manifest_path,
            verify_path=verify_path,
            agent=a5,
            attempt=attempt,
        )
        all_final_candidates.append(opt_path)

        decision = json.loads(opt_path.read_text(encoding="utf-8"))
        next_step = str(decision.get("next_step", "2"))
        if next_step not in ("1", "2"):
            next_step = "2"
        continue

    # 6) Memory：无论成功与否都归档（包含所有 attempt 的 artifacts+events）
    final_zip, merged_events = agent6_memory(
        run_id=run_id,
        base_paths=stage_paths(run_dir),  # 归档用 base paths
        user_input_path=user_input_dst,
        final_files_to_package=all_final_candidates,
        agent=a6,
    )

    print("\n=== DONE ===")
    print("run_id:", run_id)
    print("success:", success)
    print("final_model.zip:", final_zip)
    print("events_merged.json:", merged_events)
    print("memory_dir:", stage_paths(run_dir)["memory"])


if __name__ == "__main__":
    main()
