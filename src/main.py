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
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name for agents except 4A")
    parser.add_argument("--model_4a", default="gpt-4o", help="Model name for Agent4A_Verifier")
    args = parser.parse_args()

    run_id = new_run_id()
    ws = Path(args.workspace)
    run_dir = ensure_dir(ws / "runs" / run_id)

    # base paths (for input/memory root)
    base_paths = stage_paths(run_dir)

    # copy user input into run/input
    user_input_src = Path(args.input).resolve()
    user_input_dst = base_paths["input"] / user_input_src.name
    copy_file(user_input_src, user_input_dst)

    # LLM config
    llm_config = {
        "temperature": 0.2,
        "config_list": [{"model": args.model, "api_type": "openai"}],
    }
    llm_config_4a = {
        "temperature": 0.2,
        "config_list": [{"model": args.model_4a, "api_type": "openai"}],
    }

    user, a1, a2, a3, a4, a5, a6 = create_mainpath_agents(llm_config, llm_config_4a=llm_config_4a)

    MAX_ATTEMPTS = 3

    next_step = "1"  # "1" or "2"
    plan_path: Path | None = None
    success = False

    all_final_candidates: list[Path] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        paths = stage_paths(run_dir, attempt=attempt)

        # 1) Planner
        if next_step == "1" or plan_path is None or not plan_path.exists():
            plan_path = agent1_planner(run_id, paths, user_input_dst, a1, attempt=attempt)

        # 2) CADWriter
        cad_script_path = agent2_cad_writer(run_id, paths, plan_path, a2, attempt=attempt)

        # 3) Executor
        manifest_path = agent3_executor(run_id, paths, cad_script_path, a3, attempt=attempt)

        artifacts = paths["artifacts"]
        all_final_candidates += [
            artifacts / "plan.json",
            artifacts / "cad_script.py",
            artifacts / "output_manifest.json",
            artifacts / "exec.log.txt",
        ]
        all_final_candidates += list(artifacts.glob("*.step")) + list(artifacts.glob("*.stp")) + list(artifacts.glob("*.stl"))
        all_final_candidates += list((artifacts / "render").glob("*.png"))

        manifest_data = json.loads((artifacts / "output_manifest.json").read_text(encoding="utf-8"))
        if manifest_data.get("status") != "success":
            # 3 fail -> 5 (no 4A)
            opt_path = agent5_optimizer(
                run_id=run_id,
                paths=paths,
                plan_path=plan_path,
                cad_script_path=cad_script_path,
                manifest_path=manifest_path,
                verify_path=None,
                agent=a5,
                attempt=attempt,
            )
            all_final_candidates.append(opt_path)

            decision = json.loads(opt_path.read_text(encoding="utf-8"))
            next_step = str(decision.get("next_step", "2"))
            if next_step not in ("1", "2"):
                next_step = "2"
            continue

        # 3 success -> 4A
        verify_path = agent4a_verifier(run_id, paths, plan_path, manifest_path, a4, attempt=attempt)
        all_final_candidates.append(verify_path)

        report = json.loads(verify_path.read_text(encoding="utf-8"))
        if report.get("status") == "pass":
            success = True
            break

        # 4A fail -> 5
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

    # 6) Memory: always archive
    final_zip, merged_events = agent6_memory(
        run_id=run_id,
        base_paths=stage_paths(run_dir),  # base paths
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

