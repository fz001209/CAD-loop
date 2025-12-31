from __future__ import annotations

import sys
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.fs import ensure_dir, write_json, write_text, read_json, copy_file, copy_tree
from src.utils.events import build_event
from src.utils.render_stub import list_rendered_images


def _extract_first_json(text: str) -> Dict[str, Any]:
    """
    允许模型输出纯 JSON，或夹带少量文字；这里做稳健提取。
    """
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def _safe_run(cmd: List[str], cwd: str | Path, timeout: int = 120) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        text=True,
        check=False,
    )
    return p.returncode, p.stdout


def _read_text_tail(path: Path, max_chars: int = 6000) -> str:
    if not path.exists():
        return ""
    t = path.read_text(encoding="utf-8", errors="ignore")
    if len(t) <= max_chars:
        return t
    return t[-max_chars:]


def stage_paths(base_run_dir: Path, attempt: int | None = None) -> Dict[str, Path]:
    """
    attempt=None: base paths (input/memory root)
    attempt=k:    artifacts/events under attempt subfolders to avoid overwrite
    """
    run_dir = ensure_dir(base_run_dir)
    input_dir = ensure_dir(run_dir / "input")
    memory_dir = ensure_dir(run_dir / "memory")

    if attempt is None:
        artifacts_dir = ensure_dir(run_dir / "artifacts")
        events_dir = ensure_dir(memory_dir / "events")
    else:
        artifacts_dir = ensure_dir(run_dir / "artifacts" / f"attempt_{attempt:02d}")
        events_dir = ensure_dir(memory_dir / "events" / f"attempt_{attempt:02d}")

    return {
        "run": run_dir,
        "input": input_dir,
        "artifacts": artifacts_dir,
        "memory": memory_dir,
        "events": events_dir,
    }


def agent1_planner(run_id: str, paths: Dict[str, Path], anforderungsliste_path: Path, agent, attempt: int) -> Path:
    plan_path = paths["artifacts"] / "plan.json"
    event_path = paths["events"] / "event1.json"

    ev_start = build_event(
        run_id, "1", "Agent1_Planner", "start",
        inputs={"anforderungsliste": str(anforderungsliste_path), "attempt": attempt},
        outputs={"plan_json": str(plan_path), "event": str(event_path)},
    )
    write_json(event_path, ev_start)

    # 给模型：你可以让它只返回 JSON 内容；文件由代码写。
    prompt = {
        "run_id": run_id,
        "attempt": attempt,
        "input_file": str(anforderungsliste_path),
        "instructions": "Read the YAML file content and return plan.json as STRICT JSON only.",
    }
    resp = agent.generate_reply(messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])
    plan = _extract_first_json(resp if isinstance(resp, str) else resp.get("content", ""))

    plan["run_id"] = run_id
    plan["attempt"] = attempt

    write_json(plan_path, plan)

    ev_done = build_event(
        run_id, "1", "Agent1_Planner", "success",
        inputs={"anforderungsliste": str(anforderungsliste_path), "attempt": attempt},
        outputs={"plan_json": str(plan_path)},
        message="plan.json created",
    )
    write_json(event_path, ev_done)
    return plan_path


def _auto_fix_cadquery_exports(code: str) -> str:
    """
    兜底修复：把旧 CadQuery 写法 exportStep/exportStl 替换为 2.6+ 的 export_step/export_stl。
    仅作为最后防线，推荐你同时修改 system_message.yaml 的 Agent2。
    """
    fixed = code

    # 常见旧写法：result.exportStep("x.step")
    fixed = re.sub(r"\.exportStep\s*\(", ".val().export_step(", fixed)
    fixed = re.sub(r"\.exportStl\s*\(", ".val().export_stl(", fixed)

    # 也有人写 export_step 但少了 val()
    fixed = re.sub(r"(\bresult)\.export_step\s*\(", r"\1.val().export_step(", fixed)
    fixed = re.sub(r"(\bresult)\.export_stl\s*\(", r"\1.val().export_stl(", fixed)

    return fixed


def agent2_cad_writer(run_id: str, paths: Dict[str, Path], plan_path: Path, agent, attempt: int) -> Path:
    cad_script_path = paths["artifacts"] / "cad_script.py"
    event_path = paths["events"] / "event2.json"

    write_json(event_path, build_event(
        run_id, "2", "Agent2_CADWriter", "start",
        inputs={"plan_json": str(plan_path), "attempt": attempt},
        outputs={"cad_script": str(cad_script_path), "event": str(event_path)},
    ))

    plan = read_json(plan_path)

    # 读取上一轮 opt_patch（如果存在），注入给 Agent2 用于修复脚本
    prev_opt_patch = None
    if attempt > 1:
        prev_opt = paths["run"] / "artifacts" / f"attempt_{attempt-1:02d}" / "opt_patch.json"
        if prev_opt.exists():
            try:
                prev_opt_patch = read_json(prev_opt)
            except Exception:
                prev_opt_patch = None

    prompt = {
        "run_id": run_id,
        "attempt": attempt,
        "plan_json": plan,
        "previous_opt_patch": prev_opt_patch,
        "instructions": (
            "Generate a single, complete CadQuery 2.6+ Python script that creates the model. "
            "The script will be executed with CWD=artifacts folder. Therefore export to filenames without any directory prefix. "
            "MUST export STEP and STL using CadQuery 2.6+ API: result.val().export_step('model.step') and result.val().export_stl('model.stl'). "
            "If previous_opt_patch exists, you MUST apply it."
        ),
    }

    resp = agent.generate_reply(messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])
    text = resp if isinstance(resp, str) else resp.get("content", "")

    code = text
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    if m:
        code = m.group(1).strip()

    # 兜底：自动修正旧导出 API
    code = _auto_fix_cadquery_exports(code)

    ensure_dir(paths["artifacts"] / "render")
    write_text(cad_script_path, code)

    write_json(event_path, build_event(
        run_id, "2", "Agent2_CADWriter", "success",
        inputs={"plan_json": str(plan_path), "attempt": attempt},
        outputs={"cad_script": str(cad_script_path)},
        message="cad_script.py created",
    ))
    return cad_script_path


def agent3_executor(run_id: str, paths: Dict[str, Path], cad_script_path: Path, agent, attempt: int) -> Path:
    manifest_path = paths["artifacts"] / "output_manifest.json"
    exec_log_path = paths["artifacts"] / "exec.log.txt"
    event_path = paths["events"] / "event3.json"

    write_json(event_path, build_event(
        run_id, "3", "Agent3_Executor", "start",
        inputs={"cad_script": str(cad_script_path), "attempt": attempt},
        outputs={"output_manifest": str(manifest_path), "exec_log": str(exec_log_path), "event": str(event_path)},
    ))

    # 关键修复：用 cwd=artifacts + 只传脚本文件名，避免路径重复
    script_to_run = Path(cad_script_path)
    rc, out = _safe_run([sys.executable, str(Path(cad_script_path).resolve())], cwd=Path(cad_script_path).parent, timeout=180)
    write_text(exec_log_path, out)

    render_dir = paths["artifacts"] / "render"
    imgs = list_rendered_images(render_dir)

    step_files = [str(p) for p in paths["artifacts"].glob("*.step")] + [str(p) for p in paths["artifacts"].glob("*.stp")]
    stl_files = [str(p) for p in paths["artifacts"].glob("*.stl")]

    manifest = {
        "run_id": run_id,
        "attempt": attempt,
        "status": "success" if rc == 0 else "fail",
        "cad_script": str(cad_script_path),
        "step_files": step_files,
        "stl_files": stl_files,
        "render_images": imgs,
        "exec_log_path": str(exec_log_path),
        "artifacts_dir": str(paths["artifacts"]),
        "return_code": rc,
    }
    write_json(manifest_path, manifest)

    write_json(event_path, build_event(
        run_id, "3", "Agent3_Executor", "success" if rc == 0 else "fail",
        inputs={"cad_script": str(cad_script_path), "attempt": attempt},
        outputs={"output_manifest": str(manifest_path), "exec_log": str(exec_log_path)},
        message="execution finished",
        error="" if rc == 0 else f"Execution failed, return code {rc}",
    ))

    return manifest_path


def agent4a_verifier(run_id: str, paths: Dict[str, Path], plan_path: Path, manifest_path: Path, agent, attempt: int) -> Path:
    verify_path = paths["artifacts"] / "verify_report.json"
    event_path = paths["events"] / "event4A.json"

    write_json(event_path, build_event(
        run_id, "4A", "Agent4A_Verifier", "start",
        inputs={"plan_json": str(plan_path), "output_manifest": str(manifest_path), "attempt": attempt},
        outputs={"verify_report": str(verify_path), "event": str(event_path)},
    ))

    plan = read_json(plan_path)
    manifest = read_json(manifest_path)

    # Prepare multimodal messages if images exist
    image_paths = manifest.get("render_images", []) or list_rendered_images(paths["artifacts"] / "render")

    exec_log_tail = _read_text_tail(Path(manifest.get("exec_log_path", paths["artifacts"] / "exec.log.txt")))

    content_payload: List[Dict[str, Any]] = [{
        "type": "text",
        "text": json.dumps({
            "run_id": run_id,
            "attempt": attempt,
            "plan_json": plan,
            "output_manifest": manifest,
            "exec_log_tail": exec_log_tail,
            "instructions": (
                "Return STRICT JSON for verify_report.json. "
                "If render_images exist, prioritize geometry/feature checks. "
                "Also check constraints/logic using plan + manifest + exec_log_tail. "
                "Output schema: {status: pass|fail, summary, issues[], checks[] }"
            )
        }, ensure_ascii=False)
    }]

    # Attach images (multimodal)
    for p in image_paths[:8]:
        content_payload.append({"type": "image_url", "image_url": {"url": f"file://{p}"}})

    resp = agent.generate_reply(messages=[{"role": "user", "content": content_payload}])
    text = resp if isinstance(resp, str) else resp.get("content", "")

    report = _extract_first_json(text)
    report.setdefault("status", "fail")
    report.setdefault("summary", "")
    report.setdefault("issues", [])
    report.setdefault("checks", [])
    report["attempt"] = attempt

    # 记录证据
    if isinstance(report.get("checks"), list):
        report["checks"].append(f"render_images_used={len(image_paths)}")
    report["evidence"] = {
        "render_images_used": image_paths,
        "exec_log_tail_included": bool(exec_log_tail),
    }

    write_json(verify_path, report)

    write_json(event_path, build_event(
        run_id, "4A", "Agent4A_Verifier", "success",
        inputs={"plan_json": str(plan_path), "output_manifest": str(manifest_path), "attempt": attempt},
        outputs={"verify_report": str(verify_path)},
        message=f"verify status: {report.get('status')}",
    ))
    return verify_path


def agent5_optimizer(
    run_id: str,
    paths: Dict[str, Path],
    plan_path: Path,
    cad_script_path: Path,
    manifest_path: Path,
    verify_path: Optional[Path],
    agent,
    attempt: int,
) -> Path:
    opt_path = paths["artifacts"] / "opt_patch.json"
    event_path = paths["events"] / "event5.json"

    write_json(event_path, build_event(
        run_id, "5", "Agent5_Optimizer", "start",
        inputs={
            "plan_json": str(plan_path),
            "cad_script": str(cad_script_path),
            "output_manifest": str(manifest_path),
            "verify_report": str(verify_path) if verify_path else "",
            "attempt": attempt,
        },
        outputs={"opt_patch": str(opt_path), "event": str(event_path)},
    ))

    plan = read_json(plan_path)
    manifest = read_json(manifest_path)

    cad_script_text = ""
    try:
        cad_script_text = Path(cad_script_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        cad_script_text = ""

    exec_log_tail = _read_text_tail(Path(manifest.get("exec_log_path", paths["artifacts"] / "exec.log.txt")))

    verify = None
    if verify_path and Path(verify_path).exists():
        try:
            verify = read_json(verify_path)
        except Exception:
            verify = None

    prompt = {
        "run_id": run_id,
        "attempt": attempt,
        "plan_json": plan,
        "output_manifest": manifest,
        "exec_log_tail": exec_log_tail,
        "cad_script": cad_script_text,
        "verify_report": verify,
        "instructions": (
            "You must decide next action for convergence. "
            "If execution failed (manifest.status=fail), prioritize need_fix_script with next_step='2' unless the plan itself is wrong. "
            "If verify_report exists and status=fail, decide need_fix_script or need_replan. "
            "Allowed next_step only: '1' or '2'. "
            "Return STRICT JSON only."
        ),
        "output_schema": {
            "status": "need_fix_script|need_replan",
            "next_step": "1|2",
            "suggestions": [],
            "patch": {"type": "instructions|text_diff", "content": "..."}
        }
    }

    resp = agent.generate_reply(messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])
    patch = _extract_first_json(resp if isinstance(resp, str) else resp.get("content", ""))

    # 强制兜底：不能输出 6；你要求失败必须回 1/2
    patch.setdefault("status", "need_fix_script")
    patch.setdefault("next_step", "2")
    if str(patch.get("next_step")) not in ("1", "2"):
        patch["next_step"] = "2"
    if patch.get("status") not in ("need_fix_script", "need_replan"):
        patch["status"] = "need_fix_script"

    patch["attempt"] = attempt
    write_json(opt_path, patch)

    write_json(event_path, build_event(
        run_id, "5", "Agent5_Optimizer", "success",
        inputs={"attempt": attempt},
        outputs={"opt_patch": str(opt_path)},
        message=f"optimizer next_step: {patch.get('next_step')}",
    ))
    return opt_path


def agent6_memory(
    run_id: str,
    base_paths: Dict[str, Path],
    user_input_path: Path,
    final_files_to_package: List[Path],
    agent,
) -> Tuple[Path, Path]:
    """
    归档策略：
    - memory/input：复制用户输入
    - memory/artifacts：复制整个 run/artifacts（包含 attempt 子目录）
    - memory/events：复制整个 run/memory/events（包含 attempt 子目录）
    - artifacts/events_merged.json：合并所有 event*.json（rglob）
    - artifacts/final_model.zip：把关键产物 + 合并事件 + 所有渲染图打包
    """
    merged_events_path = base_paths["run"] / "artifacts" / "events_merged.json"
    final_zip_path = base_paths["run"] / "artifacts" / "final_model.zip"

    # ensure memory subdirs
    ensure_dir(base_paths["memory"] / "input")
    ensure_dir(base_paths["memory"] / "artifacts")
    ensure_dir(base_paths["memory"] / "events")

    # 1) copy input + artifacts + events
    copy_file(user_input_path, base_paths["memory"] / "input" / user_input_path.name)
    copy_tree(base_paths["run"] / "artifacts", base_paths["memory"] / "artifacts")
    
    # 2) merge events (all attempts)
    events_root = base_paths["memory"] / "events"
    event_files = sorted(events_root.rglob("event*.json"))
    merged = {
        "run_id": run_id,
        "events": [],
    }
    for p in event_files:
        try:
            merged["events"].append(read_json(p))
        except Exception:
            merged["events"].append({
                "run_id": run_id,
                "agent_id": "unknown",
                "step_name": "unknown",
                "status": "fail",
                "timestamp": "",
                "inputs": {},
                "outputs": {},
                "message": "failed to read event json",
                "error": str(p),
            })

    write_json(merged_events_path, merged)
    copy_file(merged_events_path, base_paths["memory"] / "events_merged.json")

    # 3) zip package
    import zipfile
    
    def _safe_arcname(p: Path, base_dir: Path) -> str:
        """把真实路径映射为 zip 内部路径，统一用相对路径，避免重复/混乱。"""
        p = p.resolve()
        try:
            rel = p.relative_to(base_dir.resolve())
            return str(rel).replace("\\", "/")
        except Exception:
            # 兜底：只放文件名（不推荐，但不会炸）
            return p.name

    with zipfile.ZipFile(final_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        seen = set()

        def add_file(file_path: Path, arcname: str):
            arcname = arcname.replace("\\", "/")
            if arcname in seen:
                return
            if not file_path.exists() or not file_path.is_file():
                return
            z.write(file_path, arcname=arcname)
            seen.add(arcname)

        # 1) 打包“memory/artifacts”整棵树（这里最完整，包含 attempt_01/02/03）
        mem_artifacts_dir = (base_paths["memory"] / "artifacts").resolve()  # 注意：应该是base_paths不是paths
        if mem_artifacts_dir.exists():
            for p in mem_artifacts_dir.rglob("*"):
                if p.is_file():
                    arc = "artifacts/" + _safe_arcname(p, mem_artifacts_dir)
                    add_file(p, arc)

        # 2) 打包渲染图（如果你渲染图不在 memory/artifacts 里，也可以保留）
        render_dir = base_paths["artifacts"] / "render"  # 注意：应该是base_paths不是paths
        if render_dir.exists():
            for p in sorted(render_dir.glob("*.png")):
                add_file(p, f"artifacts/render/{p.name}")

        # 3) 打包 merged events
        add_file(merged_events_path, "events_merged.json")

        # 4) （可选）如果你仍想额外塞 final_files_to_package，必须去重 arcname
        #    建议：只塞“顶层最终产物”，不要塞 attempt_x 里的东西，否则必然重复。
        #    这里我按“只塞文件名到 artifacts/”处理，避免重复 attempt_* 路径。
        for f in final_files_to_package:
            f = Path(f)
            if f.exists() and f.is_file():
                add_file(f, f"artifacts/{f.name}")

    copy_file(final_zip_path, base_paths["memory"] / "final_model.zip")
    return final_zip_path, merged_events_path


