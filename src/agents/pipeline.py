from __future__ import annotations

import sys
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.fs import ensure_dir, write_json, write_text, read_json, copy_file, copy_tree
from src.utils.events import build_event
from src.utils.render_stub import list_rendered_images


def _extract_first_json(text: str) -> Dict[str, Any]:
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
    return t if len(t) <= max_chars else t[-max_chars:]


def stage_paths(base_run_dir: Path, attempt: int | None = None) -> Dict[str, Path]:
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

    write_json(event_path, build_event(
        run_id, "1", "Agent1_Planner", "start",
        inputs={"anforderungsliste": str(anforderungsliste_path), "attempt": attempt},
        outputs={"plan_json": str(plan_path), "event": str(event_path)},
    ))

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

    write_json(event_path, build_event(
        run_id, "1", "Agent1_Planner", "success",
        inputs={"anforderungsliste": str(anforderungsliste_path), "attempt": attempt},
        outputs={"plan_json": str(plan_path)},
        message="plan.json created",
    ))
    return plan_path


def _normalize_cadquery_export(code: str) -> str:
    """
    关键：强制统一为 CadQuery 2.x 最稳健的 exporters.export()。
    目标：
      - 不再依赖 result.val().export_step/export_stl（容易遇到 Solid 路径问题）
      - 不再依赖 cq.exporters.export 的多种写法
    """
    s = code

    # 确保有 sys
    if "import sys" not in s:
        s = "import sys\n" + s

    # 确保 exporters 导入
    if "from cadquery import exporters" not in s:
        # 放在 import cadquery as cq 之后更自然
        if "import cadquery as cq" in s:
            s = s.replace("import cadquery as cq", "import cadquery as cq\nfrom cadquery import exporters", 1)
        else:
            s = "import cadquery as cq\nfrom cadquery import exporters\n" + s

    # 把各种旧导出 API 统一替换掉
    s = re.sub(r"\bresult\.val\(\)\.export_step\s*\(\s*['\"].*?['\"]\s*\)", "exporters.export(result, 'model.step')", s)
    s = re.sub(r"\bresult\.val\(\)\.export_stl\s*\(\s*['\"].*?['\"]\s*\)", "exporters.export(result, 'model.stl')", s)

    s = re.sub(r"\bresult\.exportStep\s*\(\s*['\"].*?['\"]\s*\)", "exporters.export(result, 'model.step')", s)
    s = re.sub(r"\bresult\.exportStl\s*\(\s*['\"].*?['\"]\s*\)", "exporters.export(result, 'model.stl')", s)

    s = re.sub(r"\bcq\.exporters\.export\s*\(\s*result\s*,\s*['\"].*?['\"]\s*\)", "exporters.export(result, 'model.step')", s, count=1)
    # 第二个 exporters.export 可能被替换成 step，这里补一个更稳健的兜底：如果只有一个 export，则追加 stl
    # 这一步不做“智能解析”，而是直接在脚本末尾追加“强制导出校验块”，保证收敛。

    footer = r"""
# ---- MAAS enforced export block (do not remove) ----
try:
    exporters.export(result, "model.step")
    exporters.export(result, "model.stl")
    import os
    if not os.path.exists("model.step") or not os.path.exists("model.stl"):
        print("Export did not create model.step/model.stl")
        sys.exit(1)
except Exception as e:
    print(f"Export failed: {e}")
    sys.exit(1)
"""
    if "MAAS enforced export block" not in s:
        s = s.rstrip() + "\n" + footer.lstrip()

    return s


def agent2_cad_writer(run_id: str, paths: Dict[str, Path], plan_path: Path, agent, attempt: int) -> Path:
    cad_script_path = paths["artifacts"] / "cad_script.py"
    event_path = paths["events"] / "event2.json"

    write_json(event_path, build_event(
        run_id, "2", "Agent2_CADWriter", "start",
        inputs={"plan_json": str(plan_path), "attempt": attempt},
        outputs={"cad_script": str(cad_script_path), "event": str(event_path)},
    ))

    plan = read_json(plan_path)

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
            "Generate a single, complete CadQuery 2.x Python script that defines a Workplane variable named `result`. "
            "The script will be executed with CWD = artifacts folder. DO NOT use any directory prefix when exporting. "
            "If previous_opt_patch exists, apply it."
        ),
    }

    resp = agent.generate_reply(messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])
    text = resp if isinstance(resp, str) else resp.get("content", "")

    code = text
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    if m:
        code = m.group(1).strip()

    # 强制规范化导出（这一条是“保证不再失败”的关键）
    code = _normalize_cadquery_export(code)

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

    # 用 venv 的 python 执行，cwd=脚本所在目录（attempt_xx）
    rc, out = _safe_run([sys.executable, str(Path(cad_script_path).resolve())], cwd=Path(cad_script_path).parent, timeout=180)
    write_text(exec_log_path, out)

    render_dir = paths["artifacts"] / "render"
    imgs = list_rendered_images(render_dir)

    step_files = [str(p) for p in paths["artifacts"].glob("*.step")] + [str(p) for p in paths["artifacts"].glob("*.stp")]
    stl_files = [str(p) for p in paths["artifacts"].glob("*.stl")]

    # ✅ 关键修复：rc=0 但没输出文件 => 仍然算 fail
    has_outputs = bool(step_files or stl_files)
    status = "success" if (rc == 0 and has_outputs) else "fail"
    error_msg = ""
    if rc != 0:
        error_msg = f"Execution failed, return code {rc}"
    elif not has_outputs:
        error_msg = "Execution returned 0 but produced no STEP/STL outputs."

    manifest = {
        "run_id": run_id,
        "attempt": attempt,
        "status": status,
        "cad_script": str(cad_script_path),
        "step_files": step_files,
        "stl_files": stl_files,
        "render_images": imgs,
        "exec_log_path": str(exec_log_path),
        "artifacts_dir": str(paths["artifacts"]),
        "return_code": rc,
        "error": error_msg,
    }
    write_json(manifest_path, manifest)

    write_json(event_path, build_event(
        run_id, "3", "Agent3_Executor", status,
        inputs={"cad_script": str(cad_script_path), "attempt": attempt},
        outputs={"output_manifest": str(manifest_path), "exec_log": str(exec_log_path)},
        message="execution finished",
        error=error_msg,
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

    image_paths = manifest.get("render_images", []) or list_rendered_images(paths["artifacts"] / "render")
    exec_log_tail = _read_text_tail(Path(manifest.get("exec_log_path", paths["artifacts"] / "exec.log.txt")))

    # ✅ 关键：明确告诉 4A：无渲染图时允许“文件级验收 pass”
    payload = {
        "run_id": run_id,
        "attempt": attempt,
        "plan_json": plan,
        "output_manifest": manifest,
        "exec_log_tail": exec_log_tail,
        "instructions": (
            "Return STRICT JSON for verify_report.json.\n"
            "If render_images exist, do geometry/feature checks.\n"
            "If NO render_images exist, you MUST use file-level acceptance:\n"
            "- If manifest.status=='success' AND step_files not empty => you may return status='pass' unless plan explicitly demands visual-confirmable features.\n"
            "- Mention in summary that no images were available.\n"
            "Output schema: {status: pass|fail, summary, issues[], checks[] }"
        )
    }

    content_payload: List[Dict[str, Any]] = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
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
        inputs={"attempt": attempt},
        outputs={"opt_patch": str(opt_path), "event": str(event_path)},
    ))

    plan = read_json(plan_path)
    manifest = read_json(manifest_path)

    cad_script_text = Path(cad_script_path).read_text(encoding="utf-8", errors="ignore")
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
            "Decide next_step for convergence.\n"
            "Allowed next_step only: '1' or '2'.\n"
            "If execution failed => prefer need_fix_script next_step='2' unless plan wrong.\n"
            "If verify failed => decide need_fix_script or need_replan.\n"
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
    merged_events_path = base_paths["run"] / "artifacts" / "events_merged.json"
    final_zip_path = base_paths["run"] / "artifacts" / "final_model.zip"

    ensure_dir(base_paths["memory"] / "input")
    ensure_dir(base_paths["memory"] / "artifacts")
    ensure_dir(base_paths["memory"] / "events")

    copy_file(user_input_path, base_paths["memory"] / "input" / user_input_path.name)
    copy_tree(base_paths["run"] / "artifacts", base_paths["memory"] / "artifacts")
    copy_tree(base_paths["run"] / "memory" / "events", base_paths["memory"] / "events")

    events_root = base_paths["memory"] / "events"
    event_files = sorted(events_root.rglob("event*.json"))
    merged = {"run_id": run_id, "events": []}
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

    import zipfile

    def _safe_arcname(p: Path, base_dir: Path) -> str:
        p = p.resolve()
        try:
            rel = p.relative_to(base_dir.resolve())
            return str(rel).replace("\\", "/")
        except Exception:
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

        mem_artifacts_dir = (base_paths["memory"] / "artifacts").resolve()
        if mem_artifacts_dir.exists():
            for p in mem_artifacts_dir.rglob("*"):
                if p.is_file():
                    arc = "artifacts/" + _safe_arcname(p, mem_artifacts_dir)
                    add_file(p, arc)

        add_file(merged_events_path, "events_merged.json")
        copy_file(final_zip_path, base_paths["memory"] / "final_model.zip")

    return final_zip_path, merged_events_path



