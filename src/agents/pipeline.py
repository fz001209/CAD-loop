from __future__ import annotations

import sys
import json
import re
import struct
import subprocess
from PIL import Image
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


# ---------------------------
# STL -> PNG rendering (minimal, matplotlib)
# ---------------------------
def _parse_stl_triangles(stl_path: Path) -> List[List[List[float]]]:
    """
    Return triangles as [[[x,y,z],[x,y,z],[x,y,z]], ...]
    Supports binary STL and ASCII STL (best-effort).
    """
    data = stl_path.read_bytes()
    if len(data) < 84:
        return []

    tri_count = struct.unpack("<I", data[80:84])[0]
    expected_len = 84 + tri_count * 50
    triangles: List[List[List[float]]] = []

    # binary STL
    if expected_len == len(data):
        off = 84
        for _ in range(tri_count):
            off += 12  # normal
            v1 = struct.unpack("<fff", data[off:off + 12]); off += 12
            v2 = struct.unpack("<fff", data[off:off + 12]); off += 12
            v3 = struct.unpack("<fff", data[off:off + 12]); off += 12
            off += 2  # attribute
            triangles.append([[v1[0], v1[1], v1[2]], [v2[0], v2[1], v2[2]], [v3[0], v3[1], v3[2]]])
        return triangles

    # ASCII best-effort
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return []

    verts: List[List[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("vertex "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    verts.append([x, y, z])
                    if len(verts) == 3:
                        triangles.append([verts[0], verts[1], verts[2]])
                        verts = []
                except Exception:
                    pass
    return triangles


def _render_stl_to_png(stl_path: Path, png_path: Path, elev: float, azim: float) -> Tuple[bool, str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception as e:
        return False, f"matplotlib not available: {e}"

    try:
        tris = _parse_stl_triangles(stl_path)
        if not tris:
            return False, "no triangles parsed from STL"

        fig = plt.figure(figsize=(6, 6), dpi=180)
        ax = fig.add_subplot(111, projection="3d")

        poly = Poly3DCollection(
            tris,
            linewidths=0.2,
            edgecolors="black",
        )
        ax.add_collection3d(poly)

        xs = [p[0] for tri in tris for p in tri]
        ys = [p[1] for tri in tris for p in tri]
        zs = [p[2] for tri in tris for p in tri]

        if not xs or not ys or not zs:
            return False, "empty bounds"

        # equal-ish scale
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        dz = max(zs) - min(zs)
        m = max(dx, dy, dz) if max(dx, dy, dz) > 0 else 1.0
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        cz = (max(zs) + min(zs)) / 2
        ax.set_xlim(cx - m / 2, cx + m / 2)
        ax.set_ylim(cy - m / 2, cy + m / 2)
        ax.set_zlim(cz - m / 2, cz + m / 2)

        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        ensure_dir(png_path.parent)
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return True, "rendered"
    except Exception as e:
        return False, f"render failed: {e}"


def _render_stl_to_png_multi6(stl_path: Path, render_dir: Path) -> Tuple[bool, str]:
    """
    Generate 6 standard views to improve 4A robustness.
    """
    ensure_dir(render_dir)

    views = [
        ("view_iso_1.png", 25, 45),
        ("view_iso_2.png", 25, 135),
        ("view_front.png", 0, 0),
        ("view_back.png", 0, 180),
        ("view_left.png", 0, 90),
        ("view_top.png", 90, 0),
        ("view_bottom.png", -90, 0),
    ]

    ok_any = False
    msgs: List[str] = []
    for name, elev, azim in views:
        ok, msg = _render_stl_to_png(stl_path, render_dir / name, elev=elev, azim=azim)
        ok_any = ok_any or ok
        msgs.append(f"{name}:{'ok' if ok else 'fail'}({msg})")

    return ok_any, "; ".join(msgs)


# ---------------------------
# Agents
# ---------------------------
def agent1_planner(run_id: str, paths: Dict[str, Path], anforderungsliste_path: Path, agent, attempt: int) -> Path:
    plan_path = paths["artifacts"] / "plan.json"
    event_path = paths["events"] / "event1.json"

    write_json(event_path, build_event(
        run_id, "1", "Agent1_Planner", "start",
        inputs={"anforderungsliste": str(anforderungsliste_path), "attempt": attempt},
        outputs={"plan_json": str(plan_path), "event": str(event_path)},
    ))
    yaml_text = ""
    try:
        yaml_text = Path(anforderungsliste_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        yaml_text = ""

    prompt = {
        "run_id": run_id,
        "attempt": attempt,
        "input_file": str(anforderungsliste_path),
        "anforderungsliste_yaml": yaml_text,  # ✅关键：把原文给模型
        "instructions": (
            "Parse the provided YAML TEXT (anforderungsliste_yaml) and return plan.json as STRICT JSON only. "
            "DO NOT invent requirements not present in the YAML."
        ),
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
    强制统一为 CadQuery 2.x 最稳健的 exporters.export() 并追加强制导出块。
    """
    s = code

    if "import sys" not in s:
        s = "import sys\n" + s

    if "from cadquery import exporters" not in s:
        if "import cadquery as cq" in s:
            s = s.replace("import cadquery as cq", "import cadquery as cq\nfrom cadquery import exporters", 1)
        else:
            s = "import cadquery as cq\nfrom cadquery import exporters\n" + s

    s = re.sub(r"\bresult\.val\(\)\.export_step\s*\(\s*['\"].*?['\"]\s*\)", "exporters.export(result, 'model.step')", s)
    s = re.sub(r"\bresult\.val\(\)\.export_stl\s*\(\s*['\"].*?['\"]\s*\)", "exporters.export(result, 'model.stl')", s)

    s = re.sub(r"\bresult\.exportStep\s*\(\s*['\"].*?['\"]\s*\)", "exporters.export(result, 'model.step')", s)
    s = re.sub(r"\bresult\.exportStl\s*\(\s*['\"].*?['\"]\s*\)", "exporters.export(result, 'model.stl')", s)

    footer = r"""
# ---- MAAS enforced export block (do not remove) ----
try:
    # ---- geometry self-check (fail fast) ----
    # 1) must have at least 1 solid
    try:
        solids = result.val().Solids()
        if solids is None or len(solids) == 0:
            print("Self-check failed: result has no solids")
            sys.exit(1)
        # too many solids often means floating parts / unmerged pieces
        if len(solids) > 1:
            print(f"Self-check failed: result has {len(solids)} solids (possible floating parts)")
            sys.exit(1)
    except Exception as e:
        print(f"Self-check warning: cannot inspect solids: {e}")

    # 2) bounding box sanity check (avoid absurd geometry)
    try:
        bb = result.val().BoundingBox()
        dx = float(bb.xlen); dy = float(bb.ylen); dz = float(bb.zlen)
        if dx <= 0 or dy <= 0 or dz <= 0:
            print("Self-check failed: invalid bounding box")
            sys.exit(1)
        # prevent extreme aspect artifacts
        max_dim = max(dx, dy, dz)
        min_dim = max(1e-6, min(dx, dy, dz))
        if max_dim / min_dim > 200:
            print(f"Self-check failed: extreme aspect ratio bbox=({dx},{dy},{dz})")
            sys.exit(1)
    except Exception as e:
        print(f"Self-check warning: cannot inspect bounding box: {e}")

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

    rc, out = _safe_run([sys.executable, str(Path(cad_script_path).resolve())], cwd=Path(cad_script_path).parent, timeout=180)
    write_text(exec_log_path, out)

    render_dir = ensure_dir(paths["artifacts"] / "render")

    step_files = [str(p) for p in paths["artifacts"].glob("*.step")] + [str(p) for p in paths["artifacts"].glob("*.stp")]
    stl_files = [str(p) for p in paths["artifacts"].glob("*.stl")]

    # ---- NEW: render 6 views if possible ----
    render_msg = ""
    if rc == 0 and stl_files:
        # 只要有 STL，就生成/刷新 6 视角图（提升 4A 稳健性）
        ok, msg = _render_stl_to_png_multi6(Path(stl_files[0]), render_dir)
        render_msg = msg if ok else f"render_failed: {msg}"

    imgs = list_rendered_images(render_dir)

    # 原逻辑：rc=0 且有 step/stl 才算成功
    has_outputs = bool(step_files or stl_files)
    status = "success" if (rc == 0 and has_outputs) else "fail"
    error_msg = ""
    if rc != 0:
        error_msg = f"Execution failed, return code {rc}"
    elif not has_outputs:
        error_msg = "Execution returned 0 but produced no STEP/STL outputs."

    # ---- NEW: 你要求 4A 必须基于渲染图，所以这里补一条“无图则 fail” ----
    if status == "success" and not imgs:
        status = "fail"
        error_msg = "Execution ok but produced no render images (PNG required for 4A). " + (render_msg or "")

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

    # 你的硬要求：无图则 fail
    if not image_paths:
        report = {
            "status": "fail",
            "summary": "No render images provided; geometry verification is impossible.",
            "issues": [{
                "severity": "high",
                "type": "execution",
                "message": "render_images is empty; PNG render is mandatory for 4A verification",
                "evidence": [str(manifest_path)]
            }],
            "checks": ["render_images_required=true", "render_images_found=0"],
            "attempt": attempt,
            "evidence": {
                "render_images_used": [],
                "exec_log_tail_included": bool(exec_log_tail),
            }
        }
        write_json(verify_path, report)
        write_json(event_path, build_event(
            run_id, "4A", "Agent4A_Verifier", "success",
            inputs={"plan_json": str(plan_path), "output_manifest": str(manifest_path), "attempt": attempt},
            outputs={"verify_report": str(verify_path)},
            message="verify status: fail",
        ))
        return verify_path

    content_payload: List[Dict[str, Any]] = [{"type": "text", "text": json.dumps({
        "run_id": run_id,
        "attempt": attempt,
        "plan_json": plan,
        "output_manifest": manifest,
        "exec_log_tail": exec_log_tail,
        "instructions": (
            "You MUST verify geometry STRICTLY based on the provided render images AND the plan.\n"
            "HARD RULES:\n"
            "A) If ANY required feature is missing => status MUST be 'fail'.\n"
            "B) If you cannot CONFIRM a required feature from the images => status MUST be 'fail'.\n"
            "C) If there is any unrelated/extra floating geometry/part => status MUST be 'fail'.\n"
            "\n"
            "You must produce a field `feature_evidence_map` that maps each required feature to:\n"
            "{confirmed: true|false, evidence_images: [filenames], note: string}\n"
            "\n"
            "Minimum required features for a mug-like object:\n"
            "- cylindrical body\n"
            "- hollow opening at top\n"
            "- handle attached to body (NOT a separate floating piece)\n"
            "- flat bottom\n"
            "\n"
            "PASS condition is very strict:\n"
            "- issues must be empty AND\n"
            "- every required feature must have confirmed=true AND evidence_images not empty.\n"
            "\n"
            "Return STRICT JSON only with schema:\n"
            "{status: pass|fail, summary: string, issues: [], checks: [], feature_evidence_map: { ... }}\n"
        )

    }, ensure_ascii=False)}]
        
    for p in image_paths[:8]:
        try:
            img = Image.open(p)  # p 是普通文件路径字符串/Path 都行
            content_payload.append({"type": "image_url", "image_url": {"url": img}})
        except Exception as e:
            # 如果某张图损坏/打不开，不要让整个流程崩掉；把原因写进文本证据即可
            content_payload.append({
                "type": "text",
                "text": f"[WARN] Failed to open image {p}: {e}"
            })

    resp = agent.generate_reply(messages=[{"role": "user", "content": content_payload}])
    text = resp if isinstance(resp, str) else resp.get("content", "")

    report = _extract_first_json(text)
    report.setdefault("status", "fail")
    report.setdefault("summary", "")
    report.setdefault("issues", [])
    report.setdefault("checks", [])
    report.setdefault("feature_evidence_map", {})
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



