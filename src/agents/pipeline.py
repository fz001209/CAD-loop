from __future__ import annotations

import json
import re
import struct
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src.utils.fs import ensure_dir, write_json, write_text, read_json, copy_file, copy_tree
from src.utils.events import build_event
from src.utils.render_stub import list_rendered_images

# Route-B deterministic IR executor
from src.agents.ir_executor import execute_ir, sanitize_ir


# ============================================================
# Common helpers
# ============================================================

def _extract_first_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


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


# ============================================================
# STL -> PNG rendering (matplotlib, minimal)
# ============================================================

def _parse_stl_triangles(stl_path: Path) -> List[List[List[float]]]:
    data = stl_path.read_bytes()
    if len(data) < 84:
        return []

    tri_count = struct.unpack("<I", data[80:84])[0]
    expected_len = 84 + tri_count * 50
    triangles: List[List[List[float]]] = []

    if expected_len == len(data):
        off = 84
        for _ in range(tri_count):
            off += 12
            v1 = struct.unpack("<fff", data[off:off + 12]); off += 12
            v2 = struct.unpack("<fff", data[off:off + 12]); off += 12
            v3 = struct.unpack("<fff", data[off:off + 12]); off += 12
            off += 2
            triangles.append([[v1[0], v1[1], v1[2]], [v2[0], v2[1], v2[2]], [v3[0], v3[1], v3[2]]])
        return triangles

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
        poly = Poly3DCollection(tris, linewidths=0.2, edgecolors="black")
        ax.add_collection3d(poly)

        xs = [p[0] for tri in tris for p in tri]
        ys = [p[1] for tri in tris for p in tri]
        zs = [p[2] for tri in tris for p in tri]
        if not xs or not ys or not zs:
            return False, "empty bounds"

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


def _compute_geometry_metrics_from_stl(stl_path: Path) -> Dict[str, Any]:
    tris = _parse_stl_triangles(stl_path)
    if not tris:
        return {"stl_triangle_count": 0, "error": "no triangles"}

    xs = [p[0] for tri in tris for p in tri]
    ys = [p[1] for tri in tris for p in tri]
    zs = [p[2] for tri in tris for p in tri]

    return {
        "stl_triangle_count": len(tris),
        "bbox": {
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
            "z_min": min(zs), "z_max": max(zs),
            "dx": max(xs) - min(xs),
            "dy": max(ys) - min(ys),
            "dz": max(zs) - min(zs),
        }
    }


# ============================================================
# unified diff apply (single file, minimal)
# ============================================================

def _apply_unified_diff(original: str, diff_text: str) -> Tuple[bool, str, str]:
    if not diff_text or not diff_text.strip():
        return True, "empty diff (no-op)", original

    src_lines = original.splitlines(keepends=True)
    diff_lines = diff_text.splitlines(keepends=True)

    j = 0
    while j < len(diff_lines) and not diff_lines[j].startswith("@@"):
        j += 1
    if j >= len(diff_lines):
        return False, "no hunks found in unified diff", original

    out: List[str] = []
    i = 0

    hunk_re = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
    while j < len(diff_lines):
        m = hunk_re.match(diff_lines[j])
        if not m:
            return False, f"invalid unified diff header at line {j+1}", original

        old_start = int(m.group(1))
        target = max(0, old_start - 1)
        if target < i:
            return False, "overlapping hunks or invalid order", original

        out.extend(src_lines[i:target])
        i = target

        j += 1
        while j < len(diff_lines) and not diff_lines[j].startswith("@@"):
            dl = diff_lines[j]

            if dl.startswith("\\"):
                j += 1
                continue

            if not dl:
                j += 1
                continue

            tag = dl[:1]
            content = dl[1:]

            if tag == " ":
                if i >= len(src_lines):
                    return False, "context beyond EOF", original
                if src_lines[i] != content:
                    return False, "context mismatch while applying diff", original
                out.append(src_lines[i])
                i += 1

            elif tag == "-":
                if i >= len(src_lines):
                    return False, "deletion beyond EOF", original
                if src_lines[i] != content:
                    return False, "deletion mismatch while applying diff", original
                i += 1

            elif tag == "+":
                out.append(content)

            else:
                return False, f"unknown diff tag '{tag}'", original

            j += 1

    out.extend(src_lines[i:])
    return True, "applied", "".join(out)


# ============================================================
# Agents
# ============================================================

def agent1_planner(run_id: str, paths: Dict[str, Path], anforderungsliste_path: Path, agent, attempt: int) -> Path:
    plan_path = paths["artifacts"] / "plan.json"
    event_path = paths["events"] / "event1.json"

    write_json(event_path, build_event(
        run_id, "1", "Agent1_Planner", "start",
        inputs={"anforderungsliste": str(anforderungsliste_path), "attempt": attempt},
        outputs={"plan_json": str(plan_path), "event": str(event_path)},
    ))

    try:
        yaml_text = Path(anforderungsliste_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        yaml_text = ""

    prompt = {
        "run_id": run_id,
        "attempt": attempt,
        "input_file": str(anforderungsliste_path),
        "anforderungsliste_yaml": yaml_text,
        "instructions": (
            "Parse the YAML TEXT and output STRICT JSON only.\n"
            "Do not invent requirements not present in the YAML.\n"
            "\n"
            "You MUST output an IR-like plan with these top-level keys:\n"
            "- object: string\n"
            "- required_features: list of {id,name,must,notes}\n"
            "- params: dict (numeric params in mm where applicable)\n"
            "- operations: list of {id,op,args}\n"
            "\n"
            "IMPORTANT:\n"
            "- operations are conceptual IR (not CadQuery code).\n"
            "- Every operation MUST have a stable unique 'id' (string).\n"
            "- If you cannot confidently map a requirement to operations, list it under required_features with must=true.\n"
        ),
    }

    resp = agent.generate_reply(messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])
    plan = _extract_first_json(resp if isinstance(resp, str) else resp.get("content", ""))

    plan["run_id"] = run_id
    plan["attempt"] = attempt

    plan = sanitize_ir(plan)

    ops = plan.get("operations", []) or []
    for i, op in enumerate(ops, start=1):
        if not op.get("id"):
            op["id"] = f"OP{i:02d}"
    plan["operations"] = ops

    write_json(plan_path, plan)

    write_json(event_path, build_event(
        run_id, "1", "Agent1_Planner", "success",
        inputs={"anforderungsliste": str(anforderungsliste_path), "attempt": attempt},
        outputs={"plan_json": str(plan_path)},
        message="plan.json created",
    ))
    return plan_path


def agent2_cad_writer(run_id: str, paths: Dict[str, Path], plan_path: Path, agent, attempt: int) -> Path:
    ir_path = paths["artifacts"] / "ir.json"
    event_path = paths["events"] / "event2.json"

    write_json(event_path, build_event(
        run_id, "2", "Agent2_CADWriter", "start",
        inputs={"plan_json": str(plan_path), "attempt": attempt},
        outputs={"ir_json": str(ir_path), "event": str(event_path)},
    ))

    plan = read_json(plan_path)

    # -----------------------------
    # apply unified diff from previous opt_patch to previous ir.json
    # -----------------------------
    if attempt > 1:
        prev_opt_path = paths["run"] / "artifacts" / f"attempt_{attempt-1:02d}" / "opt_patch.json"
        prev_ir_path = paths["run"] / "artifacts" / f"attempt_{attempt-1:02d}" / "ir.json"

        if prev_opt_path.exists() and prev_ir_path.exists():
            try:
                prev_opt = read_json(prev_opt_path)
                patch_obj = prev_opt.get("patch") if isinstance(prev_opt, dict) else None
                if isinstance(patch_obj, dict) and patch_obj.get("type") == "unified_diff":
                    diff_text = patch_obj.get("content") or ""
                    if diff_text.strip():
                        base_text = prev_ir_path.read_text(encoding="utf-8", errors="ignore")
                        ok, msg, patched_text = _apply_unified_diff(base_text, diff_text)
                        if ok:
                            patched_json_raw = json.loads(patched_text)

                            # hard validate no unknown ops introduced (MUST include primitive_torus)
                            allowed = set([
                                "primitive_box", "primitive_cylinder", "primitive_sphere", "primitive_torus",
                                "sketch_rect", "sketch_circle", "sketch_polygon",
                                "extrude", "cut_extrude",
                                "translate", "rotate",
                                "union", "cut", "intersect",
                                "hole", "fillet", "chamfer", "shell",
                            ])
                            raw_ops = patched_json_raw.get("operations", [])
                            unknown = []
                            for op in raw_ops:
                                if isinstance(op, dict):
                                    name = str(op.get("op", "")).strip()
                                    if name and name not in allowed:
                                        unknown.append(name)
                            if unknown:
                                raise ValueError(f"Patch introduced unknown ops: {sorted(set(unknown))}")

                            patched_json_raw["run_id"] = run_id
                            patched_json_raw["attempt"] = attempt
                            patched_json = sanitize_ir(patched_json_raw)

                            ops = patched_json.get("operations", []) or []
                            for i, op in enumerate(ops, start=1):
                                if not op.get("id"):
                                    op["id"] = f"OP{i:02d}"
                            patched_json["operations"] = ops

                            write_json(ir_path, patched_json)
                            write_json(event_path, build_event(
                                run_id, "2", "Agent2_CADWriter", "success",
                                inputs={"plan_json": str(plan_path), "attempt": attempt, "patched_from": str(prev_ir_path)},
                                outputs={"ir_json": str(ir_path)},
                                message="ir.json created by applying unified_diff patch",
                            ))
                            return ir_path
            except Exception:
                pass

    # -----------------------------
    # LLM generation (fallback)
    # -----------------------------
    allowed_ops_list = [
        "primitive_box","primitive_cylinder","primitive_sphere","primitive_torus",
        "sketch_rect","sketch_circle","sketch_polygon",
        "extrude","cut_extrude",
        "translate","rotate",
        "union","cut","intersect",
        "hole","fillet","chamfer","shell"
    ]

    prompt = {
        "run_id": run_id,
        "attempt": attempt,
        "plan_json": plan,
        "allowed_ops": allowed_ops_list,
        "instructions": (
            "Output STRICT JSON only for ir.json.\n"
            "ir.json must be executable by the deterministic executor.\n"
            "Each operation MUST have fields: {id, op, args}.\n"
            "Use args.out_id to name intermediate results.\n"
            "\n"
            "Reference rules:\n"
            "- union/cut/intersect MUST use args.a_id and args.b_id (existing work ids).\n"
            "- extrude MUST use args.sketch_id (existing sketch id).\n"
            "- cut_extrude MUST use args.base_id and args.sketch_id.\n"
            "\n"
            "Hard rule: operations[*].op MUST be one of allowed_ops. Do NOT invent op names.\n"
        ),
    }

    resp = agent.generate_reply(messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])
    ir = _extract_first_json(resp if isinstance(resp, str) else resp.get("content", ""))

    ir["run_id"] = run_id
    ir["attempt"] = attempt
    ir = sanitize_ir(ir)

    ops = ir.get("operations", []) or []
    for i, op in enumerate(ops, start=1):
        if not op.get("id"):
            op["id"] = f"OP{i:02d}"
    ir["operations"] = ops

    write_json(ir_path, ir)

    write_json(event_path, build_event(
        run_id, "2", "Agent2_CADWriter", "success",
        inputs={"plan_json": str(plan_path), "attempt": attempt},
        outputs={"ir_json": str(ir_path)},
        message="ir.json created",
    ))
    return ir_path


def agent3_executor(run_id: str, paths: Dict[str, Path], ir_path: Path, agent, attempt: int) -> Path:
    manifest_path = paths["artifacts"] / "output_manifest.json"
    exec_log_path = paths["artifacts"] / "exec.log.txt"
    event_path = paths["events"] / "event3.json"

    write_json(event_path, build_event(
        run_id, "3", "Agent3_Executor", "start",
        inputs={"ir_json": str(ir_path), "attempt": attempt},
        outputs={"output_manifest": str(manifest_path), "exec_log": str(exec_log_path), "event": str(event_path)},
    ))

    render_dir = ensure_dir(paths["artifacts"] / "render")

    ir = read_json(ir_path)
    ir = sanitize_ir(ir)

    warnings: List[str] = []
    ops = ir.get("operations", []) or []
    for i, op in enumerate(ops, start=1):
        if not op.get("id"):
            op["id"] = f"OP{i:02d}"
            warnings.append(f"WARNING: operation at index {i} missing id -> filled as {op['id']}")
    ir["operations"] = ops

    exec_result, work_registry = execute_ir(ir)

    log_lines: List[str] = []
    log_lines.append(f"IR execution ok: {exec_result.ok}")
    log_lines.append(f"Message: {exec_result.message}")
    if warnings:
        log_lines.append("---- warnings ----")
        log_lines.extend(warnings)
    log_lines.append(f"Final work_id: {exec_result.work_id}")
    log_lines.append("---- op_trace ----")
    for t in exec_result.op_trace:
        log_lines.append(json.dumps(t, ensure_ascii=False))
    if exec_result.exception:
        log_lines.append("---- exception ----")
        log_lines.append(exec_result.exception)
    write_text(exec_log_path, "\n".join(log_lines))

    step_files: List[str] = []
    stl_files: List[str] = []
    imgs: List[str] = []
    geometry_metrics: Dict[str, Any] = {}
    error_msg = ""

    if exec_result.ok and exec_result.work_id and exec_result.work_id in work_registry:
        try:
            from cadquery import exporters  # type: ignore
            final_obj = work_registry[exec_result.work_id]
            exporters.export(final_obj, str(paths["artifacts"] / "model.step"))
            exporters.export(final_obj, str(paths["artifacts"] / "model.stl"))
            step_files = [str(p) for p in paths["artifacts"].glob("*.step")] + [str(p) for p in paths["artifacts"].glob("*.stp")]
            stl_files = [str(p) for p in paths["artifacts"].glob("*.stl")]
        except Exception as e:
            error_msg = f"Export failed: {e}"
    else:
        if not exec_result.ok:
            error_msg = exec_result.message
        else:
            error_msg = "Execution ok but no final work_id produced."

    render_msg = ""
    if stl_files:
        ok_r, msg_r = _render_stl_to_png_multi6(Path(stl_files[0]), render_dir)
        render_msg = msg_r if ok_r else f"render_failed: {msg_r}"
        imgs = list_rendered_images(render_dir)
        try:
            geometry_metrics = _compute_geometry_metrics_from_stl(Path(stl_files[0]))
        except Exception as e:
            geometry_metrics = {"error": f"metrics_failed: {e}"}

    has_outputs = bool(step_files or stl_files)
    status = "success" if (exec_result.ok and has_outputs and imgs) else "fail"

    if status == "fail" and not error_msg:
        if not exec_result.ok:
            error_msg = exec_result.message
        elif not has_outputs:
            error_msg = "Execution produced no STEP/STL outputs."
        elif not imgs:
            error_msg = "Execution produced geometry but no render images (PNG required for 4A). " + (render_msg or "")

    manifest = {
        "run_id": run_id,
        "attempt": attempt,
        "status": status,
        "ir_json": str(ir_path),
        "step_files": step_files,
        "stl_files": stl_files,
        "render_images": imgs,
        "geometry_metrics": geometry_metrics,
        "executor_debug": {
            "ok": exec_result.ok,
            "message": exec_result.message,
            "work_id": exec_result.work_id,
            "solid_count": exec_result.solid_count,
            "op_trace": exec_result.op_trace,
            "exception": exec_result.exception,
        },
        "exec_log_path": str(exec_log_path),
        "artifacts_dir": str(paths["artifacts"]),
        "error": error_msg,
    }
    write_json(manifest_path, manifest)

    write_json(event_path, build_event(
        run_id, "3", "Agent3_Executor", status,
        inputs={"ir_json": str(ir_path), "attempt": attempt},
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

    required_features = plan.get("required_features", [])
    content_payload: List[Dict[str, Any]] = [{
        "type": "text",
        "text": json.dumps({
            "run_id": run_id,
            "attempt": attempt,
            "plan_json": plan,
            "required_features": required_features,
            "output_manifest": manifest,
            "exec_log_tail": exec_log_tail,
            "instructions": (
                "Verify geometry strictly based on render images AND plan.required_features.\n"
                "HARD RULES:\n"
                "A) Any must=true required feature missing => fail.\n"
                "B) If you cannot confirm a must=true feature from images => fail.\n"
                "C) Any unrelated extra geometry => fail.\n"
                "\n"
                "Output feature_evidence_map mapping feature.id -> {confirmed, evidence_images, note}.\n"
                "Return STRICT JSON only.\n"
            )
        }, ensure_ascii=False)
    }]

    for p in image_paths[:8]:
        try:
            img = Image.open(p)
            content_payload.append({"type": "image_url", "image_url": {"url": img}})
        except Exception as e:
            content_payload.append({"type": "text", "text": f"[WARN] Failed to open image {p}: {e}"})

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
    ir_path: Path,
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
    ir = read_json(ir_path)
    manifest = read_json(manifest_path)

    exec_log_tail = _read_text_tail(Path(manifest.get("exec_log_path", paths["artifacts"] / "exec.log.txt")))

    verify = None
    if verify_path and Path(verify_path).exists():
        try:
            verify = read_json(verify_path)
        except Exception:
            verify = None

    allowed_ops_list = [
        "primitive_box","primitive_cylinder","primitive_sphere","primitive_torus",
        "sketch_rect","sketch_circle","sketch_polygon",
        "extrude","cut_extrude",
        "translate","rotate",
        "union","cut","intersect",
        "hole","fillet","chamfer","shell"
    ]

    prompt = {
        "run_id": run_id,
        "attempt": attempt,
        "plan_json": plan,
        "ir_json": ir,
        "output_manifest": manifest,
        "exec_log_tail": exec_log_tail,
        "verify_report": verify,
        "allowed_ops": allowed_ops_list,
        "instructions": (
            "Decide next_step for convergence.\n"
            "Allowed next_step only: '1' or '2'.\n"
            "If verifier failed => prefer next_step='2' and patch ir.json to satisfy required_features.\n"
            "If execution failed => prefer next_step='2' and patch the failing op (see executor_debug/op_trace).\n"
            "\n"
            "HARD RULE: You MUST NOT invent any operation names. "
            "Any operation you add must be from allowed_ops.\n"
            "\n"
            "IMPORTANT: patch MUST be a unified diff for the file ir.json (single file).\n"
            "Return STRICT JSON only.\n"
        ),
    }

    resp = agent.generate_reply(messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])
    patch = _extract_first_json(resp if isinstance(resp, str) else resp.get("content", ""))

    patch.setdefault("status", "need_fix_ir")
    patch.setdefault("next_step", "2")
    if str(patch.get("next_step")) not in ("1", "2"):
        patch["next_step"] = "2"
    if patch.get("status") not in ("need_fix_ir", "need_replan", "noop"):
        patch["status"] = "need_fix_ir"

    if not isinstance(patch.get("patch"), dict):
        patch["patch"] = {"type": "unified_diff", "content": ""}
    else:
        if patch["patch"].get("type") != "unified_diff":
            patch["patch"]["type"] = "unified_diff"
        if patch["patch"].get("content") is None:
            patch["patch"]["content"] = ""

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




