from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import traceback


# ============================================================
# Allowed ops (executor contract)
# ============================================================

_ALLOWED_OPS = {
    # primitives
    "primitive_box",
    "primitive_cylinder",
    "primitive_sphere",
    "primitive_torus",   # NEW: enables arched/ring-like features generically

    # sketch
    "sketch_rect",
    "sketch_circle",
    "sketch_polygon",

    # sketch->solid
    "extrude",
    "cut_extrude",

    # transforms
    "translate",
    "rotate",

    # booleans
    "union",
    "cut",
    "intersect",

    # feature ops
    "hole",
    "fillet",
    "chamfer",
    "shell",
}

_DEFAULT_PLANE = "XY"


# ============================================================
# Utils
# ============================================================

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def sanitize_ir(ir: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic guardrail:
    - Ensure required keys exist
    - Filter operations to allowed ops
    - Normalize missing fields
    NOTE: this does NOT “fix” semantics; it only enforces schema/whitelist.
    """
    out = dict(ir or {})
    out.setdefault("project_name", out.get("name") or "unnamed_project")
    out.setdefault("object", out.get("object") or "unknown_object")

    if not isinstance(out.get("required_features"), list):
        out["required_features"] = []
    if not isinstance(out.get("params"), dict):
        out["params"] = {}
    if not isinstance(out.get("operations"), list):
        out["operations"] = []

    cleaned_ops: List[Dict[str, Any]] = []
    for item in out["operations"]:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op", "")).strip()
        if op not in _ALLOWED_OPS:
            # drop unknown op (caller may choose to treat as error elsewhere)
            continue
        cleaned_ops.append({
            "id": item.get("id"),
            "op": op,
            "args": item.get("args") if isinstance(item.get("args"), dict) else {},
        })
    out["operations"] = cleaned_ops
    return out


@dataclass
class ExecResult:
    ok: bool
    message: str
    work_id: Optional[str]
    op_trace: List[Dict[str, Any]]
    solid_count: Optional[int]
    exception: str = ""


# ============================================================
# Executor
# ============================================================

def execute_ir(ir: Dict[str, Any]) -> Tuple[ExecResult, Dict[str, Any]]:
    """
    Deterministic execution: interpret IR operations with a fixed CadQuery backend.
    Returns:
      - ExecResult (with op_trace, final work_id, exception)
      - work_registry: Dict[work_id -> cadquery.Workplane/Shape-like]
    """
    ir = sanitize_ir(ir)

    try:
        import cadquery as cq
    except Exception as e:
        res = ExecResult(
            ok=False,
            message=f"CadQuery import failed: {e}",
            work_id=None,
            op_trace=[],
            solid_count=None,
            exception=str(e),
        )
        return res, {}

    work: Dict[str, Any] = {}      # out_id -> Workplane/solid container
    sketches: Dict[str, Any] = {}  # out_id -> Workplane (2D)
    op_trace: List[Dict[str, Any]] = []
    last_work_id: Optional[str] = None

    def trace(status: str, op: str, args: Dict[str, Any], out_id: Optional[str] = None, note: str = ""):
        op_trace.append({
            "status": status,
            "op": op,
            "args": args,
            "out_id": out_id,
            "note": note,
        })

    def solids_count(obj: Any) -> Optional[int]:
        try:
            s = obj.val().Solids()
            return 0 if s is None else len(s)
        except Exception:
            return None

    def ensure_has_solids(obj: Any, op_name: str, out_id: str):
        sc = solids_count(obj)
        if sc == 0:
            raise RuntimeError(f"result has no solids after op '{op_name}' (out_id={out_id})")

    ops = ir.get("operations", []) or []

    try:
        for idx, item in enumerate(ops, start=1):
            op = item.get("op")
            args = item.get("args", {}) or {}
            op_id = item.get("id") or f"OP{idx:02d}"

            # determine output id
            out_id = str(args.get("out_id") or op_id)

            try:
                # --------------------
                # primitives
                # --------------------
                if op == "primitive_box":
                    x = _to_float(args.get("x_mm", args.get("x", 10)))
                    y = _to_float(args.get("y_mm", args.get("y", 10)))
                    z = _to_float(args.get("z_mm", args.get("z", 10)))
                    centered = bool(args.get("centered", True))
                    obj = cq.Workplane(_DEFAULT_PLANE).box(x, y, z, centered=centered)
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                elif op == "primitive_cylinder":
                    r = _to_float(args.get("radius_mm", args.get("radius", 5)))
                    h = _to_float(args.get("height_mm", args.get("height", 10)))
                    centered = bool(args.get("centered", True))
                    # CadQuery cylinder by circle+extrude
                    obj = cq.Workplane(_DEFAULT_PLANE).circle(r).extrude(h, both=centered)
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                elif op == "primitive_sphere":
                    r = _to_float(args.get("radius_mm", args.get("radius", 5)))
                    obj = cq.Workplane(_DEFAULT_PLANE).sphere(r)
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                elif op == "primitive_torus":
                    # args: major_radius_mm, minor_radius_mm
                    major = _to_float(args.get("major_radius_mm", 20))
                    minor = _to_float(args.get("minor_radius_mm", 5))
                    # CadQuery: Solid.makeTorus(majorRadius, minorRadius)
                    tor = cq.Solid.makeTorus(major, minor)
                    obj = cq.Workplane(_DEFAULT_PLANE).newObject([tor])
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                # --------------------
                # sketches
                # --------------------
                elif op == "sketch_rect":
                    x = _to_float(args.get("x_mm", args.get("w", 10)))
                    y = _to_float(args.get("y_mm", args.get("l", 10)))
                    plane = str(args.get("plane", _DEFAULT_PLANE))
                    sk = cq.Workplane(plane).rect(x, y)
                    sketches[out_id] = sk
                    trace("ok", op, args, out_id)

                elif op == "sketch_circle":
                    r = _to_float(args.get("radius_mm", args.get("r", 5)))
                    plane = str(args.get("plane", _DEFAULT_PLANE))
                    sk = cq.Workplane(plane).circle(r)
                    sketches[out_id] = sk
                    trace("ok", op, args, out_id)

                elif op == "sketch_polygon":
                    n = _to_int(args.get("n", 6), 6)
                    r = _to_float(args.get("radius_mm", args.get("r", 5)))
                    plane = str(args.get("plane", _DEFAULT_PLANE))
                    # polygon uses diameter-like size; use 2*r
                    sk = cq.Workplane(plane).polygon(n, 2 * r)
                    sketches[out_id] = sk
                    trace("ok", op, args, out_id)

                # --------------------
                # sketch -> solid
                # --------------------
                elif op == "extrude":
                    sk_id = str(args.get("sketch_id") or "")
                    if not sk_id or sk_id not in sketches:
                        raise RuntimeError(f"extrude requires args.sketch_id referencing an existing sketch (got '{sk_id}').")
                    dist = _to_float(args.get("distance_mm", args.get("h", 10)))
                    obj = sketches[sk_id].extrude(dist)
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                elif op == "cut_extrude":
                    base_id = str(args.get("base_id") or "")
                    sk_id = str(args.get("sketch_id") or "")
                    if not base_id or base_id not in work:
                        raise RuntimeError(f"cut_extrude requires args.base_id referencing an existing work (got '{base_id}').")
                    if not sk_id or sk_id not in sketches:
                        raise RuntimeError(f"cut_extrude requires args.sketch_id referencing an existing sketch (got '{sk_id}').")
                    dist = _to_float(args.get("distance_mm", args.get("h", 10)))
                    cutter = sketches[sk_id].extrude(dist)
                    obj = work[base_id].cut(cutter)
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                # --------------------
                # booleans
                # --------------------
                elif op == "union":
                    a_id = str(args.get("a_id") or "")
                    b_id = str(args.get("b_id") or "")
                    if not a_id or a_id not in work or not b_id or b_id not in work:
                        raise RuntimeError("union requires args.a_id and args.b_id referencing existing work ids.")
                    obj = work[a_id].union(work[b_id])
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                elif op == "cut":
                    a_id = str(args.get("a_id") or "")
                    b_id = str(args.get("b_id") or "")
                    if not a_id or a_id not in work or not b_id or b_id not in work:
                        raise RuntimeError("cut requires args.a_id and args.b_id referencing existing work ids.")
                    obj = work[a_id].cut(work[b_id])
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                elif op == "intersect":
                    a_id = str(args.get("a_id") or "")
                    b_id = str(args.get("b_id") or "")
                    if not a_id or a_id not in work or not b_id or b_id not in work:
                        raise RuntimeError("intersect requires args.a_id and args.b_id referencing existing work ids.")
                    obj = work[a_id].intersect(work[b_id])
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                # --------------------
                # transforms
                # --------------------
                elif op == "translate":
                    base_id = str(args.get("base_id") or last_work_id or "")
                    if not base_id or base_id not in work:
                        raise RuntimeError("translate requires args.base_id referencing an existing work id.")
                    dx = _to_float(args.get("x_mm", args.get("dx", 0)))
                    dy = _to_float(args.get("y_mm", args.get("dy", 0)))
                    dz = _to_float(args.get("z_mm", args.get("dz", 0)))
                    obj = work[base_id].translate((dx, dy, dz))
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                elif op == "rotate":
                    base_id = str(args.get("base_id") or last_work_id or "")
                    if not base_id or base_id not in work:
                        raise RuntimeError("rotate requires args.base_id referencing an existing work id.")
                    axis = str(args.get("axis", "Z")).upper()
                    angle = _to_float(args.get("angle_deg", 0))
                    if axis == "X":
                        obj = work[base_id].rotate((0, 0, 0), (1, 0, 0), angle)
                    elif axis == "Y":
                        obj = work[base_id].rotate((0, 0, 0), (0, 1, 0), angle)
                    else:
                        obj = work[base_id].rotate((0, 0, 0), (0, 0, 1), angle)
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                # --------------------
                # feature ops
                # --------------------
                elif op == "hole":
                    base_id = str(args.get("base_id") or last_work_id or "")
                    if not base_id or base_id not in work:
                        raise RuntimeError("hole requires args.base_id referencing an existing work id.")
                    face_sel = str(args.get("face", ">Z"))
                    d = _to_float(args.get("diameter_mm", args.get("d", 2)))
                    depth = args.get("depth_mm", None)
                    wp = work[base_id].faces(face_sel).workplane()
                    if depth is None:
                        obj = wp.hole(d)
                    else:
                        obj = wp.hole(d, _to_float(depth, 1))
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                elif op == "shell":
                    base_id = str(args.get("base_id") or last_work_id or "")
                    if not base_id or base_id not in work:
                        raise RuntimeError("shell requires args.base_id referencing an existing work id.")
                    t = _to_float(args.get("thickness_mm", args.get("t", 1)))
                    face_sel = str(args.get("open_face_selector", args.get("face", ">Z")))
                    obj = work[base_id].faces(face_sel).shell(-abs(t))
                    work[out_id] = obj
                    last_work_id = out_id
                    trace("ok", op, args, out_id)
                    ensure_has_solids(obj, op, out_id)

                elif op == "fillet":
                    base_id = str(args.get("base_id") or last_work_id or "")
                    if not base_id or base_id not in work:
                        raise RuntimeError("fillet requires args.base_id referencing an existing work id.")
                    r0 = _to_float(args.get("radius_mm", args.get("r", 1)))
                    selector = str(args.get("selector", "all_edges"))

                    radii = [r0, r0 * 0.5, r0 * 0.25]
                    last_err: Optional[Exception] = None
                    for r in radii:
                        try:
                            if selector == "all_edges":
                                obj = work[base_id].edges().fillet(r)
                            else:
                                obj = work[base_id].edges(selector).fillet(r)
                            work[out_id] = obj
                            last_work_id = out_id
                            trace("ok", op, {**args, "radius_mm": r}, out_id, note=f"fillet_ok_radius={r}")
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e

                    if last_err is not None:
                        raise RuntimeError(f"fillet failed after retries: {last_err}")
                    ensure_has_solids(work[out_id], op, out_id)

                elif op == "chamfer":
                    base_id = str(args.get("base_id") or last_work_id or "")
                    if not base_id or base_id not in work:
                        raise RuntimeError("chamfer requires args.base_id referencing an existing work id.")
                    d0 = _to_float(args.get("distance_mm", args.get("d", 1)))
                    selector = str(args.get("selector", "all_edges"))

                    ds = [d0, d0 * 0.5, d0 * 0.25]
                    last_err = None
                    for d in ds:
                        try:
                            if selector == "all_edges":
                                obj = work[base_id].edges().chamfer(d)
                            else:
                                obj = work[base_id].edges(selector).chamfer(d)
                            work[out_id] = obj
                            last_work_id = out_id
                            trace("ok", op, {**args, "distance_mm": d}, out_id, note=f"chamfer_ok_distance={d}")
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e

                    if last_err is not None:
                        raise RuntimeError(f"chamfer failed after retries: {last_err}")
                    ensure_has_solids(work[out_id], op, out_id)

                else:
                    raise RuntimeError(f"Unsupported op: {op}")

            except Exception as e:
                trace("fail", str(op), args, out_id, note=f"{type(e).__name__}: {e}")
                res = ExecResult(
                    ok=False,
                    message=f"Op#{idx}({op_id}) failed: {op}: {e}",
                    work_id=last_work_id,
                    op_trace=op_trace,
                    solid_count=None,
                    exception=traceback.format_exc(),
                )
                return res, work

        # final result
        if last_work_id is None or last_work_id not in work:
            res = ExecResult(
                ok=False,
                message="No final work_id produced.",
                work_id=last_work_id,
                op_trace=op_trace,
                solid_count=None,
                exception="",
            )
            return res, work

        sc = solids_count(work[last_work_id])
        res = ExecResult(
            ok=True,
            message="executed",
            work_id=last_work_id,
            op_trace=op_trace,
            solid_count=sc,
            exception="",
        )
        return res, work

    except Exception as e:
        res = ExecResult(
            ok=False,
            message=f"Executor crashed: {e}",
            work_id=last_work_id,
            op_trace=op_trace,
            solid_count=None,
            exception=traceback.format_exc(),
        )
        return res, work
