"""
渲染策略（可替换）：
- 后续可以把这里换成 FreeCAD headless等。
- 现在的 stub：只检查 artifacts/render 下是否已有 png；没有则不强制生成。
"""
from pathlib import Path
from typing import List

def list_rendered_images(render_dir: str | Path) -> List[str]:
    render_dir = Path(render_dir)
    if not render_dir.exists():
        return []
    return [str(p) for p in sorted(render_dir.glob("*.png"))]
