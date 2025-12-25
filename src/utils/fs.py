import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

def ensure_dir(p:str|Path)->Path:
    p=Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_text(path:str|Path, content:str, encoding:str="utf-8")-> None:
    path=Path(path)
    ensure_dir(path.parent)
    path.write_text(content, encoding=encoding)


def write_json(path:str|Path, data:Dict[str, Any], encoding:str="utf-8")->None:
    path=Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding=encoding)


def read_json(path:str|Path, encoding:str="utf-8")->Dict[str, Any]:
    path=Path(path)
    return json.loads(path.read_text(encoding=encoding))


def copy_file(src: str|Path, dst: str|Path) -> None:
    src=Path(src)
    dst=Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_tree(src_dir: str|Path, dst_dir: str|Path) -> None:
    src_dir=Path(src_dir)
    dst_dir=Path(dst_dir)
    ensure_dir(dst_dir)
    if not src_dir.exists():
        return
    for root, _, files in os.walk(src_dir):
        root_p=Path(root)
        rel=root_p.relative_to(src_dir)
        for f in files:
            s=root_p/f
            d=dst_dir/rel/ f
            ensure_dir(d.parent)
            shutil.copy2(s, d)
