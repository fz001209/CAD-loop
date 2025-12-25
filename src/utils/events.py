from __future__ import annotations
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def new_run_id() -> str:
    # short + stable enough for folders
    return time.strftime("%Y%m%d_%H%M%S", time.localtime()) + "_" + uuid.uuid4().hex[:8]

@dataclass
class EventRecord:
    run_id: str
    agent_id: str
    step_name: str
    status: str  # "start" | "success" | "fail" | "info"
    timestamp: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    message: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "step_name": self.step_name,
            "status": self.status,
            "timestamp": self.timestamp,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "message": self.message,
            "error": self.error,
          }

def build_event(
    run_id: str,
    agent_id: str,
    step_name: str,
    status: str,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    message: str = "",
    error: str = "",
) -> Dict[str, Any]:
    ev = EventRecord(
        run_id=run_id,
        agent_id=agent_id,
        step_name=step_name,
        status=status,
        timestamp=now_iso(),
        inputs=inputs or {},
        outputs=outputs or {},
        message=message,
        error=error,
    )
    return ev.to_dict()
