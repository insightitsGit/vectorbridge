import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class Checkpoint:
    job_id: str
    source: str
    target: str
    mode: str
    vectors_transferred: int = 0
    batches_completed: int = 0
    last_offset: int = 0
    last_id: str = ""
    last_timestamp: str = ""
    started_at: str = ""
    updated_at: str = ""
    completed: bool = False

    @classmethod
    def new(cls, job_id: str, source: str, target: str, mode: str) -> "Checkpoint":
        now = datetime.now(timezone.utc).isoformat()
        return cls(job_id=job_id, source=source, target=target,
                   mode=mode, started_at=now, updated_at=now)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        with open(path) as f:
            return cls(**json.load(f))

    def advance(self, count: int, offset: int, last_id: str = "") -> None:
        self.vectors_transferred += count
        self.batches_completed += 1
        self.last_offset = offset
        self.last_id = last_id
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_complete(self) -> None:
        self.completed = True
        self.updated_at = datetime.now(timezone.utc).isoformat()


def checkpoint_path(job_id: str) -> str:
    return os.path.join(".vectorbridge", f"{job_id}.json")
