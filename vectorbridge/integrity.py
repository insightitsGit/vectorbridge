import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class IntegrityReport:
    job_id: str
    source: str
    target: str
    mode: str
    started_at: str
    completed_at: str = ""
    total_vectors: int = 0
    transferred: int = 0
    verified: int = 0
    failed_watermark: int = 0
    batches: int = 0
    wire_bytes: int = 0
    raw_bytes: int = 0
    errors: list[str] = field(default_factory=list)
    semantic_verify: dict = field(default_factory=dict)

    @property
    def verification_rate(self) -> float:
        return (self.verified / self.transferred * 100) if self.transferred else 0.0

    @property
    def bandwidth_savings(self) -> float:
        return (self.raw_bytes / self.wire_bytes) if self.wire_bytes else 0.0

    def record_batch(self, stats: dict) -> None:
        self.transferred += stats.get("received", 0)
        self.verified += stats.get("verified", 0)
        self.failed_watermark += stats.get("failed", 0)
        self.batches += 1
        self.wire_bytes += stats.get("wire_bytes", 0)
        self.raw_bytes += stats.get("raw_bytes", 0)

    def finalize(self) -> None:
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verification_rate_pct"] = round(self.verification_rate, 4)
        d["bandwidth_savings_x"] = round(self.bandwidth_savings, 2)
        return d

    def summary(self) -> str:
        SEP = "-" * 56
        sv = self.semantic_verify
        if sv:
            status = "PASS" if sv.get("passed") else "FAIL"
            semantic_line = (
                f"  Semantic OK  : {sv.get('avg_overlap_pct', 0):.2f}% neighbour overlap "
                f"({sv.get('n_probes', 0)} probes x top-{sv.get('top_k', 5)}) [{status}]\n"
            )
        else:
            semantic_line = "  Semantic     : not run\n"

        return (
            f"\n{SEP}\n"
            f"  VectorBridge Integrity Report\n"
            f"{SEP}\n"
            f"  Job ID       : {self.job_id}\n"
            f"  Source       : {self.source}\n"
            f"  Target       : {self.target}\n"
            f"  Mode         : {self.mode}\n"
            f"  Transferred  : {self.transferred:,} vectors\n"
            f"  CHORUS verify: {self.verified:,} / {self.transferred:,} "
            f"({self.verification_rate:.2f}%)\n"
            f"  Failed WM    : {self.failed_watermark}\n"
            f"  Wire bytes   : {self.wire_bytes:,}\n"
            f"  Bandwidth    : {self.bandwidth_savings:.2f}x savings vs raw float32\n"
            f"  Batches      : {self.batches}\n"
            f"{semantic_line}"
            f"  Errors       : {len(self.errors)}\n"
            f"{SEP}\n"
        )
