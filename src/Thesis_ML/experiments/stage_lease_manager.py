from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import uuid4

LeaseClass = Literal["cpu", "gpu"]


@dataclass(frozen=True)
class StageLeaseRequest:
    run_id: str
    stage_key: str
    owner_identity: str
    lease_class: LeaseClass = "gpu"
    lease_reason: str | None = None
    expected_backend_family: str | None = None
    expected_executor_id: str | None = None
    expected_compute_lane: str | None = None


@dataclass(frozen=True)
class StageLeaseHandle:
    lease_id: str
    lease_class: LeaseClass
    run_id: str
    stage_key: str
    owner_identity: str
    acquired_at_utc: str
    wait_seconds: float
    queue_depth_at_acquire: int
    lease_path: str | None


@dataclass(frozen=True)
class StageLeaseReleaseResult:
    lease_id: str
    lease_class: LeaseClass
    released: bool
    released_at_utc: str
    hold_seconds: float | None


class StageLeaseManager:
    """File-backed stage lease manager for scarce resources.

    Phase 6 intentionally scopes this manager to GPU lease coordination only.
    """

    def __init__(
        self,
        *,
        lease_root: Path | str,
        max_parallel_gpu_leases: int,
        poll_interval_seconds: float = 0.05,
        stale_lock_seconds: float = 120.0,
    ) -> None:
        resolved_max = int(max_parallel_gpu_leases)
        if resolved_max < 0:
            raise ValueError("max_parallel_gpu_leases must be >= 0.")
        resolved_poll = float(poll_interval_seconds)
        if resolved_poll <= 0.0:
            raise ValueError("poll_interval_seconds must be > 0.")

        self.lease_root = Path(lease_root)
        self.max_parallel_gpu_leases = resolved_max
        self.poll_interval_seconds = resolved_poll
        self.stale_lock_seconds = float(max(1.0, stale_lock_seconds))

        self._gpu_lease_dir = self.lease_root / "gpu"
        self._lock_dir = self.lease_root / "locks"
        self._coordinator_lock_path = self._lock_dir / "coordinator.lock"

        self._gpu_lease_dir.mkdir(parents=True, exist_ok=True)
        self._lock_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).replace(microsecond=0).isoformat()

    @staticmethod
    def _parse_utc(value: str | None) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _read_lease_payload(self, lease_path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(lease_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(payload, dict):
            return payload
        return None

    def _list_active_gpu_lease_paths(self) -> list[Path]:
        paths = [path for path in self._gpu_lease_dir.glob("*.json") if path.is_file()]
        paths.sort(key=lambda item: item.name)
        return paths

    @contextmanager
    def _coordinator_lock(self) -> Iterator[None]:
        fd: int | None = None
        while fd is None:
            try:
                fd = os.open(
                    str(self._coordinator_lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                )
                os.write(fd, f"pid={os.getpid()} at={self._utc_now()}\n".encode("utf-8"))
            except FileExistsError:
                try:
                    lock_mtime = self._coordinator_lock_path.stat().st_mtime
                    lock_age = time.time() - float(lock_mtime)
                    if lock_age > self.stale_lock_seconds:
                        self._coordinator_lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass
                time.sleep(self.poll_interval_seconds)

        try:
            yield
        finally:
            try:
                if fd is not None:
                    os.close(fd)
            finally:
                try:
                    self._coordinator_lock_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _build_gpu_lease_path(self, lease_id: str) -> Path:
        return self._gpu_lease_dir / f"{lease_id}.json"

    def acquire(
        self,
        request: StageLeaseRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> StageLeaseHandle:
        resolved_lease_class = str(request.lease_class).strip().lower()
        if resolved_lease_class not in {"cpu", "gpu"}:
            raise ValueError("lease_class must be one of: cpu, gpu.")

        lease_class: LeaseClass = "gpu" if resolved_lease_class == "gpu" else "cpu"
        if lease_class == "cpu":
            acquired_at = self._utc_now()
            return StageLeaseHandle(
                lease_id=f"cpu-noop-{uuid4().hex}",
                lease_class="cpu",
                run_id=str(request.run_id),
                stage_key=str(request.stage_key),
                owner_identity=str(request.owner_identity),
                acquired_at_utc=acquired_at,
                wait_seconds=0.0,
                queue_depth_at_acquire=0,
                lease_path=None,
            )

        resolved_timeout = float(timeout_seconds) if timeout_seconds is not None else None
        if resolved_timeout is not None and resolved_timeout <= 0.0:
            raise ValueError("timeout_seconds must be > 0 when provided.")

        wait_start = time.perf_counter()
        while True:
            acquired_at_utc: str | None = None
            queue_depth_at_acquire = 0
            lease_id: str | None = None
            lease_path: Path | None = None

            with self._coordinator_lock():
                active_lease_paths = self._list_active_gpu_lease_paths()
                active_count = int(len(active_lease_paths))
                if active_count < int(self.max_parallel_gpu_leases):
                    lease_id = uuid4().hex
                    lease_path = self._build_gpu_lease_path(lease_id)
                    acquired_at_utc = self._utc_now()
                    queue_depth_at_acquire = int(active_count)
                    payload = {
                        "lease_id": str(lease_id),
                        "lease_class": "gpu",
                        "run_id": str(request.run_id),
                        "stage_key": str(request.stage_key),
                        "owner_identity": str(request.owner_identity),
                        "lease_reason": (
                            str(request.lease_reason) if request.lease_reason is not None else None
                        ),
                        "expected_backend_family": (
                            str(request.expected_backend_family)
                            if request.expected_backend_family is not None
                            else None
                        ),
                        "expected_executor_id": (
                            str(request.expected_executor_id)
                            if request.expected_executor_id is not None
                            else None
                        ),
                        "expected_compute_lane": (
                            str(request.expected_compute_lane)
                            if request.expected_compute_lane is not None
                            else None
                        ),
                        "acquired_at_utc": str(acquired_at_utc),
                    }
                    lease_path.write_text(
                        f"{json.dumps(payload, ensure_ascii=True)}\n",
                        encoding="utf-8",
                    )

            if lease_id is not None and lease_path is not None and acquired_at_utc is not None:
                wait_seconds = float(max(0.0, time.perf_counter() - wait_start))
                return StageLeaseHandle(
                    lease_id=str(lease_id),
                    lease_class="gpu",
                    run_id=str(request.run_id),
                    stage_key=str(request.stage_key),
                    owner_identity=str(request.owner_identity),
                    acquired_at_utc=str(acquired_at_utc),
                    wait_seconds=float(wait_seconds),
                    queue_depth_at_acquire=int(queue_depth_at_acquire),
                    lease_path=str(lease_path.resolve()),
                )

            if resolved_timeout is not None:
                elapsed = float(time.perf_counter() - wait_start)
                if elapsed >= resolved_timeout:
                    raise TimeoutError("stage_gpu_lease_acquire_timeout")
            time.sleep(self.poll_interval_seconds)

    def release(self, lease: StageLeaseHandle | str) -> StageLeaseReleaseResult:
        if isinstance(lease, StageLeaseHandle):
            lease_id = str(lease.lease_id)
            lease_class = str(lease.lease_class).strip().lower()
            acquired_at_utc = str(lease.acquired_at_utc)
        else:
            lease_id = str(lease)
            lease_class = "gpu"
            acquired_at_utc = None

        released_at_utc = self._utc_now()
        if lease_class == "cpu" or lease_id.startswith("cpu-noop-"):
            return StageLeaseReleaseResult(
                lease_id=str(lease_id),
                lease_class="cpu",
                released=True,
                released_at_utc=str(released_at_utc),
                hold_seconds=0.0,
            )

        lease_path = self._build_gpu_lease_path(lease_id)
        released = False
        hold_seconds: float | None = None

        with self._coordinator_lock():
            payload = self._read_lease_payload(lease_path) if lease_path.exists() else None
            payload_acquired_at = payload.get("acquired_at_utc") if isinstance(payload, dict) else None
            if isinstance(payload_acquired_at, str):
                acquired_at_utc = payload_acquired_at
            try:
                lease_path.unlink(missing_ok=True)
                released = True
            except Exception:
                released = False

        acquired_dt = self._parse_utc(acquired_at_utc)
        released_dt = self._parse_utc(released_at_utc)
        if acquired_dt is not None and released_dt is not None:
            hold_seconds = max(0.0, float((released_dt - acquired_dt).total_seconds()))

        return StageLeaseReleaseResult(
            lease_id=str(lease_id),
            lease_class="gpu",
            released=bool(released),
            released_at_utc=str(released_at_utc),
            hold_seconds=(float(hold_seconds) if hold_seconds is not None else None),
        )

    def cleanup_owner_leases(self, *, owner_identity: str) -> int:
        released_count = 0
        normalized_owner = str(owner_identity)
        with self._coordinator_lock():
            for lease_path in self._list_active_gpu_lease_paths():
                payload = self._read_lease_payload(lease_path)
                payload_owner = (
                    str(payload.get("owner_identity"))
                    if isinstance(payload, dict) and payload.get("owner_identity") is not None
                    else None
                )
                if payload_owner != normalized_owner:
                    continue
                try:
                    lease_path.unlink(missing_ok=True)
                    released_count += 1
                except Exception:
                    continue
        return int(released_count)

    def cleanup_run_leases(self, *, run_id: str) -> int:
        released_count = 0
        normalized_run_id = str(run_id)
        with self._coordinator_lock():
            for lease_path in self._list_active_gpu_lease_paths():
                payload = self._read_lease_payload(lease_path)
                payload_run_id = (
                    str(payload.get("run_id"))
                    if isinstance(payload, dict) and payload.get("run_id") is not None
                    else None
                )
                if payload_run_id != normalized_run_id:
                    continue
                try:
                    lease_path.unlink(missing_ok=True)
                    released_count += 1
                except Exception:
                    continue
        return int(released_count)

    def snapshot(self) -> dict[str, Any]:
        active_payloads: list[dict[str, Any]] = []
        with self._coordinator_lock():
            for lease_path in self._list_active_gpu_lease_paths():
                payload = self._read_lease_payload(lease_path)
                if isinstance(payload, dict):
                    active_payloads.append(payload)
        return {
            "lease_root": str(self.lease_root.resolve()),
            "max_parallel_gpu_leases": int(self.max_parallel_gpu_leases),
            "active_gpu_lease_count": int(len(active_payloads)),
            "active_gpu_leases": active_payloads,
        }


def build_stage_lease_manager_from_context(
    *,
    stage_lease_context: Mapping[str, Any] | None,
    reports_root: Path,
    run_id: str,
    max_parallel_gpu_leases: int,
) -> StageLeaseManager:
    context = dict(stage_lease_context) if isinstance(stage_lease_context, Mapping) else {}

    lease_root_raw = context.get("lease_root")
    if isinstance(lease_root_raw, str) and lease_root_raw.strip():
        lease_root = Path(lease_root_raw)
    else:
        lease_root = Path(reports_root) / ".stage_leases" / str(run_id)

    context_max_parallel_gpu_leases = context.get("max_parallel_gpu_leases")
    resolved_max_parallel_gpu_leases = (
        int(context_max_parallel_gpu_leases)
        if context_max_parallel_gpu_leases is not None
        else int(max_parallel_gpu_leases)
    )
    poll_interval_seconds = float(context.get("poll_interval_seconds", 0.05))

    return StageLeaseManager(
        lease_root=lease_root,
        max_parallel_gpu_leases=resolved_max_parallel_gpu_leases,
        poll_interval_seconds=poll_interval_seconds,
    )


__all__ = [
    "LeaseClass",
    "StageLeaseHandle",
    "StageLeaseManager",
    "StageLeaseReleaseResult",
    "StageLeaseRequest",
    "build_stage_lease_manager_from_context",
]
