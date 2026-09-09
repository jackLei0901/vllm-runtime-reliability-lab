from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dfxlab.external_schema import (
    ExternalIncidentArtifact,
    WriterErrorKind,
    WriterHealth,
)

DEFAULT_MAX_ARTIFACT_BYTES = 256 * 1024
DEFAULT_MAX_ARTIFACTS = 4


@dataclass(frozen=True, slots=True)
class WriteOutcome:
    path: Path | None
    size_bytes: int
    error_kind: WriterErrorKind | None

    @property
    def ok(self) -> bool:
        return self.path is not None and self.error_kind is None


class IncidentWriter:
    """Bounded, fail-open writer for shareable external incident artifacts."""

    def __init__(
        self,
        output_dir: Path,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
    ) -> None:
        if max_artifact_bytes <= 0 or max_artifacts <= 0:
            raise ValueError("writer limits must be positive")
        self.output_dir = output_dir
        self.max_artifact_bytes = max_artifact_bytes
        self.max_artifacts = max_artifacts
        self.artifacts_written_total = 0
        self.artifacts_failed_total = 0
        self.last_error_kind: WriterErrorKind | None = None
        self._initialization_error = False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                self.output_dir.chmod(0o700)
            self._remove_stale_temporaries()
            self._rotate(self.max_artifacts)
        except OSError:
            self._initialization_error = True

    def health(self) -> WriterHealth:
        return WriterHealth(
            artifacts_written_total=self.artifacts_written_total,
            artifacts_failed_total=self.artifacts_failed_total,
            last_error_kind=self.last_error_kind,
        )

    def try_write(self, artifact: ExternalIncidentArtifact) -> WriteOutcome:
        if self._initialization_error:
            return self._failure(WriterErrorKind.IO)
        try:
            payload = artifact.to_dict()
        except (TypeError, ValueError):
            return self._failure(WriterErrorKind.VALIDATION)

        encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        if len(encoded) > self.max_artifact_bytes:
            return self._failure(WriterErrorKind.SIZE_LIMIT, len(encoded))

        safe_id = artifact.incident_id.replace(":", "-")
        final_path = self.output_dir / f"incident-{safe_id}.json"
        temporary = self.output_dir / f".{final_path.name}.tmp"
        try:
            self._rotate(self.max_artifacts - 1)
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600,
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise
            os.replace(temporary, final_path)
            if os.name != "nt":
                final_path.chmod(0o600)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return self._failure(WriterErrorKind.IO, len(encoded))

        self.artifacts_written_total += 1
        self.last_error_kind = None
        return WriteOutcome(final_path, len(encoded), None)

    def _failure(
        self, error_kind: WriterErrorKind, size_bytes: int = 0
    ) -> WriteOutcome:
        self.artifacts_failed_total += 1
        self.last_error_kind = error_kind
        return WriteOutcome(None, size_bytes, error_kind)

    def _remove_stale_temporaries(self) -> None:
        for path in self.output_dir.glob(".incident-*.tmp"):
            path.unlink()

    def _rotate(self, keep: int) -> None:
        paths = sorted(
            self.output_dir.glob("incident-*.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        for path in paths[max(keep, 0) :]:
            path.unlink()
