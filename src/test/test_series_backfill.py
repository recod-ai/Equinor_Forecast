# tests/test_series_backfill.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import types

import pytest

# We import the module under test
import common.series_backfill as sb


class _DummySettings:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_missing_artifacts_finds_only_success_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    - Creates 3 JSONs: (success,success,failed)
    - Simulates that one Parquet already exists and the other does not.
    - Expects discover_missing_artifacts to yield only the truly missing success.
    """
    results_root = tmp_path / "results_root"
    series_root = tmp_path / "series_root"

    # Minimal valid job_meta
    jm = {
        "group": "G",
        "arch": "seq2",
        "dataset": "UNISIM_IV",
        "well": "P16",
        "campaign": "CAMP1",
        "job_hash": "abcd1234",
    }
    jm2 = {**jm, "job_hash": "efgh5678"}

    # Create JSON files
    success1 = results_root / "a" / "job1.json"
    success2 = results_root / "b" / "job2.json"
    failed1  = results_root / "c" / "job3.json"

    _write_json(success1, {"status": "success", "job_meta": jm})
    _write_json(success2, {"status": "success", "job_meta": jm2})
    _write_json(failed1,  {"status": "failed",  "job_meta": jm})

    # Monkeypatch derive_series_path to a deterministic path
    def fake_derive_series_path(settings, group, arch, dataset, well, campaign, job_hash):
        return series_root / f"group={group}/arch={arch}/dataset={dataset}/well={well}/campaign={campaign}/job={job_hash}.parquet"

    monkeypatch.setattr("common.series_backfill.derive_series_path", fake_derive_series_path)

    # Pretend job1 parquet exists; job2 parquet missing
    existing_parquet = series_root / "group=G/arch=seq2/dataset=UNISIM_IV/well=P16/campaign=CAMP1/job=abcd1234.parquet"
    existing_parquet.parent.mkdir(parents=True, exist_ok=True)
    existing_parquet.write_bytes(b"PARQUET")

    found = list(sb.discover_missing_artifacts(results_root, series_root))
    assert len(found) == 1
    assert found[0].job_hash == "efgh5678"
    assert found[0].json_path == success2


def test_run_backfill_dry_run_and_real(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verifies that:
    - dry_run returns DRY_RUN_CREATE without writing
    - real run calls build_series_record and persist_series once per artifact
    """
    results_root = tmp_path / "results_root"
    series_root = tmp_path / "series_root"

    jm = {
        "group": "G",
        "arch": "seq2",
        "dataset": "UNISIM_IV",
        "well": "P16",
        "campaign": "CAMP1",
        "job_hash": "abcd1234",
    }
    json_path = results_root / "ok.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"status": "success", "job_meta": jm, "series_context": {"forecast_agg": {}}}), encoding="utf-8")

    # Fake path derivation to a deterministic place
    def fake_derive_series_path(settings, group, arch, dataset, well, campaign, job_hash):
        p = series_root / f"group={group}/arch={arch}/dataset={dataset}/well={well}/campaign={campaign}/job={job_hash}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr("common.series_backfill.derive_series_path", fake_derive_series_path)

    # Fake writer contracts
    calls = {"build": 0, "persist": 0}

    def fake_build_series_record(result_dict):
        calls["build"] += 1
        return {"_mock": True}  # anything non-empty

    def fake_persist_series(record, settings):
        calls["persist"] += 1
        # write a small file to mark success
        out = fake_derive_series_path(settings, **jm)
        out.write_bytes(b"PARQUET")

    monkeypatch.setattr("common.series_backfill.build_series_record", fake_build_series_record)
    monkeypatch.setattr("common.series_backfill.persist_series", fake_persist_series)

    # Dry run: should not write, should not call writer
    res_dry = sb.run_backfill(results_root, series_root, dry_run=True, concurrency=2)
    assert len(res_dry) == 1
    assert res_dry[0].action == "DRY_RUN_CREATE"
    assert calls["build"] == 0 and calls["persist"] == 0

    # Real run: should create parquet and call writer exactly once
    res_real = sb.run_backfill(results_root, series_root, dry_run=False, concurrency=2)
    assert len(res_real) == 1
    assert res_real[0].action == "CREATED"
    assert calls["build"] == 1 and calls["persist"] == 1

    # Re-run real: now nothing missing → returns empty list (idempotent at discovery stage)
    res_again = sb.run_backfill(results_root, series_root, dry_run=False, concurrency=2)
    assert res_again == []
