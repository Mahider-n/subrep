"""Atomic admission snapshots and persistent rejection history.

The authoritative snapshot commits certificates, library identities, manifests
and evidence together. Native MeTTa/SkillLibrary instances are reconstructed and
reverified before every commit/restart. Exported files are derived artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from baseline.improvement_calculator import ImprovementCalculator
from certification.certificate_schema import Certificate
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from certification.metta_storage import CertificateStore
from library.skill_library import SkillLibrary
from .contracts import EvaluationEvidence, EvaluationProtocol, canonical, digest, nonnegative


def now():
    return datetime.now(timezone.utc).isoformat()


def improvements(baseline, candidate):
    if baseline.protocol != candidate.protocol:
        raise ValueError("evaluation_protocol_mismatch")
    calculator = ImprovementCalculator({"baseline_payoff": baseline.mean_payoff,
                                        "baseline_motives": baseline.mean_objectives})
    return calculator.compute_improvements(candidate.mean_payoff, candidate.mean_objectives)


class AdmissionStore:
    def __init__(self, path, protocol: EvaluationProtocol):
        self.path = Path(path)
        self.protocol = protocol
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._read()
        self.certificates, self.library = self._materialize(self._data)

    def _empty(self):
        return {"format": "metamo_subrep_phase1_v1", "protocol": asdict(self.protocol),
                "baseline": None, "entries": {}, "rejections": [], "events": []}

    def _read(self):
        if not self.path.exists():
            return self._empty()
        data = json.loads(self.path.read_text())
        if data.get("format") != "metamo_subrep_phase1_v1":
            raise ValueError("unknown_snapshot_format")
        if EvaluationProtocol.from_dict(data["protocol"]) != self.protocol:
            raise ValueError("snapshot_reward_schema_or_protocol_mismatch")
        return data

    @contextmanager
    def _locked(self):
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = self._read()
                self._materialize(data)
                yield data
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _materialize(self, data):
        store = CertificateStore()
        library = SkillLibrary(cert_store=store)
        baseline = EvaluationEvidence.from_dict(data["baseline"]) if data["baseline"] else None
        if baseline and baseline.protocol != self.protocol:
            raise ValueError("baseline_protocol_mismatch")
        for sid, item in data["entries"].items():
            if baseline is None:
                raise ValueError("certificate_missing_baseline")
            cert = Certificate.from_dict(item["certificate"])
            evidence = EvaluationEvidence.from_dict(item["evidence"])
            manifest = item["registration"]
            dr, dn = improvements(baseline, evidence) if baseline else (None, None)
            if (sid != cert.skill_id or sid != evidence.behavior_id
                    or cert.weight_region_type != "FULL_SIMPLEX"
                    or manifest["schema_id"] != self.protocol.schema.identity
                    or item["certificate_digest"] != digest(cert.to_dict())
                    or item["evidence_id"] != evidence.identity
                    or cert.baseline_id != baseline.identity
                    or cert.environment != self.protocol.schema.environment
                    or cert.version != self.protocol.schema.version
                    or cert.seed != self.protocol.seeds[0]
                    or cert.gamma != self.protocol.gamma
                    or cert.episode_length != self.protocol.horizon
                    or not np.isclose(cert.delta_r, dr, atol=1e-9, rtol=0)
                    or not np.allclose(cert.delta_n, dn, atol=1e-9, rtol=0)):
                raise ValueError("certificate_manifest_or_evidence_mismatch")
            expected_sid = f"{manifest['action_id']}.{manifest['behavior_version']}.{digest(manifest)}"
            if sid != expected_sid:
                raise ValueError("behavior_identity_mismatch")
            margin = cert.delta_r + min(cert.delta_n) + cert.epsilon
            if not math.isclose(cert.admission_margin, margin, abs_tol=1e-9, rel_tol=0):
                raise ValueError("certificate_margin_mismatch")
            if not store.add(cert) or not library.add_skill(sid, cert, policy=None):
                raise ValueError("certificate_library_reverification_failed")
        if {c.skill_id for c in store.load_all()} != {s.skill_id for s in library.get_admitted_skills()}:
            raise ValueError("certificate_library_identity_mismatch")
        event_ids = set()
        for record in data["rejections"]:
            identity = record["record_id"]
            if identity in event_ids or record["record_digest"] != digest(
                    {k: v for k, v in record.items() if k != "record_digest"}):
                raise ValueError("invalid_rejection_history")
            event_ids.add(identity)
        return store, library

    def _atomic_write(self, data):
        descriptor, temporary = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w") as file:
                file.write(canonical(data) + "\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _commit(self, data):
        store, library = self._materialize(data)
        self._preserve_policies(library)
        self._atomic_write(data)
        self._data, self.certificates, self.library = data, store, library

    def _preserve_policies(self, library):
        for entry in library.get_admitted_skills():
            previous = self.library.get_skill(entry.skill_id)
            if previous and previous.certificate == entry.certificate:
                library.register_policy(entry.skill_id, previous.policy)

    def set_baseline(self, baseline: EvaluationEvidence):
        if baseline.protocol != self.protocol:
            raise ValueError("baseline_protocol_mismatch")
        with self._locked() as data:
            if data["baseline"] and digest(data["baseline"]) != baseline.identity:
                raise ValueError("immutable_baseline_conflict")
            data["baseline"] = asdict(baseline)
            self._commit(data)

    @property
    def rejections(self):
        return copy.deepcopy(self._data["rejections"])

    @property
    def events(self):
        return copy.deepcopy(self._data["events"])

    def record_rejection(self, category, *, action_id=None, behavior_id=None,
                         scope="admission", context_id=None, evidence=None, details=None):
        record = {"record_id": uuid.uuid4().hex, "timestamp": now(), "category": category,
                  "action_id": action_id, "behavior_id": behavior_id,
                  "protocol_id": self.protocol.identity, "scope": scope,
                  "context_id": context_id, "evidence": evidence, "details": details}
        record["record_digest"] = digest(record)
        with self._locked() as data:
            data["rejections"].append(record)
            self._commit(data)
        return record["record_id"]

    def record_event(self, event):
        with self._locked() as data:
            data["events"].append(copy.deepcopy(event))
            self._commit(data)

    def admit(self, registration, evidence: EvaluationEvidence, *, epsilon=0):
        epsilon = nonnegative(epsilon, "epsilon")
        try:
            registration.verify()
            if registration.schema != self.protocol.schema:
                raise ValueError("reward_schema_mismatch")
            if evidence.behavior_id != registration.skill_id:
                raise ValueError("evidence_behavior_mismatch")
            with self._locked() as data:
                if not data["baseline"]:
                    raise ValueError("missing_baseline")
                baseline = EvaluationEvidence.from_dict(data["baseline"])
                dr, dn = improvements(baseline, evidence)
                cds = CDSGate().admit(dr, dn)
                pds = PDSGate(epsilon).admit(dr, dn)
                if not cds and not pds:
                    raise ValueError("gate_failed")
                existing = data["entries"].get(registration.skill_id)
                budget = 0.0 if cds else epsilon
                if existing:
                    cert = Certificate.from_dict(existing["certificate"])
                    if (existing["evidence_id"] != evidence.identity
                            or existing["registration"] != registration.manifest()
                            or cert.epsilon != budget):
                        raise ValueError("immutable_admission_conflict")
                    self._data = data
                    store, library = self._materialize(data)
                    self._preserve_policies(library)
                    self.certificates, self.library = store, library
                    return {"admitted": True, "certificate": cert, "idempotent": True}
                cert = Certificate(
                    skill_id=registration.skill_id, gate_type="CDS" if cds else "PDS",
                    delta_r=dr, delta_n=tuple(dn), admission_margin=dr + float(min(dn)) + budget,
                    epsilon=budget, timestamp=now(), seed=self.protocol.seeds[0],
                    gamma=self.protocol.gamma, baseline_id=baseline.identity,
                    environment=self.protocol.schema.environment,
                    episode_length=self.protocol.horizon, version=self.protocol.schema.version,
                )
                data["entries"][registration.skill_id] = {
                    "certificate": cert.to_dict(), "certificate_digest": digest(cert.to_dict()),
                    "registration": registration.manifest(), "evidence": asdict(evidence),
                    "evidence_id": evidence.identity,
                }
                self._commit(data)
                return {"admitted": True, "certificate": cert, "idempotent": False}
        except Exception as exc:
            reason = str(exc)
            try:
                record_id = self.record_rejection(
                    reason, action_id=registration.action_id, behavior_id=registration.skill_id,
                    evidence=asdict(evidence), details={"epsilon": epsilon})
                persisted = True
            except Exception:
                record_id, persisted = None, False
            return {"admitted": False, "reason": reason, "rejection_id": record_id,
                    "rejection_persisted": persisted}

    def resolve_certificate(self, registration):
        registration.verify()
        if registration.schema != self.protocol.schema:
            raise ValueError("reward_schema_mismatch")
        item = self._data["entries"].get(registration.skill_id)
        if item is None:
            raise ValueError("no_certificate")
        if item["registration"] != registration.manifest():
            raise ValueError("registration_manifest_mismatch")
        cert = self.certificates.get_certificate(registration.skill_id)
        entry = self.library.get_skill(registration.skill_id)
        if cert is None or entry is None or cert != entry.certificate:
            raise ValueError("certificate_library_identity_mismatch")
        return cert

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.certificates.save_to_file(directory / "certificates.metta")
        self.library.save(str(directory / "library.json"))
