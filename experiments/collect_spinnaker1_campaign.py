"""Collect existing NMPI jobs and analyze complete label-free hardware captures.

Run with an authenticated NMPI client in the experiment notebook. This module
does not submit jobs, request resources, or load ground-truth labels.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from urllib.request import urlretrieve
import zipfile

from analyze_spinnaker1_matrix import analyze as analyze_shd
from analyze_spinnaker1_dvs_matrix import analyze as analyze_dvs
from analyze_spinnaker1_repeats import analyze as analyze_repeats
from summarize_spinnaker1_reports import summarize as summarize_reports


TERMINAL = {"finished", "error", "cancelled", "canceled"}


def collect_campaign(client, jobs, destination, audits, *, poll_seconds=300,
                     download=urlretrieve, sleep=time.sleep):
    """Wait on existing handles, retain outputs, and aggregate each complete task.

    Each job entry has ``job``, ``task`` (shd/dvs), and a unique ``name``.
    Session/account identifiers remain in this local collection ledger, not in
    the scientific analysis reports. A notebook can run this in a background
    thread and read ``progress.json`` without polling the service again.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    states, captured, analyzed = {}, {}, set()
    while True:
        for entry in jobs:
            name = entry["name"]
            if name in captured:
                continue
            try:
                # The service has intermittently rejected either request form.
                # A log failure is not evidence that hardware execution stopped.
                log_error = None
                try:
                    state = client.get_job(entry["job"], with_log=True)
                except Exception as error:
                    log_error = str(error)
                    state = client.get_job(entry["job"], with_log=False)
                status = state.get("status", "unknown")
                log = state.get("log", "")
                lines = [line for line in log.splitlines()
                         if " shd-test-" in line or line.startswith("input ")]
                states[name] = dict(status=status, last_progress=lines[-1:] or [])
                if log_error is not None:
                    states[name]["log_retrieval_error"] = log_error
                if status not in TERMINAL:
                    continue
                target = destination / name
                target.mkdir(exist_ok=True)
                if log_error is None:
                    (target / "service.log").write_text(log, encoding="utf-8")
                else:
                    (target / "service_log_unavailable.txt").write_text(log_error, encoding="utf-8")
                files = (state.get("output_data") or {}).get("files", [])
                capture_urls = [item["url"] for item in files
                                if item["url"].endswith("_capture.zip")]
                if not capture_urls:
                    captured[name] = None
                    states[name]["capture"] = "no capture archive returned"
                    continue
                archive = target / "capture.zip"
                download(capture_urls[0], archive)
                capture = target / "capture"
                with zipfile.ZipFile(archive) as packed:
                    packed.extractall(capture)
                for item in files:
                    if item["url"].endswith("reports.zip"):
                        download(item["url"], target / "reports.zip")
                        summary = summarize_reports([target / "reports.zip"])
                        (target / "execution_log_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
                captured[name] = capture
                states[name]["capture"] = "downloaded"
            except Exception as error:
                states[name] = dict(status="collection_retry", error=str(error))

        for task, analyzer in (("shd", analyze_shd), ("dvs", analyze_dvs)):
            selected = [entry for entry in jobs if entry["task"] == task]
            if not selected or task in analyzed or not all(e["name"] in captured for e in selected):
                continue
            paths = [captured[e["name"]] for e in selected if captured[e["name"]] is not None]
            report = analyzer(paths, Path(audits[task]))
            (destination / f"{task}_label_free_analysis.json").write_text(json.dumps(report, indent=2)+"\n")
            repeats = analyze_repeats(paths, task)
            (destination / f"{task}_repeat_analysis.json").write_text(json.dumps(repeats, indent=2)+"\n")
            analyzed.add(task)

        complete = all(entry["name"] in captured for entry in jobs)
        progress = dict(status="collection_finished" if complete else "waiting_for_hardware",
                        updated_at=time.time(), jobs=states, analyzed_tasks=sorted(analyzed))
        (destination / "progress.json").write_text(json.dumps(progress, indent=2)+"\n")
        if complete:
            return progress
        sleep(poll_seconds)
