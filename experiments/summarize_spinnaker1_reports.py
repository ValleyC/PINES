"""Summarize full execution-log warnings from SpiNNaker report archives.

Per-input diagnostic queries can be empty before shutdown retrieves provenance.
This reads the final log database without changing predictions or sample counts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import zipfile


def warning_category(message):
    if "SpikeSourceArray sending too many spikes" in message:
        return "dense_spike_source"
    if "packets were dumped from outgoing links" in message:
        return "router_packet_dump"
    if "transmission buffer" in message and "was blocked" in message:
        return "transmission_backpressure"
    if "cfg has no version" in message:
        return "configuration_deprecation"
    if "spinnaker_get_data is non-standard PyNN" in message:
        return "reader_api_portability"
    return "other"


def summarize(archives):
    rows = []
    for index, path in enumerate(archives):
        with zipfile.ZipFile(path) as archive:
            databases = [name for name in archive.namelist()
                         if name.endswith("global_provenance.sqlite3")]
            for database_index, name in enumerate(databases):
                connection = sqlite3.connect(":memory:")
                try:
                    connection.deserialize(archive.read(name))
                    entries = connection.execute(
                        "SELECT level,message FROM p_log_provenance WHERE level>=30"
                    ).fetchall()
                finally:
                    connection.close()
                rows.append(dict(
                    archive_index=index, database_index=database_index,
                    warning_log_entries=sum(level == 30 for level, _ in entries),
                    error_or_critical_log_entries=sum(level >= 40 for level, _ in entries),
                    categories=dict(Counter(warning_category(message)
                                            for _, message in entries))))
            if not databases:
                rows.append(dict(archive_index=index, status="no_global_log_database"))
    return dict(
        scope="Execution log summary, not a certificate or packet-loss estimate.",
        counts="Log entries, not distinct hardware incidents or lost packets. "
               "Counters can be reported repeatedly across resets and reinjections.",
        interpretation="Retain all primary predictions. These warnings do not identify "
                       "which predictions changed or prove that an execution is invalid.",
        review="Review raw service logs and other-category entries alongside this summary.",
        archives=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
