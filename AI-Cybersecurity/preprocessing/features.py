"""
Module 3: Feature Engineering
-------------------------------
Derives security-relevant features from whatever raw columns are actually
present. Every derived feature is conditional -- we never fabricate a
feature from data that doesn't exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add derived features where source columns support them.
    Returns (df_with_features, list_of_features_actually_added).
    """
    df = df.copy()
    added: list[str] = []

    has = lambda col: col in df.columns  # noqa: E731

    # Packets per second
    if has("packet_count") and has("flow_duration"):
        df["packets_per_second"] = df["packet_count"] / df["flow_duration"].replace(0, np.nan)
        added.append("packets_per_second")

    # Bytes per second (total traffic / duration)
    if has("bytes_sent") and has("bytes_received") and has("flow_duration"):
        total_bytes = df["bytes_sent"] + df["bytes_received"]
        df["bytes_per_second"] = total_bytes / df["flow_duration"].replace(0, np.nan)
        added.append("bytes_per_second")

    # Average packet size
    if has("bytes_sent") and has("bytes_received") and has("packet_count"):
        total_bytes = df["bytes_sent"] + df["bytes_received"]
        df["avg_packet_size"] = total_bytes / df["packet_count"].replace(0, np.nan)
        added.append("avg_packet_size")

    # Send/receive ratio (helps flag exfiltration-like or scan-like asymmetry)
    if has("bytes_sent") and has("bytes_received"):
        df["send_receive_ratio"] = df["bytes_sent"] / df["bytes_received"].replace(0, np.nan)
        added.append("send_receive_ratio")

    # Failed login rate relative to flow duration
    if has("failed_logins") and has("flow_duration"):
        df["failed_logins_per_second"] = df["failed_logins"] / df["flow_duration"].replace(0, np.nan)
        added.append("failed_logins_per_second")

    # Connection frequency per source IP (how many events from same source)
    if has("source_ip"):
        freq = df["source_ip"].value_counts()
        df["connection_frequency"] = df["source_ip"].map(freq)
        added.append("connection_frequency")

    # Unique destination ports per source IP (port-scan indicator)
    if has("source_ip") and has("destination_port"):
        unique_ports = df.groupby("source_ip")["destination_port"].transform("nunique")
        df["unique_dest_ports_by_source"] = unique_ports
        added.append("unique_dest_ports_by_source")

    # Clean up any inf/NaN produced by the derived ratios above
    new_cols = added
    if new_cols:
        df[new_cols] = df[new_cols].replace([np.inf, -np.inf], np.nan)

    return df, added


def get_available_features_report(df: pd.DataFrame, added: list[str]) -> dict:
    """Document which engineered features were available for this dataset."""
    all_possible = [
        "packets_per_second", "bytes_per_second", "avg_packet_size",
        "send_receive_ratio", "failed_logins_per_second",
        "connection_frequency", "unique_dest_ports_by_source",
    ]
    return {
        "available_features": added,
        "unavailable_features": [f for f in all_possible if f not in added],
        "reason": "Unavailable features require source columns not present in this dataset.",
    }
