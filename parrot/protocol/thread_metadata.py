from dataclasses import dataclass


@dataclass
class ThreadMetadata:
    """Thread metadata includes useful information of the thread passed to OS,
    enabling better thread scheduling (such as DAG-aware scheduling)."""

    is_latency_critical: bool
