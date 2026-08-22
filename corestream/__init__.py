"""CoreStream: tiered-memory streaming inference for consumer GPUs."""

from .engine import EngineConfig, RunReport, StreamingEngine, zipf_router
from .hardware import HardwareProfile, Roofline, profile, roofline
from .inspector import ModelKind, ModelTopology
from .loaders import DensePlan, ExecutionPlan, MoEPlan
from .scheduler import PrefetchScheduler
from .sources import SafetensorsSource, SyntheticSource
from .store import LFUAdmission, Tier, TieredWeightStore

__version__ = "0.1.0"

__all__ = [
    "DensePlan",
    "EngineConfig",
    "ExecutionPlan",
    "HardwareProfile",
    "LFUAdmission",
    "MoEPlan",
    "ModelKind",
    "ModelTopology",
    "PrefetchScheduler",
    "Roofline",
    "RunReport",
    "SafetensorsSource",
    "StreamingEngine",
    "SyntheticSource",
    "Tier",
    "TieredWeightStore",
    "profile",
    "roofline",
    "zipf_router",
]
