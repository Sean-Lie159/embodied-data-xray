"""领域工具包：Agent 的"手"。每个工具一个模块，返回精简结果。"""

from app.tools.check_sensor_sanity import check_sensor_sanity, check_sensor_sanity_impl
from app.tools.check_temporal_sync import check_temporal_sync, check_temporal_sync_impl
from app.tools.compute_stats import compute_stats, compute_stats_impl
from app.tools.generate_report import generate_report, generate_report_impl
from app.tools.inspect_streams import inspect_streams, inspect_streams_impl
from app.tools.load_dataset import load_dataset, load_dataset_impl
from app.tools.plot_chart import plot_chart, plot_chart_impl
from app.tools.profile_data import profile_data, profile_data_impl

__all__ = [
    "load_dataset",
    "load_dataset_impl",
    "profile_data",
    "profile_data_impl",
    "inspect_streams",
    "inspect_streams_impl",
    "check_temporal_sync",
    "check_temporal_sync_impl",
    "check_sensor_sanity",
    "check_sensor_sanity_impl",
    "compute_stats",
    "compute_stats_impl",
    "plot_chart",
    "plot_chart_impl",
    "generate_report",
    "generate_report_impl",
]
