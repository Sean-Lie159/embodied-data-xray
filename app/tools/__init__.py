"""领域工具包：Agent 的"手"。每个工具一个模块，返回精简结果。"""

from app.tools.align_container import (
    align_container_streams,
    align_container_streams_impl,
)
from app.tools.check_sensor_sanity import check_sensor_sanity, check_sensor_sanity_impl
from app.tools.compare_datasets import compare_datasets, compare_datasets_impl
from app.tools.check_temporal_sync import check_temporal_sync, check_temporal_sync_impl
from app.tools.compute_stats import compute_stats, compute_stats_impl
from app.tools.generate_report import generate_report, generate_report_impl
from app.tools.inspect_streams import inspect_streams, inspect_streams_impl
from app.tools.inspect_video_frame import (
    inspect_video_frame,
    inspect_video_frame_impl,
)
from app.tools.load_dataset import (
    confirm_stream_semantic_impl,
    load_dataset,
    load_dataset_impl,
)
from app.tools.plot_chart import plot_chart, plot_chart_impl
from app.tools.profile_data import profile_data, profile_data_impl
from app.tools.propose_semantics import (
    propose_stream_semantics,
    propose_stream_semantics_impl,
)
from app.tools.mcap_reader import (
    unpack_mcap,
    unpack_mcap_to_dir,
    unpack_mcap_tool_impl,
)

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
    "propose_stream_semantics",
    "propose_stream_semantics_impl",
    "unpack_mcap",
    "unpack_mcap_to_dir",
    "unpack_mcap_tool_impl",
    "align_container_streams",
    "align_container_streams_impl",
    "compare_datasets",
    "compare_datasets_impl",
    "inspect_video_frame",
    "inspect_video_frame_impl",
]
