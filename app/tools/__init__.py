"""领域工具包：Agent 的"手"。每个工具一个模块，返回精简结果。"""

from app.tools.align_container import (
    align_container_streams,
    align_container_streams_impl,
)
from app.tools.annotation_store import (
    EpisodeAnchor,
    append_annotations,
    check_source_for_use,
    delete_annotation,
    diff_snapshots,
    format_timestamp,
    list_snapshots,
    load_annotation_history,
    load_annotations,
    normalize_record,
    parse_timestamp,
    resolve_anchors,
    snapshot_annotations,
)
from app.tools.annotate import (
    annotate_task,
    annotate_task_impl,
    check_annotation_qc,
    check_annotation_qc_impl,
    save_annotations,
    save_annotations_impl,
)
from app.tools.confirm_dataset import (
    confirm_dataset_profile,
    confirm_dataset_profile_impl,
)
from app.tools.read_file_content import (
    read_file_content,
    read_file_content_impl,
)
from app.tools.segment_actions import segment_actions, segment_actions_impl
from app.tools.check_dataset_quality import (
    check_dataset_quality,
    check_dataset_quality_impl,
)
from app.tools.check_sensor_sanity import check_sensor_sanity, check_sensor_sanity_impl
from app.tools.compare_datasets import compare_datasets, compare_datasets_impl
from app.tools.compare_table_columns import (
    compare_table_columns,
    compare_table_columns_impl,
)
from app.tools.check_temporal_sync import check_temporal_sync, check_temporal_sync_impl
from app.tools.compute_stats import compute_stats, compute_stats_impl
from app.tools.generate_report import generate_report, generate_report_impl
from app.tools.inspect_streams import inspect_streams, inspect_streams_impl
from app.tools.inspect_video_frame import (
    inspect_video_frame,
    inspect_video_frame_impl,
)
from app.tools.list_tables import list_tables, list_tables_impl
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
    "list_tables",
    "list_tables_impl",
    "check_temporal_sync",
    "check_temporal_sync_impl",
    "check_sensor_sanity",
    "check_sensor_sanity_impl",
    "check_dataset_quality",
    "check_dataset_quality_impl",
    "segment_actions",
    "segment_actions_impl",
    "confirm_dataset_profile",
    "confirm_dataset_profile_impl",
    "read_file_content",
    "read_file_content_impl",
    "annotate_task",
    "annotate_task_impl",
    "save_annotations",
    "save_annotations_impl",
    "check_annotation_qc",
    "check_annotation_qc_impl",
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
    "compare_table_columns",
    "compare_table_columns_impl",
    "inspect_video_frame",
    "inspect_video_frame_impl",
    # 标注存储与规范化层（格式无关锚点 + 读写 + 版本管理）。
    "EpisodeAnchor",
    "resolve_anchors",
    "normalize_record",
    "check_source_for_use",
    "load_annotations",
    "append_annotations",
    "delete_annotation",
    "snapshot_annotations",
    "list_snapshots",
    "load_annotation_history",
    "diff_snapshots",
    "format_timestamp",
    "parse_timestamp",
]
