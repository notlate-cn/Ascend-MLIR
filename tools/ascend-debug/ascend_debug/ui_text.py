from __future__ import annotations


# Central place for user-facing ascend-debug UI copy.
#
# Keep labels here instead of hardcoding them in renderers. This lets a user
# customize visible wording without tracing through generated HTML templates.
UI_TEXT: dict[str, str] = {
    "advanced_commands_summary": "高级信息：执行命令",
    "args_column": "Args",
    "debug_graph_column": "调试图",
    "debug_graph_format_label": "图形化格式",
    "exists_status": "存在",
    "missing_status": "缺失",
    "mlir_column": "MLIR",
    "open_debug_workbench": "打开调试工作台",
    "overview_heading": "运行概览",
    "primary_debug_description": "统一查看 Stage 演进、Kernel DAG、Tensor Diff、Locate 和 Memory。",
    "report_column": "报告",
    "stage_column": "Stage",
    "stage_order_column": "顺序",
    "stage_timeline_heading": "Stage Timeline",
    "status_column": "状态",
    "step_column": "Step",
    "text_format_label": "文本格式",
    "tool_column": "Tool",
}


def text(key: str) -> str:
    return UI_TEXT[key]
