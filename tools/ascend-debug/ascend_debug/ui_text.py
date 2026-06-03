from __future__ import annotations


# Central place for user-facing ascend-debug UI copy.
#
# Keep labels here instead of hardcoding them in renderers. This lets a user
# customize visible wording without tracing through generated HTML templates.
UI_TEXT: dict[str, str] = {
    "advanced_commands_summary": "高级信息：执行命令",
    "args_column": "Args",
    "command_column": "Command",
    "debug_graph_column": "Graph",
    "debug_graph_format_label": "Graph",
    "exists_status": "存在",
    "missing_status": "缺失",
    "mlir_column": "MLIR",
    "open_debug_workbench": "打开调试工作台",
    "overview_heading": "运行概览",
    "primary_debug_description": "统一查看 Stage 演进、Kernel DAG、Tensor Diff、Locate 和 Memory。",
    "report_column": "Report",
    "stage_column": "Stage",
    "stage_order_column": "Order",
    "stage_timeline_heading": "Stage Timeline",
    "status_column": "状态",
    "step_column": "Step / Per pass",
    "text_format_label": "Text",
    "tool_column": "Tool",
    "view_column": "View",
}


def text(key: str) -> str:
    return UI_TEXT[key]
