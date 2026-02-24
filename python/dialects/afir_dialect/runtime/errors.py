# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
from typing import List, Optional


class CodegenError(Exception):
    context_lines = 3

    def __init__(self, node: ast.AST, src: Optional[List[str]] = None, message: Optional[str] = None):
        super().__init__()
        self.src = src
        self.node = node
        self.error_message = message
        self.message = self.format_message()

    def __str__(self) -> str:
        return self.message

    def format_message(self) -> str:
        node = self.node
        message = []
        if self.src is None:
            message.append(" <source unavailable>")
        else:
            if hasattr(node, "lineno") and hasattr(node, "col_offset"):
                message.append(f"at <source>:{node.lineno}:{node.col_offset}:")
                excerpt = self.src[(node.lineno - self.context_lines - 1):node.lineno]
                if excerpt:
                    message.extend(excerpt)
                    message.append(" " * node.col_offset + "^")
                    message.extend(self.src[node.lineno:node.lineno + self.context_lines])
                else:
                    message.append(" <source empty>")
            else:
                message.extend(self.src)
        if self.error_message:
            message.append(self.error_message)
        return "\n".join(message)


class UnsupportedSyntaxError(CodegenError):
    pass
