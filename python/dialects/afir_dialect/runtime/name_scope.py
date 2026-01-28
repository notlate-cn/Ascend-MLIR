# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Dict, Optional, Set


class NameScope:
    builtins: Dict[str, Any] = {
        b.__name__: b
        for b in (
            dict,
            float,
            int,
            isinstance,
            issubclass,
            len,
            list,
            range,
            repr,
            str,
            tuple,
            type,
        )
    }

    def __init__(self, global_vars: Dict[str, Any], local_vars: Optional[Dict[str, Any]] = None):
        self.global_vars = global_vars
        self.local_vars = {} if local_vars is None else local_vars
        self.sentinel = object()
        self.defined: Set[str] = set()
        self.redefined: Set[str] = set()

    def __repr__(self) -> str:
        return f"NameScope(globals={self.global_vars}, locals={self.local_vars})"

    def inherit(self, copy_globals=False):
        global_vars = self.global_vars.copy() if copy_globals else self.global_vars
        return NameScope(global_vars, self.local_vars.copy())

    def save(self, name: str, value: Any) -> None:
        if name not in self.local_vars:
            self.defined.add(name)
        elif name not in self.defined:
            self.redefined.add(name)
        self.local_vars[name] = value

    def lookup(self, name: str) -> Optional[Any]:
        for storage in self.local_vars, self.global_vars, self.builtins:
            val = storage.get(name, self.sentinel)
            if val is not self.sentinel:
                return val
        raise NameError(f"{name} is not defined")

    def reset_def(self) -> None:
        self.defined.clear()
        self.redefined.clear()
