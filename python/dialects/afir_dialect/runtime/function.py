# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import inspect
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, get_args, get_origin, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class FunctionLocation:
    filename: str = "<source>"
    line_offset: int = 0


class Function(Generic[P, T]):
    fn: Callable[P, T]
    node: ast.FunctionDef
    location: FunctionLocation
    src: Optional[List[str]]

    def __init__(self, fn: Callable[P, T]):
        if not callable(fn):
            raise TypeError(f"{fn.__class__.__name__} instance is not callable")
        self.fn = fn
        self.node = self.get_function_node(fn)
        self.location = self.get_location(fn)

        self.raw_src, self.starting_line_number = self.get_source_lines(fn)
        self.src = "".join(self.raw_src).splitlines()

        self.fn_name = self.get_full_name(fn)
        src_temp = textwrap.dedent("".join(self.raw_src))
        self.src_without_decorator = src_temp[re.search(r"^def\s+\w+\s*\(", src_temp, re.MULTILINE).start():]
        self.hash = None

        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

        # reuse docs of wrapped function
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__qualname__ = fn.__qualname__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.fn(*args, **kwargs)

    # @property
    # def cache_key(self):
    #     # NOTE: hash should be attribute of `self`
    #     if self.hash is None:
    #         # Set a placeholder hash to break recursion in case the function
    #         # transitively calls itself. The full hash is set after.
    #         self.hash = f"recursion:{self.fn_name}"
    #         nonlocals = inspect.getclosurevars(self.fn).nonlocals
    #         from .dependencies_finder import DependenciesFinder
    #         dependencies_finder = DependenciesFinder(
    #             name_=self.fn_name,
    #             globals_=self.__globals__,
    #             nonlocals_=nonlocals,
    #             src_=self.src_without_decorator,
    #         )
    #         dependencies_finder.visit(self.parse())
    #         self.hash = dependencies_finder.ret + str(self.starting_line_number)
    #         self.used_global_vals = dict(sorted(dependencies_finder.used_global_vals.items()))
    #         self.hash += str([(name, val)
    #                           for (name, _), (val, _) in self.used_global_vals.items()
    #                           if isinstance(val, ConstExpr)])
    #         self.hash = hashlib.sha256(self.hash.encode("utf-8")).hexdigest()
    #     return self.hash

    @staticmethod
    def get_function_node(fn: Callable) -> ast.FunctionDef:
        source = inspect.getsource(fn)
        source = textwrap.dedent("".join(source))
        source = source[re.search(r"^def\s+\w+\s*\(", source, re.MULTILINE).start():]
        node = ast.parse(source)
        if not isinstance(node, ast.Module) or len(node.body) != 1:
            raise RuntimeError("Unexpected function definition, must be ast.Module node with a single child")
        def_node = node.body[0]
        if not isinstance(def_node, ast.FunctionDef):
            raise TypeError(f"JIT compilation is applicable to functions only, got {def_node.__class__.__name__}")
        return def_node

    @staticmethod
    def get_location(fn: Callable) -> FunctionLocation:
        code = fn.__code__
        return FunctionLocation(code.co_filename, code.co_firstlineno)

    @staticmethod
    def get_source_lines(fn: Callable) -> Optional[Tuple[List[str], int]]:
        try:
            lines, lnum = inspect.getsourcelines(fn)
            return lines, lnum
        except OSError:
            return None

    @staticmethod
    def get_full_name(fn: Callable) -> str:
        return f"{fn.__module__}.{fn.__qualname__}"


    # we do not parse `src` in the constructor because
    # the user might want to monkey-patch self.src dynamically.
    # Our unit tests do this, for example.
    def parse(self):
        tree = ast.parse(self.src_without_decorator)
        if not isinstance(tree, ast.Module):
            raise TypeError("tree must be type of ast.Module")
        if len(tree.body) != 1:
            raise ValueError("the length of tree.body must be 1")
        if not isinstance(tree.body[0], ast.FunctionDef):
            raise TypeError("tree.body[0] must be type of ast.FunctionDef")
        return tree
