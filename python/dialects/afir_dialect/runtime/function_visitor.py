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
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Generator, Iterable, List, Literal, NoReturn, Optional, Tuple, Type, TypeVar,
                    Union, ParamSpec, TypeAlias)

from .function import Function, FunctionLocation

from .errors import CodegenError, UnsupportedSyntaxError

from .dtype import KnownTypes

from .ir_value import IRHandle, IRValue, PlainValue, materialize_ir_value
from .name_scope import NameScope

from .utils import ConvertPythonTypeToMLIRType

import mlir.ir as ir

import mlir.dialects.func as func
import mlir.dialects.scf as scf
import mlir.dialects.afir as afir
import mlir.dialects.arith as arith
import mlir.dialects.cf as cf


T = TypeVar("T")
P = ParamSpec("P")
@dataclass
class CodegenOptions:
    capture_exceptions: bool = True
    ir_multithreading: bool = True


@dataclass
class ReturnType:
    py_type: Type[IRValue]
    ir_type: ir.Type


ReturnTypesDict: TypeAlias = Dict[str, List[ReturnType]]


@dataclass
class VisitorState:
    discard_everything: bool = False
    inside_function: bool = False
    return_allowed: bool = True
    return_types: List[ReturnType] = field(default_factory=list)
    visited_return_types: ReturnTypesDict = field(default_factory=dict)


@dataclass
class BlockInOut:
    block: scf.ExecuteRegionOp
    init_handles: Dict[str, IRHandle]
    yield_handles: Dict[str, IRHandle]


class FunctionVisitor(ast.NodeVisitor):
    def __init__(
        self,
        source_lines: Optional[List[str]],
        args,
        global_vars: Dict[str, Any],
        location: FunctionLocation,
        options: CodegenOptions,
        visited_return_types: Optional[ReturnTypesDict] = None,
        is_kernel: bool = True,
        use_new_module = True
    ):
        super().__init__()
        self.src = source_lines
        self.ir_function: Optional[ir.FuncOp] = None
        self.args = args
        self.scope = NameScope(global_vars)
        self.location = location
        if use_new_module:
            self.context = ir.Context()
            afir.register_dialect(self.context)
            self.loc = ir.Location.file(self.location.filename, self.location.line_offset, 0,context = self.context)
            self.module = ir.Module.create(loc = self.loc)
            self.insertPoint = None
        self.state = VisitorState()
        self.options = options
        self.is_kernel = is_kernel
        if visited_return_types:
            self.state.visited_return_types = visited_return_types

    @staticmethod
    def get_binary_method_name(op_class: Type[ast.operator]) -> str:
        names: Dict[Type[ast.operator], str] = {
            ast.Add: '__add__',
            ast.Sub: '__sub__',
            ast.Mult: '__mul__',
            ast.Div: '__truediv__',
            ast.FloorDiv: '__floordiv__',
            ast.Mod: '__mod__',
            ast.Pow: '__pow__',
            ast.LShift: '__lshift__',
            ast.RShift: '__rshift__',
            ast.BitAnd: '__and__',
            ast.BitOr: '__or__',
            ast.BitXor: '__xor__',
        }
        name = names.get(op_class)
        if name:
            return name
        raise NotImplementedError(f"Method for {op_class.__class__.__name__} is not implemented")

    @staticmethod
    def get_bool_method_name(op_class: Type[ast.boolop]) -> str:
        names: Dict[Type[ast.boolop], str] = {
            ast.And: 'logical_and',
            ast.Or: 'logical_or',
        }
        name = names.get(op_class)
        if name:
            return name
        raise NotImplementedError(f"Method for {op_class.__name__} is not implemented")

    @staticmethod
    def get_unary_method_name(op_class: Type[ast.unaryop]) -> str:
        names: Dict[Type[ast.unaryop], str] = {
            ast.USub: '__neg__',
            ast.UAdd: '__pos__',
            ast.Not: '__not__',
            ast.Invert: '__invert__',
        }
        name = names.get(op_class)
        if name:
            return name
        raise NotImplementedError(f"Method for {op_class.__name__} is not implemented")

    @staticmethod
    def get_compare_method_name(op_class: Type[ast.cmpop]) -> str:
        names: Dict[Type[ast.cmpop], str] = {
            ast.Eq: '__eq__',
            ast.NotEq: '__ne__',
            ast.Gt: '__gt__',
            ast.GtE: '__ge__',
            ast.Lt: '__lt__',
            ast.LtE: '__le__',
        }
        name = names.get(op_class)
        if name:
            return name
        raise NotImplementedError(f"Method for {op_class.__name__} is not implemented")

    @staticmethod
    def has_builder_support(value) -> bool:
        return isinstance(value, (ir.Value))

    def raise_unsupported(self, node: ast.AST, message: Optional[str] = None) -> NoReturn:
        error = UnsupportedSyntaxError(node, self.src, message)
        raise error from None

    def apply_binary_method(self, method_name, lhs, rhs) -> Any:
        reverse_method_name = re.sub(r"__(.*)__", r"__r\1__", method_name)
        if not self.has_builder_support(lhs) and self.has_builder_support(rhs):
            return getattr(rhs, reverse_method_name)(lhs)
        result = getattr(lhs, method_name)(rhs)
        if result is NotImplemented:
            result = getattr(rhs, reverse_method_name)(lhs)
        return result

    def get_call_args(self, func: Callable, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return bound_args.arguments

    def call_jit_function(self, fn: Function, args: Tuple[Any], kwargs: Dict[str, Any]) -> Optional[Any]:
        base_fn = fn.fn
        call_args = self.get_call_args(base_fn, *args, **kwargs)
        arg_values: Dict[str, IRValue] = {}
        for name, value in call_args.items():
            arg_values[name] = value
        arg_types = {name: value.type for name, value in arg_values.items()}
        fn_name = fn.node.name
        ret_types = []
        if fn_name in ir.SymbolTable(self.module.operation):
            ret_types = self.state.visited_return_types[fn_name]
        else:
            visitor = FunctionVisitor(fn.src, arg_types, base_fn.__globals__, fn.location, self.options,
                                      self.state.visited_return_types, is_kernel=False, use_new_module=False)
            visitor.context = self.context
            visitor.loc = self.loc
            visitor.module = self.module
            visitor.insertPoint = ir.InsertionPoint(self.module.body)
            visitor.visit(fn.node)
            ret_types = visitor.state.return_types
            self.state.visited_return_types[fn_name] = ret_types
        ir_operands = list(arg_values.values())
        op = func.CallOp([ret_type for ret_type in ret_types], fn.node.name, ir_operands)
        if len(ret_types) == 0:
            return None
        if len(ret_types) == 1:
            return op.result  
        return [op.results[i] for i in range(len(op.results))]

    def compute_inout(self, stmts: List[ast.stmt], ind_var: Optional[Tuple[str, ir.Type]] = None,
                      make_args: bool = False) -> BlockInOut:
        with self.visit_region() as (outer_scope, _):
            executeRegion = scf.ExecuteRegionOp([])
            block = ir.Block.create_at_start(executeRegion.region)
            if ind_var is not None:
                name, ir_type = ind_var
                arg = block.add_argument(ir_type, self.loc)
                self.scope.save(name, arg)
            with ir.InsertionPoint(block):
                self.visit_statements(stmts)
            init_handles = [outer_scope.lookup(name) for name in self.scope.redefined]
            if make_args:
                for handle in init_handles:
                    if isinstance(handle, (ir.Operation, ir.OpView)):
                        handle = handle.result
                    arg = block.add_argument(handle.type, self.loc)
                    unExceptOp = list()
                    for user in handle.uses:
                        if user.owner.operation.block != block:
                            unExceptOp.append(user.owner.operation)
                    handle.replace_all_uses_except(arg, unExceptOp)
            yield_handles = [self.scope.lookup(name) for name in self.scope.redefined]
            return BlockInOut(
                block=executeRegion,
                init_handles=dict(zip(self.scope.redefined, init_handles)),
                yield_handles=dict(zip(self.scope.redefined, yield_handles)),
            )

    def dereference_name(self, name: str) -> Optional[Any]:
        return self.scope.lookup(name)

    @contextmanager
    def nest_scope(self) -> Generator[None, Any, None]:
        outer_scope = self.scope
        self.scope = outer_scope.inherit()
        try:
            yield
        finally:
            self.scope = outer_scope

    def generic_visit(self, node: ast.AST) -> NoReturn:
        self.raise_unsupported(node, f"{node.__class__.__name__} syntax is not supported in JIT function")

    def visit(self, node: Optional[ast.AST]) -> Optional[Any]:
        if node is None:
            return None
        if self.state.discard_everything:
            return None
        if self.state.inside_function and isinstance(node, ast.FunctionDef):
            self.raise_unsupported(node, "Nested functions are not supported")
        if not self.state.inside_function and not isinstance(node, ast.FunctionDef):
            raise RuntimeError(f"JIT compilation is applicable to functions only, got {node.__class__.__name__} node")
        if hasattr(node, "lineno") and hasattr(node, "col_offset"):
            self.loc = ir.Location.file(self.location.filename, self.location.line_offset + node.lineno,
                                                    node.col_offset, context = self.context)
        try:
            return super().visit(node)
        except CodegenError:
            raise
        except Exception as e:
            if self.options.capture_exceptions:
                raise CodegenError(node, self.src, f"{e.__class__.__name__}: {e}") from e
            raise

    def visit_arguments(self, node: ast.arguments) -> Tuple[List[str], str]:
        if node.defaults or node.kw_defaults:
            self.raise_unsupported(node, "Default values for function arguments are not supported")
        if node.posonlyargs:
            self.raise_unsupported(node, "Positional-only arguments are not supported")
        if node.kwonlyargs:
            self.raise_unsupported(node, "Keyword-only arguments are not supported")
        arg_names = [str(self.visit(arg)) for arg in node.args]
        kwarg_name = str(self.visit(node.kwarg))
        return arg_names, kwarg_name

    def visit_arg(self, node: ast.arg) -> str:
        return node.arg

    @contextmanager
    def visit_region(self) -> Generator[Tuple[NameScope, ir.InsertionPoint], Any, None]:
        outer_scope = self.scope
        self.scope = outer_scope.inherit()
        return_allowed = self.state.return_allowed
        self.state.return_allowed = False
        try:
            yield outer_scope.inherit(), self.insertPoint
        finally:
            self.scope = outer_scope
            self.state.return_allowed = return_allowed

    def visit_statements(self, stmts: List[ast.stmt]) -> None:
        for stmt in stmts:
            self.visit(stmt)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        return self.visit_Assign(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        test = self.visit(node.test)
        try:
            cf.AssertOp(test, self.visit(node.msg))
        except TypeError as e:
            self.raise_unsupported(
                node,
                f"An assertion turned out to test a runtime value {test!r}, only compile-time values are supported")

    def visit_Assign(self, node: ast.Assign) -> None:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        if len(targets) != 1:
            self.raise_unsupported(node, "Assignment operator must have exactly one target")
        lhs = targets[0]
        rhs = self.visit(node.value)
        if isinstance(lhs, ast.Subscript) and isinstance(lhs.ctx, ast.Store):
            base = self.visit(lhs.value)
            subscript = self.visit(lhs.slice)
            base.__setitem__(subscript, rhs)
            return
        if isinstance(lhs, ast.Attribute) and isinstance(lhs.ctx, ast.Store):
            base = self.visit(lhs.value)
            if setter := getattr(base, "__setattrjit__", None):
                setter(lhs.attr, rhs)
            else:
                setattr(base, lhs.attr, rhs)
            return
        lhs_names = []
        if isinstance(lhs, ast.Name):
            lhs_names.append(self.visit(lhs))
        elif isinstance(lhs, ast.Tuple) and all(isinstance(elt, ast.Name) for elt in lhs.elts):
            lhs_names.extend(self.visit(lhs))
        else:
            self.raise_unsupported(node, "Assignment target must be name or tuple of names")
        rhs_values = []
        if isinstance(rhs, Iterable) and len(lhs_names) != 1:
            if len(rhs) != len(lhs_names):
                self.raise_unsupported(
                    node, "Assignment operator must have equal number of names and values, "
                    f"got {len(lhs_names)} names and {len(rhs)} values")
            rhs_values.extend(rhs)
        else:
            if isinstance(rhs, (ir.Operation, ir.OpView)):
                rhs = rhs.result
            rhs_values.append(rhs)
        if len(lhs_names) != len(rhs_values):
            raise RuntimeError("Assignment operator must have equal number of names and values")
        for name, value in zip(lhs_names, rhs_values):
            self.scope.save(name, value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        lhs = node.target
        if isinstance(lhs, ast.Name):
            lhs = ast.Name(lhs.id, ctx=ast.Load())
        elif isinstance(lhs, ast.Subscript):
            lhs = ast.Subscript(lhs.value, lhs.slice, ctx=ast.Load())
        else:
            self.raise_unsupported(node, f"{lhs.__class__.__name__} is not supported as left operand of AugAssign")
        rhs = ast.BinOp(lhs, node.op, node.value)
        assign = ast.Assign(targets=[node.target], value=rhs)
        self.visit(assign)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        lhs = self.visit(node.value)
        attr = str(node.attr)
        try:
            value = getattr(lhs, attr)
            return value
        except Exception as e:
            raise e

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        method_name = self.get_binary_method_name(type(node.op))
        if isinstance(lhs, int):
            nodeType = ir.IntegerType.get_signless(32)
            lhs = arith.ConstantOp(nodeType, lhs)
        if isinstance(rhs, int):
            nodeType = ir.IntegerType.get_signless(32)
            rhs = arith.ConstantOp(nodeType, rhs)
        if isinstance(rhs, float):
            nodeType = ir.F32Type.get()
            rhs = arith.ConstantOp(nodeType, rhs)
        # if isinstance(lhs, (ir.Operation, ir.OpView)):
        #     lhs = lhs.result
        # if isinstance(rhs, (ir.Operation, ir.OpView)):
        #     rhs = rhs.result
        return self.apply_binary_method(method_name, PlainValue(lhs, lhs.type), PlainValue(rhs, rhs.type))

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if len(node.values) != 2:
            self.raise_unsupported(node, "Chained boolean operators are not supported, group pairs with parentheses")
        lhs = self.visit(node.values[0])
        rhs = self.visit(node.values[1])
        method_name = self.get_bool_method_name(type(node.op))
        return self.apply_binary_method(method_name, PlainValue(lhs, lhs.type), PlainValue(rhs, rhs.type))

    def visit_Call(self, node: ast.Call) -> Optional[Any]:
        fn = self.visit(node.func)
        if not callable(fn):
            self.raise_unsupported(node, f"{fn.__class__.__name__} instance is not callable")
        args = [self.visit(arg) for arg in node.args]
        kwargs = dict(self.visit(keyword) for keyword in node.keywords)
        if isinstance(fn, Function):
            return self.call_jit_function(fn, args, kwargs)
        return fn(*args, **kwargs)

    def visit_Compare(self, node: ast.Compare) -> Any:
        if len(node.comparators) != 1 or len(node.ops) != 1:
            self.raise_unsupported(node, "Only simple comparison is supported (one operation, one comparator)")
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Is):
            return lhs is rhs
        if isinstance(op, ast.IsNot):
            return lhs is not rhs
        # if isinstance(lhs, (ir.Operation, ir.OpView)):
        #     lhs = lhs.result
        # if isinstance(rhs, (ir.Operation, ir.OpView)):
        #     rhs = rhs.result
        method_name = self.get_compare_method_name(type(node.ops[0]))
        return self.apply_binary_method(method_name, PlainValue(lhs, lhs.type), PlainValue(rhs, rhs.type))

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Expr(self, node: ast.Expr) -> Optional[Any]:
        return self.visit(node.value)
    
    def visit_FormattedValue(self, node: ast.FormattedValue) -> str:
        value = self.visit(node.value)
        template = "{"
        if node.conversion >= 0:
            template += f"!{chr(node.conversion)}"
        spec = self.visit(node.format_spec)
        if spec:
            template += f":{spec}"
        template += "}"
        return template.format(value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.state.inside_function = True
        arg_types = self.args.values()
        input_ir_types = [ConvertPythonTypeToMLIRType(arg_type, self.context, self.loc) for arg_type in arg_types]
        with self.context, self.loc:
            with ir.InsertionPoint(self.module.body):
                self.ir_function = func.FuncOp(node.name, ir.FunctionType.get(input_ir_types, []))
                arg_names = list(self.args.keys())
                with ir.InsertionPoint(self.ir_function.add_entry_block()): 
                    for i, name in enumerate(arg_names):
                        value = self.ir_function.arguments[i]
                        self.scope.save(name, value)
                    self.visit_statements(node.body)
                    if self.ir_function.body.blocks[0].operations[-1].OPERATION_NAME != "func.return":
                        func.ReturnOp([])
                    if self.state.return_types:
                        self.ir_function.function_type = ir.TypeAttr.get(ir.FunctionType.get(input_ir_types, self.state.return_types))
        self.state.inside_function = False
        self.state.discard_everything = False

    def parse_iterator(self, node: ast.For) -> tuple:
        func = self.visit(node.iter.func)
        args = [self.visit(arg) for arg in node.iter.args]
        kwargs = dict(self.visit(keyword) for keyword in node.iter.keywords)
        return func, args, kwargs


    def visit_For(self, node: ast.For) -> None:
        if len(node.orelse) != 0:
            self.raise_unsupported(node, "else statement is not allowed after for-loop")
        target = self.visit(node.target)
        if not isinstance(target, str):
            self.raise_unsupported(node, f"For-loop target must be an identifier, got {target.__class__.__name__}")
        try:
            func, args, kwargs = self.parse_iterator(node)
        except Exception as e:
            self.raise_unsupported(
                node,
                "Only for-loops with range are supported",
            )
        if func is range:
            totalNum = len(args) + len(kwargs)
            if totalNum < 1 or totalNum > 3:
                raise ValueError(f"range expects from 1 to 3 arguments, got {totalNum}")
            start = 0
            stop = 0
            step = 1
            if len(args) == 1:
                stop = args[0]
            elif len(args) >= 2:
                start = args[0]
                stop = args[1]
            if len(args) == 3:
                step = args[2]
            elif len(kwargs) == 1:
                stop = kwargs['step']
            iter_args = start, stop, step
        else:
            self.raise_unsupported(
                node,
                "Only for-loops with range are supported",
            )
        def getHandleValue(s):
            if isinstance(s, int):
                return materialize_ir_value(s, KnownTypes.int_).handle
            else:
                if not hasattr(s, "type"):
                    return s.operation.result
                return s
        start, stop, step = map(getHandleValue, iter_args)
        if start.type != stop.type or start.type != step.type:
            self.raise_unsupported(node, "Loop bounds must have the same DataType")
        yields = {}
        with self.visit_region():
            block_inout = self.compute_inout(node.body, (target, start.type), make_args=True)
            init_Args = list(block_inout.init_handles.values())
            op = scf.ForOp(start, stop, step,init_Args)
            self.scope.save(target, op.induction_variable)
            with ir.InsertionPoint(op.body):
                self.visit_statements(node.body)
                init_Args = list(block_inout.init_handles.values())
                for i, handle in enumerate(init_Args):
                    unExceptOp = list()
                    for user in handle.uses:
                        if user.owner.operation.block != op.body:
                            unExceptOp.append(user.owner.operation)
                    handle.replace_all_uses_except(op.inner_iter_args[i], unExceptOp)
                scf.YieldOp([self.scope.lookup(name) for name in block_inout.yield_handles.keys()])
            for i, name in enumerate(block_inout.yield_handles.keys()):
                yields[name] = op.results[i]
            block_inout.block.erase()
        for name, value in yields.items():
            self.scope.save(name, value)

    def visit_If(self, node: ast.If) -> None:
        cond = self.visit(node.test)
        yields = {}
        then_inout = self.compute_inout(node.body)
        else_inout = self.compute_inout(node.orelse)
        def merge_sorted(dict1: Dict[str, T], dict2: Dict[str, T]) -> Dict[str, T]:
            dicts = dict1 | dict2
            return {key: dicts[key] for key in sorted(dicts.keys())}
        yield_handles = merge_sorted(then_inout.yield_handles, else_inout.yield_handles)
        ret_types = [value.type for value in yield_handles.values()]
        op = scf.IfOp(cond, ret_types, hasElse=True)

        with self.visit_region() as (outer_scope, _):
            with ir.InsertionPoint(op.then_block):
                self.visit_statements(node.body)
                scf.YieldOp([self.scope.lookup(name) for name in yield_handles.keys()])
        with self.visit_region() as (outer_scope, _):
            with ir.InsertionPoint(op.else_block):
                self.visit_statements(node.orelse)
                scf.YieldOp([self.scope.lookup(name) for name in yield_handles.keys()])
        for i, name in enumerate(yield_handles.keys()):
            yields[name] = op.results_[i]
        for name, value in yields.items():
            self.scope.save(name, value)
        then_inout.block.operation.erase()
        else_inout.block.operation.erase()


    def visit_Return(self, node: ast.Return) -> None:
        if not self.state.return_allowed:
            self.raise_unsupported(node, "Return statement is not allowed in nested blocks")
        value = self.visit(node.value)
        self.state.discard_everything = True
        if value is None:
            return
        # if self.is_kernel:
        #     self.raise_unsupported(node, "JIT kernel function cannot return objects")
        values = []
        if isinstance(value, Iterable):
            values.extend(value)
        else:
            values.append(value)
        func.ReturnOp(values)
        self.state.return_types = [value.type if isinstance(value, ir.Value) else value.result.type
                                    for value in values ]


    def visit_IfExp(self, node: ast.IfExp) -> Any:
        cond = self.visit(node.test)
        with self.visit_region():
            then_value = self.visit(node.body)
            else_value = self.visit(node.orelse)
            then_value_type = then_value.type if hasattr(then_value, "type") else then_value.result.type
            else_value_type = else_value.type if hasattr(else_value, "type") else else_value.result.type
            if then_value_type != else_value_type:
                self.raise_unsupported(
                    node,
                    f"Conditional operator has inconsistent result types: {then_value.dtype} / {else_value.dtype}")
            ret_type = then_value_type
            op = scf.IfOp(cond, [ret_type], hasElse=True)
            if isinstance(then_value, ir.OpResult):
                then_value.owner.operation.erase()
            else:
                then_value.operation.erase()
            if isinstance(else_value, ir.OpResult):
                else_value.owner.operation.erase()
            else:
                else_value.operation.erase()
            with self.visit_region():
                with ir.InsertionPoint(op.then_block):
                    scf.YieldOp([self.visit(node.body)])
            with self.visit_region():
                with ir.InsertionPoint(op.else_block):
                    scf.YieldOp([self.visit(node.orelse)])
            return op.result

    def visit_While(self, node: ast.While) -> None:
        if len(node.orelse) != 0:
            self.raise_unsupported(node, "else statement is not allowed after while-loop")
        after_inout = self.compute_inout(node.body, make_args=True)
        ret_types = [handle.type # if isinstance(handle, ir.Value) else handle.result.type 
                     for handle in after_inout.init_handles.values()]
        op = scf.WhileOp(ret_types, list(after_inout.init_handles.values()))
        with self.visit_region():
            before_block = ir.Block.create_at_start(op.before, ret_types)
            with ir.InsertionPoint(before_block):
                cond = self.visit(node.test)
                scf.ConditionOp(cond, before_block.arguments)
        with self.visit_region():
            after_block = ir.Block.create_at_start(op.after, ret_types)
            with ir.InsertionPoint(after_block):
                self.visit_statements(node.body)
                init_Args = list(after_inout.init_handles.values())
                for i, handle in enumerate(init_Args):
                    unExceptOp = list()
                    # if isinstance(handle, (ir.Operation, ir.OpView)):
                    #     handle = handle.result
                    for user in handle.uses:
                        if user.owner.operation.block != after_block:
                            unExceptOp.append(user.owner.operation)
                    handle.replace_all_uses_except(after_block.arguments[i], unExceptOp)
                scf.YieldOp([self.scope.lookup(name) for name in after_inout.yield_handles.keys()])
        after_inout.block.erase()
        for i, name in enumerate(after_inout.yield_handles.keys()):
            self.scope.save(name, op.results[i])


    def visit_JoinedStr(self, node: ast.JoinedStr) -> str:
        values = (self.visit(value) for value in node.values)
        return "".join(values)

    def visit_keyword(self, node: ast.keyword) -> Tuple[str, Any]:
        return node.arg, self.visit(node.value)

    def visit_List(self, node: ast.List) -> List[Optional[Any]]:
        return [self.visit(elt) for elt in node.elts]

    def visit_Name(self, node: ast.Name) -> Union[str, Optional[Any]]:
        if isinstance(node.ctx, ast.Store):
            return node.id
        value = self.dereference_name(node.id)
        return value

    def visit_Pass(self, node: ast.Pass) -> None:
        pass
    def visit_Slice(self, node: ast.Slice) -> slice:
        return slice(self.visit(node.lower), self.visit(node.upper), self.visit(node.step))

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        if not isinstance(node.ctx, ast.Load):
            self.raise_unsupported(node, "Subscript operation is not allowed (must be Load context or assignment)")
        value = self.visit(node.value)
        
        if isinstance(value, list):
            if isinstance(node.slice, ast.Constant):
                slices = node.slice.value
        else: 
            slices = self.visit(node.slice)
        return value.__getitem__(slices)

    def visit_Tuple(self, node: ast.Tuple) -> Tuple[Optional[Any], ...]:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        method_name = self.get_unary_method_name(type(node.op))
        return getattr(operand, method_name)()

    def visit_With(self, node: ast.With) -> None:
        if len(node.items) != 1:
            self.raise_unsupported(node, "Only one item in with-statement is supported")
        item = node.items[0]
        context = self.visit(item.context_expr)
        with self.nest_scope():
            entered = context.__enter__()
            if isinstance(item.optional_vars, ast.Name):
                self.scope.save(self.visit(item.optional_vars), entered)
            self.visit_statements(node.body)
            context.__exit__(None, None, None)
