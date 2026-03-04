import pickle
import inspect
from dataclasses import fields
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, overload, Union
from .function import Function, P, T

from .function_visitor import CodegenOptions, FunctionVisitor


import mlir.ir as ir

import mlir.dialects.afir as afir

class JITFunction(Function[P, T]):
    codegen: Type[FunctionVisitor] = FunctionVisitor
    # compiler: Type[Compiler] = Compiler
    # launcher: Type[Launcher] = Launcher

    def __init__(self, fn: Callable[P, T], **options):
        super().__init__(fn)
        # clashed_args = self.get_clashed_args(fn)
        # if clashed_args:
        #     raise RuntimeError("The following argument names conflict with JIT configuration options: " +
        #                        ", ".join(clashed_args))
        # unknown_options = set(options).difference(self.get_config_keywords())
        # if unknown_options:
        #     raise RuntimeError("The following option names are unknown: " + ", ".join(unknown_options))
        self.default_options: Dict[str, Any] = options
        # self.launch_options = LaunchOptions()
        self.kernel_cache = {}

    def __getitem__(self, user_launch_options: Union[int, tuple]) -> Callable:
        # try:
        #     if isinstance(user_launch_options, int):
        #         self.launch_options = LaunchOptions(core_num=user_launch_options)
        #     else:
        #         self.launch_options = LaunchOptions(*user_launch_options)
        # except Exception as e:
        #     raise TypeError("Parse user launch options failed") from e
        # else:
            return self._run

    @staticmethod
    def get_clashed_args(fn: Callable) -> List[str]:
        keywords = set(__class__.get_config_keywords())
        signature = inspect.signature(fn)
        keywords.intersection_update(signature.parameters)
        return list(keywords)

    # @staticmethod
    # def get_arg_type(value: Any) -> BaseArgType:
    #     if isinstance(value, bool):
    #         return PlainArgType(KT.bool_)
    #     if isinstance(value, int):
    #         return PlainArgType(KT.int_)
    #     if isinstance(value, float):
    #         return PlainArgType(KT.float_)
    #     if isinstance(value, np.ndarray):
    #         return PointerArgType(DataType(str(np.dtype(value.dtype))))
    #     if isinstance(value, np.generic):
    #         return PlainArgType(DataType(str(value.dtype)))
    #     if isinstance(value, Struct):
    #         return StructArgType(type(value))
    #     try:
    #         import torch
    #         if isinstance(value, torch.Tensor):
    #             dtype_str = str(value.dtype)
    #             if dtype_str.startswith("torch."):
    #                 dtype_str = dtype_str[6:]
    #             return PointerArgType(DataType(dtype_str))
    #     except ModuleNotFoundError:
    #         pass
    #     # Mock arguments
    #     if isinstance(value, MockTensor):
    #         return PointerArgType(value.dtype)
    #     if isinstance(value, MockValue):
    #         return PlainArgType(value.dtype)
    #     raise TypeError(f"Argument type in JIT function is not supported: {value.__class__.__name__}")

    @staticmethod
    def extract_kwargs(dataclass: Type, base_kwargs: Dict[str, Any]) -> Any:
        keywords = map(lambda f: f.name, fields(dataclass))
        kwargs = {}
        for keyword in keywords:
            if keyword in base_kwargs:
                kwargs[keyword] = base_kwargs[keyword]
                del base_kwargs[keyword]
        return dataclass(**kwargs)

    # @staticmethod
    # def create_context() -> ir.Context:
    #     context = ir.Context()
    #     context.disable_multithreading()
    #     ir.load_dialects(context)
    #     return context

    # @staticmethod
    # def get_arg_dtype(arg_type: BaseArgType) -> str:
    #     if isinstance(arg_type, PointerArgType):
    #         return str(arg_type.dtype)
    #     if isinstance(arg_type, PlainArgType):
    #         return str(arg_type.dtype)
    #     if isinstance(arg_type, IRArgType):
    #         return str(arg_type.py_type)
    #     if isinstance(arg_type, StructArgType):
    #         return str(arg_type.py_type)
    #     raise NotImplementedError(f"Not implemented for {arg_type.__class__.__name__}")

    @classmethod
    def get_config_keywords(cls) -> List[str]:
        attr = "_config_keywords"
        cached_keywords = getattr(cls, attr, None)
        if cached_keywords is not None:
            return cached_keywords
        keywords = []
        # for dataclass in CodegenOptions, CompileOptions, LaunchOptions:
        #     keywords.extend(f.name for f in fields(dataclass))
        setattr(cls, attr, keywords)
        return keywords

    def _gen_cache_factors(self, arg_types, constexprs, codegen_options, compile_options) -> str:
        cache_factors = []
        separator = ";"
        key = separator.join([f"{name}={val}" for name, val in vars(codegen_options).items()])
        cache_factors.append(key)
        key = separator.join([f"{name}={val}" for name, val in vars(compile_options).items()])
        cache_factors.append(key)
        key = separator.join([f"{name}={repr(val)}" for name, val in constexprs.items()])
        cache_factors.append(key)
        key = separator.join(
            [f"{name}={val.__class__.__name__}:{self.get_arg_dtype(val)}" for name, val in arg_types.items()])
        cache_factors.append(key)
        key = f"fn_name={self.fn_name}"
        cache_factors.append(key)

        separator = "__"
        cache_factors = separator.join(cache_factors)
        return cache_factors

    # def _cache_kernel(self, runtime_args, constexprs, codegen_options, compile_options) -> CompiledKernel:
    #     arg_types = {name: self.get_arg_type(value) for name, value in runtime_args.items()}
    #     cache_factors = self._gen_cache_factors(arg_types, constexprs, codegen_options, compile_options)
    #     mem_cache_key = get_mem_cache_key(cache_factors)
    #     kernel = self.kernel_cache.get(mem_cache_key, None)
    #     if not compile_options.always_compile and kernel is not None:
    #         return kernel
    #     file_cache_key = get_file_cache_key(self.cache_key, cache_factors)
    #     file_cache_manager = get_cache_manager(file_cache_key)
    #     kernel_file_name = self.fn.__name__ + ".o"
    #     cached_kernel_file = file_cache_manager.get_file(kernel_file_name)
    #     if not compile_options.always_compile and cached_kernel_file is not None:
    #         dst = Path(cached_kernel_file)
    #         with open(dst, 'rb') as file:
    #             kernel = pickle.load(file)
    #     else:
    #         mod = self._run_codegen(Specialization(arg_types, constexprs), codegen_options)
    #         kernel = self._run_compiler(mod, compile_options)
    #         kernel_bin = pickle.dumps(kernel)
    #     if not compile_options.always_compile and cached_kernel_file is None:
    #         file_cache_manager.put(kernel_bin, kernel_file_name)
    #         self.kernel_cache[mem_cache_key] = kernel
    #     return kernel

    def _run_codegen(self, spec, options: CodegenOptions) -> ir.Module:
        self.context = ir.Context()
        if options.ir_multithreading:
            self.context.enable_multithreading(False)
        try:
            visitor = self.codegen(self.src, spec, self.fn.__globals__, self.location, options, is_kernel=True, use_new_module=True)
            visitor.visit(self.node)
            # visitor.module.dump()
            # print(list(visitor.module.body.operations[0].body.blocks[0].operations))
            return visitor.module
        finally:
            pass
            # global_builder.teardown()

    # def _run_compiler(self, mod: ir.ModuleOp, options: CompileOptions) -> CompiledKernel:
    #     compiler = self.compiler(options)
    #     return compiler.run(mod, self.fn.__name__)

    # def _run_launcher(self, kernel: CompiledKernel, options: LaunchOptions, runtime_args: Tuple[Any]) -> None:
    #     launcher = self.launcher(options)
    #     launcher.run(kernel, self.fn.__name__, runtime_args)

    def _run(self, *args: P.args, **kwargs: P.kwargs) -> None:
        kwargs = self.default_options | kwargs
        codegen_options = self.extract_kwargs(CodegenOptions, kwargs)
        compile_options = None #self.extract_kwargs(CompileOptions, kwargs)
        call_args = inspect.signature(self.fn).bind(*args, **kwargs).arguments
        # annotations = inspect.get_annotations(self.fn)
        return self._run_codegen(call_args, codegen_options)
        # binary = self._cache_kernel(runtime_args, constexprs, codegen_options, compile_options)
        # self._run_launcher(binary, self.launch_options, tuple(runtime_args.values()))


@overload
def jit(fn: Callable[P, T]) -> JITFunction[P, T]:
    """Instantiate JIT function without additional options"""
    ...


@overload
def jit(**options) -> Callable[[Callable[P, T]], JITFunction[P, T]]:
    """Instantiate JIT function and provide options (see CodegenOptions, CompileOptions, LaunchOptions)"""
    """for exmaple, use @jit(debug = True) to pass True to CompileOptions.debug"""
    ...


def jit(fn: Optional[Callable[P, T]] = None, **options):

    def decorator(fn: Callable[P, T]) -> JITFunction[P, T]:
        return JITFunction(fn, **options)

    if fn is None:
        return decorator
    return decorator(fn)
