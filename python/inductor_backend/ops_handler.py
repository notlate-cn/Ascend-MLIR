from torch._inductor.ops_handler import OpsHandler


class MLIROpsHandler(OpsHandler):
    """
    拦截 inner_fn 的标量操作，构建 linalg.generic body。

    inner_fn 调用约定：
    - Pointwise: inner_fn(index)          — index 是 sympy 表达式序列
    - Reduction: inner_fn(index, rindex)  — 多一个 reduction 索引参数
    """

    def __init__(self):
        self.operations = []
        self.ssa_counter = 0

    def _new_ssa(self):
        op_id = f"%{self.ssa_counter}"
        self.ssa_counter += 1
        return op_id

    def constant(self, value, dtype):
        op_id = self._new_ssa()
        self.operations.append({'type': 'constant', 'id': op_id, 'value': value, 'dtype': dtype})
        return op_id

    def load(self, name, index):
        op_id = self._new_ssa()
        self.operations.append({'type': 'load', 'id': op_id, 'name': name, 'index': index})
        return op_id

    def store(self, name, index, value, mode=None):
        self.operations.append({'type': 'store', 'name': name, 'index': index, 'value': value})

    def store_reduction(self, name, index, value):
        self.operations.append({'type': 'store_reduction', 'name': name, 'index': index, 'value': value})

    def add(self, a, b):
        op_id = self._new_ssa()
        self.operations.append({'type': 'add', 'id': op_id, 'lhs': a, 'rhs': b})
        return op_id

    def mul(self, a, b):
        op_id = self._new_ssa()
        self.operations.append({'type': 'mul', 'id': op_id, 'lhs': a, 'rhs': b})
        return op_id

    def reduction(self, dtype, src_dtype, reduction_type, value):
        op_id = self._new_ssa()
        self.operations.append({
            'type': 'reduction', 'id': op_id,
            'dtype': dtype, 'src_dtype': src_dtype,
            'reduction_type': reduction_type,
            'value': value
        })
        return op_id

    def index_expr(self, expr, dtype):
        op_id = self._new_ssa()
        self.operations.append({'type': 'index_expr', 'id': op_id, 'expr': str(expr), 'dtype': dtype})
        return op_id

    def indirect_indexing(self, x, size, check=True, wrap_neg=True):
        import sympy
        return sympy.Symbol(f"indirect_{self.ssa_counter}")

    def masked(self, mask, body, other):
        return body()

    def to_dtype(self, value, dtype):
        return value

    def where(self, condition, input, other):
        op_id = self._new_ssa()
        self.operations.append({'type': 'where', 'id': op_id, 'cond': condition, 'a': input, 'b': other})
        return op_id