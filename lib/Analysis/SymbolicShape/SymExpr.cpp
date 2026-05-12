//===- SymExpr.cpp - Tiny symbolic-shape expression type --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Analysis/SymbolicShape/SymExpr.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cctype>

using namespace mlir::afir::symshape;

namespace mlir::afir::symshape::detail {
struct SymExprNode {
  SymExpr::Kind kind;
  SymId sym = 0;        // valid iff kind == Sym
  int64_t cst = 0;      // valid iff kind == Const
  SymExpr lhs, rhs;     // valid for binary kinds
};
} // namespace mlir::afir::symshape::detail

//===----------------------------------------------------------------------===//
// Construction + folding
//===----------------------------------------------------------------------===//

SymExpr SymExpr::make(Kind k, SymId s, int64_t c, SymExpr l, SymExpr r) {
  auto n = std::make_shared<detail::SymExprNode>();
  n->kind = k;
  n->sym = s;
  n->cst = c;
  n->lhs = std::move(l);
  n->rhs = std::move(r);
  return SymExpr(std::move(n));
}

SymExpr::Kind SymExpr::getKind() const { return node->kind; }
SymId SymExpr::getSym() const { return node->sym; }
int64_t SymExpr::getCstValue() const { return node->cst; }
SymExpr SymExpr::getLhs() const { return node->lhs; }
SymExpr SymExpr::getRhs() const { return node->rhs; }

SymExpr SymExpr::sym(SymId id) { return make(Kind::Sym, id, 0, {}, {}); }

SymExpr SymExpr::constant(int64_t v) { return make(Kind::Const, 0, v, {}, {}); }

static std::optional<int64_t> cst(const SymExpr &e) { return e.getConst(); }

SymExpr SymExpr::add(SymExpr a, SymExpr b) {
  if (auto x = cst(a), y = cst(b); x && y)
    return constant(*x + *y);
  if (auto x = cst(a); x && *x == 0)
    return b;
  if (auto y = cst(b); y && *y == 0)
    return a;
  return make(Kind::Add, 0, 0, std::move(a), std::move(b));
}

SymExpr SymExpr::sub(SymExpr a, SymExpr b) {
  if (auto x = cst(a), y = cst(b); x && y)
    return constant(*x - *y);
  if (auto y = cst(b); y && *y == 0)
    return a;
  if (a == b)
    return constant(0);
  return make(Kind::Sub, 0, 0, std::move(a), std::move(b));
}

SymExpr SymExpr::mul(SymExpr a, SymExpr b) {
  if (auto x = cst(a), y = cst(b); x && y)
    return constant(*x * *y);
  if (auto x = cst(a); x && *x == 0)
    return constant(0);
  if (auto y = cst(b); y && *y == 0)
    return constant(0);
  if (auto x = cst(a); x && *x == 1)
    return b;
  if (auto y = cst(b); y && *y == 1)
    return a;
  return make(Kind::Mul, 0, 0, std::move(a), std::move(b));
}

SymExpr SymExpr::ceilDiv(SymExpr a, SymExpr b) {
  if (auto x = cst(a), y = cst(b); x && y) {
    assert(*y > 0 && "ceilDiv by non-positive constant");
    return constant((*x + *y - 1) / *y);
  }
  if (auto x = cst(a); x && *x == 0)
    return constant(0);
  if (auto y = cst(b); y && *y == 1)
    return a;
  return make(Kind::CeilDiv, 0, 0, std::move(a), std::move(b));
}

SymExpr SymExpr::mod(SymExpr a, SymExpr b) {
  if (auto x = cst(a), y = cst(b); x && y) {
    assert(*y > 0 && "mod by non-positive constant");
    return constant(*x % *y);
  }
  if (auto x = cst(a); x && *x == 0)
    return constant(0);
  if (auto y = cst(b); y && *y == 1)
    return constant(0);
  return make(Kind::Mod, 0, 0, std::move(a), std::move(b));
}

SymExpr SymExpr::min(SymExpr a, SymExpr b) {
  if (auto x = cst(a), y = cst(b); x && y)
    return constant(*x < *y ? *x : *y);
  if (a == b)
    return a;
  return make(Kind::Min, 0, 0, std::move(a), std::move(b));
}

SymExpr SymExpr::max(SymExpr a, SymExpr b) {
  if (auto x = cst(a), y = cst(b); x && y)
    return constant(*x > *y ? *x : *y);
  if (a == b)
    return a;
  return make(Kind::Max, 0, 0, std::move(a), std::move(b));
}

std::optional<int64_t> SymExpr::getConst() const {
  if (node->kind == Kind::Const)
    return node->cst;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// eval / emitC / walkSymbols
//===----------------------------------------------------------------------===//

int64_t SymExpr::eval(const llvm::DenseMap<SymId, int64_t> &env) const {
  switch (node->kind) {
  case Kind::Sym: {
    auto it = env.find(node->sym);
    assert(it != env.end() && "SymExpr::eval: symbol missing from env");
    return it->second;
  }
  case Kind::Const:
    return node->cst;
  case Kind::Add:
    return node->lhs.eval(env) + node->rhs.eval(env);
  case Kind::Sub:
    return node->lhs.eval(env) - node->rhs.eval(env);
  case Kind::Mul:
    return node->lhs.eval(env) * node->rhs.eval(env);
  case Kind::CeilDiv: {
    int64_t a = node->lhs.eval(env), b = node->rhs.eval(env);
    assert(b > 0 && "SymExpr::eval: ceildiv by non-positive value");
    return (a + b - 1) / b;
  }
  case Kind::Mod: {
    int64_t a = node->lhs.eval(env), b = node->rhs.eval(env);
    assert(b > 0 && "SymExpr::eval: mod by non-positive value");
    return a % b;
  }
  case Kind::Min: {
    int64_t a = node->lhs.eval(env), b = node->rhs.eval(env);
    return a < b ? a : b;
  }
  case Kind::Max: {
    int64_t a = node->lhs.eval(env), b = node->rhs.eval(env);
    return a > b ? a : b;
  }
  }
  llvm_unreachable("unhandled SymExpr::Kind in eval");
}

std::string SymExpr::emitC(llvm::function_ref<std::string(SymId)> name) const {
  switch (node->kind) {
  case Kind::Sym:
    return name(node->sym);
  case Kind::Const:
    return std::to_string(node->cst);
  case Kind::Add:
    return "(" + node->lhs.emitC(name) + " + " + node->rhs.emitC(name) + ")";
  case Kind::Sub:
    return "(" + node->lhs.emitC(name) + " - " + node->rhs.emitC(name) + ")";
  case Kind::Mul:
    return "(" + node->lhs.emitC(name) + " * " + node->rhs.emitC(name) + ")";
  case Kind::CeilDiv: {
    std::string a = node->lhs.emitC(name), b = node->rhs.emitC(name);
    return "((" + a + " + " + b + " - 1) / " + b + ")";
  }
  case Kind::Mod:
    return "(" + node->lhs.emitC(name) + " % " + node->rhs.emitC(name) + ")";
  case Kind::Min: {
    std::string a = node->lhs.emitC(name), b = node->rhs.emitC(name);
    return "((" + a + ") < (" + b + ") ? (" + a + ") : (" + b + "))";
  }
  case Kind::Max: {
    std::string a = node->lhs.emitC(name), b = node->rhs.emitC(name);
    return "((" + a + ") > (" + b + ") ? (" + a + ") : (" + b + "))";
  }
  }
  llvm_unreachable("unhandled SymExpr::Kind in emitC");
}

void SymExpr::walkSymbols(llvm::function_ref<void(SymId)> cb) const {
  switch (node->kind) {
  case Kind::Sym:
    cb(node->sym);
    return;
  case Kind::Const:
    return;
  default:
    node->lhs.walkSymbols(cb);
    node->rhs.walkSymbols(cb);
    return;
  }
}

SymExpr SymExpr::mapSymbols(llvm::function_ref<SymId(SymId)> f) const {
  switch (node->kind) {
  case Kind::Sym:
    return sym(f(node->sym));
  case Kind::Const:
    return *this;
  case Kind::Add:
    return add(node->lhs.mapSymbols(f), node->rhs.mapSymbols(f));
  case Kind::Sub:
    return sub(node->lhs.mapSymbols(f), node->rhs.mapSymbols(f));
  case Kind::Mul:
    return mul(node->lhs.mapSymbols(f), node->rhs.mapSymbols(f));
  case Kind::CeilDiv:
    return ceilDiv(node->lhs.mapSymbols(f), node->rhs.mapSymbols(f));
  case Kind::Mod:
    return mod(node->lhs.mapSymbols(f), node->rhs.mapSymbols(f));
  case Kind::Min:
    return min(node->lhs.mapSymbols(f), node->rhs.mapSymbols(f));
  case Kind::Max:
    return max(node->lhs.mapSymbols(f), node->rhs.mapSymbols(f));
  }
  llvm_unreachable("unhandled SymExpr::Kind in mapSymbols");
}

//===----------------------------------------------------------------------===//
// print
//===----------------------------------------------------------------------===//

void SymExpr::print(llvm::raw_ostream &os) const {
  auto bin = [&](const char *op) {
    os << '(';
    node->lhs.print(os);
    os << op;
    node->rhs.print(os);
    os << ')';
  };
  auto call = [&](const char *fn) {
    os << fn << '(';
    node->lhs.print(os);
    os << ',';
    node->rhs.print(os);
    os << ')';
  };
  switch (node->kind) {
  case Kind::Sym:
    os << 's' << node->sym;
    return;
  case Kind::Const:
    os << node->cst;
    return;
  case Kind::Add:
    return bin("+");
  case Kind::Sub:
    return bin("-");
  case Kind::Mul:
    return bin("*");
  case Kind::CeilDiv:
    return call("ceildiv");
  case Kind::Mod:
    return call("mod");
  case Kind::Min:
    return call("min");
  case Kind::Max:
    return call("max");
  }
  llvm_unreachable("unhandled SymExpr::Kind in print");
}

std::string SymExpr::str() const {
  std::string s;
  llvm::raw_string_ostream os(s);
  print(os);
  return os.str();
}

//===----------------------------------------------------------------------===//
// equality / hash
//===----------------------------------------------------------------------===//

bool SymExpr::operator==(const SymExpr &other) const {
  if (node == other.node)
    return true;
  if (!node || !other.node)
    return false;
  if (node->kind != other.node->kind)
    return false;
  switch (node->kind) {
  case Kind::Sym:
    return node->sym == other.node->sym;
  case Kind::Const:
    return node->cst == other.node->cst;
  default:
    return node->lhs == other.node->lhs && node->rhs == other.node->rhs;
  }
}

llvm::hash_code SymExpr::hashValue() const {
  switch (node->kind) {
  case Kind::Sym:
    return llvm::hash_combine(node->kind, node->sym);
  case Kind::Const:
    return llvm::hash_combine(node->kind, node->cst);
  default:
    return llvm::hash_combine(node->kind, node->lhs.hashValue(),
                              node->rhs.hashValue());
  }
}

//===----------------------------------------------------------------------===//
// parsing
//===----------------------------------------------------------------------===//

namespace {

class Parser {
public:
  explicit Parser(llvm::StringRef s) : s(s), pos(0) {}

  std::optional<SymExpr> parseToplevel() {
    skipWs();
    auto e = parseExpr();
    if (!e)
      return std::nullopt;
    skipWs();
    if (pos != s.size())
      return std::nullopt; // trailing garbage
    return e;
  }

  // Parses one expr; on success leaves pos just past it (no trailing-ws skip).
  std::optional<SymExpr> parseExpr() {
    skipWs();
    if (pos >= s.size())
      return std::nullopt;
    char c = s[pos];

    if (c == '(')
      return parseParen();
    if (c == '-' || isdigit(static_cast<unsigned char>(c)))
      return parseNumber();
    if (consumeKeyword("ceildiv"))
      return parseCall(SymExpr::Kind::CeilDiv);
    if (consumeKeyword("mod"))
      return parseCall(SymExpr::Kind::Mod);
    if (consumeKeyword("min"))
      return parseCall(SymExpr::Kind::Min);
    if (consumeKeyword("max"))
      return parseCall(SymExpr::Kind::Max);
    if (c == 's')
      return parseSym();
    return std::nullopt;
  }

private:
  llvm::StringRef s;
  size_t pos;

  void skipWs() {
    while (pos < s.size() && isspace(static_cast<unsigned char>(s[pos])))
      ++pos;
  }

  bool consumeKeyword(llvm::StringRef kw) {
    // Must be followed by '(' (after optional ws) to be unambiguous; we don't
    // skip ws between the keyword and '(' for simplicity (printer never emits
    // any), but allow it.
    if (s.substr(pos).starts_with(kw)) {
      size_t after = pos + kw.size();
      // peek next non-ws
      size_t p = after;
      while (p < s.size() && isspace(static_cast<unsigned char>(s[p])))
        ++p;
      if (p < s.size() && s[p] == '(') {
        pos = after;
        return true;
      }
    }
    return false;
  }

  std::optional<SymExpr> parseNumber() {
    size_t start = pos;
    if (pos < s.size() && s[pos] == '-')
      ++pos;
    size_t digitsStart = pos;
    while (pos < s.size() && isdigit(static_cast<unsigned char>(s[pos])))
      ++pos;
    if (pos == digitsStart)
      return std::nullopt; // "-" with no digits
    int64_t v;
    if (s.substr(start, pos - start).getAsInteger(10, v))
      return std::nullopt;
    return SymExpr::constant(v);
  }

  std::optional<SymExpr> parseSym() {
    assert(pos < s.size() && s[pos] == 's');
    ++pos;
    size_t digitsStart = pos;
    while (pos < s.size() && isdigit(static_cast<unsigned char>(s[pos])))
      ++pos;
    if (pos == digitsStart)
      return std::nullopt;
    uint32_t id;
    if (s.substr(digitsStart, pos - digitsStart).getAsInteger(10, id))
      return std::nullopt;
    return SymExpr::sym(id);
  }

  std::optional<SymExpr> parseParen() {
    assert(pos < s.size() && s[pos] == '(');
    ++pos; // '('
    auto a = parseExpr();
    if (!a)
      return std::nullopt;
    skipWs();
    if (pos >= s.size())
      return std::nullopt;
    char op = s[pos++];
    auto b = parseExpr();
    if (!b)
      return std::nullopt;
    skipWs();
    if (pos >= s.size() || s[pos] != ')')
      return std::nullopt;
    ++pos; // ')'
    switch (op) {
    case '+':
      return SymExpr::add(*a, *b);
    case '-':
      return SymExpr::sub(*a, *b);
    case '*':
      return SymExpr::mul(*a, *b);
    default:
      return std::nullopt;
    }
  }

  std::optional<SymExpr> parseCall(SymExpr::Kind k) {
    skipWs();
    if (pos >= s.size() || s[pos] != '(')
      return std::nullopt;
    ++pos; // '('
    auto a = parseExpr();
    if (!a)
      return std::nullopt;
    skipWs();
    if (pos >= s.size() || s[pos] != ',')
      return std::nullopt;
    ++pos; // ','
    auto b = parseExpr();
    if (!b)
      return std::nullopt;
    skipWs();
    if (pos >= s.size() || s[pos] != ')')
      return std::nullopt;
    ++pos; // ')'
    switch (k) {
    case SymExpr::Kind::CeilDiv:
      return SymExpr::ceilDiv(*a, *b);
    case SymExpr::Kind::Mod:
      return SymExpr::mod(*a, *b);
    case SymExpr::Kind::Min:
      return SymExpr::min(*a, *b);
    case SymExpr::Kind::Max:
      return SymExpr::max(*a, *b);
    default:
      return std::nullopt;
    }
  }
};

} // namespace

std::optional<SymExpr> mlir::afir::symshape::parseSymExpr(llvm::StringRef text) {
  return Parser(text).parseToplevel();
}

std::optional<llvm::SmallVector<SymExpr, 4>>
mlir::afir::symshape::parseSymExprList(llvm::StringRef text) {
  llvm::SmallVector<SymExpr, 4> out;
  text = text.trim();
  if (text.empty())
    return out;
  // Split on top-level commas (commas inside ceildiv(...,...) etc. are nested).
  int depth = 0;
  size_t start = 0;
  for (size_t i = 0; i <= text.size(); ++i) {
    char c = i < text.size() ? text[i] : ',';
    if (c == '(')
      ++depth;
    else if (c == ')')
      --depth;
    if (i == text.size() || (c == ',' && depth == 0)) {
      auto e = parseSymExpr(text.substr(start, i - start));
      if (!e)
        return std::nullopt;
      out.push_back(*e);
      start = i + 1;
    }
  }
  if (depth != 0)
    return std::nullopt;
  return out;
}

void mlir::afir::symshape::printSymExprList(llvm::ArrayRef<SymExpr> exprs,
                                            llvm::raw_ostream &os) {
  for (auto [i, e] : llvm::enumerate(exprs)) {
    if (i)
      os << ',';
    e.print(os);
  }
}

std::string mlir::afir::symshape::symExprListStr(llvm::ArrayRef<SymExpr> exprs) {
  std::string s;
  llvm::raw_string_ostream os(s);
  printSymExprList(exprs, os);
  return os.str();
}
