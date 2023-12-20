# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implementing Algebraic Reasoning (AR)."""

from collections import defaultdict  # pylint: disable=g-importing-member
from fractions import Fraction as frac  # pylint: disable=g-importing-member
from typing import Any, Generator

import geometry as gm
import numpy as np
import problem as pr
from scipy import optimize


class InfQuotientError(Exception):
  pass


def _gcd(x: int, y: int) -> int:
  while y:
    x, y = y, x % y
  return x


def simplify(n: int, d: int) -> tuple[int, int]:
  g = _gcd(n, d)
  return (n // g, d // g)


# maximum denominator for a fraction.
MAX_DENOMINATOR = 1000000

# tolerance for fraction approximation
TOL = 1e-15


def get_quotient(v: float) -> tuple[int, int]:
  n = v
  d = 1
  while abs(n - round(n)) > TOL:
    d += 1
    n += v
    if d > MAX_DENOMINATOR:
      e = InfQuotientError(v)
      raise e

  n = int(round(n))
  return simplify(n, d)


def fix_v(v: float) -> float:
  n, d = get_quotient(v)
  return n / d


def fix(e: dict[str, float]) -> dict[str, float]:
  return {k: fix_v(v) for k, v in e.items()}


def frac_string(f: frac) -> str:
  n, d = get_quotient(f)
  return f'{n}/{d}'


def hashed(e: dict[str, float]) -> tuple[tuple[str, float], ...]:
  return tuple(sorted(list(e.items())))


def is_zero(e: dict[str, float]) -> bool:
  return len(strip(e)) == 0  # pylint: disable=g-explicit-length-test


def strip(e: dict[str, float]) -> dict[str, float]:
  return {v: c for v, c in e.items() if c != 0}


def plus(e1: dict[str, float], e2: dict[str, float]) -> dict[str, float]:
  e = dict(e1)
  for v, c in e2.items():
    if v in e:
      e[v] += c
    else:
      e[v] = c
  return strip(e)


def plus_all(*es: list[dict[str, float]]) -> dict[str, float]:
  result = {}
  for e in es:
    result = plus(result, e)
  return result


def mult(e: dict[str, float], m: float) -> dict[str, float]:
  return {v: m * c for v, c in e.items()}


def minus(e1: dict[str, float], e2: dict[str, float]) -> dict[str, float]:
  return plus(e1, mult(e2, -1))


def div(e1: dict[str, float], e2: dict[str, float]) -> float:
  """Divide e1 by e2."""
  e1 = strip(e1)
  e2 = strip(e2)
  if set(e1.keys()) != set(e2.keys()):
    return None

  n, d = None, None

  for v, c1 in e1.items():
    c2 = e2[v]  # we want c1/c2 = n/d => c1*d=c2*n
    if n is not None and c1 * d != c2 * n:
      return None
    n, d = c1, c2
  return frac(n) / frac(d)


def recon(e: dict[str, float], const: str) -> tuple[str, dict[str, float]]:
  """Reconcile one variable in the expression e=0, given const."""
  e = strip(e)
  if len(e) == 0:  # pylint: disable=g-explicit-length-test
    return None

  v0 = None
  for v in e:
    if v != const:
      v0 = v
      break
  if v0 is None:
    return v0

  c0 = e.pop(v0)
  return v0, {v: -c / c0 for v, c in e.items()}


def replace(
    e: dict[str, float], v0: str, e0: dict[str, float]
) -> dict[str, float]:
  if v0 not in e:
    return e
  e = dict(e)
  m = e.pop(v0)
  return plus(e, mult(e0, m))


def comb2(elems: list[Any]) -> Generator[tuple[Any, Any], None, None]:
  if len(elems) < 1:
    return
  for i, e1 in enumerate(elems[:-1]):
    for e2 in elems[i + 1 :]:
      yield e1, e2


def perm2(elems: list[Any]) -> Generator[tuple[Any, Any], None, None]:
  for e1, e2 in comb2(elems):
    yield e1, e2
    yield e2, e1


def chain2(elems: list[Any]) -> Generator[tuple[Any, Any], None, None]:
  if len(elems) < 2:
    return
  for i, e1 in enumerate(elems[:-1]):
    yield e1, elems[i + 1]


def update_groups(
    groups1: list[Any], groups2: list[Any]
) -> tuple[list[Any], list[tuple[Any, Any]], list[list[Any]]]:
  """Update groups of equivalent elements.

  Given groups1 = [set1, set2, set3, ..]
  where all elems within each set_i is defined to be "equivalent" to each other.
  (but not across the sets)

  Incoming groups2 = [set1, set2, ...] similar to set1 - it is the
  additional equivalent information on elements in groups1.

  Return the new updated groups1 and the set of links
  that make it that way.

  Example:
    groups1 = [{1, 2}, {3, 4, 5}, {6, 7}]
    groups2 = [{2, 3, 8}, {9, 10, 11}]

  => new groups1 and links:
    groups1 = [{1, 2, 3, 4, 5, 8}, {6, 7}, {9, 10, 11}]
    links = (2, 3), (3, 8), (9, 10), (10, 11)

  Explain: since groups2 says 2 and 3 are equivalent (with {2, 3, 8}),
  then {1, 2} and {3, 4, 5} in groups1 will be merged,
  because 2 and 3 each belong to those 2 groups.
  Additionally 8 also belong to this same group.
  {3, 4, 5} is left alone, while {9, 10, 11} is a completely new set.

  The links to make this all happens is:
  (2, 3): to merge {1, 2} and {3, 4, 5}
  (3, 8): to link 8 into the merged({1, 2, 3, 4, 5})
  (9, 10) and (10, 11): to make the new group {9, 10, 11}

  Args:
    groups1: a list of sets.
    groups2: a list of sets.

  Returns:
    groups1, links, history: result of the update.
  """
  history = []
  links = []
  for g2 in groups2:
    joins = [None] * len(groups1)  # mark which one in groups1 is merged
    merged_g1 = set()  # merge them into this.
    old = None  # any elem in g2 that belong to any set in groups1 (old)
    new = []  # all elem in g2 that is new

    for e in g2:
      found = False
      for i, g1 in enumerate(groups1):
        if e not in g1:
          continue

        found = True
        if joins[i]:
          continue

        joins[i] = True
        merged_g1.update(g1)

        if old is not None:
          links.append((old, e))  # link to make merging happen.
        old = e

      if not found:  # e is new!
        new.append(e)

    # now chain elems in new together.
    if old is not None and new:
      links.append((old, new[0]))
      merged_g1.update(new)

    links += chain2(new)

    new_groups1 = []
    if merged_g1:  # put the merged_g1 in first
      new_groups1.append(merged_g1)

    # put the remaining (unjoined) groups in
    new_groups1 += [g1 for j, g1 in zip(joins, groups1) if not j]

    if old is None and new:
      new_groups1 += [set(new)]

    groups1 = new_groups1
    history.append(groups1)

  return groups1, links, history


class Table:
  """The coefficient matrix."""

  def __init__(self, const: str = '1'):
    self.const = const
    self.v2e = {}
    self.add_free(const)  # the table {var: expression}

    # to cache what is already derived/inputted
    self.eqs = set()
    self.groups = []  # groups of equal pairs.

    # for why (linprog)
    self.c = []
    self.v2i = {}  # v -> index of row in A.
    self.deps = []  # equal number of columns.
    self.A = np.zeros([0, 0])  # pylint: disable=invalid-name
    self.do_why = True

  def add_free(self, v: str) -> None:
    self.v2e[v] = {v: frac(1)}

  def replace(self, v0: str, e0: dict[str, float]) -> None:
    for v, e in list(self.v2e.items()):
      self.v2e[v] = replace(e, v0, e0)

  def add_expr(self, vc: list[tuple[str, float]]) -> bool:
    """Add a new equality, represented by the list of tuples vc=[(v, c), ..]."""
    result = {}
    free = []

    for v, c in vc:
      c = frac(c)
      if v in self.v2e:
        result = plus(result, mult(self.v2e[v], c))
      else:
        free += [(v, c)]

    if free == []:  # pylint: disable=g-explicit-bool-comparison
      if is_zero(self.modulo(result)):
        return False
      result = recon(result, self.const)
      if result is None:
        return False
      v, e = result
      self.replace(v, e)

    elif len(free) == 1:
      v, m = free[0]
      self.v2e[v] = mult(result, frac(-1, m))

    else:
      dependent_v = None
      for v, m in free:
        if dependent_v is None and v != self.const:
          dependent_v = (v, m)
          continue

        self.add_free(v)
        result = plus(result, {v: m})

      v, m = dependent_v
      self.v2e[v] = mult(result, frac(-1, m))

    return True

  def register(self, vc: list[tuple[str, float]], dep: pr.Dependency) -> None:
    """Register a new equality vc=[(v, c), ..] with traceback dependency dep."""
    result = plus_all(*[{v: c} for v, c in vc])
    if is_zero(result):
      return

    vs, _ = zip(*vc)
    for v in vs:
      if v not in self.v2i:
        self.v2i[v] = len(self.v2i)

    (m, n), l = self.A.shape, len(self.v2i)
    if l > m:
      self.A = np.concatenate([self.A, np.zeros([l - m, n])], 0)

    new_column = np.zeros([len(self.v2i), 2])  # N, 2
    for v, c in vc:
      new_column[self.v2i[v], 0] += float(c)
      new_column[self.v2i[v], 1] -= float(c)

    self.A = np.concatenate([self.A, new_column], 1)
    self.c += [1.0, -1.0]
    self.deps += [dep]

  def register2(
      self, a: str, b: str, m: float, n: float, dep: pr.Dependency
  ) -> None:
    self.register([(a, m), (b, -n)], dep)

  def register3(self, a: str, b: str, f: float, dep: pr.Dependency) -> None:
    self.register([(a, 1), (b, -1), (self.const, -f)], dep)

  def register4(
      self, a: str, b: str, c: str, d: str, dep: pr.Dependency
  ) -> None:
    self.register([(a, 1), (b, -1), (c, -1), (d, 1)], dep)

  def why(self, e: dict[str, float]) -> list[Any]:
    """AR traceback == MILP."""
    if not self.do_why:
      return []
    # why expr == 0?
    # Solve min(c^Tx) s.t. A_eq * x = b_eq, x >= 0
    e = strip(e)
    if not e:
      return []

    b_eq = [0] * len(self.v2i)
    for v, c in e.items():
      b_eq[self.v2i[v]] += float(c)

    try:
      x = optimize.linprog(c=self.c, A_eq=self.A, b_eq=b_eq, method='highs')[
          'x'
      ]
    except:  # pylint: disable=bare-except
      x = optimize.linprog(
          c=self.c,
          A_eq=self.A,
          b_eq=b_eq,
      )['x']

    deps = []
    for i, dep in enumerate(self.deps):
      if x[2 * i] > 1e-12 or x[2 * i + 1] > 1e-12:
        if dep not in deps:
          deps.append(dep)
    return deps

  def record_eq(self, v1: str, v2: str, v3: str, v4: str) -> None:
    self.eqs.add((v1, v2, v3, v4))
    self.eqs.add((v2, v1, v4, v3))
    self.eqs.add((v3, v4, v1, v2))
    self.eqs.add((v4, v3, v2, v1))

  def check_record_eq(self, v1: str, v2: str, v3: str, v4: str) -> bool:
    if (v1, v2, v3, v4) in self.eqs:
      return True
    if (v2, v1, v4, v3) in self.eqs:
      return True
    if (v3, v4, v1, v2) in self.eqs:
      return True
    if (v4, v3, v2, v1) in self.eqs:
      return True
    return False

  def add_eq2(
      self, a: str, b: str, m: float, n: float, dep: pr.Dependency
  ) -> None:
    # a/b = m/n
    if not self.add_expr([(a, m), (b, -n)]):
      return []
    self.register2(a, b, m, n, dep)

  def add_eq3(self, a: str, b: str, f: float, dep: pr.Dependency) -> None:
    # a - b = f * constant
    self.eqs.add((a, b, frac(f)))
    self.eqs.add((b, a, frac(1 - f)))

    if not self.add_expr([(a, 1), (b, -1), (self.const, -f)]):
      return []

    self.register3(a, b, f, dep)

  def add_eq4(self, a: str, b: str, c: str, d: str, dep: pr.Dependency) -> None:
    # a - b = c - d
    self.record_eq(a, b, c, d)
    self.record_eq(a, c, b, d)

    expr = list(minus({a: 1, b: -1}, {c: 1, d: -1}).items())

    if not self.add_expr(expr):
      return []

    self.register4(a, b, c, d, dep)
    self.groups, _, _ = update_groups(
        self.groups, [{(a, b), (c, d)}, {(b, a), (d, c)}]
    )

  def pairs(self) -> Generator[list[tuple[str, str]], None, None]:
    for v1, v2 in perm2(list(self.v2e.keys())):  # pylint: disable=g-builtin-op
      if v1 == self.const or v2 == self.const:
        continue
      yield v1, v2

  def modulo(self, e: dict[str, float]) -> dict[str, float]:
    return strip(e)

  def get_all_eqs(
      self,
  ) -> dict[tuple[tuple[str, float], ...], list[tuple[str, str]]]:
    h2pairs = defaultdict(list)
    for v1, v2 in self.pairs():
      e1, e2 = self.v2e[v1], self.v2e[v2]
      e12 = minus(e1, e2)
      h12 = hashed(self.modulo(e12))
      h2pairs[h12].append((v1, v2))
    return h2pairs

  def get_all_eqs_and_why(
      self, return_quads: bool = True
  ) -> Generator[Any, None, None]:
    """Check all 4/3/2-permutations for new equalities."""
    groups = []

    for h, vv in self.get_all_eqs().items():
      if h == ():  # pylint: disable=g-explicit-bool-comparison
        for v1, v2 in vv:
          if (v1, v2) in self.eqs or (v2, v1) in self.eqs:
            continue
          self.eqs.add((v1, v2))
          # why v1 - v2 = e12 ?  (note modulo(e12) == 0)
          why_dict = minus({v1: 1, v2: -1}, minus(self.v2e[v1], self.v2e[v2]))
          yield v1, v2, self.why(why_dict)
        continue

      if len(h) == 1 and h[0][0] == self.const:
        for v1, v2 in vv:
          frac = h[0][1]  # pylint: disable=redefined-outer-name
          if (v1, v2, frac) in self.eqs:
            continue
          self.eqs.add((v1, v2, frac))
          # why v1 - v2 = e12 ?  (note modulo(e12) == 0)
          why_dict = minus({v1: 1, v2: -1}, minus(self.v2e[v1], self.v2e[v2]))
          value = simplify(frac.numerator, frac.denominator)
          yield v1, v2, value, self.why(why_dict)
        continue

      groups.append(vv)

    if not return_quads:
      return

    self.groups, links, _ = update_groups(self.groups, groups)
    for (v1, v2), (v3, v4) in links:
      if self.check_record_eq(v1, v2, v3, v4):
        continue
      e12 = minus(self.v2e[v1], self.v2e[v2])
      e34 = minus(self.v2e[v3], self.v2e[v4])

      why_dict = minus(  # why (v1-v2)-(v3-v4)=e12-e34?
          minus({v1: 1, v2: -1}, {v3: 1, v4: -1}), minus(e12, e34)
      )
      self.record_eq(v1, v2, v3, v4)
      yield v1, v2, v3, v4, self.why(why_dict)


class GeometricTable(Table):
  """Abstract class representing the coefficient matrix (table) A."""

  def __init__(self, name: str = ''):
    super().__init__(name)
    self.v2obj = {}

  def get_name(self, objs: list[Any]) -> list[str]:
    self.v2obj.update({o.name: o for o in objs})
    return [o.name for o in objs]

  def map2obj(self, names: list[str]) -> list[Any]:
    return [self.v2obj[n] for n in names]

  def get_all_eqs_and_why(
      self, return_quads: bool
  ) -> Generator[Any, None, None]:
    for out in super().get_all_eqs_and_why(return_quads):
      if len(out) == 3:
        x, y, why = out
        x, y = self.map2obj([x, y])
        yield x, y, why
      if len(out) == 4:
        x, y, f, why = out
        x, y = self.map2obj([x, y])
        yield x, y, f, why
      if len(out) == 5:
        a, b, x, y, why = out
        a, b, x, y = self.map2obj([a, b, x, y])
        yield a, b, x, y, why


class RatioTable(GeometricTable):
  """Coefficient matrix A for log(distance)."""

  def __init__(self, name: str = ''):
    name = name or '1'
    super().__init__(name)
    self.one = self.const

  def add_eq(self, l1: gm.Length, l2: gm.Length, dep: pr.Dependency) -> None:
    l1, l2 = self.get_name([l1, l2])
    return super().add_eq3(l1, l2, 0.0, dep)

  def add_const_ratio(
      self, l1: gm.Length, l2: gm.Length, m: float, n: float, dep: pr.Dependency
  ) -> None:
    l1, l2 = self.get_name([l1, l2])
    return super().add_eq2(l1, l2, m, n, dep)

  def add_eqratio(
      self,
      l1: gm.Length,
      l2: gm.Length,
      l3: gm.Length,
      l4: gm.Length,
      dep: pr.Dependency,
  ) -> None:
    l1, l2, l3, l4 = self.get_name([l1, l2, l3, l4])
    return self.add_eq4(l1, l2, l3, l4, dep)

  def get_all_eqs_and_why(self) -> Generator[Any, None, None]:
    return super().get_all_eqs_and_why(True)


class AngleTable(GeometricTable):
  """Coefficient matrix A for slope(direction)."""

  def __init__(self, name: str = ''):
    name = name or 'pi'
    super().__init__(name)
    self.pi = self.const

  def modulo(self, e: dict[str, float]) -> dict[str, float]:
    e = strip(e)
    if self.pi not in e:
      return super().modulo(e)

    e[self.pi] = e[self.pi] % 1
    return strip(e)

  def add_para(
      self, d1: gm.Direction, d2: gm.Direction, dep: pr.Dependency
  ) -> None:
    return self.add_const_angle(d1, d2, 0, dep)

  def add_const_angle(
      self, d1: gm.Direction, d2: gm.Direction, ang: float, dep: pr.Dependency
  ) -> None:
    if ang and d2._obj.num > d1._obj.num:  # pylint: disable=protected-access
      d1, d2 = d2, d1
      ang = 180 - ang

    d1, d2 = self.get_name([d1, d2])

    num, den = simplify(ang, 180)
    ang = frac(int(num), int(den))
    return super().add_eq3(d1, d2, ang, dep)

  def add_eqangle(
      self,
      d1: gm.Direction,
      d2: gm.Direction,
      d3: gm.Direction,
      d4: gm.Direction,
      dep: pr.Dependency,
  ) -> None:
    """Add the inequality d1-d2=d3-d4."""
    # Use string as variables.
    l1, l2, l3, l4 = [d._obj.num for d in [d1, d2, d3, d4]]  # pylint: disable=protected-access
    d1, d2, d3, d4 = self.get_name([d1, d2, d3, d4])
    ang1 = {d1: 1, d2: -1}
    ang2 = {d3: 1, d4: -1}

    if l2 > l1:
      ang1 = plus({self.pi: 1}, ang1)
    if l4 > l3:
      ang2 = plus({self.pi: 1}, ang2)

    ang12 = minus(ang1, ang2)
    self.record_eq(d1, d2, d3, d4)
    self.record_eq(d1, d3, d2, d4)

    expr = list(ang12.items())
    if not self.add_expr(expr):
      return []

    self.register(expr, dep)

  def get_all_eqs_and_why(self) -> Generator[Any, None, None]:
    return super().get_all_eqs_and_why(True)


class DistanceTable(GeometricTable):
  """Coefficient matrix A for position(point, line)."""

  def __init__(self, name: str = ''):
    name = name or '1:1'
    self.merged = {}
    self.ratios = set()
    super().__init__(name)

  def pairs(self) -> Generator[tuple[str, str], None, None]:
    l2vs = defaultdict(list)
    for v in list(self.v2e.keys()):  # pylint: disable=g-builtin-op
      if v == self.const:
        continue
      l, p = v.split(':')
      l2vs[l].append(p)

    for l, ps in l2vs.items():
      for p1, p2 in perm2(ps):
        yield l + ':' + p1, l + ':' + p2

  def name(self, l: gm.Line, p: gm.Point) -> str:
    v = l.name + ':' + p.name
    self.v2obj[v] = (l, p)
    return v

  def map2obj(self, names: list[str]) -> list[gm.Point]:
    return [self.v2obj[n][1] for n in names]

  def add_cong(
      self,
      l12: gm.Line,
      l34: gm.Line,
      p1: gm.Point,
      p2: gm.Point,
      p3: gm.Point,
      p4: gm.Point,
      dep: pr.Dependency,
  ) -> None:
    """Add that distance between p1 and p2 (on l12) == p3 and p4 (on l34)."""
    if p2.num > p1.num:
      p1, p2 = p2, p1
    if p4.num > p3.num:
      p3, p4 = p4, p3

    p1 = self.name(l12, p1)
    p2 = self.name(l12, p2)
    p3 = self.name(l34, p3)
    p4 = self.name(l34, p4)
    return super().add_eq4(p1, p2, p3, p4, dep)

  def get_all_eqs_and_why(self) -> Generator[Any, None, None]:
    for x in super().get_all_eqs_and_why(True):
      yield x

    # Now we figure out all the const ratios.
    h2pairs = defaultdict(list)
    for v1, v2 in self.pairs():
      if (v1, v2) in self.merged:
        continue
      e1, e2 = self.v2e[v1], self.v2e[v2]
      e12 = minus(e1, e2)
      h12 = hashed(e12)
      h2pairs[h12].append((v1, v2, e12))

    for (_, vves1), (_, vves2) in perm2(list(h2pairs.items())):
      v1, v2, e12 = vves1[0]
      for v1_, v2_, _ in vves1[1:]:
        self.merged[(v1_, v2_)] = (v1, v2)

      v3, v4, e34 = vves2[0]
      for v3_, v4_, _ in vves2[1:]:
        self.merged[(v3_, v4_)] = (v3, v4)

      if (v1, v2, v3, v4) in self.ratios:
        continue

      d12 = div(e12, e34)
      if d12 is None or d12 > 1 or d12 < 0:
        continue

      self.ratios.add((v1, v2, v3, v4))
      self.ratios.add((v2, v1, v4, v3))

      n, d = d12.numerator, d12.denominator

      # (v1 - v2) * d = (v3 - v4) * n
      why_dict = minus(
          minus({v1: d, v2: -d}, {v3: n, v4: -n}),
          minus(mult(e12, d), mult(e34, n)),  # there is no modulo, so this is 0
      )

      v1, v2, v3, v4 = self.map2obj([v1, v2, v3, v4])
      yield v1, v2, v3, v4, abs(n), abs(d), self.why(why_dict)
