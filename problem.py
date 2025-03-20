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

"""Implements objects to represent problems, theorems, proofs, traceback."""

from __future__ import annotations

from collections import defaultdict  # pylint: disable=g-importing-member
from typing import Any

import geometry as gm
import pretty as pt


# pylint: disable=protected-access
# pylint: disable=unused-variable
# pylint: disable=unused-argument
# pylint: disable=unused-assignment


def reshape(l: list[Any], n: int = 1) -> list[list[Any]]:
  assert len(l) % n == 0
  columns = [[] for i in range(n)]
  for i, x in enumerate(l):
    columns[i % n].append(x)
  return zip(*columns)


def isint(x: str) -> bool:
  try:
    int(x)
    return True
  except:  # pylint: disable=bare-except
    return False


class Construction:
  """One predicate."""

  @classmethod
  def from_txt(cls, data: str) -> Construction:
    data = data.split(' ')
    return Construction(data[0], data[1:])

  def __init__(self, name: str, args: list[str]):
    self.name = name
    self.args = args

  def translate(self, mapping: dict[str, str]) -> Construction:
    args = [a if isint(a) else mapping[a] for a in self.args]
    return Construction(self.name, args)

  def txt(self) -> str:
    return ' '.join([self.name] + list(self.args))


class Clause:
  """One construction (>= 1 predicate)."""

  @classmethod
  def from_txt(cls, data: str) -> Clause:
    if data == ' =':
      return Clause([], [])
    points, constructions = data.split(' = ')
    return Clause(
        points.split(' '),
        [Construction.from_txt(c) for c in constructions.split(', ')],
    )

  def __init__(self, points: list[str], constructions: list[Construction]):
    self.points = []
    self.nums = []

    for p in points:
      num = None
      if isinstance(p, str) and '@' in p:
        p, num = p.split('@')
        x, y = num.split('_')
        num = float(x), float(y)
      self.points.append(p)
      self.nums.append(num)

    self.constructions = constructions

  def translate(self, mapping: dict[str, str]) -> Clause:
    points0 = []
    for p in self.points:
      pcount = len(mapping) + 1
      name = chr(96 + pcount)
      if name > 'z':  # pcount = 26 -> name = 'z'
        name = chr(97 + (pcount - 1) % 26) + str((pcount - 1) // 26)

      p0 = mapping.get(p, name)
      mapping[p] = p0
      points0.append(p0)
    return Clause(points0, [c.translate(mapping) for c in self.constructions])

  def add(self, name: str, args: list[str]) -> None:
    self.constructions.append(Construction(name, args))

  def txt(self) -> str:
    return (
        ' '.join(self.points)
        + ' = '
        + ', '.join(c.txt() for c in self.constructions)
    )


def _gcd(x: int, y: int) -> int:
  while y:
    x, y = y, x % y
  return x


def simplify(n: int, d: int) -> tuple[int, int]:
  g = _gcd(n, d)
  return (n // g, d // g)


def compare_fn(dep: Dependency) -> tuple[Dependency, str]:
  return (dep, pt.pretty(dep))


def sort_deps(deps: list[Dependency]) -> list[Dependency]:
  return sorted(deps, key=compare_fn)


class Problem:
  """Describe one problem to solve."""

  @classmethod
  def from_txt_file(
      cls, fname: str, to_dict: bool = False, translate: bool = True
  ):
    """Load a problem from a text file."""
    with open(fname, 'r') as f:
      lines = f.read().split('\n')

    lines = [l for l in lines if l]
    data = [
        cls.from_txt(url + '\n' + problem, translate)
        for (url, problem) in reshape(lines, 2)
    ]
    if to_dict:
      return cls.to_dict(data)
    return data

  @classmethod
  def from_txt(cls, data: str, translate: bool = True) -> Problem:
    """Load a problem from a str object."""
    url = ''
    if '\n' in data:
      url, data = data.split('\n')

    if ' ? ' in data:
      clauses, goal = data.split(' ? ')
      goal = Construction.from_txt(goal)
    else:
      clauses, goal = data, None

    clauses = clauses.split('; ')
    problem = Problem(
        url=url, clauses=[Clause.from_txt(c) for c in clauses], goal=goal
    )
    if translate:
      return problem.translate()
    return problem

  @classmethod
  def to_dict(cls, data: list[Problem]) -> dict[str, Problem]:
    return {p.url: p for p in data}

  def __init__(self, url: str, clauses: list[Clause], goal: Construction):
    self.url = url
    self.clauses = clauses
    self.goal = goal

  def copy(self) -> Problem:
    return Problem(self.url, list(self.clauses), self.goal)

  def translate(self) -> Problem:  # to single-char point names
    """Translate point names into alphabetical."""
    mapping = {}
    clauses = []

    for clause in self.clauses:
      clauses.append(clause.translate(mapping))

    if self.goal:
      goal = self.goal.translate(mapping)
    else:
      goal = self.goal

    p = Problem(self.url, clauses, goal)
    p.mapping = mapping
    return p

  def txt(self) -> str:
    return (
        '; '.join([c.txt() for c in self.clauses]) + ' ? ' + self.goal.txt()
        if self.goal
        else ''
    )

  def setup_str_from_problem(self, definitions: list[Definition]) -> str:
    """Construct the <theorem_premises> string from Problem object."""
    ref = 0

    string = []
    for clause in self.clauses:
      group = {}
      p2deps = defaultdict(list)
      for c in clause.constructions:
        cdef = definitions[c.name]

        if len(c.args) != len(cdef.construction.args):
          assert len(c.args) + len(clause.points) == len(cdef.construction.args)
          c.args = clause.points + c.args

        mapping = dict(zip(cdef.construction.args, c.args))
        for points, bs in cdef.basics:
          points = tuple([mapping[x] for x in points])
          for p in points:
            group[p] = points

          for b in bs:
            args = [mapping[a] for a in b.args]
            name = b.name
            if b.name in ['s_angle', 'aconst']:
              x, y, z, v = args
              name = 'aconst'
              v = int(v)

              if v < 0:
                v = -v
                x, z = z, x

              m, n = simplify(int(v), 180)
              args = [y, z, y, x, f'{m}pi/{n}']

            p2deps[points].append(hashed_txt(name, args))

      for k, v in p2deps.items():
        p2deps[k] = sort_deps(v)

      points = clause.points
      while points:
        p = points[0]
        gr = group[p]
        points = [x for x in points if x not in gr]

        deps_str = []
        for dep in p2deps[gr]:
          ref_str = '{:02}'.format(ref)
          dep_str = pt.pretty(dep)

          if dep[0] == 'aconst':
            m, n = map(int, dep[-1].split('pi/'))
            mn = f'{m}. pi / {n}.'
            dep_str = ' '.join(dep_str.split()[:-1] + [mn])

          deps_str.append(dep_str + ' ' + ref_str)
          ref += 1

        string.append(' '.join(gr) + ' : ' + ' '.join(deps_str))

    string = '{S} ' + ' ; '.join([s.strip() for s in string])
    goal = self.goal
    string += ' ? ' + pt.pretty([goal.name] + goal.args)
    return string


def parse_rely(s: str) -> dict[str, str]:
  result = {}
  if not s:
    return result
  s = [x.strip() for x in s.split(',')]
  for x in s:
    a, b = x.split(':')
    a, b = a.strip().split(), b.strip().split()
    result.update({m: b for m in a})
  return result


class Definition:
  """Definitions of construction statements."""

  @classmethod
  def from_txt_file(cls, fname: str, to_dict: bool = False) -> Definition:
    with open(fname, 'r') as f:
      lines = f.read()
    return cls.from_string(lines, to_dict)

  @classmethod
  def from_string(cls, string: str, to_dict: bool = False) -> Definition:
    lines = string.split('\n')
    data = [cls.from_txt('\n'.join(group)) for group in reshape(lines, 6)]
    if to_dict:
      return cls.to_dict(data)
    return data

  @classmethod
  def to_dict(cls, data: list[Definition]) -> dict[str, Definition]:
    return {d.construction.name: d for d in data}

  @classmethod
  def from_txt(cls, data: str) -> Definition:
    """Load definitions from a str object."""
    construction, rely, deps, basics, numerics, _ = data.split('\n')
    basics = [] if not basics else [b.strip() for b in basics.split(';')]

    levels = []
    for bs in basics:
      if ':' in bs:
        points, bs = bs.split(':')
        points = points.strip().split()
      else:
        points = []
      if bs.strip():
        bs = [Construction.from_txt(b.strip()) for b in bs.strip().split(',')]
      else:
        bs = []
      levels.append((points, bs))

    numerics = [] if not numerics else numerics.split(', ')

    return Definition(
        construction=Construction.from_txt(construction),
        rely=parse_rely(rely),
        deps=Clause.from_txt(deps),
        basics=levels,
        numerics=[Construction.from_txt(c) for c in numerics],
    )

  def __init__(
      self,
      construction: Construction,
      rely: dict[str, str],
      deps: Clause,
      basics: list[tuple[list[str], list[Construction]]],
      numerics: list[Construction],
  ):
    self.construction = construction
    self.rely = rely
    self.deps = deps
    self.basics = basics
    self.numerics = numerics

    args = set()
    for num in numerics:
      args.update(num.args)

    self.points = []
    self.args = []
    for p in self.construction.args:
      if p in args:
        self.args.append(p)
      else:
        self.points.append(p)


class Theorem:
  """Deduction rule."""

  @classmethod
  def from_txt_file(cls, fname: str, to_dict: bool = False) -> Theorem:
    with open(fname, 'r') as f:
      theorems = f.read()
    return cls.from_string(theorems, to_dict)

  @classmethod
  def from_string(cls, string: str, to_dict: bool = False) -> Theorem:
    """Load deduction rule from a str object."""
    theorems = string.split('\n')
    theorems = [l for l in theorems if l and not l.startswith('#')]
    theorems = [cls.from_txt(l) for l in theorems]

    for i, th in enumerate(theorems):
      th.rule_name = 'r{:02}'.format(i)

    if to_dict:
      result = {}
      for t in theorems:
        if t.name in result:
          t.name += '_'
        result[t.rule_name] = t

      return result

    return theorems

  @classmethod
  def from_txt(cls, data: str) -> Theorem:
    premises, conclusion = data.split(' => ')
    premises = premises.split(', ')
    conclusion = conclusion.split(', ')
    return Theorem(
        premise=[Construction.from_txt(p) for p in premises],
        conclusion=[Construction.from_txt(c) for c in conclusion],
    )

  def __init__(
      self, premise: list[Construction], conclusion: list[Construction]
  ):
    if len(conclusion) != 1:
      raise ValueError('Cannot have more than one conclusion')
    self.name = '_'.join([p.name for p in premise + conclusion])
    self.premise = premise
    self.conclusion = conclusion
    self.is_arg_reduce = False

    assert len(self.conclusion) == 1
    con = self.conclusion[0]

    if con.name in [
        'eqratio3',
        'midp',
        'contri',
        'simtri',
        'contri2',
        'simtri2',
        'simtri*',
        'contri*',
    ]:
      return

    prem_args = set(sum([p.args for p in self.premise], []))
    con_args = set(con.args)
    if len(prem_args) <= len(con_args):
      self.is_arg_reduce = True

  def txt(self) -> str:
    premise_txt = ', '.join([clause.txt() for clause in self.premise])
    conclusion_txt = ', '.join([clause.txt() for clause in self.conclusion])
    return f'{premise_txt} => {conclusion_txt}'

  def conclusion_name_args(
      self, mapping: dict[str, gm.Point]
  ) -> tuple[str, list[gm.Point]]:
    mapping = {arg: p for arg, p in mapping.items() if isinstance(arg, str)}
    c = self.conclusion[0]
    args = [mapping[a] for a in c.args]
    return c.name, args


def why_eqratio(
    l1: gm.Length,
    l2: gm.Length,
    l3: gm.Length,
    l4: gm.Length,
    level: int,
) -> list[Dependency]:
  """Why two ratios are equal, returns a Dependency objects."""
  all12 = list(gm.all_ratios(l1, l2, level))
  all34 = list(gm.all_ratios(l3, l4, level))

  min_why = None
  for rat12, l1s, l2s in all12:
    for rat34, l3s, l4s in all34:
      why0 = gm.why_equal(rat12, rat34, level)
      if why0 is None:
        continue
      l1_, l2_ = rat12._l
      l3_, l4_ = rat34._l
      why1 = gm.bfs_backtrack(l1, [l1_], l1s)
      why2 = gm.bfs_backtrack(l2, [l2_], l2s)
      why3 = gm.bfs_backtrack(l3, [l3_], l3s)
      why4 = gm.bfs_backtrack(l4, [l4_], l4s)
      why = why0 + why1 + why2 + why3 + why4
      if min_why is None or len(why) < len(min_why[0]):
        min_why = why, rat12, rat34, why0, why1, why2, why3, why4

  if min_why is None:
    return None

  _, rat12, rat34, why0, why1, why2, why3, why4 = min_why
  l1_, l2_ = rat12._l
  l3_, l4_ = rat34._l

  if l1 == l1_ and l2 == l2_ and l3 == l3_ and l4 == l4_:
    return why0

  (a_, b_), (c_, d_) = l1_._obj.points, l2_._obj.points
  (e_, f_), (g_, h_) = l3_._obj.points, l4_._obj.points
  deps = []
  if why0:
    dep = Dependency('eqratio', [a_, b_, c_, d_, e_, f_, g_, h_], '', level)
    dep.why = why0
    deps.append(dep)

  (a, b), (c, d) = l1._obj.points, l2._obj.points
  (e, f), (g, h) = l3._obj.points, l4._obj.points
  for why, (x, y), (x_, y_) in zip(
      [why1, why2, why3, why4],
      [(a, b), (c, d), (e, f), (g, h)],
      [(a_, b_), (c_, d_), (e_, f_), (g_, h_)],
  ):
    if why:
      dep = Dependency('cong', [x, y, x_, y_], '', level)
      dep.why = why
      deps.append(dep)

  return deps

def why_eqratio30(
    l1: gm.Length,
    l2: gm.Length,
    lp46: gm.Length_Pro,
    lp35: gm.Length_Pro,
    level: int,
) -> list[Dependency]:
  """Why two ratios are equal, returns a Dependency objects."""
  all12 = list(gm.all_ratios(l1, l2, level))
  all43 = list(gm.all_ratios2(lp46, lp35, level))

  min_why = None
  for rat12, l1s, l2s in all12:
    for ratp43, lp4s, lp3s in all43:
      why0 = gm.why_equal(rat12, ratp43, level)
      if why0 is None:
        continue
      l1_, l2_ = rat12._l
      lp46_, lp35_ = ratp43._l
      why1 = gm.bfs_backtrack(l1, [l1_], l1s)
      why2 = gm.bfs_backtrack(l2, [l2_], l2s)
      why3 = gm.bfs_backtrack(lp35, [lp35_], lp3s)
      why4 = gm.bfs_backtrack(lp46, [lp46_], lp4s)
      why = why0 + why1 + why2 + why3 + why4
      if min_why is None or len(why) < len(min_why[0]):
        min_why = why, rat12, ratp43, why0, why1, why2, why3, why4

  if min_why is None:
    return None

  _, rat12, ratp43, why0, why1, why2, why3, why4 = min_why
  l1_, l2_ = rat12._l
  lp46_, lp35_ = ratp43._l

  if l1 == l1_ and l2 == l2_ and lp35 == lp35_ and lp46 == lp46_:
    return why0
  
  l3_, l5_ = lp35._l
  l4_, l6_ = lp46._l

  (a_, b_), (c_, d_) = l1_._obj.points, l2_._obj.points
  (m_, n_), (p_, q_) = l3_._obj.points, l4_._obj.points
  (x_, y_), (z_, w_) = l5_._obj.points, l6_._obj.points
  deps = []
  if why0:
    dep = Dependency('eqratio30', [a_, b_, c_, d_, m_, n_, p_, q_, x_, y_, z_, w_], '', level)
    dep.why = why0
    deps.append(dep)

  l3, l5 = lp35._l
  l4, l6 = lp46._l

  (a, b), (c, d) = l1._obj.points, l2._obj.points
  (m, n), (p, q) = l3._obj.points, l4._obj.points
  (x, y), (z, w) = l5._obj.points, l6._obj.points

  if why1:
    dep = Dependency('cong', [a, b, a_, b_], '', level)
    dep.why = why1
    deps.append(dep)
  if why2:
    dep = Dependency('cong', [c, d, c_, d_], '', level)
    dep.why = why2
    deps.append(dep)
  if why3:
    dep = Dependency('eqratio', [m, n, m_, n_, x_, y_, x, y], '', level)
    dep.why = why3
    deps.append(dep)
  if why4:
    dep = Dependency('eqratio', [p, q, p_, q_, z_, w_, z, w], '', level)
    dep.why = why4
    deps.append(dep)
  return deps


def why_eqangle(
    d1: gm.Direction,
    d2: gm.Direction,
    d3: gm.Direction,
    d4: gm.Direction,
    level: int,
    verbose: bool = False,
) -> list[Dependency]:
  """Why two angles are equal, returns a Dependency objects."""
  all12 = list(gm.all_angles(d1, d2, level))
  all34 = list(gm.all_angles(d3, d4, level))

  min_why = None
  for ang12, d1s, d2s in all12:
    for ang34, d3s, d4s in all34:
      why0 = gm.why_equal(ang12, ang34, level)
      if why0 is None:
        continue
      d1_, d2_ = ang12._d
      d3_, d4_ = ang34._d
      why1 = gm.bfs_backtrack(d1, [d1_], d1s)
      why2 = gm.bfs_backtrack(d2, [d2_], d2s)
      why3 = gm.bfs_backtrack(d3, [d3_], d3s)
      why4 = gm.bfs_backtrack(d4, [d4_], d4s)
      why = why0 + why1 + why2 + why3 + why4
      if min_why is None or len(why) < len(min_why[0]):
        min_why = why, ang12, ang34, why0, why1, why2, why3, why4

  if min_why is None:
    return None

  _, ang12, ang34, why0, why1, why2, why3, why4 = min_why
  why0 = gm.why_equal(ang12, ang34, level)
  d1_, d2_ = ang12._d
  d3_, d4_ = ang34._d

  if d1 == d1_ and d2 == d2_ and d3 == d3_ and d4 == d4_:
    return (d1_, d2_, d3_, d4_), why0

  (a_, b_), (c_, d_) = d1_._obj.points, d2_._obj.points
  (e_, f_), (g_, h_) = d3_._obj.points, d4_._obj.points
  deps = []
  if why0:
    dep = Dependency('eqangle', [a_, b_, c_, d_, e_, f_, g_, h_], '', None)
    dep.why = why0
    deps.append(dep)

  (a, b), (c, d) = d1._obj.points, d2._obj.points
  (e, f), (g, h) = d3._obj.points, d4._obj.points
  for why, d_xy, (x, y), d_xy_, (x_, y_) in zip(
      [why1, why2, why3, why4],
      [d1, d2, d3, d4],
      [(a, b), (c, d), (e, f), (g, h)],
      [d1_, d2_, d3_, d4_],
      [(a_, b_), (c_, d_), (e_, f_), (g_, h_)],
  ):
    xy, xy_ = d_xy._obj, d_xy_._obj
    if why:
      if xy == xy_:
        name = 'collx'
      else:
        name = 'para'
      dep = Dependency(name, [x_, y_, x, y], '', None)
      dep.why = why
      deps.append(dep)

  return (d1_, d2_, d3_, d4_), deps


CONSTRUCTION_RULE = 'c0'


class EmptyDependency:
  """Empty dependency predicate ready to get filled up."""

  def __init__(self, level: int, rule_name: str):
    self.level = level
    self.rule_name = rule_name or ''
    self.empty = True
    self.why = []
    self.trace = None

  def populate(self, name: str, args: list[gm.Point]) -> Dependency:
    dep = Dependency(name, args, self.rule_name, self.level)
    dep.trace2 = self.trace
    dep.why = list(self.why)
    return dep

  def copy(self) -> EmptyDependency:
    other = EmptyDependency(self.level, self.rule_name)
    other.why = list(self.why)
    return other

  def extend(
      self,
      g: Any,
      name0: str,
      args0: list[gm.Point],
      name: str,
      args: list[gm.Point],
  ) -> EmptyDependency:
    """Extend the dependency list by (name, args)."""
    dep0 = self.populate(name0, args0)
    deps = EmptyDependency(level=self.level, rule_name=None)
    dep = Dependency(name, args, None, deps.level)
    deps.why = [dep0, dep.why_me_or_cache(g, None)]
    return deps

  def extend_many(
      self,
      g: Any,
      name0: str,
      args0: list[gm.Point],
      name_args: list[tuple[str, list[gm.Point]]],
  ) -> EmptyDependency:
    """Extend the dependency list by many name_args."""
    if not name_args:
      return self
    dep0 = self.populate(name0, args0)
    deps = EmptyDependency(level=self.level, rule_name=None)
    deps.why = [dep0]
    for name, args in name_args:
      dep = Dependency(name, args, None, deps.level)
      deps.why += [dep.why_me_or_cache(g, None)]
    return deps


def maybe_make_equal_pairs(
    a: gm.Point,
    b: gm.Point,
    c: gm.Point,
    d: gm.Point,
    m: gm.Point,
    n: gm.Point,
    p: gm.Point,
    q: gm.Point,
    ab: gm.Line,
    mn: gm.Line,
    g: Any,
    level: int,
) -> list[Dependency]:
  """Make a-b:c-d==m-n:p-q in case a-b==m-n or c-d==p-q."""
  if ab != mn:
    return
  why = []
  eqname = 'para' if isinstance(ab, gm.Line) else 'cong'
  colls = [a, b, m, n]
  if len(set(colls)) > 2 and eqname == 'para':
    dep = Dependency('collx', colls, None, level)
    dep.why_me(g, level)
    why += [dep]

  dep = Dependency(eqname, [c, d, p, q], None, level)
  dep.why_me(g, level)
  why += [dep]
  return why


class Dependency(Construction):
  """Dependency is a predicate that other predicates depend on."""

  def __init__(
      self, name: str, args: list[gm.Point], rule_name: str, level: int
  ):
    super().__init__(name, args)
    self.rule_name = rule_name or ''
    self.level = level
    self.why = []

    self._stat = None
    self.trace = None

  def _find(self, dep_hashed: tuple[str, ...]) -> Dependency:
    self.why = [w for w in self.why if type(w) is not EmptyDependency]
    for w in self.why:
      f = w._find(dep_hashed)
      if f:
        return f
      if w.hashed() == dep_hashed:
        return w

  def remove_loop(self) -> Dependency:
    f = self._find(self.hashed())
    if f:
      return f
    return self

  def copy(self) -> Dependency:
    dep = Dependency(self.name, self.args, self.rule_name, self.level)
    dep.trace = self.trace
    dep.why = list(self.why)
    return dep

  def why_me_or_cache(self, g: Any, level: int) -> Dependency:
    if self.hashed() in g.cache:
      return g.cache[self.hashed()]
    self.why_me(g, level)
    return self

  def populate(self, name: str, args: list[gm.Point]) -> Dependency:
    assert self.rule_name == CONSTRUCTION_RULE, self.rule_name
    dep = Dependency(self.name, self.args, self.rule_name, self.level)
    dep.why = list(self.why)
    return dep

  def why_me(self, g: Any, level: int) -> None:
    """Figure out the dependencies predicates of self."""
    name, args = self.name, self.args

    hashed_me = hashed(name, args)
    if hashed_me in g.cache:
      dep = g.cache[hashed_me]
      self.why = dep.why
      self.rule_name = dep.rule_name
      return

    if self.name == 'para':
      a, b, c, d = self.args
      if {a, b} == {c, d}:
        self.why = []
        return

      ab = g._get_line(a, b)
      cd = g._get_line(c, d)
      if ab == cd:
        if {a, b} == {c, d}:
          self.why = []
          self.rule_name = ''
          return
        dep = Dependency('coll', list({a, b, c, d}), 't??', None)
        self.why = [dep.why_me_or_cache(g, level)]
        return

      for (x, y), xy in zip([(a, b), (c, d)], [ab, cd]):
        x_, y_ = xy.points
        if {x, y} == {x_, y_}:
          continue
        d = Dependency('collx', [x, y, x_, y_], None, level)
        self.why += [d.why_me_or_cache(g, level)]

      whypara = g.why_equal(ab, cd, None)
      self.why += whypara

    elif self.name == 'midp':
      m, a, b = self.args
      ma = g._get_segment(m, a)
      mb = g._get_segment(m, b)
      dep = Dependency('coll', [m, a, b], None, None).why_me_or_cache(g, None)
      self.why = [dep] + g.why_equal(ma, mb, level)

    elif self.name == 'perp':
      a, b, c, d = self.args
      ab = g._get_line(a, b)
      cd = g._get_line(c, d)
      for (x, y), xy in zip([(a, b), (c, d)], [ab, cd]):
        x_, y_ = xy.points
        if {x, y} == {x_, y_}:
          continue
        d = Dependency('collx', [x, y, x_, y_], None, level)
        self.why += [d.why_me_or_cache(g, level)]

      _, why = why_eqangle(ab._val, cd._val, cd._val, ab._val, level)
      a, b = ab.points
      c, d = cd.points

      if hashed(self.name, [a, b, c, d]) != self.hashed():
        d = Dependency(self.name, [a, b, c, d], None, level)
        d.why = why
        why = [d]

      self.why += why

    elif self.name == 'cong':
      a, b, c, d = self.args
      ab = g._get_segment(a, b)
      cd = g._get_segment(c, d)

      self.why = g.why_equal(ab, cd, level)

    elif self.name == 'coll':
      _, why = gm.line_of_and_why(self.args, level)
      self.why = why

    elif self.name == 'collx':
      if g.check_coll(self.args):
        args = list(set(self.args))
        hashed_me = hashed('coll', args)
        if hashed_me in g.cache:
          dep = g.cache[hashed_me]
          self.why = [dep]
          self.rule_name = ''
          return
        _, self.why = gm.line_of_and_why(args, level)
      else:
        self.name = 'para'
        self.why_me(g, level)

    elif self.name == 'cyclic':
      _, why = gm.circle_of_and_why(self.args, level)
      self.why = why

    elif self.name == 'circle':
      o, a, b, c = self.args
      oa = g._get_segment(o, a)
      ob = g._get_segment(o, b)
      oc = g._get_segment(o, c)
      self.why = g.why_equal(oa, ob, level) + g.why_equal(oa, oc, level)

    elif self.name in ['eqangle', 'eqangle6']:
      a, b, c, d, m, n, p, q = self.args

      ab, why1 = g.get_line_thru_pair_why(a, b)
      cd, why2 = g.get_line_thru_pair_why(c, d)
      mn, why3 = g.get_line_thru_pair_why(m, n)
      pq, why4 = g.get_line_thru_pair_why(p, q)

      if ab is None or cd is None or mn is None or pq is None:
        if {a, b} == {m, n}:
          d = Dependency('para', [c, d, p, q], None, level)
          self.why = [d.why_me_or_cache(g, level)]
        if {a, b} == {c, d}:
          d = Dependency('para', [p, q, m, n], None, level)
          self.why = [d.why_me_or_cache(g, level)]
        if {c, d} == {p, q}:
          d = Dependency('para', [a, b, m, n], None, level)
          self.why = [d.why_me_or_cache(g, level)]
        if {p, q} == {m, n}:
          d = Dependency('para', [a, b, c, d], None, level)
          self.why = [d.why_me_or_cache(g, level)]
        return

      for (x, y), xy, whyxy in zip(
          [(a, b), (c, d), (m, n), (p, q)],
          [ab, cd, mn, pq],
          [why1, why2, why3, why4],
      ):
        x_, y_ = xy.points
        if {x, y} == {x_, y_}:
          continue
        d = Dependency('collx', [x, y, x_, y_], None, level)
        d.why = whyxy
        self.why += [d]

      a, b = ab.points
      c, d = cd.points
      m, n = mn.points
      p, q = pq.points
      diff = hashed(self.name, [a, b, c, d, m, n, p, q]) != self.hashed()

      whyeqangle = None
      if ab._val and cd._val and mn._val and pq._val:
        whyeqangle = why_eqangle(ab._val, cd._val, mn._val, pq._val, level)

      if whyeqangle:
        (dab, dcd, dmn, dpq), whyeqangle = whyeqangle
        if diff:
          d = Dependency('eqangle', [a, b, c, d, m, n, p, q], None, level)
          d.why = whyeqangle
          whyeqangle = [d]
        self.why += whyeqangle

      else:
        if (ab == cd and mn == pq) or (ab == mn and cd == pq):
          self.why += []
        elif ab == mn:
          self.why += maybe_make_equal_pairs(
              a, b, c, d, m, n, p, q, ab, mn, g, level
          )
        elif cd == pq:
          self.why += maybe_make_equal_pairs(
              c, d, a, b, p, q, m, n, cd, pq, g, level
          )
        elif ab == cd:
          self.why += maybe_make_equal_pairs(
              a, b, m, n, c, d, p, q, ab, cd, g, level
          )
        elif mn == pq:
          self.why += maybe_make_equal_pairs(
              m, n, a, b, p, q, c, d, mn, pq, g, level
          )
        elif g.is_equal(ab, mn) or g.is_equal(cd, pq):
          dep1 = Dependency('para', [a, b, m, n], None, level)
          dep1.why_me(g, level)
          dep2 = Dependency('para', [c, d, p, q], None, level)
          dep2.why_me(g, level)
          self.why += [dep1, dep2]
        elif g.is_equal(ab, cd) or g.is_equal(mn, pq):
          dep1 = Dependency('para', [a, b, c, d], None, level)
          dep1.why_me(g, level)
          dep2 = Dependency('para', [m, n, p, q], None, level)
          dep2.why_me(g, level)
          self.why += [dep1, dep2]
        elif ab._val and cd._val and mn._val and pq._val:
          self.why = why_eqangle(ab._val, cd._val, mn._val, pq._val, level)

    elif self.name in ['eqratio', 'eqratio6']:
      a, b, c, d, m, n, p, q = self.args
      ab = g._get_segment(a, b)
      cd = g._get_segment(c, d)
      mn = g._get_segment(m, n)
      pq = g._get_segment(p, q)

      if ab is None or cd is None or mn is None or pq is None:
        if {a, b} == {m, n}:
          d = Dependency('cong', [c, d, p, q], None, level)
          self.why = [d.why_me_or_cache(g, level)]
        if {a, b} == {c, d}:
          d = Dependency('cong', [p, q, m, n], None, level)
          self.why = [d.why_me_or_cache(g, level)]
        if {c, d} == {p, q}:
          d = Dependency('cong', [a, b, m, n], None, level)
          self.why = [d.why_me_or_cache(g, level)]
        if {p, q} == {m, n}:
          d = Dependency('cong', [a, b, c, d], None, level)
          self.why = [d.why_me_or_cache(g, level)]
        return

      if ab._val and cd._val and mn._val and pq._val:
        self.why = why_eqratio(ab._val, cd._val, mn._val, pq._val, level)

      if self.why is None:
        self.why = []
        if (ab == cd and mn == pq) or (ab == mn and cd == pq):
          self.why = []
        elif ab == mn:
          self.why += maybe_make_equal_pairs(
              a, b, c, d, m, n, p, q, ab, mn, g, level
          )
        elif cd == pq:
          self.why += maybe_make_equal_pairs(
              c, d, a, b, p, q, m, n, cd, pq, g, level
          )
        elif ab == cd:
          self.why += maybe_make_equal_pairs(
              a, b, m, n, c, d, p, q, ab, cd, g, level
          )
        elif mn == pq:
          self.why += maybe_make_equal_pairs(
              m, n, a, b, p, q, c, d, mn, pq, g, level
          )
        elif g.is_equal(ab, mn) or g.is_equal(cd, pq):
          dep1 = Dependency('cong', [a, b, m, n], None, level)
          dep1.why_me(g, level)
          dep2 = Dependency('cong', [c, d, p, q], None, level)
          dep2.why_me(g, level)
          self.why += [dep1, dep2]
        elif g.is_equal(ab, cd) or g.is_equal(mn, pq):
          dep1 = Dependency('cong', [a, b, c, d], None, level)
          dep1.why_me(g, level)
          dep2 = Dependency('cong', [m, n, p, q], None, level)
          dep2.why_me(g, level)
          self.why += [dep1, dep2]
        elif ab._val and cd._val and mn._val and pq._val:
          self.why = why_eqangle(ab._val, cd._val, mn._val, pq._val, level)

    elif self.name == 'eqratio30':
      a, b, c, d, m, n, p, q, x, y, z, w = self.args
      ab = g._get_segment(a, b)
      cd = g._get_segment(c, d)
      mn = g._get_segment(m, n)
      pq = g._get_segment(p, q)
      xy = g._get_segment(x, y)
      zw = g._get_segment(z, w)
      if ab is None or cd is None or mn is None or pq is None or xy is None or zw is None:
        if {a, b} == {c, d}:
          dp = Dependency('eqratio', [m, n, p, q, z, w, x, y], None, level)
          self.why = [dp.why_me_or_cache(g, level)]
        if {a, b} == {p, q}:
          dp = Dependency('eqratio', [m, n, c, d, z, w, x, y], None, level)
          self.why = [dp.why_me_or_cache(g, level)]
        if {a, b} == {z, w}:
          dp = Dependency('eqratio', [m, n, p, q, c, d, x, y], None, level)
          self.why = [dp.why_me_or_cache(g, level)]  
        if {m, n} == {c, d}:
          dp = Dependency('eqratio', [a, b, p, q, z, w, x, y], None, level)
          self.why = [dp.why_me_or_cache(g, level)]
        if {m, n} == {p, q}:
          dp = Dependency('eqratio', [a, b, c, d, z, w, x, y], None, level)
          self.why = [dp.why_me_or_cache(g, level)]
        if {m, n} == {z, w}:
          dp = Dependency('eqratio', [a, b, p, q, c, d, x, y], None, level)
          self.why = [dp.why_me_or_cache(g, level)]  
        if {x, y} == {c, d}:
          dp = Dependency('eqratio', [a, b, p, q, z, w, m, n], None, level)
          self.why = [dp.why_me_or_cache(g, level)]
        if {x, y} == {p, q}:
          dp = Dependency('eqratio', [a, b, c, d, z, w, m, n], None, level)
          self.why = [dp.why_me_or_cache(g, level)]
        if {x, y} == {z, w}:
          dp = Dependency('eqratio', [a, b, p, q, c, d, m, n], None, level)
          self.why = [dp.why_me_or_cache(g, level)]  
        return
      
      ab_mn, _ = g._get_or_create_length_pro(ab, mn, deps=None)
      ab_xy, _ = g._get_or_create_length_pro(ab, xy, deps=None)
      mn_xy, _ = g._get_or_create_length_pro(mn, xy, deps=None)
      cd_pq, _ = g._get_or_create_length_pro(cd, pq, deps=None)
      cd_zw, _ = g._get_or_create_length_pro(cd, zw, deps=None)
      pq_zw, _ = g._get_or_create_length_pro(pq, zw, deps=None)

      self.why = []
      if g.is_equal(ab, cd) or g.is_equal(mn_xy, pq_zw):
        dep1 = Dependency('cong', [a, b, c, d], None, level)
        dep1.why_me(g, level)
        dep2 = Dependency('eqratio', [m, n, p, q, z, w, x, y], None, level)
        dep2.why_me(g, level)
        self.why += [dep1, dep2]
      elif g.is_equal(ab, pq) or g.is_equal(mn_xy, cd_zw):
        dep1 = Dependency('cong', [a, b, p, q], None, level)
        dep1.why_me(g, level)
        dep2 = Dependency('eqratio', [m, n, c, d, z, w, x, y], None, level)
        dep2.why_me(g, level)
        self.why += [dep1, dep2]
      elif g.is_equal(ab, zw) or g.is_equal(mn_xy, cd_pq):
        dep1 = Dependency('cong', [a, b, z, w], None, level)
        dep1.why_me(g, level)
        dep2 = Dependency('eqratio', [m, n, c, d, p, q, x, y], None, level)
        dep2.why_me(g, level)
        self.why += [dep1, dep2]
      elif g.is_equal(mn, cd) or g.is_equal(ab_xy, pq_zw):
        dep1 = Dependency('cong', [m, n, c, d], None, level)
        dep1.why_me(g, level)
        dep2 = Dependency('eqratio', [a, b, p, q, z, w, x, y], None, level)
        dep2.why_me(g, level)
        self.why += [dep1, dep2]
      elif g.is_equal(mn, pq) or g.is_equal(ab_xy, cd_zw):
        dep1 = Dependency('cong', [m, n, p, q], None, level)
        dep1.why_me(g, level)
        dep2 = Dependency('eqratio', [a, b, c, d, z, w, x, y], None, level)
        dep2.why_me(g, level)
        self.why += [dep1, dep2]
      elif g.is_equal(mn, zw) or g.is_equal(ab_xy, cd_pq):
        dep1 = Dependency('cong', [m, n, z, w], None, level)
        dep1.why_me(g, level)
        dep2 = Dependency('eqratio', [a, b, c, d, p, q, x, y], None, level)
        dep2.why_me(g, level)
        self.why += [dep1, dep2]
      elif g.is_equal(xy, cd) or g.is_equal(ab_mn, pq_zw):
        dep1 = Dependency('cong', [x, y, c, d], None, level)
        dep1.why_me(g, level)
        dep2 = Dependency('eqratio', [a, b, p, q, z, w, m, n], None, level)
        dep2.why_me(g, level)
        self.why += [dep1, dep2]
      elif g.is_equal(xy, pq) or g.is_equal(ab_mn, cd_zw):
        dep1 = Dependency('cong', [x, y, p, q], None, level)
        dep1.why_me(g, level)
        dep2 = Dependency('eqratio', [a, b, c, d, z, w, m, n], None, level)
        dep2.why_me(g, level)
        self.why += [dep1, dep2]
      elif g.is_equal(xy, zw) or g.is_equal(ab_mn, cd_pq):
        dep1 = Dependency('cong', [x, y, z, w], None, level)
        dep1.why_me(g, level)
        dep2 = Dependency('eqratio', [a, b, c, d, p, q, m, n], None, level)
        dep2.why_me(g, level)
        self.why += [dep1, dep2]
      else:
        self.why = None

      if self.why is None:
        self.why = why_eqratio30(ab._val, cd._val, mn_xy, pq_zw, level)

      if self.why is None:
        self.why = []


    elif self.name in ['diff', 'npara', 'nperp', 'ncoll', 'sameside']:
      self.why = []

    elif self.name == 'simtri':
      a, b, c, x, y, z = self.args
      dep1 = Dependency('eqangle', [a, b, a, c, x, y, x, z], '', level)
      dep1.why_me(g, level)
      dep2 = Dependency('eqangle', [b, a, b, c, y, x, y, z], '', level)
      dep2.why_me(g, level)
      self.rule_name = 'r34'
      self.why = [dep1, dep2]

    elif self.name == 'contri':
      a, b, c, x, y, z = self.args
      dep1 = Dependency('cong', [a, b, x, y], '', level)
      dep1.why_me(g, level)
      dep2 = Dependency('cong', [b, c, y, z], '', level)
      dep2.why_me(g, level)
      dep3 = Dependency('cong', [c, a, z, x], '', level)
      dep3.why_me(g, level)
      self.rule_name = 'r32'
      self.why = [dep1, dep2, dep3]

    elif self.name == 'ind':
      pass

    elif self.name == 'aconst':
      a, b, c, d, ang0 = self.args

      measure = ang0._val

      for ang in measure.neighbors(gm.Angle):
        if ang == ang0:
          continue
        d1, d2 = ang._d
        l1, l2 = d1._obj, d2._obj
        (a1, b1), (c1, d1) = l1.points, l2.points

        if not g.check_para_or_coll([a, b, a1, b1]) or not g.check_para_or_coll(
            [c, d, c1, d1]
        ):
          continue

        self.why = []
        for args in [(a, b, a1, b1), (c, d, c1, d1)]:
          if g.check_coll(args):
            if len(set(args)) > 2:
              dep = Dependency('coll', args, None, None)
              self.why.append(dep.why_me_or_cache(g, level))
          else:
            dep = Dependency('para', args, None, None)
            self.why.append(dep.why_me_or_cache(g, level))

        self.why += gm.why_equal(ang, ang0)
        break

    elif self.name == 'rconst':
      a, b, c, d, rat0 = self.args

      val = rat0._val

      for rat in val.neighbors(gm.Ratio):
        if rat == rat0:
          continue
        l1, l2 = rat._l
        s1, s2 = l1._obj, l2._obj
        (a1, b1), (c1, d1) = list(s1.points), list(s2.points)

        if not g.check_cong([a, b, a1, b1]) or not g.check_cong([c, d, c1, d1]):
          continue

        self.why = []
        for args in [(a, b, a1, b1), (c, d, c1, d1)]:
          if len(set(args)) > 2:
            dep = Dependency('cong', args, None, None)
            self.why.append(dep.why_me_or_cache(g, level))

        self.why += gm.why_equal(rat, rat0)
        break

    else:
      raise ValueError('Not recognize', self.name)

  def hashed(self, rename: bool = False) -> tuple[str, ...]:
    return hashed(self.name, self.args, rename=rename)


def hashed(
    name: str, args: list[gm.Point], rename: bool = False
) -> tuple[str, ...]:
  if name == 's_angle':
    args = [p.name if not rename else p.new_name for p in args[:-1]] + [
        str(args[-1])
    ]
  else:
    args = [p.name if not rename else p.new_name for p in args]
  return hashed_txt(name, args)


def hashed_txt(name: str, args: list[str]) -> tuple[str, ...]:
  """Return a tuple unique to name and args upto arg permutation equivariant."""

  if name in ['const', 'aconst', 'rconst']:
    a, b, c, d, y = args
    a, b = sorted([a, b])
    c, d = sorted([c, d])
    return name, a, b, c, d, y

  if name in ['npara', 'nperp', 'para', 'cong', 'perp', 'collx']:
    a, b, c, d = args

    a, b = sorted([a, b])
    c, d = sorted([c, d])
    (a, b), (c, d) = sorted([(a, b), (c, d)])

    return (name, a, b, c, d)

  if name in ['midp', 'midpoint']:
    a, b, c = args
    b, c = sorted([b, c])
    return (name, a, b, c)

  if name in ['coll', 'cyclic', 'ncoll', 'diff', 'triangle']:
    return (name,) + tuple(sorted(list(set(args))))

  if name == 'circle':
    x, a, b, c = args
    return (name, x) + tuple(sorted([a, b, c]))

  if name in ['eqangle', 'eqratio', 'eqangle6', 'eqratio6']:
    a, b, c, d, e, f, g, h = args
    a, b = sorted([a, b])
    c, d = sorted([c, d])
    e, f = sorted([e, f])
    g, h = sorted([g, h])
    if tuple(sorted([a, b, e, f])) > tuple(sorted([c, d, g, h])):
      a, b, e, f, c, d, g, h = c, d, g, h, a, b, e, f
    if (a, b, c, d) > (e, f, g, h):
      a, b, c, d, e, f, g, h = e, f, g, h, a, b, c, d

    if name == 'eqangle6':
      name = 'eqangle'
    if name == 'eqratio6':
      name = 'eqratio'
    return (name,) + (a, b, c, d, e, f, g, h)

  if name in ['contri', 'simtri', 'simtri2', 'contri2', 'contri*', 'simtri*']:
    a, b, c, x, y, z = args
    (a, x), (b, y), (c, z) = sorted([(a, x), (b, y), (c, z)], key=sorted)
    (a, b, c), (x, y, z) = sorted([(a, b, c), (x, y, z)], key=sorted)
    return (name, a, b, c, x, y, z)

  if name in ['eqratio3']:
    a, b, c, d, o, o = args  # pylint: disable=redeclared-assigned-name
    (a, c), (b, d) = sorted([(a, c), (b, d)], key=sorted)
    (a, b), (c, d) = sorted([(a, b), (c, d)], key=sorted)
    return (name, a, b, c, d, o, o)

  if name in ['sameside', 's_angle']:
    return (name,) + tuple(args)
  
  if name in ['eqratio30']:
    a, b, c, d, e, f, g, h, i, j, k, l = args
    a, b = sorted([a, b])
    c, d = sorted([c, d])
    e, f = sorted([e, f])
    g, h = sorted([g, h])
    i, j = sorted([i, j])
    k, l = sorted([k, l])
    ratios1 = [(a, b),(e, f),(i, j)]
    ratios1.sort()
    ratios2 = [(c, d),(g, h),(k, l)]
    ratios2.sort()
    if ratios1 > ratios2:
      ratios1, ratios2 = ratios2, ratios1
    a, b = ratios1[0]
    e, f = ratios1[1]
    i, j = ratios1[2]
    c, d = ratios2[0]
    g, h = ratios2[1]
    k, l = ratios2[2]
    
    return (name, a, b, c, d, e, f, g, h, i, j, k, l)

  raise ValueError(f'Not recognize {name} to hash.')
