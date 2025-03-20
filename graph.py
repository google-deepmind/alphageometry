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

"""Implements the graph representation of the proof state."""

# pylint: disable=g-multiple-import
from __future__ import annotations

from collections import defaultdict  # pylint: disable=g-importing-member
from typing import Callable, Generator, Optional, Type, Union

from absl import logging
import ar
import geometry as gm
from geometry import Angle, Direction, Length, Ratio, Length_Pro, Ratio_Pro
from geometry import Circle, Line, Point, Segment
from geometry import Measure, Value
import graph_utils as utils
import numericals as nm
import problem
from problem import Dependency, EmptyDependency
from multiset import Multiset

np = nm.np


FREE = [
    'free',
    'segment',
    'r_triangle',
    'risos',
    'triangle',
    'triangle12',
    'ieq_triangle',
    'eq_quadrangle',
    'eq_trapezoid',
    'eqdia_quadrangle',
    'quadrangle',
    'r_trapezoid',
    'rectangle',
    'isquare',
    'trapezoid',
    'pentagon',
    'iso_triangle',
]

INTERSECT = [
    'angle_bisector',
    'angle_mirror',
    'eqdistance',
    'lc_tangent',
    'on_aline',
    'on_bline',
    'on_circle',
    'on_line',
    'on_pline',
    'on_tline',
    'on_dia',
    's_angle',
    'on_opline',
    'eqangle3',
]


# pylint: disable=protected-access
# pylint: disable=unused-argument


class DepCheckFailError(Exception):
  pass


class PointTooCloseError(Exception):
  pass


class PointTooFarError(Exception):
  pass


class Graph:
  """Graph data structure representing proof state."""

  def __init__(self):
    self.type2nodes = {
        Point: [],
        Line: [],
        Segment: [],
        Circle: [],
        Direction: [],
        Length: [],
        Angle: [],
        Ratio: [],
        Measure: [],
        Value: [],
        Length_Pro: [],
        Ratio_Pro: [],
    }
    self._name2point = {}
    self._name2node = {}

    self.rconst = {}  # contains all constant ratios
    self.aconst = {}  # contains all constant angles.

    self.halfpi, _ = self.get_or_create_const_ang(1, 2)
    self.vhalfpi = self.halfpi.val

    self.atable = ar.AngleTable()
    self.dtable = ar.DistanceTable()
    self.rtable = ar.RatioTable()

    # to quick access deps.
    self.cache = {}

    self._pair2line = {}
    self._triplet2circle = {}

  def copy(self) -> Graph:
    """Make a copy of self."""
    p, definitions = self.build_def

    p = p.copy()
    for clause in p.clauses:
      clause.nums = []
      for pname in clause.points:
        clause.nums.append(self._name2node[pname].num)

    g, _ = Graph.build_problem(p, definitions, verbose=False, init_copy=False)

    g.build_clauses = list(getattr(self, 'build_clauses', []))
    return g

  def _create_const_ang(self, n: int, d: int) -> None:
    n, d = ar.simplify(n, d)
    ang = self.aconst[(n, d)] = self.new_node(Angle, f'{n}pi/{d}')
    ang.set_directions(None, None)
    self.connect_val(ang, deps=None)

  def _create_const_rat(self, n: int, d: int) -> None:
    n, d = ar.simplify(n, d)
    rat = self.rconst[(n, d)] = self.new_node(Ratio, f'{n}/{d}')
    rat.set_lengths(None, None)
    self.connect_val(rat, deps=None)

  def get_or_create_const_ang(self, n: int, d: int) -> None:
    n, d = ar.simplify(n, d)
    if (n, d) not in self.aconst:
      self._create_const_ang(n, d)
    ang1 = self.aconst[(n, d)]

    n, d = ar.simplify(d - n, d)
    if (n, d) not in self.aconst:
      self._create_const_ang(n, d)
    ang2 = self.aconst[(n, d)]
    return ang1, ang2

  def get_or_create_const_rat(self, n: int, d: int) -> None:
    n, d = ar.simplify(n, d)
    if (n, d) not in self.rconst:
      self._create_const_rat(n, d)
    rat1 = self.rconst[(n, d)]

    if (d, n) not in self.rconst:
      self._create_const_rat(d, n)  # pylint: disable=arguments-out-of-order
    rat2 = self.rconst[(d, n)]
    return rat1, rat2

  def add_algebra(self, dep: Dependency, level: int) -> None:
    """Add new algebraic predicates."""
    _ = level
    if dep.name not in [
        'para',
        'perp',
        'eqangle',
        'eqratio',
        'aconst',
        'rconst',
        'cong',
        'eqratio30'
    ]:
      return

    name, args = dep.name, dep.args

    if name == 'para':
      ab, cd = dep.algebra
      self.atable.add_para(ab, cd, dep)

    if name == 'perp':
      ab, cd = dep.algebra
      self.atable.add_const_angle(ab, cd, 90, dep)

    if name == 'eqangle':
      ab, cd, mn, pq = dep.algebra
      if (ab, cd) == (pq, mn):
        self.atable.add_const_angle(ab, cd, 90, dep)
      else:
        self.atable.add_eqangle(ab, cd, mn, pq, dep)

    if name == 'eqratio':
      ab, cd, mn, pq = dep.algebra
      if (ab, cd) == (pq, mn):
        self.rtable.add_eq(ab, cd, dep)
      else:
        self.rtable.add_eqratio(ab, cd, mn, pq, dep)
    
    if name == 'eqratio30': #TODO
      ab, cd, mn, pq, xy, zw = dep.algebra
      _abmn_, _ = self._get_or_create_length_pro_l(ab, mn)
      _abxy_, _ = self._get_or_create_length_pro_l(ab, xy)
      _mnxy_, _ = self._get_or_create_length_pro_l(mn, xy)
      _cdpq_, _ = self._get_or_create_length_pro_l(cd, pq)
      _cdzw_, _ = self._get_or_create_length_pro_l(cd, zw)
      _pqzw_, _ = self._get_or_create_length_pro_l(pq, zw)
      
      
      self.rtable.add_length_pro(_abmn_, ab, mn)
      self.rtable.add_length_pro(_abxy_, ab, xy)
      self.rtable.add_length_pro(_mnxy_, mn, xy)
      self.rtable.add_length_pro(_cdpq_, cd, pq)
      self.rtable.add_length_pro(_cdzw_, cd, zw)
      self.rtable.add_length_pro(_pqzw_, pq, zw)

      self.rtable.add_eqratio(ab, cd, _pqzw_, _mnxy_ , dep)
      self.rtable.add_eqratio(ab, pq, _cdzw_, _mnxy_ , dep)
      self.rtable.add_eqratio(ab, zw, _cdpq_, _mnxy_ , dep)
      self.rtable.add_eqratio(mn, cd, _pqzw_, _abxy_ , dep)
      self.rtable.add_eqratio(mn, pq, _cdzw_, _abxy_ , dep)
      self.rtable.add_eqratio(mn, zw, _cdpq_, _abxy_ , dep)
      self.rtable.add_eqratio(xy, cd, _pqzw_, _abmn_ , dep)
      self.rtable.add_eqratio(xy, pq, _cdzw_, _abmn_ , dep)
      self.rtable.add_eqratio(xy, zw, _cdpq_, _abmn_ , dep)
      
    if name == 'aconst':
      bx, ab, y = dep.algebra
      self.atable.add_const_angle(bx, ab, y, dep)

    if name == 'rconst':
      l1, l2, m, n = dep.algebra
      self.rtable.add_const_ratio(l1, l2, m, n, dep)

    if name == 'cong':
      a, b, c, d = args
      ab, _ = self.get_line_thru_pair_why(a, b)
      cd, _ = self.get_line_thru_pair_why(c, d)
      self.dtable.add_cong(ab, cd, a, b, c, d, dep)

      ab, cd = dep.algebra
      self.rtable.add_eq(ab, cd, dep)

  def add_eqrat_const(
      self, args: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add new algebraic predicates of type eqratio-constant."""
    a, b, c, d, num, den = args
    nd, dn = self.get_or_create_const_rat(num, den)

    if num == den:
      return self.add_cong([a, b, c, d], deps)

    ab = self._get_or_create_segment(a, b, deps=None)
    cd = self._get_or_create_segment(c, d, deps=None)

    self.connect_val(ab, deps=None)
    self.connect_val(cd, deps=None)

    if ab.val == cd.val:
      raise ValueError(f'{ab.name} and {cd.name} cannot be equal')

    args = [a, b, c, d, nd]
    i = 0
    for x, y, xy in [(a, b, ab), (c, d, cd)]:
      i += 1
      x_, y_ = list(xy._val._obj.points)
      if {x, y} == {x_, y_}:
        continue
      if deps:
        deps = deps.extend(self, 'rconst', list(args), 'cong', [x, y, x_, y_])
      args[2 * i - 2] = x_
      args[2 * i - 1] = y_

    ab_cd, cd_ab, why = self._get_or_create_ratio(ab, cd, deps=None)
    if why:
      dep0 = deps.populate('rconst', [a, b, c, d, nd])
      deps = EmptyDependency(level=deps.level, rule_name=None)
      deps.why = [dep0] + why

    lab, lcd = ab_cd._l
    a, b = list(lab._obj.points)
    c, d = list(lcd._obj.points)

    add = []
    if not self.is_equal(ab_cd, nd):
      args = [a, b, c, d, nd]
      dep1 = deps.populate('rconst', args)
      dep1.algebra = ab._val, cd._val, num, den
      self.make_equal(nd, ab_cd, deps=dep1)
      self.cache_dep('rconst', [a, b, c, d, nd], dep1)
      add += [dep1]

    if not self.is_equal(cd_ab, dn):
      args = [c, d, a, b, dn]
      dep2 = deps.populate('rconst', args)
      dep2.algebra = cd._val, ab._val, num, den
      self.make_equal(dn, cd_ab, deps=dep2)
      self.cache_dep('rconst', [c, d, a, b, dn], dep2)
      add += [dep2]

    return add

  def do_algebra(self, name: str, args: list[Point]) -> list[Dependency]:
    """Derive (but not add) new algebraic predicates."""
    if name == 'para':
      a, b, dep = args
      if gm.is_equiv(a, b):
        return []
      (x, y), (m, n) = a._obj.points, b._obj.points
      return self.add_para([x, y, m, n], dep)

    if name == 'aconst':
      a, b, n, d, dep = args
      ab, ba, why = self.get_or_create_angle_d(a, b, deps=None)
      nd, dn = self.get_or_create_const_ang(n, d)

      (x, y), (m, n) = a._obj.points, b._obj.points

      if why:
        dep0 = dep.populate('aconst', [x, y, m, n, nd])
        dep = EmptyDependency(level=dep.level, rule_name=None)
        dep.why = [dep0] + why

      a, b = ab._d
      (x, y), (m, n) = a._obj.points, b._obj.points

      added = []
      if not self.is_equal(ab, nd):
        if nd == self.halfpi:
          added += self.add_perp([x, y, m, n], dep)
        # else:
        name = 'aconst'
        args = [x, y, m, n, nd]
        dep1 = dep.populate(name, args)
        self.cache_dep(name, args, dep1)
        self.make_equal(nd, ab, deps=dep1)
        added += [dep1]

      if not self.is_equal(ba, dn):
        if dn == self.halfpi:
          added += self.add_perp([m, n, x, y], dep)
        name = 'aconst'
        args = [m, n, x, y, dn]
        dep2 = dep.populate(name, args)
        self.cache_dep(name, args, dep2)
        self.make_equal(dn, ba, deps=dep2)
        added += [dep2]
      return added

    if name == 'rconst':
      a, b, c, d, num, den, dep = args
      return self.add_eqrat_const([a, b, c, d, num, den], dep)

    if name == 'eqangle':
      d1, d2, d3, d4, dep = args
      a, b = d1._obj.points
      c, d = d2._obj.points
      e, f = d3._obj.points
      g, h = d4._obj.points

      return self.add_eqangle([a, b, c, d, e, f, g, h], dep)

    if name == 'eqratio':
      d1, d2, d3, d4, dep = args
      a, b = d1._obj.points
      c, d = d2._obj.points
      e, f = d3._obj.points
      g, h = d4._obj.points

      return self.add_eqratio([a, b, c, d, e, f, g, h], dep)

    if name in ['cong', 'cong2']:
      a, b, c, d, dep = args
      if not (a != b and c != d and (a != c or b != d)):
        return []
      return self.add_cong([a, b, c, d], dep)
    
    if name == 'eqratio30': # TODO
      a, b, c, d, e, f, dep = args
      a1, a2 = a._obj.points
      b1, b2 = b._obj.points
      c1, c2 = c._obj.points
      d1, d2 = d._obj.points
      e1, e2 = e._obj.points
      f1, f2 = f._obj.points
      return self.add_eqratio30([a1, a2, b1, b2, c1, c2, d1, d2, e1, e2, f1, f2], dep)

    return []

  def derive_algebra(
      self, level: int, verbose: bool = False
  ) -> tuple[
      dict[str, list[tuple[Point, ...]]], dict[str, [tuple[Point, ...]]]
  ]:
    """Derive new algebraic predicates."""
    derives = {}
    ang_derives = self.derive_angle_algebra(level, verbose=verbose)
    dist_derives = self.derive_distance_algebra(level, verbose=verbose)
    rat_derives = self.derive_ratio_algebra(level, verbose=verbose)

    derives.update(ang_derives)
    derives.update(dist_derives)
    derives.update(rat_derives)

    # Separate eqangle and eqratio derivations
    # As they are too numerous => slow down DD+AR.
    # & reserve them only for last effort.
    eqs = {'eqangle': derives.pop('eqangle'), 'eqratio': derives.pop('eqratio'), 'eqratio30': derives.pop('eqratio30')}
    return derives, eqs

  def derive_ratio_algebra(
      self, level: int, verbose: bool = False
  ) -> dict[str, list[tuple[Point, ...]]]:
    """Derive new eqratio predicates."""
    added = {'cong2': [], 'eqratio': [], 'eqratio30': []}
    new_products = set()
    for x in self.rtable.get_all_eqs_and_why():
      x, why = x[:-1], x[-1]
      dep = EmptyDependency(level=level, rule_name='a01')
      dep.why = why
      A = []
      B = []

      if len(x) == 2:
        a, b = x
        if type(a) is gm.Length and type(b) is gm.Length:
          A = [a]
          B = [b]
        else:
          assert type(a) is gm.Length_Pro and type(b) is gm.Length_Pro
          a1, a2 = a._l
          b1, b2 = b._l
          A = [a1, a2]
          B = [b1, b2]

      if len(x) == 4:
        a, b, c, d = x
        if type(a) is gm.Length and type(d) is gm.Length:
          assert type(b) is gm.Length, f"b.type:{type(b)}"
          assert type(c) is gm.Length, f"c.type:{type(c)}"
          A = [a,d]
          B = [b,c]
        if {type(a),type(d)} == {gm.Length, gm.Length_Pro}:
          assert {type(b), type(c)} == {gm.Length, gm.Length_Pro}, f"b.type:{type(b)}, c.type:{type(c)}"
          if type(a) is gm.Length: # type(d) is gm.Length_Pro
            a, d = d, a
          if type(b) is gm.Length: # type(c) is gm.Length_Pro
            b, c = c, b
          a1, a2 = a._l
          b1, b2 = b._l
          A = [a1, a2, d]
          B = [b1, b2, c]

        if type(a) is gm.Length_Pro and type(d) is gm.Length_Pro:
          assert type(b) is gm.Length_Pro, f"b.type:{type(b)}"
          assert type(c) is gm.Length_Pro, f"c.type:{type(c)}"
          a1, a2 = a._l
          b1, b2 = b._l
          c1, c2 = c._l
          d1, d2 = d._l
          A = [a1, a2, d1, d2]
          B = [b1, b2, c1, c2]

      print('before',[ag.name for ag in A],[ag.name for ag in B])
      A_ = []
      for x in A:
        for y in B:
          if gm.is_equiv(x, y):
            B.remove(y)
            A_.append(x)
            break
      for x in A_:
        A.remove(x)

      print('after',[ag.name for ag in A],[ag.name for ag in B])
      if len(A) == 0:
        continue
      elif len(A) == 1:
        a = A[0]
        b = B[0]
        m, n = a._obj.points
        p, q = b._obj.points
        added['cong2'].append((m, n, p, q, dep))
      elif len(A) == 2:
        a, d = A
        b, c = B
        added['eqratio'].append((a, b, c, d, dep))
      elif len(A) == 3:
        a1, a2, d = A
        b1, b2, c = B
        _a1a2_, _ = self._get_or_create_length_pro_l(a1, a2)
        _a1d_, _ = self._get_or_create_length_pro_l(a1, d)
        _a2d_, _ = self._get_or_create_length_pro_l(a2, d)
        _b1b2_, _ = self._get_or_create_length_pro_l(b1, b2)
        _b1c_, _ = self._get_or_create_length_pro_l(b1, c)
        _b2c_, _ = self._get_or_create_length_pro_l(b2, c)
        new_products.update([_a1a2_,_a1d_,_a2d_,_b1b2_,_b1c_,_b2c_])
        added['eqratio30'].append((a1, b1, a2, b2, d, c, dep))
      else : #len(A) == 4
        a1, a2, d1, d2 = A
        b1, b2, c1, c2 = B
        _a1a2_, _ = self._get_or_create_length_pro_l(a1, a2)
        _a1d1_, _ = self._get_or_create_length_pro_l(a1, d1)
        _a1d2_, _ = self._get_or_create_length_pro_l(a1, d2)
        _a2d1_, _ = self._get_or_create_length_pro_l(a2, d1)
        _a2d2_, _ = self._get_or_create_length_pro_l(a2, d2)
        _d1d2_, _ = self._get_or_create_length_pro_l(d1, d2)
        _b1b2_, _ = self._get_or_create_length_pro_l(b1, b2)
        _b1c1_, _ = self._get_or_create_length_pro_l(b1, c1)
        _b1c2_, _ = self._get_or_create_length_pro_l(b1, c2)
        _b2c1_, _ = self._get_or_create_length_pro_l(b2, c1)
        _b2c2_, _ = self._get_or_create_length_pro_l(b2, c2)
        _c1c2_, _ = self._get_or_create_length_pro_l(c1, c2)
        new_products.update([_a1a2_,_a1d1_,_a1d2_,_a2d1_,_a2d2_,_d1d2_,_b1b2_,_b1c1_,_b1c2_,_b2c1_,_b2c2_,_d1d2_])
      
        self._set_ratio_pro_equal(_a1a2_, _b1b2_, _c1c2_, _d1d2_, dep)
        self._set_ratio_pro_equal(_a1a2_, _b1c1_, _b2c2_, _d1d2_, dep)
        self._set_ratio_pro_equal(_a1a2_, _b1c2_, _b2c1_, _d1d2_, dep)
        self._set_ratio_pro_equal(_a1d1_, _b1b2_, _c1c2_, _a2d2_, dep)
        self._set_ratio_pro_equal(_a1d1_, _b1c1_, _b2c2_, _a2d2_, dep)
        self._set_ratio_pro_equal(_a1d1_, _b1c2_, _b2c1_, _a2d2_, dep)
        self._set_ratio_pro_equal(_a1d2_, _b1b2_, _c1c2_, _a2d1_, dep)
        self._set_ratio_pro_equal(_a1d2_, _b1c1_, _b2c2_, _a2d1_, dep)
        self._set_ratio_pro_equal(_a1d2_, _b1c2_, _b2c1_, _a2d1_, dep)

    for lp in new_products:
      l1, l2 = lp._l
      self.rtable.add_length_pro(lp, l1, l2)

    return added

  def derive_angle_algebra(
      self, level: int, verbose: bool = False
  ) -> dict[str, list[tuple[Point, ...]]]:
    """Derive new eqangles predicates."""
    added = {'eqangle': [], 'aconst': [], 'para': []}

    for x in self.atable.get_all_eqs_and_why():
      x, why = x[:-1], x[-1]
      dep = EmptyDependency(level=level, rule_name='a02')
      dep.why = why

      if len(x) == 2:
        a, b = x
        if gm.is_equiv(a, b):
          continue

        (e, f), (p, q) = a._obj.points, b._obj.points
        if not nm.check('para', [e, f, p, q]):
          continue

        added['para'].append((a, b, dep))

      if len(x) == 3:
        a, b, (n, d) = x

        (e, f), (p, q) = a._obj.points, b._obj.points
        if not nm.check('aconst', [e, f, p, q, n, d]):
          continue

        added['aconst'].append((a, b, n, d, dep))

      if len(x) == 4:
        a, b, c, d = x
        added['eqangle'].append((a, b, c, d, dep))

    return added

  def derive_distance_algebra(
      self, level: int, verbose: bool = False
  ) -> dict[str, list[tuple[Point, ...]]]:
    """Derive new cong predicates."""
    added = {'inci': [], 'cong': [], 'rconst': []}
    for x in self.dtable.get_all_eqs_and_why():
      x, why = x[:-1], x[-1]
      dep = EmptyDependency(level=level, rule_name='a00')
      dep.why = why

      if len(x) == 2:
        a, b = x
        if a == b:
          continue

        dep.name = f'inci {a.name} {b.name}'
        added['inci'].append((x, dep))

      if len(x) == 4:
        a, b, c, d = x
        if not (a != b and c != d and (a != c or b != d)):
          continue
        added['cong'].append((a, b, c, d, dep))

      if len(x) == 6:
        a, b, c, d, num, den = x
        if not (a != b and c != d and (a != c or b != d)):
          continue
        added['rconst'].append((a, b, c, d, num, den, dep))

    return added

  @classmethod
  def build_problem(
      cls,
      pr: problem.Problem,
      definitions: dict[str, problem.Definition],
      verbose: bool = True,
      init_copy: bool = True,
  ) -> tuple[Graph, list[Dependency]]:
    """Build a problem into a gr.Graph object."""
    check = False
    g = None
    added = None
    if verbose:
      logging.info(pr.url)
      logging.info(pr.txt())
    while not check:
      try:
        g = Graph()
        added = []
        plevel = 0
        for clause in pr.clauses:
          adds, plevel = g.add_clause(
              clause, plevel, definitions, verbose=verbose
          )
          added += adds
        g.plevel = plevel

      except (nm.InvalidLineIntersectError, nm.InvalidQuadSolveError):
        continue
      except DepCheckFailError:
        continue
      except (PointTooCloseError, PointTooFarError):
        continue

      if not pr.goal:
        break

      args = list(map(lambda x: g.get(x, lambda: int(x)), pr.goal.args))
      check = nm.check(pr.goal.name, args)
      print(check)

    g.url = pr.url
    g.build_def = (pr, definitions)
    for add in added:
      g.add_algebra(add, level=0)

    return g, added

  def all_points(self) -> list[Point]:
    """Return all nodes of type Point."""
    return list(self.type2nodes[Point])

  def all_nodes(self) -> list[gm.Node]:
    """Return all nodes."""
    return list(self._name2node.values())

  def add_points(self, pnames: list[str]) -> list[Point]:
    """Add new points with given names in list pnames."""
    result = [self.new_node(Point, name) for name in pnames]
    self._name2point.update(zip(pnames, result))
    return result

  def names2nodes(self, pnames: list[str]) -> list[gm.Node]:
    return [self._name2node[name] for name in pnames]

  def names2points(
      self, pnames: list[str], create_new_point: bool = False
  ) -> list[Point]:
    """Return Point objects given names."""
    result = []
    for name in pnames:
      if name not in self._name2node and not create_new_point:
        raise ValueError(f'Cannot find point {name} in graph')
      elif name in self._name2node:
        obj = self._name2node[name]
      else:
        obj = self.new_node(Point, name)
      result.append(obj)

    return result

  def names2points_or_int(self, pnames: list[str]) -> list[Point]:
    """Return Point objects given names."""
    result = []
    for name in pnames:
      if name.isdigit():
        result += [int(name)]
      elif 'pi/' in name:
        n, d = name.split('pi/')
        ang, _ = self.get_or_create_const_ang(int(n), int(d))
        result += [ang]
      elif '/' in name:
        n, d = name.split('/')
        rat, _ = self.get_or_create_const_rat(int(n), int(d))
        result += [rat]
      else:
        result += [self._name2point[name]]

    return result

  def get(self, pointname: str, default_fn: Callable[str, Point]) -> Point:
    if pointname in self._name2point:
      return self._name2point[pointname]
    if pointname in self._name2node:
      return self._name2node[pointname]
    return default_fn()

  def new_node(self, oftype: Type[gm.Node], name: str = '') -> gm.Node:
    node = oftype(name, self)

    self.type2nodes[oftype].append(node)
    self._name2node[name] = node

    if isinstance(node, Point):
      self._name2point[name] = node

    return node

  def merge(self, nodes: list[gm.Node], deps: Dependency) -> gm.Node:
    """Merge all nodes."""
    if len(nodes) < 2:
      return

    node0, *nodes1 = nodes
    all_nodes = self.type2nodes[type(node0)]

    # find node0 that exists in all_nodes to be the rep
    # and merge all other nodes into node0
    for node in nodes:
      if node in all_nodes:
        node0 = node
        nodes1 = [n for n in nodes if n != node0]
        break
    return self.merge_into(node0, nodes1, deps)

  def merge_into(
      self, node0: gm.Node, nodes1: list[gm.Node], deps: Dependency
  ) -> gm.Node:
    """Merge nodes1 into a single node0."""
    node0.merge(nodes1, deps)
    for n in nodes1:
      if n.rep() != n:
        self.remove([n])

    nodes = [node0] + nodes1
    if any([node._val for node in nodes]):
      for node in nodes:
        self.connect_val(node, deps=None)

      vals1 = [n._val for n in nodes1]
      node0._val.merge(vals1, deps)

      for v in vals1:
        if v.rep() != v:
          self.remove([v])

    return node0

  def remove(self, nodes: list[gm.Node]) -> None:
    """Remove nodes out of self because they are merged."""
    if not nodes:
      return

    for node in nodes:
      all_nodes = self.type2nodes[type(nodes[0])]

      if node in all_nodes:
        all_nodes.remove(node)

      if node.name in self._name2node.values():
        self._name2node.pop(node.name)

  def connect(self, a: gm.Node, b: gm.Node, deps: Dependency) -> None:
    a.connect_to(b, deps)
    b.connect_to(a, deps)

  def connect_val(self, node: gm.Node, deps: Dependency) -> gm.Node:
    """Connect a node into its value (equality) node."""
    if node._val:
      return node._val
    name = None
    if isinstance(node, Line):
      name = 'd(' + node.name + ')'
    if isinstance(node, Angle):
      name = 'm(' + node.name + ')'
    if isinstance(node, Segment):
      name = 'l(' + node.name + ')'
    if isinstance(node, Ratio):
      name = 'r(' + node.name + ')'
    if isinstance(node, Ratio_Pro):
      name = 'rp(' + node.name + ')'
    if isinstance(node, Length_Pro):
      name = 'lp(' + node.name + ')'
    v = self.new_node(gm.val_type(node), name)
    self.connect(node, v, deps=deps)
    return v

  def is_equal(self, x: gm.Node, y: gm.Node, level: int = None) -> bool:
    return gm.is_equal(x, y, level)

  def add_piece(
      self, name: str, args: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add a new predicate."""
    if name in ['coll', 'collx']:
      return self.add_coll(args, deps)
    elif name == 'para':
      return self.add_para(args, deps)
    elif name == 'perp':
      return self.add_perp(args, deps)
    elif name == 'midp':
      return self.add_midp(args, deps)
    elif name == 'cong':
      return self.add_cong(args, deps)
    elif name == 'circle':
      return self.add_circle(args, deps)
    elif name == 'cyclic':
      return self.add_cyclic(args, deps)
    elif name in ['eqangle', 'eqangle6']:
      return self.add_eqangle(args, deps)
    elif name in ['eqratio', 'eqratio6']:
      return self.add_eqratio(args, deps)
    elif name == 'eqratio30':
      return self.add_eqratio30(args, deps)
    # numerical!
    elif name == 's_angle':
      return self.add_s_angle(args, deps)
    elif name == 'aconst':
      a, b, c, d, ang = args

      if isinstance(ang, str):
        name = ang
      else:
        name = ang.name

      num, den = name.split('pi/')
      num, den = int(num), int(den)
      return self.add_aconst([a, b, c, d, num, den], deps)
    elif name == 's_angle':
      b, x, a, b, ang = (  # pylint: disable=redeclared-assigned-name,unused-variable
          args
      )

      if isinstance(ang, str):
        name = ang
      else:
        name = ang.name

      n, d = name.split('pi/')
      ang = int(n) * 180 / int(d)
      return self.add_s_angle([a, b, x, ang], deps)
    elif name == 'rconst':
      a, b, c, d, rat = args

      if isinstance(rat, str):
        name = rat
      else:
        name = rat.name

      num, den = name.split('/')
      num, den = int(num), int(den)
      return self.add_eqrat_const([a, b, c, d, num, den], deps)

    # composite pieces:
    elif name == 'cong2':
      return self.add_cong2(args, deps)
    elif name == 'eqratio3':
      return self.add_eqratio3(args, deps)
    elif name == 'eqratio4':
      return self.add_eqratio4(args, deps)
    elif name == 'simtri':
      return self.add_simtri(args, deps)
    elif name == 'contri':
      return self.add_contri(args, deps)
    elif name == 'simtri2':
      return self.add_simtri2(args, deps)
    elif name == 'contri2':
      return self.add_contri2(args, deps)
    elif name == 'simtri*':
      return self.add_simtri_check(args, deps)
    elif name == 'contri*':
      return self.add_contri_check(args, deps)
    elif name in ['acompute', 'rcompute']:
      dep = deps.populate(name, args)
      self.cache_dep(name, args, dep)
      return [dep]
    elif name in ['fixl', 'fixc', 'fixb', 'fixt', 'fixp']:
      dep = deps.populate(name, args)
      self.cache_dep(name, args, dep)
      return [dep]
    elif name in ['ind']:
      return []
    raise ValueError(f'Not recognize {name}')

  def check(self, name: str, args: list[Point]) -> bool:
    """Symbolically check if a predicate is True."""
    if name == 'ncoll':
      return self.check_ncoll(args)
    if name == 'npara':
      return self.check_npara(args)
    if name == 'nperp':
      return self.check_nperp(args)
    if name == 'midp':
      return self.check_midp(args)
    if name == 'cong':
      return self.check_cong(args)
    if name == 'perp':
      return self.check_perp(args)
    if name == 'para':
      return self.check_para(args)
    if name == 'coll':
      return self.check_coll(args)
    if name == 'cyclic':
      return self.check_cyclic(args)
    if name == 'circle':
      return self.check_circle(args)
    if name == 'aconst':
      return self.check_aconst(args)
    if name == 'rconst':
      return self.check_rconst(args)
    if name == 'acompute':
      return self.check_acompute(args)
    if name == 'rcompute':
      return self.check_rcompute(args)
    if name in ['eqangle', 'eqangle6']:
      if len(args) == 5:
        return self.check_aconst(args)
      return self.check_eqangle(args)
    if name in ['eqratio', 'eqratio6']:
      if len(args) == 5:
        return self.check_rconst(args)
      return self.check_eqratio(args)
    if name in ['eqratio30']:
      return self.check_eqratio30(args)
    if name in ['simtri', 'simtri2', 'simtri*']:
      return self.check_simtri(args)
    if name in ['contri', 'contri2', 'contri*']:
      return self.check_contri(args)
    if name == 'sameside':
      return self.check_sameside(args)
    if name in 'diff':
      a, b = args
      return not a.num.close(b.num)
    if name in ['fixl', 'fixc', 'fixb', 'fixt', 'fixp']:
      return self.in_cache(name, args)
    if name in ['ind']:
      return True
    raise ValueError(f'Not recognize {name}')

  def get_lines_thru_all(self, *points: list[gm.Point]) -> list[Line]:
    line2count = defaultdict(lambda: 0)
    points = set(points)
    for p in points:
      for l in p.neighbors(Line):
        line2count[l] += 1
    return [l for l, count in line2count.items() if count == len(points)]

  def _get_line(self, a: Point, b: Point) -> Optional[Line]:
    linesa = a.neighbors(Line)
    for l in b.neighbors(Line):
      if l in linesa:
        return l
    return None

  def _get_line_all(self, a: Point, b: Point) -> Generator[Line, None, None]:
    linesa = a.neighbors(Line, do_rep=False)
    linesb = b.neighbors(Line, do_rep=False)
    for l in linesb:
      if l in linesa:
        yield l

  def _get_lines(self, *points: list[Point]) -> list[Line]:
    """Return all lines that connect to >= 2 points."""
    line2count = defaultdict(lambda: 0)
    for p in points:
      for l in p.neighbors(Line):
        line2count[l] += 1
    return [l for l, count in line2count.items() if count >= 2]

  def get_circle_thru_triplet(self, p1: Point, p2: Point, p3: Point) -> Circle:
    p1, p2, p3 = sorted([p1, p2, p3], key=lambda x: x.name)
    if (p1, p2, p3) in self._triplet2circle:
      return self._triplet2circle[(p1, p2, p3)]
    return self.get_new_circle_thru_triplet(p1, p2, p3)

  def get_new_circle_thru_triplet(
      self, p1: Point, p2: Point, p3: Point
  ) -> Circle:
    """Get a new Circle that goes thru three given Points."""
    p1, p2, p3 = sorted([p1, p2, p3], key=lambda x: x.name)
    name = p1.name.lower() + p2.name.lower() + p3.name.lower()
    circle = self.new_node(Circle, f'({name})')
    circle.num = nm.Circle(p1=p1.num, p2=p2.num, p3=p3.num)
    circle.points = p1, p2, p3

    self.connect(p1, circle, deps=None)
    self.connect(p2, circle, deps=None)
    self.connect(p3, circle, deps=None)
    self._triplet2circle[(p1, p2, p3)] = circle
    return circle

  def get_line_thru_pair(self, p1: Point, p2: Point) -> Line:
    if (p1, p2) in self._pair2line:
      return self._pair2line[(p1, p2)]
    if (p2, p1) in self._pair2line:
      return self._pair2line[(p2, p1)]
    return self.get_new_line_thru_pair(p1, p2)

  def get_new_line_thru_pair(self, p1: Point, p2: Point) -> Line:
    if p1.name.lower() > p2.name.lower():
      p1, p2 = p2, p1
    name = p1.name.lower() + p2.name.lower()
    line = self.new_node(Line, name)
    line.num = nm.Line(p1.num, p2.num)
    line.points = p1, p2

    self.connect(p1, line, deps=None)
    self.connect(p2, line, deps=None)
    self._pair2line[(p1, p2)] = line
    return line

  def get_line_thru_pair_why(
      self, p1: Point, p2: Point
  ) -> tuple[Line, list[Dependency]]:
    """Get one line thru two given points and the corresponding dependency list."""
    if p1.name.lower() > p2.name.lower():
      p1, p2 = p2, p1
    if (p1, p2) in self._pair2line:
      return self._pair2line[(p1, p2)].rep_and_why()

    l, why = gm.line_of_and_why([p1, p2])
    if l is None:
      l = self.get_new_line_thru_pair(p1, p2)
      why = []
    return l, why

  def coll_dep(self, points: list[Point], p: Point) -> list[Dependency]:
    """Return the dep(.why) explaining why p is coll with points."""
    for p1, p2 in utils.comb2(points):
      if self.check_coll([p1, p2, p]):
        dep = Dependency('coll', [p1, p2, p], None, None)
        return dep.why_me_or_cache(self, None)

  def add_coll(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add a predicate that `points` are collinear."""
    points = list(set(points))
    og_points = list(points)

    all_lines = []
    for p1, p2 in utils.comb2(points):
      all_lines.append(self.get_line_thru_pair(p1, p2))
    points = sum([l.neighbors(Point) for l in all_lines], [])
    points = list(set(points))

    existed = set()
    new = set()
    for p1, p2 in utils.comb2(points):
      if p1.name > p2.name:
        p1, p2 = p2, p1
      if (p1, p2) in self._pair2line:
        line = self._pair2line[(p1, p2)]
        existed.add(line)
      else:
        line = self.get_new_line_thru_pair(p1, p2)
        new.add(line)

    existed = sorted(existed, key=lambda l: l.name)
    new = sorted(new, key=lambda l: l.name)

    existed, new = list(existed), list(new)
    if not existed:
      line0, *lines = new
    else:
      line0, lines = existed[0], existed[1:] + new

    add = []
    line0, why0 = line0.rep_and_why()
    a, b = line0.points
    for line in lines:
      c, d = line.points
      args = list({a, b, c, d})
      if len(args) < 3:
        continue

      whys = []
      for x in args:
        if x not in og_points:
          whys.append(self.coll_dep(og_points, x))

      abcd_deps = deps
      if whys + why0:
        dep0 = deps.populate('coll', og_points)
        abcd_deps = EmptyDependency(level=deps.level, rule_name=None)
        abcd_deps.why = [dep0] + whys

      is_coll = self.check_coll(args)
      dep = abcd_deps.populate('coll', args)
      self.cache_dep('coll', args, dep)
      self.merge_into(line0, [line], dep)

      if not is_coll:
        add += [dep]

    return add

  def check_coll(self, points: list[Point]) -> bool:
    points = list(set(points))
    if len(points) < 3:
      return True
    line2count = defaultdict(lambda: 0)
    for p in points:
      for l in p.neighbors(Line):
        line2count[l] += 1
    return any([count == len(points) for _, count in line2count.items()])

  def why_coll(self, args: tuple[Line, list[Point]]) -> list[Dependency]:
    line, points = args
    return line.why_coll(points)

  def check_ncoll(self, points: list[Point]) -> bool:
    if self.check_coll(points):
      return False
    return not nm.check_coll([p.num for p in points])

  def check_sameside(self, points: list[Point]) -> bool:
    return nm.check_sameside([p.num for p in points])

  def check_onseg(self, points: list[Point]) -> bool:
    return nm.check_onseg([p.num for p in points])

  def check_offseg(self, points: list[Point]) -> bool:
    return nm.check_offseg([p.num for p in points])

  def make_equal(self, x: gm.Node, y: gm.Node, deps: Dependency) -> None:
    """Make that two nodes x and y are equal, i.e. merge their value node."""
    if x.val is None:
      x, y = y, x
    self.connect_val(x, deps=None)
    self.connect_val(y, deps=None)
    vx = x._val
    vy = y._val

    if vx == vy:
      return

    merges = [vx, vy]

    if (
        isinstance(x, Angle)
        and x not in self.aconst.values()
        and y not in self.aconst.values()
        and x.directions == y.directions[::-1]
        and x.directions[0] != x.directions[1]
    ):
      merges = [self.vhalfpi, vx, vy]

    self.merge(merges, deps)

  def merge_vals(self, vx: gm.Node, vy: gm.Node, deps: Dependency) -> None:
    if vx == vy:
      return
    merges = [vx, vy]
    self.merge(merges, deps)

  def why_equal(self, x: gm.Node, y: gm.Node, level: int) -> list[Dependency]:
    return gm.why_equal(x, y, level)

  def _why_coll4(
      self,
      a: Point,
      b: Point,
      ab: Line,
      c: Point,
      d: Point,
      cd: Line,
      level: int,
  ) -> list[Dependency]:
    return self._why_coll2(a, b, ab, level) + self._why_coll2(c, d, cd, level)

  def _why_coll8(
      self,
      a: Point,
      b: Point,
      ab: Line,
      c: Point,
      d: Point,
      cd: Line,
      m: Point,
      n: Point,
      mn: Line,
      p: Point,
      q: Point,
      pq: Line,
      level: int,
  ) -> list[Dependency]:
    """Dependency list of why 8 points are collinear."""
    why8 = self._why_coll4(a, b, ab, c, d, cd, level)
    why8 += self._why_coll4(m, n, mn, p, q, pq, level)
    return why8

  def add_para(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add a new predicate that 4 points (2 lines) are parallel."""
    a, b, c, d = points
    ab, why1 = self.get_line_thru_pair_why(a, b)
    cd, why2 = self.get_line_thru_pair_why(c, d)

    is_equal = self.is_equal(ab, cd)

    (a, b), (c, d) = ab.points, cd.points

    dep0 = deps.populate('para', points)
    deps = EmptyDependency(level=deps.level, rule_name=None)

    deps = deps.populate('para', [a, b, c, d])
    deps.why = [dep0] + why1 + why2

    self.make_equal(ab, cd, deps)
    deps.algebra = ab._val, cd._val

    self.cache_dep('para', [a, b, c, d], deps)
    if not is_equal:
      return [deps]
    return []

  def why_para(self, args: list[Point]) -> list[Dependency]:
    ab, cd, lvl = args
    return self.why_equal(ab, cd, lvl)

  def check_para_or_coll(self, points: list[Point]) -> bool:
    return self.check_para(points) or self.check_coll(points)

  def check_para(self, points: list[Point]) -> bool:
    a, b, c, d = points
    if (a == b) or (c == d):
      return False
    ab = self._get_line(a, b)
    cd = self._get_line(c, d)
    if not ab or not cd:
      return False

    return self.is_equal(ab, cd)

  def check_npara(self, points: list[Point]) -> bool:
    if self.check_para(points):
      return False
    return not nm.check_para([p.num for p in points])

  def _get_angle(
      self, d1: Direction, d2: Direction
  ) -> tuple[Angle, Optional[Angle]]:
    for a in self.type2nodes[Angle]:
      if a.directions == (d1, d2):
        return a, a.opposite
    return None, None

  def get_first_angle(
      self, l1: Line, l2: Line
  ) -> tuple[Angle, list[Dependency]]:
    """Get a first angle between line l1 and line l2."""
    d1, d2 = l1._val, l2._val

    d1s = d1.all_reps()
    d2s = d2.all_reps()

    found = d1.first_angle(d2s)
    if found is None:
      found = d2.first_angle(d1s)
      if found is None:
        return None, []
      ang, x2, x1 = found
      found = ang.opposite, x1, x2

    ang, x1, x2 = found
    return ang, d1.deps_upto(x1) + d2.deps_upto(x2)

  def _get_or_create_angle(
      self, l1: Line, l2: Line, deps: Dependency
  ) -> tuple[Angle, Angle, list[Dependency]]:
    return self.get_or_create_angle_d(l1._val, l2._val, deps)

  def get_or_create_angle_d(
      self, d1: Direction, d2: Direction, deps: Dependency
  ) -> tuple[Angle, Angle, list[Dependency]]:
    """Get or create an angle between two Direction d1 and d2."""
    for a in self.type2nodes[Angle]:
      if a.directions == (d1.rep(), d2.rep()):  # directions = _d.rep()
        d1_, d2_ = a._d
        why1 = d1.why_equal([d1_], None) + d1_.why_rep()
        why2 = d2.why_equal([d2_], None) + d2_.why_rep()
        return a, a.opposite, why1 + why2

    d1, why1 = d1.rep_and_why()
    d2, why2 = d2.rep_and_why()
    a12 = self.new_node(Angle, f'{d1.name}-{d2.name}')
    a21 = self.new_node(Angle, f'{d2.name}-{d1.name}')
    self.connect(d1, a12, deps)
    self.connect(d2, a21, deps)
    self.connect(a12, a21, deps)
    a12.set_directions(d1, d2)
    a21.set_directions(d2, d1)
    a12.opposite = a21
    a21.opposite = a12
    return a12, a21, why1 + why2

  def _add_para_or_coll(
      self,
      a: Point,
      b: Point,
      c: Point,
      d: Point,
      x: Point,
      y: Point,
      m: Point,
      n: Point,
      deps: EmptyDependency,
  ) -> list[Dependency]:
    """Add a new parallel or collinear predicate."""
    extends = [('perp', [x, y, m, n])]
    if {a, b} == {x, y}:
      pass
    elif self.check_para([a, b, x, y]):
      extends.append(('para', [a, b, x, y]))
    elif self.check_coll([a, b, x, y]):
      extends.append(('coll', set(list([a, b, x, y]))))
    else:
      return None

    if m in [c, d] or n in [c, d] or c in [m, n] or d in [m, n]:
      pass
    elif self.check_coll([c, d, m]):
      extends.append(('coll', [c, d, m]))
    elif self.check_coll([c, d, n]):
      extends.append(('coll', [c, d, n]))
    elif self.check_coll([c, m, n]):
      extends.append(('coll', [c, m, n]))
    elif self.check_coll([d, m, n]):
      extends.append(('coll', [d, m, n]))
    else:
      deps = deps.extend_many(self, 'perp', [a, b, c, d], extends)
      return self.add_para([c, d, m, n], deps)

    deps = deps.extend_many(self, 'perp', [a, b, c, d], extends)
    return self.add_coll(list(set([c, d, m, n])), deps)

  def maybe_make_para_from_perp(
      self, points: list[Point], deps: EmptyDependency
  ) -> Optional[list[Dependency]]:
    """Maybe add a new parallel predicate from perp predicate."""
    a, b, c, d = points
    halfpi = self.aconst[(1, 2)]
    for ang in halfpi.val.neighbors(Angle):
      if ang == halfpi:
        continue
      d1, d2 = ang.directions
      x, y = d1._obj.points
      m, n = d2._obj.points

      for args in [
          (a, b, c, d, x, y, m, n),
          (a, b, c, d, m, n, x, y),
          (c, d, a, b, x, y, m, n),
          (c, d, a, b, m, n, x, y),
      ]:
        args = args + (deps,)
        add = self._add_para_or_coll(*args)
        if add:
          return add

    return None

  def add_perp(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add a new perpendicular predicate from 4 points (2 lines)."""
    add = self.maybe_make_para_from_perp(points, deps)
    if add is not None:
      return add

    a, b, c, d = points
    ab, why1 = self.get_line_thru_pair_why(a, b)
    cd, why2 = self.get_line_thru_pair_why(c, d)

    (a, b), (c, d) = ab.points, cd.points

    if why1 + why2:
      dep0 = deps.populate('perp', points)
      deps = EmptyDependency(level=deps.level, rule_name=None)
      deps.why = [dep0] + why1 + why2

    self.connect_val(ab, deps=None)
    self.connect_val(cd, deps=None)

    if ab.val == cd.val:
      raise ValueError(f'{ab.name} and {cd.name} Cannot be perp.')

    args = [a, b, c, d]
    i = 0
    for x, y, xy in [(a, b, ab), (c, d, cd)]:
      i += 1
      x_, y_ = xy._val._obj.points
      if {x, y} == {x_, y_}:
        continue
      if deps:
        deps = deps.extend(self, 'perp', list(args), 'para', [x, y, x_, y_])
      args[2 * i - 2] = x_
      args[2 * i - 1] = y_

    a12, a21, why = self._get_or_create_angle(ab, cd, deps=None)

    if why:
      dep0 = deps.populate('perp', [a, b, c, d])
      deps = EmptyDependency(level=deps.level, rule_name=None)
      deps.why = [dep0] + why

    dab, dcd = a12._d
    a, b = dab._obj.points
    c, d = dcd._obj.points

    is_equal = self.is_equal(a12, a21)
    deps = deps.populate('perp', [a, b, c, d])
    deps.algebra = [dab, dcd]
    self.make_equal(a12, a21, deps=deps)

    self.cache_dep('perp', [a, b, c, d], deps)
    self.cache_dep('eqangle', [a, b, c, d, c, d, a, b], deps)

    if not is_equal:
      return [deps]
    return []

  def why_perp(
      self, args: list[Union[Point, list[Dependency]]]
  ) -> list[Dependency]:
    a, b, deps = args
    return deps + self.why_equal(a, b, None)

  def check_perpl(self, ab: Line, cd: Line) -> bool:
    if ab.val is None or cd.val is None:
      return False
    if ab.val == cd.val:
      return False
    a12, a21 = self._get_angle(ab.val, cd.val)
    if a12 is None or a21 is None:
      return False
    return self.is_equal(a12, a21)

  def check_perp(self, points: list[Point]) -> bool:
    a, b, c, d = points
    ab = self._get_line(a, b)
    cd = self._get_line(c, d)
    if not ab or not cd:
      return False
    return self.check_perpl(ab, cd)

  def check_nperp(self, points: list[Point]) -> bool:
    if self.check_perp(points):
      return False
    return not nm.check_perp([p.num for p in points])

  def _get_segment(self, p1: Point, p2: Point) -> Optional[Segment]:
    for s in self.type2nodes[Segment]:
      if s.points == {p1, p2}:
        return s
    return None

  def _get_or_create_segment(
      self, p1: Point, p2: Point, deps: Dependency
  ) -> Segment:
    """Get or create a Segment object between two Points p1 and p2."""
    if p1 == p2:
      raise ValueError(f'Creating same 0-length segment {p1.name}')

    for s in self.type2nodes[Segment]:
      if s.points == {p1, p2}:
        return s

    if p1.name > p2.name:
      p1, p2 = p2, p1
    s = self.new_node(Segment, name=f'{p1.name.upper()}{p2.name.upper()}')
    self.connect(p1, s, deps=deps)
    self.connect(p2, s, deps=deps)
    s.points = {p1, p2}
    return s

  def add_cong(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add that two segments (4 points) are congruent."""
    a, b, c, d = points
    ab = self._get_or_create_segment(a, b, deps=None)
    cd = self._get_or_create_segment(c, d, deps=None)

    is_equal = self.is_equal(ab, cd)

    dep = deps.populate('cong', [a, b, c, d])
    self.make_equal(ab, cd, deps=dep)
    dep.algebra = ab._val, cd._val

    self.cache_dep('cong', [a, b, c, d], dep)

    result = []

    if not is_equal:
      result += [dep]

    if a not in [c, d] and b not in [c, d]:
      return result

    if b in [c, d]:
      a, b = b, a
    if a == d:
      c, d = d, c  # pylint: disable=unused-variable

    result += self._maybe_add_cyclic_from_cong(a, b, d, dep)
    return result

  def _maybe_add_cyclic_from_cong(
      self, a: Point, b: Point, c: Point, cong_ab_ac: Dependency
  ) -> list[Dependency]:
    """Maybe add a new cyclic predicate from given congruent segments."""
    ab = self._get_or_create_segment(a, b, deps=None)

    # all eq segs with one end being a.
    segs = [s for s in ab.val.neighbors(Segment) if a in s.points]

    # all points on circle (a, b)
    points = []
    for s in segs:
      x, y = list(s.points)
      points.append(x if y == a else y)

    # for sure both b and c are in points
    points = [p for p in points if p not in [b, c]]

    if len(points) < 2:
      return []

    x, y = points[:2]

    if self.check_cyclic([b, c, x, y]):
      return []

    ax = self._get_or_create_segment(a, x, deps=None)
    ay = self._get_or_create_segment(a, y, deps=None)
    why = ab._val.why_equal([ax._val, ay._val], level=None)
    why += [cong_ab_ac]

    deps = EmptyDependency(cong_ab_ac.level, '')
    deps.why = why

    return self.add_cyclic([b, c, x, y], deps)

  def check_cong(self, points: list[Point]) -> bool:
    a, b, c, d = points
    if {a, b} == {c, d}:
      return True

    ab = self._get_segment(a, b)
    cd = self._get_segment(c, d)
    if ab is None or cd is None:
      return False
    return self.is_equal(ab, cd)

  def why_cong(self, args: tuple[Segment, Segment]) -> list[Dependency]:
    ab, cd = args
    return self.why_equal(ab, cd, None)

  def add_midp(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    m, a, b = points
    add = self.add_coll(points, deps=deps)
    add += self.add_cong([m, a, m, b], deps)
    return add

  def why_midp(
      self, args: tuple[Line, list[Point], Segment, Segment]
  ) -> list[Dependency]:
    line, points, ma, mb = args
    return self.why_coll([line, points]) + self.why_cong([ma, mb])

  def check_midp(self, points: list[Point]) -> bool:
    if not self.check_coll(points):
      return False
    m, a, b = points
    return self.check_cong([m, a, m, b])

  def add_circle(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    o, a, b, c = points
    add = self.add_cong([o, a, o, b], deps=deps)
    add += self.add_cong([o, a, o, c], deps=deps)
    return add

  def why_circle(
      self, args: tuple[Segment, Segment, Segment]
  ) -> list[Dependency]:
    oa, ob, oc = args
    return self.why_equal(oa, ob, None) and self.why_equal(oa, oc, None)

  def check_circle(self, points: list[Point]) -> bool:
    o, a, b, c = points
    return self.check_cong([o, a, o, b]) and self.check_cong([o, a, o, c])

  def get_circles_thru_all(self, *points: list[Point]) -> list[Circle]:
    circle2count = defaultdict(lambda: 0)
    points = set(points)
    for p in points:
      for c in p.neighbors(Circle):
        circle2count[c] += 1
    return [c for c, count in circle2count.items() if count == len(points)]

  def _get_circles(self, *points: list[Point]) -> list[Circle]:
    circle2count = defaultdict(lambda: 0)
    for p in points:
      for c in p.neighbors(Circle):
        circle2count[c] += 1
    return [c for c, count in circle2count.items() if count >= 3]

  def cyclic_dep(self, points: list[Point], p: Point) -> list[Dependency]:
    for p1, p2, p3 in utils.comb3(points):
      if self.check_cyclic([p1, p2, p3, p]):
        dep = Dependency('cyclic', [p1, p2, p3, p], None, None)
        return dep.why_me_or_cache(self, None)

  def add_cyclic(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add a new cyclic predicate that 4 points are concyclic."""
    points = list(set(points))
    og_points = list(points)

    all_circles = []
    for p1, p2, p3 in utils.comb3(points):
      all_circles.append(self.get_circle_thru_triplet(p1, p2, p3))
    points = sum([c.neighbors(Point) for c in all_circles], [])
    points = list(set(points))

    existed = set()
    new = set()
    for p1, p2, p3 in utils.comb3(points):
      p1, p2, p3 = sorted([p1, p2, p3], key=lambda x: x.name)

      if (p1, p2, p3) in self._triplet2circle:
        circle = self._triplet2circle[(p1, p2, p3)]
        existed.add(circle)
      else:
        circle = self.get_new_circle_thru_triplet(p1, p2, p3)
        new.add(circle)

    existed = sorted(existed, key=lambda l: l.name)
    new = sorted(new, key=lambda l: l.name)

    existed, new = list(existed), list(new)
    if not existed:
      circle0, *circles = new
    else:
      circle0, circles = existed[0], existed[1:] + new

    add = []
    circle0, why0 = circle0.rep_and_why()
    a, b, c = circle0.points
    for circle in circles:
      d, e, f = circle.points
      args = list({a, b, c, d, e, f})
      if len(args) < 4:
        continue
      whys = []
      for x in [a, b, c, d, e, f]:
        if x not in og_points:
          whys.append(self.cyclic_dep(og_points, x))
      abcdef_deps = deps
      if whys + why0:
        dep0 = deps.populate('cyclic', og_points)
        abcdef_deps = EmptyDependency(level=deps.level, rule_name=None)
        abcdef_deps.why = [dep0] + whys

      is_cyclic = self.check_cyclic(args)

      dep = abcdef_deps.populate('cyclic', args)
      self.cache_dep('cyclic', args, dep)
      self.merge_into(circle0, [circle], dep)
      if not is_cyclic:
        add += [dep]

    return add

  def check_cyclic(self, points: list[Point]) -> bool:
    points = list(set(points))
    if len(points) < 4:
      return True
    circle2count = defaultdict(lambda: 0)
    for p in points:
      for c in p.neighbors(Circle):
        circle2count[c] += 1
    return any([count == len(points) for _, count in circle2count.items()])

  def make_equal_pairs(
      self,
      a: Point,
      b: Point,
      c: Point,
      d: Point,
      m: Point,
      n: Point,
      p: Point,
      q: Point,
      ab: Line,
      cd: Line,
      mn: Line,
      pq: Line,
      deps: EmptyDependency,
  ) -> list[Dependency]:
    """Add ab/cd = mn/pq in case either two of (ab,cd,mn,pq) are equal."""
    depname = 'eqratio' if isinstance(ab, Segment) else 'eqangle'
    eqname = 'cong' if isinstance(ab, Segment) else 'para'

    is_equal = self.is_equal(mn, pq)

    if ab != cd:
      dep0 = deps.populate(depname, [a, b, c, d, m, n, p, q])
      deps = EmptyDependency(level=deps.level, rule_name=None)

      dep = Dependency(eqname, [a, b, c, d], None, deps.level)
      deps.why = [dep0, dep.why_me_or_cache(self, None)]

    elif eqname == 'para':  # ab == cd.
      colls = [a, b, c, d]
      if len(set(colls)) > 2:
        dep0 = deps.populate(depname, [a, b, c, d, m, n, p, q])
        deps = EmptyDependency(level=deps.level, rule_name=None)

        dep = Dependency('collx', colls, None, deps.level)
        deps.why = [dep0, dep.why_me_or_cache(self, None)]

    deps = deps.populate(eqname, [m, n, p, q])
    self.make_equal(mn, pq, deps=deps)

    deps.algebra = mn._val, pq._val
    self.cache_dep(eqname, [m, n, p, q], deps)

    if is_equal:
      return []
    return [deps]

  def maybe_make_equal_pairs(
      self,
      a: Point,
      b: Point,
      c: Point,
      d: Point,
      m: Point,
      n: Point,
      p: Point,
      q: Point,
      ab: Line,
      cd: Line,
      mn: Line,
      pq: Line,
      deps: EmptyDependency,
  ) -> Optional[list[Dependency]]:
    """Add ab/cd = mn/pq in case maybe either two of (ab,cd,mn,pq) are equal."""
    level = deps.level
    if self.is_equal(ab, cd, level):
      return self.make_equal_pairs(a, b, c, d, m, n, p, q, ab, cd, mn, pq, deps)
    elif self.is_equal(mn, pq, level):
      return self.make_equal_pairs(  # pylint: disable=arguments-out-of-order
          m,
          n,
          p,
          q,
          a,
          b,
          c,
          d,
          mn,
          pq,
          ab,
          cd,
          deps,
      )
    elif self.is_equal(ab, mn, level):
      return self.make_equal_pairs(  # pylint: disable=arguments-out-of-order
          a,
          b,
          m,
          n,
          c,
          d,
          p,
          q,
          ab,
          mn,
          cd,
          pq,
          deps,
      )
    elif self.is_equal(cd, pq, level):
      return self.make_equal_pairs(  # pylint: disable=arguments-out-of-order
          c,
          d,
          p,
          q,
          a,
          b,
          m,
          n,
          cd,
          pq,
          ab,
          mn,
          deps,
      )
    else:
      return None

  def _add_eqangle(
      self,
      a: Point,
      b: Point,
      c: Point,
      d: Point,
      m: Point,
      n: Point,
      p: Point,
      q: Point,
      ab: Line,
      cd: Line,
      mn: Line,
      pq: Line,
      deps: EmptyDependency,
  ) -> list[Dependency]:
    """Add eqangle core."""
    if deps:
      deps = deps.copy()

    args = [a, b, c, d, m, n, p, q]
    i = 0
    for x, y, xy in [(a, b, ab), (c, d, cd), (m, n, mn), (p, q, pq)]:
      i += 1
      x_, y_ = xy._val._obj.points
      if {x, y} == {x_, y_}:
        continue
      if deps:
        deps = deps.extend(self, 'eqangle', list(args), 'para', [x, y, x_, y_])

        args[2 * i - 2] = x_
        args[2 * i - 1] = y_

    add = []
    ab_cd, cd_ab, why1 = self._get_or_create_angle(ab, cd, deps=None)
    mn_pq, pq_mn, why2 = self._get_or_create_angle(mn, pq, deps=None)

    why = why1 + why2
    if why:
      dep0 = deps.populate('eqangle', args)
      deps = EmptyDependency(level=deps.level, rule_name=None)
      deps.why = [dep0] + why

    dab, dcd = ab_cd._d
    dmn, dpq = mn_pq._d

    a, b = dab._obj.points
    c, d = dcd._obj.points
    m, n = dmn._obj.points
    p, q = dpq._obj.points

    is_eq1 = self.is_equal(ab_cd, mn_pq)
    deps1 = None
    if deps:
      deps1 = deps.populate('eqangle', [a, b, c, d, m, n, p, q])
      deps1.algebra = [dab, dcd, dmn, dpq]
    if not is_eq1:
      add += [deps1]
    self.cache_dep('eqangle', [a, b, c, d, m, n, p, q], deps1)
    self.make_equal(ab_cd, mn_pq, deps=deps1)

    is_eq2 = self.is_equal(cd_ab, pq_mn)
    deps2 = None
    if deps:
      deps2 = deps.populate('eqangle', [c, d, a, b, p, q, m, n])
      deps2.algebra = [dcd, dab, dpq, dmn]
    if not is_eq2:
      add += [deps2]
    self.cache_dep('eqangle', [c, d, a, b, p, q, m, n], deps2)
    self.make_equal(cd_ab, pq_mn, deps=deps2)

    return add

  def add_eqangle(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add eqangle made by 8 points in `points`."""
    if deps:
      deps = deps.copy()
    a, b, c, d, m, n, p, q = points
    ab, why1 = self.get_line_thru_pair_why(a, b)
    cd, why2 = self.get_line_thru_pair_why(c, d)
    mn, why3 = self.get_line_thru_pair_why(m, n)
    pq, why4 = self.get_line_thru_pair_why(p, q)

    a, b = ab.points
    c, d = cd.points
    m, n = mn.points
    p, q = pq.points

    if deps and why1 + why2 + why3 + why4:
      dep0 = deps.populate('eqangle', points)
      deps = EmptyDependency(level=deps.level, rule_name=None)
      deps.why = [dep0] + why1 + why2 + why3 + why4

    add = self.maybe_make_equal_pairs(
        a, b, c, d, m, n, p, q, ab, cd, mn, pq, deps
    )

    if add is not None:
      return add

    self.connect_val(ab, deps=None)
    self.connect_val(cd, deps=None)
    self.connect_val(mn, deps=None)
    self.connect_val(pq, deps=None)

    add = []
    if (
        ab.val != cd.val
        and mn.val != pq.val
        and (ab.val != mn.val or cd.val != pq.val)
    ):
      add += self._add_eqangle(a, b, c, d, m, n, p, q, ab, cd, mn, pq, deps)

    if (
        ab.val != mn.val
        and cd.val != pq.val
        and (ab.val != cd.val or mn.val != pq.val)
    ):
      add += self._add_eqangle(  # pylint: disable=arguments-out-of-order
          a,
          b,
          m,
          n,
          c,
          d,
          p,
          q,
          ab,
          mn,
          cd,
          pq,
          deps,
      )

    return add

  def add_aconst(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add that an angle is equal to some constant."""
    a, b, c, d, num, den = points
    nd, dn = self.get_or_create_const_ang(num, den)

    if nd == self.halfpi:
      return self.add_perp([a, b, c, d], deps)

    ab, why1 = self.get_line_thru_pair_why(a, b)
    cd, why2 = self.get_line_thru_pair_why(c, d)

    (a, b), (c, d) = ab.points, cd.points
    if why1 + why2:
      args = points[:-2] + [nd]
      dep0 = deps.populate('aconst', args)
      deps = EmptyDependency(level=deps.level, rule_name=None)
      deps.why = [dep0] + why1 + why2

    self.connect_val(ab, deps=None)
    self.connect_val(cd, deps=None)

    if ab.val == cd.val:
      raise ValueError(f'{ab.name} - {cd.name} cannot be {nd.name}')

    args = [a, b, c, d, nd]
    i = 0
    for x, y, xy in [(a, b, ab), (c, d, cd)]:
      i += 1
      x_, y_ = xy._val._obj.points
      if {x, y} == {x_, y_}:
        continue
      if deps:
        deps = deps.extend(self, 'aconst', list(args), 'para', [x, y, x_, y_])
      args[2 * i - 2] = x_
      args[2 * i - 1] = y_

    ab_cd, cd_ab, why = self._get_or_create_angle(ab, cd, deps=None)
    if why:
      dep0 = deps.populate('aconst', [a, b, c, d, nd])
      deps = EmptyDependency(level=deps.level, rule_name=None)
      deps.why = [dep0] + why

    dab, dcd = ab_cd._d
    a, b = dab._obj.points
    c, d = dcd._obj.points

    ang = int(num) * 180 / int(den)
    add = []
    if not self.is_equal(ab_cd, nd):
      deps1 = deps.populate('aconst', [a, b, c, d, nd])
      deps1.algebra = dab, dcd, ang % 180
      self.make_equal(ab_cd, nd, deps=deps1)
      self.cache_dep('aconst', [a, b, c, d, nd], deps1)
      add += [deps1]

    if not self.is_equal(cd_ab, dn):
      deps2 = deps.populate('aconst', [c, d, a, b, dn])
      deps2.algebra = dcd, dab, 180 - ang % 180
      self.make_equal(cd_ab, dn, deps=deps2)
      self.cache_dep('aconst', [c, d, a, b, dn], deps2)
      add += [deps2]
    return add

  def add_s_angle(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add that an angle abx is equal to constant y."""
    a, b, x, y = points

    n, d = ar.simplify(y % 180, 180)
    nd, dn = self.get_or_create_const_ang(n, d)

    if nd == self.halfpi:
      return self.add_perp([a, b, b, x], deps)

    ab, why1 = self.get_line_thru_pair_why(a, b)
    bx, why2 = self.get_line_thru_pair_why(b, x)

    self.connect_val(ab, deps=None)
    self.connect_val(bx, deps=None)
    add = []

    if ab.val == bx.val:
      return add

    deps.why += why1 + why2

    for p, q, pq in [(a, b, ab), (b, x, bx)]:
      p_, q_ = pq.val._obj.points
      if {p, q} == {p_, q_}:
        continue
      dep = Dependency('para', [p, q, p_, q_], None, deps.level)
      deps.why += [dep.why_me_or_cache(self, None)]

    xba, abx, why = self._get_or_create_angle(bx, ab, deps=None)
    if why:
      dep0 = deps.populate('aconst', [b, x, a, b, nd])
      deps = EmptyDependency(level=deps.level, rule_name=None)
      deps.why = [dep0] + why

    dab, dbx = abx._d
    a, b = dab._obj.points
    c, x = dbx._obj.points

    if not self.is_equal(xba, nd):
      deps1 = deps.populate('aconst', [c, x, a, b, nd])
      deps1.algebra = dbx, dab, y % 180

      self.make_equal(xba, nd, deps=deps1)
      self.cache_dep('aconst', [c, x, a, b, nd], deps1)
      add += [deps1]

    if not self.is_equal(abx, dn):
      deps2 = deps.populate('aconst', [a, b, c, x, dn])
      deps2.algebra = dab, dbx, 180 - (y % 180)

      self.make_equal(abx, dn, deps=deps2)
      self.cache_dep('s_angle', [a, b, c, x, dn], deps2)
      add += [deps2]
    return add

  def check_aconst(self, points: list[Point], verbose: bool = False) -> bool:
    """Check if the angle is equal to a certain constant."""
    a, b, c, d, nd = points
    _ = verbose
    if isinstance(nd, str):
      name = nd
    else:
      name = nd.name
    num, den = name.split('pi/')
    ang, _ = self.get_or_create_const_ang(int(num), int(den))

    ab = self._get_line(a, b)
    cd = self._get_line(c, d)
    if not ab or not cd:
      return False

    if not (ab.val and cd.val):
      return False

    for ang1, _, _ in gm.all_angles(ab._val, cd._val):
      if self.is_equal(ang1, ang):
        return True
    return False

  def check_acompute(self, points: list[Point]) -> bool:
    """Check if an angle has a constant value."""
    a, b, c, d = points
    ab = self._get_line(a, b)
    cd = self._get_line(c, d)
    if not ab or not cd:
      return False

    if not (ab.val and cd.val):
      return False

    for ang0 in self.aconst.values():
      for ang in ang0.val.neighbors(Angle):
        d1, d2 = ang.directions
        if ab.val == d1 and cd.val == d2:
          return True
    return False

  def check_eqangle(self, points: list[Point]) -> bool:
    """Check if two angles are equal."""
    a, b, c, d, m, n, p, q = points

    if {a, b} == {c, d} and {m, n} == {p, q}:
      return True
    if {a, b} == {m, n} and {c, d} == {p, q}:
      return True

    if (a == b) or (c == d) or (m == n) or (p == q):
      return False
    ab = self._get_line(a, b)
    cd = self._get_line(c, d)
    mn = self._get_line(m, n)
    pq = self._get_line(p, q)

    if {a, b} == {c, d} and mn and pq and self.is_equal(mn, pq):
      return True
    if {a, b} == {m, n} and cd and pq and self.is_equal(cd, pq):
      return True
    if {p, q} == {m, n} and ab and cd and self.is_equal(ab, cd):
      return True
    if {p, q} == {c, d} and ab and mn and self.is_equal(ab, mn):
      return True

    if not ab or not cd or not mn or not pq:
      return False

    if self.is_equal(ab, cd) and self.is_equal(mn, pq):
      return True
    if self.is_equal(ab, mn) and self.is_equal(cd, pq):
      return True

    if not (ab.val and cd.val and mn.val and pq.val):
      return False

    if (ab.val, cd.val) == (mn.val, pq.val) or (ab.val, mn.val) == (
        cd.val,
        pq.val,
    ):
      return True

    for ang1, _, _ in gm.all_angles(ab._val, cd._val):
      for ang2, _, _ in gm.all_angles(mn._val, pq._val):
        if self.is_equal(ang1, ang2):
          return True

    if self.check_perp([a, b, m, n]) and self.check_perp([c, d, p, q]):
      return True
    if self.check_perp([a, b, p, q]) and self.check_perp([c, d, m, n]):
      return True

    return False

  def _get_ratio(self, l1: Length, l2: Length) -> tuple[Ratio, Ratio]:
    for r in self.type2nodes[Ratio]:
      if r.lengths == (l1, l2):
        return r, r.opposite
    return None, None

  def _get_or_create_ratio(
      self, s1: Segment, s2: Segment, deps: Dependency
  ) -> tuple[Ratio, Ratio, list[Dependency]]:
    return self._get_or_create_ratio_l(s1._val, s2._val, deps)

  def _get_or_create_ratio_l(
      self, l1: Length, l2: Length, deps: Dependency
  ) -> tuple[Ratio, Ratio, list[Dependency]]:
    """Get or create a new Ratio from two Lenghts l1 and l2."""
    for r in self.type2nodes[Ratio]:
      if r.lengths == (l1.rep(), l2.rep()):
        l1_, l2_ = r._l
        why1 = l1.why_equal([l1_], None) + l1_.why_rep()
        why2 = l2.why_equal([l2_], None) + l2_.why_rep()
        return r, r.opposite, why1 + why2

    l1, why1 = l1.rep_and_why()
    l2, why2 = l2.rep_and_why()
    r12 = self.new_node(Ratio, f'{l1.name}/{l2.name}')
    r21 = self.new_node(Ratio, f'{l2.name}/{l1.name}')
    self.connect(l1, r12, deps)
    self.connect(l2, r21, deps)
    self.connect(r12, r21, deps)
    r12.set_lengths(l1, l2)
    r21.set_lengths(l2, l1)
    r12.opposite = r21
    r21.opposite = r12
    return r12, r21, why1 + why2

  def _get_or_create_length_pro(
      self, s1: Segment, s2: Segment, deps=None
  ) -> tuple[Length_Pro, list[Dependency]]:
    return self._get_or_create_length_pro_l(s1._val, s2._val, deps)

  def _get_or_create_length_pro_l(
      self, l1: Length, l2: Length, deps=None
  ) -> tuple[Length_Pro, list[Dependency]]:
    """Get or create a new Length_Pro from two Lenghts l1 and l2."""
    if l1.name > l2.name:
      l1, l2 = l2, l1
    for lp in self.type2nodes[Length_Pro]:
      if lp.lengths == Multiset([l1.rep(), l2.rep()]):
        l1_, l2_ = lp.lengths
        if l1.rep() == l1_:# l1.rep() == l1_ and l2.rep() == l2_
          why1 = l1.why_equal([l1_], None) + l1_.why_rep()
          why2 = l2.why_equal([l2_], None) + l2_.why_rep()
          return lp, why1 + why2
        else: #l2.rep() == l1_ and l1.rep == l2_
          why1 = l2.why_equal([l1_], None) + l1_.why_rep()
          why2 = l1.why_equal([l2_], None) + l2_.why_rep()

    l1, why1 = l1.rep_and_why()
    l2, why2 = l2.rep_and_why()
    lp12 = self.new_node(Length_Pro, f'{l1.name}*{l2.name}')
    self.connect(l1, lp12, deps)
    self.connect(l2, lp12, deps)
    lp12.set_lengths(l1, l2)
    return lp12, why1 + why2

  def _get_or_create_ratio_pro(
      self, s1: Segment, s2: Segment, s3: Segment, s4: Segment, deps=None
  ) -> tuple[Ratio_Pro, Ratio_Pro, list[Dependency]]:
    return self._get_or_create_ratio_pro_l(s1._val, s2._val, s3._val, s4._val, deps)

  def _get_or_create_ratio_pro_l(
      self, l1: Length, l2: Length, l3:Length, l4: Length, deps=None
  ) -> tuple[Ratio_Pro, Ratio_Pro, list[Dependency]]:
    """Get or create a new Ratio_Pro from four Lengths l1*l3 and l2*l4."""
    if l1.name > l3.name:
      l1, l3 = l3, l1
    if l2.name > l4.name:
      l2, l4 = l4, l2
    for rp in self.type2nodes[Ratio_Pro]:
      lp13_, lp24_ = rp._lp
      if lp13_.lengths == Multiset([l1.rep(), l3.rep()]) and lp24_.lengths == Multiset([l2.rep(), l4.rep()]):
        l1_, l3_ = lp13_.lengths
        l2_, l4_ = lp24_.lengths
        if l1.rep() == l1_:
          why1 = l1.why_equal([l1_], None) + l1_.why_rep()
          why3 = l3.why_equal([l3_], None) + l3_.why_rep()
        else:
          why1 = l3.why_equal([l1_], None) + l1_.why_rep()
          why3 = l1.why_equal([l3_], None) + l3_.why_rep()
        if l2.rep() == l2_:
          why2 = l2.why_equal([l2_], None) + l2_.why_rep()
          why4 = l4.why_equal([l4_], None) + l4_.why_rep()
        else:
          why2 = l4.why_equal([l2_], None) + l2_.why_rep()
          why4 = l2.why_equal([l4_], None) + l4_.why_rep()
        return rp, rp.opposite, why1 + why2 + why3 + why4

    l1, why1 = l1.rep_and_why()
    l2, why2 = l2.rep_and_why()
    l3, why3 = l3.rep_and_why()
    l4, why4 = l4.rep_and_why()
    lp13 = self.new_node(Length_Pro, f'{l1.name}*{l3.name}')
    lp24 = self.new_node(Length_Pro, f'{l2.name}*{l4.name}')
    lp13.set_lengths(l1, l3)
    lp24.set_lengths(l2, l4)
    rp1324 = self.new_node(Ratio_Pro, f'{lp13.name}/{lp24.name}')
    rp2413 = self.new_node(Ratio_Pro, f'{lp24.name}/{lp13.name}')
    self.connect(lp13, rp1324, deps)
    self.connect(lp24, rp2413, deps)
    self.connect(rp1324, rp2413, deps)
    rp1324.set_lengths(lp13, lp24)
    rp2413.set_lengths(lp24, lp13)
    rp1324.opposite = rp2413
    rp2413.opposite = rp1324
    return rp1324, rp2413, why1 + why2 + why3 + why4
  
  def _get_or_create_ratio_pro_lp(
      self, lp1: Length_Pro, lp2: Length_Pro, deps=None
  ) -> tuple[Ratio_Pro, Ratio_Pro, list[Dependency]]:
      """Get or create a new Ratio_Pro from two Length_Pro l1*l2."""
      l1, l3 = lp1._l
      l2, l4 = lp2._l
      return self._get_or_create_ratio_pro_l(l1, l2, l3, l4, deps)
  
  def _set_ratio_pro_equal(
      self, lp1: Length_Pro, lp2: Length_Pro, lp3: Length_Pro, lp4: Length_Pro, dep=None
  ) -> None:
      deps = dep
      rp12, rp21, _ = self._get_or_create_ratio_pro_lp(lp1, lp2)
      rp34, rp43, _ = self._get_or_create_ratio_pro_lp(lp3, lp4)
      self.make_equal(rp12, rp34, deps=deps)
      self.make_equal(rp21, rp43, deps=deps)
      rp13, rp31, _ = self._get_or_create_ratio_pro_lp(lp1, lp3)
      rp24, rp42, _ = self._get_or_create_ratio_pro_lp(lp2, lp4)
      self.make_equal(rp13, rp24, deps=deps)
      self.make_equal(rp31, rp42, deps=deps)

  def add_cong2(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    m, n, a, b = points
    add = []
    add += self.add_cong([m, a, n, a], deps)
    add += self.add_cong([m, b, n, b], deps)
    return add

  def add_eqratio3(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add three eqratios through a list of 6 points (due to parallel lines)."""
    a, b, c, d, m, n = points
    #   a -- b
    #  m  --  n
    # c   --   d
    add = []
    add += self.add_eqratio([m, a, m, c, n, b, n, d], deps)
    add += self.add_eqratio([a, m, a, c, b, n, b, d], deps)
    add += self.add_eqratio([c, m, c, a, d, n, d, b], deps)
    if m == n:
      add += self.add_eqratio([m, a, m, c, a, b, c, d], deps)
    return add

  def add_eqratio4(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    o, a, b, c, d = points
    #   o
    #  a b
    # c   d
    add = self.add_eqratio3([a, b, c, d, o, o], deps)
    add += self.add_eqratio([o, a, o, c, a, b, c, d], deps)
    return add

  def _add_eqratio(
      self,
      a: Point,
      b: Point,
      c: Point,
      d: Point,
      m: Point,
      n: Point,
      p: Point,
      q: Point,
      ab: Segment,
      cd: Segment,
      mn: Segment,
      pq: Segment,
      deps: EmptyDependency,
  ) -> list[Dependency]:
    """Add a new eqratio from 8 points (core)."""
    if deps:
      deps = deps.copy()

    args = [a, b, c, d, m, n, p, q]
    i = 0
    for x, y, xy in [(a, b, ab), (c, d, cd), (m, n, mn), (p, q, pq)]:
      if {x, y} == set(xy.points):
        continue
      x_, y_ = list(xy.points)
      if deps:
        deps = deps.extend(self, 'eqratio', list(args), 'cong', [x, y, x_, y_])
      args[2 * i - 2] = x_
      args[2 * i - 1] = y_

    add = []
    ab_cd, cd_ab, why1 = self._get_or_create_ratio(ab, cd, deps=None)
    mn_pq, pq_mn, why2 = self._get_or_create_ratio(mn, pq, deps=None)

    _abpq_, _ = self._get_or_create_length_pro(ab, pq, deps=None)
    _cdmn_, _ = self._get_or_create_length_pro(cd, mn, deps=None)

    why = why1 + why2
    if why:
      dep0 = deps.populate('eqratio', args)
      deps = EmptyDependency(level=deps.level, rule_name=None)
      deps.why = [dep0] + why

    lab, lcd = ab_cd._l
    lmn, lpq = mn_pq._l

    a, b = lab._obj.points
    c, d = lcd._obj.points
    m, n = lmn._obj.points
    p, q = lpq._obj.points

    is_eq1 = self.is_equal(ab_cd, mn_pq)
    deps1 = None
    if deps:
      deps1 = deps.populate('eqratio', [a, b, c, d, m, n, p, q])
      deps1.algebra = [ab._val, cd._val, mn._val, pq._val]
    if not is_eq1:
      add += [deps1]
    self.cache_dep('eqratio', [a, b, c, d, m, n, p, q], deps1)
    self.make_equal(ab_cd, mn_pq, deps=deps1)
    self.make_equal(_abpq_, _cdmn_, deps=deps1)

    is_eq2 = self.is_equal(cd_ab, pq_mn)
    deps2 = None
    if deps:
      deps2 = deps.populate('eqratio', [c, d, a, b, p, q, m, n])
      deps2.algebra = [cd._val, ab._val, pq._val, mn._val]
    if not is_eq2:
      add += [deps2]
    self.cache_dep('eqratio', [c, d, a, b, p, q, m, n], deps2)
    self.make_equal(cd_ab, pq_mn, deps=deps2)
    return add

  def add_eqratio(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add a new eqratio from 8 points."""
    if deps:
      deps = deps.copy()
    a, b, c, d, m, n, p, q = points
    ab = self._get_or_create_segment(a, b, deps=None)
    cd = self._get_or_create_segment(c, d, deps=None)
    mn = self._get_or_create_segment(m, n, deps=None)
    pq = self._get_or_create_segment(p, q, deps=None)

    add = self.maybe_make_equal_pairs(
        a, b, c, d, m, n, p, q, ab, cd, mn, pq, deps
    )

    if add is not None:
      return add

    self.connect_val(ab, deps=None)
    self.connect_val(cd, deps=None)
    self.connect_val(mn, deps=None)
    self.connect_val(pq, deps=None)

    add = []
    if (
        ab.val != cd.val
        and mn.val != pq.val
        and (ab.val != mn.val or cd.val != pq.val)
    ):
      add += self._add_eqratio(a, b, c, d, m, n, p, q, ab, cd, mn, pq, deps)

    if (
        ab.val != mn.val
        and cd.val != pq.val
        and (ab.val != cd.val or mn.val != pq.val)
    ):
      add += self._add_eqratio(  # pylint: disable=arguments-out-of-order
          a,
          b,
          m,
          n,
          c,
          d,
          p,
          q,
          ab,
          mn,
          cd,
          pq,
          deps,
      )
    return add  
  
  def make_equal_pairs30(
      self,
      a: Point, b: Point, c: Point, d: Point, m: Point, n: Point, p: Point, q: Point, x:Point, y:Point, z:Point, w:Point,
      ab: Segment, cd: Segment, mn: Segment, pq: Segment, xy: Segment, zw: Segment,
      deps: EmptyDependency,
  ) -> list[Dependency]:
    """Add ab/cd * mn/pq * xy/zw = 1 in case ab and cd are equal."""

    mn_pq, pq_mn, why1 = self._get_or_create_ratio(mn, pq, deps=None)
    xy_zw, zw_xy, why2 = self._get_or_create_ratio(xy, zw, deps=None)

    _mnzw_, _ = self._get_or_create_length_pro(mn, zw, deps=None)
    _pqxy_, _ = self._get_or_create_length_pro(pq, xy, deps=None)

    is_equal = self.is_equal(mn_pq, zw_xy)

    if ab != cd:
      dep0 = deps.populate('eqratio30', [a, b, c, d, m, n, p, q, x, y, z, w])
      deps = EmptyDependency(level=deps.level, rule_name=None)

      dep = Dependency('cong', [a, b, c, d], None, deps.level)
      deps.why = [dep0, dep.why_me_or_cache(self, None)]

    deps = deps.populate('eqratio', [m, n, p, q, z, w, x, y])
    self.make_equal(mn_pq, zw_xy, deps=deps)
    self.make_equal(xy_zw, pq_mn, deps=deps)
    self.make_equal(_mnzw_, _pqxy_, deps=deps)

    deps.algebra = mn._val, pq._val, zw._val, xy._val
    self.cache_dep('eqratio', [m, n, p, q, z, w, x, y], deps)

    if is_equal:
      return []
    return [deps]

  def maybe_make_equal_pairs30(
      self,
      a: Point, b: Point, c: Point, d: Point, m: Point, n: Point, p: Point, q: Point, x:Point, y:Point, z:Point, w:Point,
      ab: Segment, cd: Segment, mn: Segment, pq: Segment, xy: Segment, zw: Segment,
      deps: EmptyDependency,
  ) -> Optional[list[Dependency]]:
    """Add ab/cd * mn/pq * xy/zw = 1 in case either two of (ab,cd,mn,pq,xy,zw) are equal."""
    level = deps.level
    if self.is_equal(ab, cd, level):
      return self.make_equal_pairs30(a, b, c, d, m, n, p, q, x, y, z, w, ab, cd, mn, pq, xy, zw, deps)
    elif self.is_equal(ab, pq, level):
      return self.make_equal_pairs30(a, b, p, q, m, n, c, d, x, y, z, w, ab, pq, mn, cd, xy, zw, deps)
    elif self.is_equal(ab, zw, level):
      return self.make_equal_pairs30(a, b, z, w, m, n, p, q, x, y, c, d, ab, zw, mn, pq, xy, cd, deps)
    elif self.is_equal(mn, cd, level):
      return self.make_equal_pairs30(m, n, c, d, a, b, p, q, x, y, z, w, mn, cd, ab, pq, xy, zw, deps)
    elif self.is_equal(mn, pq, level):
      return self.make_equal_pairs30(m, n, p, q, a, b, c, d, x, y, z, w, mn, pq, ab, cd, xy, zw, deps)
    elif self.is_equal(mn, zw, level):
      return self.make_equal_pairs30(m, n, z, w, a, b, p, q, x, y, c, d, mn, zw, ab, pq, xy, cd, deps)
    elif self.is_equal(xy, cd, level):
      return self.make_equal_pairs30(x, y, c, d, m, n, p, q, a, b, z, w, xy, cd, mn, pq, ab, zw, deps)
    elif self.is_equal(xy, pq, level):
      return self.make_equal_pairs30(x, y, p, q, m, n, c, d, a, b, z, w, xy, pq, mn, cd, ab, zw, deps)
    elif self.is_equal(xy, zw, level):
      return self.make_equal_pairs30(x, y, z, w, m, n, p, q, a, b, c, d, xy, zw, mn, pq, ab, cd, deps)
    else:
      return None

  def _add_eqratio30(
      self,
      a: Point,
      b: Point,
      c: Point,
      d: Point,
      m: Point,
      n: Point,
      p: Point,
      q: Point,
      x: Point,
      y: Point,
      z: Point,
      w: Point,
      ab: Segment,
      cd: Segment,
      mn: Segment,
      pq: Segment,
      xy: Segment,
      zw: Segment,
      deps: EmptyDependency,
  ) -> list[Dependency]:
    """Add a new eqratio from 12 points (core)."""
    if deps:
      deps = deps.copy()

    args = [a, b, c, d, m, n, p, q, x, y, z, w]
    i = 0
    for u, v, uv in [(a, b, ab), (c, d, cd), (m, n, mn), (p, q, pq), (x, y, xy), (z, w, zw)]:
      if {u, v} == set(uv.points):
        continue
      u_, v_ = list(uv.points)
      if deps:
        deps = deps.extend(self, 'eqratio', list(args), 'cong', [u, v, u_, v_])
      args[2 * i - 2] = u_
      args[2 * i - 1] = v_

    add = []

    ab_cd, cd_ab, why1 = self._get_or_create_ratio(ab, cd, deps=None)
    mnxy_pqzw, pqzw_mnxy, why2 = self._get_or_create_ratio_pro(mn, pq, xy, zw, deps=None)

    why = why1 + why2
    if why:
      dep0 = deps.populate('eqratio30', args)
      deps = EmptyDependency(level=deps.level, rule_name=None)
      deps.why = [dep0] + why

    lab, lcd = ab_cd._l
    lpmnxy, lppqzw = mnxy_pqzw._lp
    lmn, lxy = lpmnxy._l
    lpq, lzw = lppqzw._l

    a, b = lab._obj.points
    c, d = lcd._obj.points
    m, n = lmn._obj.points
    p, q = lpq._obj.points
    x, y = lxy._obj.points
    z, w = lzw._obj.points

    is_eq1 = self.is_equal(ab_cd, mnxy_pqzw)
    deps1 = None
    if deps:
      deps1 = deps.populate('eqratio30', [a, b, c, d, m, n, p, q, x, y, z, w])
      deps1.algebra = [ab._val, cd._val, mn._val, pq._val, xy._val, zw._val]
    if not is_eq1:
      add += [deps1]
    self.cache_dep('eqratio30', [a, b, c, d, m, n, p, q, x, y, z, w], deps1)
    self.make_equal(ab_cd, mnxy_pqzw, deps=deps1)

    is_eq2 = self.is_equal(cd_ab, pqzw_mnxy)
    deps2 = None
    if deps:
      deps2 = deps.populate('eqratio30', [c, d, a, b, p, q, m, n, z, w, x, y])
      deps2.algebra = [cd._val, ab._val, pq._val, mn._val, zw._val, xy._val]
    if not is_eq2:
      add += [deps2]
    self.cache_dep('eqratio30', [c, d, a, b, p, q, m, n, z, w, x, y], deps2)
    self.make_equal(cd_ab, pqzw_mnxy, deps=deps2)
    return add

  def add_eqratio30(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add a new eqratio30 from 12 points."""
    if deps:
      deps = deps.copy()
    a, b, c, d, m, n, p, q, x, y, z, w = points
    ab = self._get_or_create_segment(a, b, deps=None)
    cd = self._get_or_create_segment(c, d, deps=None)
    mn = self._get_or_create_segment(m, n, deps=None)
    pq = self._get_or_create_segment(p, q, deps=None)
    xy = self._get_or_create_segment(x, y, deps=None)
    zw = self._get_or_create_segment(z, w, deps=None)

    
    self.connect_val(ab, deps=None)
    self.connect_val(cd, deps=None)
    self.connect_val(mn, deps=None)
    self.connect_val(pq, deps=None)
    self.connect_val(xy, deps=None)
    self.connect_val(zw, deps=None)

    add = self.maybe_make_equal_pairs30(
        a, b, c, d, m, n, p, q, x, y, z, w, ab, cd, mn, pq, xy, zw, deps
    )

    if add is not None:
      return add

    add = []

    _abmn_, why1 = self._get_or_create_length_pro(ab, mn, deps=None)
    _abxy_, why2 = self._get_or_create_length_pro(ab, xy, deps=None)
    _mnxy_, why3 = self._get_or_create_length_pro(mn, xy, deps=None)
    _cdpq_, why4 = self._get_or_create_length_pro(cd, pq, deps=None)
    _cdzw_, why5 = self._get_or_create_length_pro(cd, zw, deps=None)
    _pqzw_, why6 = self._get_or_create_length_pro(pq, zw, deps=None)

    self.connect_val(_abmn_, deps=why1)
    self.connect_val(_abxy_, deps=why2)
    self.connect_val(_mnxy_, deps=why3)
    self.connect_val(_cdpq_, deps=why4)
    self.connect_val(_cdzw_, deps=why5)
    self.connect_val(_pqzw_, deps=why6)

    add = []
    add += self._add_eqratio30(a, b, c, d, m, n, p, q, x, y, z, w, ab, cd, mn, pq, xy, zw, deps)
    add += self._add_eqratio30(a, b, p, q, m, n, c, d, x, y, z, w, ab, pq, mn, cd, xy, zw, deps)
    add += self._add_eqratio30(a, b, z, w, m, n, p, q, x, y, c, d, ab, zw, mn, pq, xy, cd, deps)
    add += self._add_eqratio30(m, n, c, d, a, b, p, q, x, y, z, w, mn, cd, ab, pq, xy, zw, deps)
    add += self._add_eqratio30(m, n, p, q, a, b, c, d, x, y, z, w, mn, pq, ab, cd, xy, zw, deps)
    add += self._add_eqratio30(m, n, z, w, a, b, c, d, x, y, p, q, mn, zw, ab, cd, xy, pq, deps)
    add += self._add_eqratio30(x, y, c, d, a, b, p, q, m, n, z, w, xy, cd, ab, pq, mn, zw, deps)
    add += self._add_eqratio30(x, y, p, q, a, b, c, d, m, n, z, w, xy, pq, ab, cd, mn, zw, deps)
    add += self._add_eqratio30(x, y, z, w, a, b, c, d, m, n, p, q, xy, zw, ab, cd, mn, pq, deps)
    return add

  def check_rconst(self, points: list[Point], verbose: bool = False) -> bool:
    """Check whether a ratio is equal to some given constant."""
    _ = verbose
    a, b, c, d, nd = points
    if isinstance(nd, str):
      name = nd
    else:
      name = nd.name
    num, den = name.split('/')
    rat, _ = self.get_or_create_const_rat(int(num), int(den))

    ab = self._get_segment(a, b)
    cd = self._get_segment(c, d)

    if not ab or not cd:
      return False

    if not (ab.val and cd.val):
      return False

    for rat1, _, _ in gm.all_ratios(ab._val, cd._val):
      if self.is_equal(rat1, rat):
        return True
    return False

  def check_rcompute(self, points: list[Point]) -> bool:
    """Check whether a ratio is equal to some constant."""
    a, b, c, d = points
    ab = self._get_segment(a, b)
    cd = self._get_segment(c, d)

    if not ab or not cd:
      return False

    if not (ab.val and cd.val):
      return False

    for rat0 in self.rconst.values():
      for rat in rat0.val.neighbors(Ratio):
        l1, l2 = rat.lengths
        if ab.val == l1 and cd.val == l2:
          return True
    return False

  def check_eqratio(self, points: list[Point]) -> bool:
    """Check if 8 points make an eqratio predicate."""
    a, b, c, d, m, n, p, q = points

    if {a, b} == {c, d} and {m, n} == {p, q}:
      return True
    if {a, b} == {m, n} and {c, d} == {p, q}:
      return True

    ab = self._get_segment(a, b)
    cd = self._get_segment(c, d)
    mn = self._get_segment(m, n)
    pq = self._get_segment(p, q)

    if {a, b} == {c, d} and mn and pq and self.is_equal(mn, pq):
      return True
    if {a, b} == {m, n} and cd and pq and self.is_equal(cd, pq):
      return True
    if {p, q} == {m, n} and ab and cd and self.is_equal(ab, cd):
      return True
    if {p, q} == {c, d} and ab and mn and self.is_equal(ab, mn):
      return True

    if not ab or not cd or not mn or not pq:
      return False

    if self.is_equal(ab, cd) and self.is_equal(mn, pq):
      return True
    if self.is_equal(ab, mn) and self.is_equal(cd, pq):
      return True

    if not (ab.val and cd.val and mn.val and pq.val):
      return False

    if (ab.val, cd.val) == (mn.val, pq.val) or (ab.val, mn.val) == (
        cd.val,
        pq.val,
    ):
      return True

    for rat1, _, _ in gm.all_ratios(ab._val, cd._val):
      for rat2, _, _ in gm.all_ratios(mn._val, pq._val):
        if self.is_equal(rat1, rat2):
          return True
    return False

  def check_eqratio30(self, points: list[Point]) -> bool:
    """Check if 12 points make an eqratio30 predicate."""
    a, b, c, d, m, n, p, q, x, y, z, w = points

    if {a, b} == {c, d}:
      return self.check_eqratio([m, n, p, q, z, w, x, y])
    if {a, b} == {p, q}:
      return self.check_eqratio([m, n, c, d, z, w, x, y])
    if {a, b} == {z, w}:
      return self.check_eqratio([m, n, p, q, c, d, x, y])
    if {m, n} == {c, d}:
      return self.check_eqratio([a, b, p, q, z, w, x, y])
    if {m, n} == {p, q}:
      return self.check_eqratio([a, b, c, d, z, w, x, y])
    if {m, n} == {z, w}:
      return self.check_eqratio([a, b, p, q, c, d, x, y])     
    if {x, y} == {c, d}:
      return self.check_eqratio([a, b, p, q, z, w, m, n])
    if {x, y} == {p, q}:
      return self.check_eqratio([a, b, c, d, z, w, m, n])
    if {x, y} == {z, w}:
      return self.check_eqratio([a, b, p, q, c, d, m, n])

    ab = self._get_segment(a, b)
    cd = self._get_segment(c, d)
    mn = self._get_segment(m, n)
    pq = self._get_segment(p, q)
    xy = self._get_segment(x, y)
    zw = self._get_segment(z, w)

    if not (ab and cd and mn and pq and xy and zw):
      return False
    
    if self.is_equal(ab, cd):
      return self.check_eqratio([m, n, p, q, z, w, x, y])
    if self.is_equal(ab, pq):
      return self.check_eqratio([m, n, c, d, z, w, x, y])
    if self.is_equal(ab, zw):
      return self.check_eqratio([m, n, p, q, c, d, x, y])
    if self.is_equal(mn, cd):
      return self.check_eqratio([a, b, p, q, z, w, x, y])
    if self.is_equal(mn, pq):
      return self.check_eqratio([a, b, c, d, z, w, x, y])
    if self.is_equal(mn, zw):
      return self.check_eqratio([a, b, p, q, c, d, x, y])     
    if self.is_equal(xy, cd):
      return self.check_eqratio([a, b, p, q, z, w, m, n])
    if self.is_equal(xy, pq):
      return self.check_eqratio([a, b, c, d, z, w, m, n])
    if self.is_equal(xy, zw):
      return self.check_eqratio([a, b, p, q, c, d, m, n])

    if not (ab.val and cd.val and mn.val and pq.val and xy.val and zw.val):
      return False

    if Multiset([ab.val, mn.val, xy.val]) == Multiset([cd.val, pq.val, zw.val]):
      return True

    _pqzw_, _ = self._get_or_create_length_pro(pq, zw, deps=None)
    _mnxy_, _ = self._get_or_create_length_pro(mn, xy, deps=None)

    for rat1, _, _ in gm.all_ratios(ab._val, cd._val):
      for ratp2, _, _ in gm.all_ratios2(_pqzw_._val, _mnxy_._val):
        if self.is_equal(rat1, ratp2):
          return True
    return False

  def add_simtri_check(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    if nm.same_clock(*[p.num for p in points]):
      return self.add_simtri(points, deps)
    return self.add_simtri2(points, deps)

  def add_contri_check(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    if nm.same_clock(*[p.num for p in points]):
      return self.add_contri(points, deps)
    return self.add_contri2(points, deps)

  def enum_sides(
      self, points: list[Point]
  ) -> Generator[list[Point], None, None]:
    a, b, c, x, y, z = points
    yield [a, b, x, y]
    yield [b, c, y, z]
    yield [c, a, z, x]

  def enum_triangle(
      self, points: list[Point]
  ) -> Generator[list[Point], None, None]:
    a, b, c, x, y, z = points
    yield [a, b, a, c, x, y, x, z]
    yield [b, a, b, c, y, x, y, z]
    yield [c, a, c, b, z, x, z, y]

  def enum_triangle2(
      self, points: list[Point]
  ) -> Generator[list[Point], None, None]:
    a, b, c, x, y, z = points
    yield [a, b, a, c, x, z, x, y]
    yield [b, a, b, c, y, z, y, x]
    yield [c, a, c, b, z, y, z, x]

  def add_simtri(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add two similar triangles."""
    add = []
    hashs = [d.hashed() for d in deps.why]

    for args in self.enum_triangle(points):
      if problem.hashed('eqangle6', args) in hashs:
        continue
      add += self.add_eqangle(args, deps=deps)

    for args in self.enum_triangle(points):
      if problem.hashed('eqratio6', args) in hashs:
        continue
      add += self.add_eqratio(args, deps=deps)

    return add

  def check_simtri(self, points: list[Point]) -> bool:
    a, b, c, x, y, z = points
    return self.check_eqangle([a, b, a, c, x, y, x, z]) and self.check_eqangle(
        [b, a, b, c, y, x, y, z]
    )

  def add_simtri2(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add two similar reflected triangles."""
    add = []
    hashs = [d.hashed() for d in deps.why]
    for args in self.enum_triangle2(points):
      if problem.hashed('eqangle6', args) in hashs:
        continue
      add += self.add_eqangle(args, deps=deps)

    for args in self.enum_triangle(points):
      if problem.hashed('eqratio6', args) in hashs:
        continue
      add += self.add_eqratio(args, deps=deps)

    return add

  def add_contri(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add two congruent triangles."""
    add = []
    hashs = [d.hashed() for d in deps.why]
    for args in self.enum_triangle(points):
      if problem.hashed('eqangle6', args) in hashs:
        continue
      add += self.add_eqangle(args, deps=deps)

    for args in self.enum_sides(points):
      if problem.hashed('cong', args) in hashs:
        continue
      add += self.add_cong(args, deps=deps)
    return add

  def check_contri(self, points: list[Point]) -> bool:
    a, b, c, x, y, z = points
    return (
        self.check_cong([a, b, x, y])
        and self.check_cong([b, c, y, z])
        and self.check_cong([c, a, z, x])
    )

  def add_contri2(
      self, points: list[Point], deps: EmptyDependency
  ) -> list[Dependency]:
    """Add two congruent reflected triangles."""
    add = []
    hashs = [d.hashed() for d in deps.why]
    for args in self.enum_triangle2(points):
      if problem.hashed('eqangle6', args) in hashs:
        continue
      add += self.add_eqangle(args, deps=deps)

    for args in self.enum_sides(points):
      if problem.hashed('cong', args) in hashs:
        continue
      add += self.add_cong(args, deps=deps)

    return add

  def in_cache(self, name: str, args: list[Point]) -> bool:
    return problem.hashed(name, args) in self.cache

  def cache_dep(
      self, name: str, args: list[Point], premises: list[Dependency]
  ) -> None:
    hashed = problem.hashed(name, args)
    if hashed in self.cache:
      return
    self.cache[hashed] = premises

  def all_same_line(
      self, a: Point, b: Point
  ) -> Generator[tuple[Point, Point], None, None]:
    ab = self._get_line(a, b)
    if ab is None:
      return
    for p1, p2 in utils.comb2(ab.neighbors(Point)):
      if {p1, p2} != {a, b}:
        yield p1, p2

  def all_same_angle(
      self, a: Point, b: Point, c: Point, d: Point
  ) -> Generator[tuple[Point, Point, Point, Point], None, None]:
    for x, y in self.all_same_line(a, b):
      for m, n in self.all_same_line(c, d):
        yield x, y, m, n

  def additionally_draw(self, name: str, args: list[Point]) -> None:
    """Draw some extra line/circles for illustration purpose."""

    if name in ['circle']:
      center, point = args[:2]
      circle = self.new_node(Circle, f'({center.name},{point.name})')
      circle.num = nm.Circle(center.num, p1=point.num)
      circle.points = center, point

    if name in ['on_circle', 'tangent']:
      center, point = args[-2:]
      circle = self.new_node(Circle, f'({center.name},{point.name})')
      circle.num = nm.Circle(center.num, p1=point.num)
      circle.points = center, point

    if name in ['incenter', 'excenter', 'incenter2', 'excenter2']:
      d, a, b, c = [x for x in args[-4:]]
      a, b, c = sorted([a, b, c], key=lambda x: x.name.lower())
      circle = self.new_node(Circle, f'({d.name},h.{a.name}{b.name})')
      p = d.num.foot(nm.Line(a.num, b.num))
      circle.num = nm.Circle(d.num, p1=p)
      circle.points = d, a, b, c

    if name in ['cc_tangent']:
      o, a, w, b = args[-4:]
      c1 = self.new_node(Circle, f'({o.name},{a.name})')
      c1.num = nm.Circle(o.num, p1=a.num)
      c1.points = o, a

      c2 = self.new_node(Circle, f'({w.name},{b.name})')
      c2.num = nm.Circle(w.num, p1=b.num)
      c2.points = w, b

    if name in ['ninepoints']:
      a, b, c = args[-3:]
      a, b, c = sorted([a, b, c], key=lambda x: x.name.lower())
      circle = self.new_node(Circle, f'(,m.{a.name}{b.name}{c.name})')
      p1 = (b.num + c.num) * 0.5
      p2 = (c.num + a.num) * 0.5
      p3 = (a.num + b.num) * 0.5
      circle.num = nm.Circle(p1=p1, p2=p2, p3=p3)
      circle.points = (None, None, a, b, c)

    if name in ['2l1c']:
      a, b, c, o = args[:4]
      a, b, c = sorted([a, b, c], key=lambda x: x.name.lower())
      circle = self.new_node(Circle, f'({o.name},{a.name}{b.name}{c.name})')
      circle.num = nm.Circle(p1=a.num, p2=b.num, p3=c.num)
      circle.points = (a, b, c)

  def add_clause(
      self,
      clause: problem.Clause,
      plevel: int,
      definitions: dict[str, problem.Definition],
      verbose: int = False,
  ) -> tuple[list[Dependency], int]:
    """Add a new clause of construction, e.g. a new excenter."""
    existing_points = self.all_points()
    new_points = [Point(name) for name in clause.points]

    new_points_dep_points = set()
    new_points_dep = []

    # Step 1: check for all deps.
    for c in clause.constructions:
      cdef = definitions[c.name]

      if len(cdef.construction.args) != len(c.args):
        if len(cdef.construction.args) - len(c.args) == len(clause.points):
          c.args = clause.points + c.args
        else:
          correct_form = ' '.join(cdef.points + ['=', c.name] + cdef.args)
          raise ValueError('Argument mismatch. ' + correct_form)

      mapping = dict(zip(cdef.construction.args, c.args))
      c_name = 'midp' if c.name == 'midpoint' else c.name
      deps = EmptyDependency(level=0, rule_name=problem.CONSTRUCTION_RULE)
      deps.construction = Dependency(c_name, c.args, rule_name=None, level=0)

      for d in cdef.deps.constructions:
        args = self.names2points([mapping[a] for a in d.args])
        new_points_dep_points.update(args)
        if not self.check(d.name, args):
          raise DepCheckFailError(
              d.name + ' ' + ' '.join([x.name for x in args])
          )
        deps.why += [
            Dependency(
                d.name, args, rule_name=problem.CONSTRUCTION_RULE, level=0
            )
        ]

      new_points_dep += [deps]

    # Step 2: draw.
    def range_fn() -> (
        list[Union[nm.Point, nm.Line, nm.Circle, nm.HalfLine, nm.HoleCircle]]
    ):
      to_be_intersected = []
      for c in clause.constructions:
        cdef = definitions[c.name]
        mapping = dict(zip(cdef.construction.args, c.args))
        for n in cdef.numerics:
          args = [mapping[a] for a in n.args]
          args = list(map(lambda x: self.get(x, lambda: int(x)), args))
          to_be_intersected += nm.sketch(n.name, args)

      return to_be_intersected

    is_total_free = (
        len(clause.constructions) == 1 and clause.constructions[0].name in FREE
    )
    is_semi_free = (
        len(clause.constructions) == 1
        and clause.constructions[0].name in INTERSECT
    )

    existing_points = [p.num for p in existing_points]

    def draw_fn() -> list[nm.Point]:
      to_be_intersected = range_fn()
      return nm.reduce(to_be_intersected, existing_points)

    rely_on = set()
    for c in clause.constructions:
      cdef = definitions[c.name]
      mapping = dict(zip(cdef.construction.args, c.args))
      for n in cdef.numerics:
        args = [mapping[a] for a in n.args]
        args = list(map(lambda x: self.get(x, lambda: int(x)), args))
        rely_on.update([a for a in args if isinstance(a, Point)])

    for p in rely_on:
      p.change.update(new_points)

    nums = draw_fn()
    for p, num, num0 in zip(new_points, nums, clause.nums):
      p.co_change = new_points
      if isinstance(num0, nm.Point):
        num = num0
      elif isinstance(num0, (tuple, list)):
        x, y = num0
        num = nm.Point(x, y)

      p.num = num

    # check two things.
    if nm.check_too_close(nums, existing_points):
      raise PointTooCloseError()
    if nm.check_too_far(nums, existing_points):
      raise PointTooFarError()

    # Commit: now that all conditions are passed.
    # add these points to current graph.
    for p in new_points:
      self._name2point[p.name] = p
      self._name2node[p.name] = p
      self.type2nodes[Point].append(p)

    for p in new_points:
      p.why = sum([d.why for d in new_points_dep], [])  # to generate txt logs.
      p.group = new_points
      p.dep_points = new_points_dep_points
      p.dep_points.update(new_points)
      p.plevel = plevel

    # movement dependency:
    rely_dict_0 = defaultdict(lambda: [])

    for c in clause.constructions:
      cdef = definitions[c.name]
      mapping = dict(zip(cdef.construction.args, c.args))
      for p, ps in cdef.rely.items():
        p = mapping[p]
        ps = [mapping[x] for x in ps]
        rely_dict_0[p].append(ps)

    rely_dict = {}
    for p, pss in rely_dict_0.items():
      ps = sum(pss, [])
      if len(pss) > 1:
        ps = [x for x in ps if x != p]

      p = self._name2point[p]
      ps = self.names2nodes(ps)
      rely_dict[p] = ps

    for p in new_points:
      p.rely_on = set(rely_dict.get(p, []))
      for x in p.rely_on:
        if not hasattr(x, 'base_rely_on'):
          x.base_rely_on = set()
      p.base_rely_on = set.union(*[x.base_rely_on for x in p.rely_on] + [set()])
      if is_total_free or is_semi_free:
        p.rely_on.add(p)
        p.base_rely_on.add(p)

    plevel_done = set()
    added = []
    basics = []
    # Step 3: build the basics.
    for c, deps in zip(clause.constructions, new_points_dep):
      cdef = definitions[c.name]
      mapping = dict(zip(cdef.construction.args, c.args))

      # not necessary for proofing, but for visualization.
      c_args = list(map(lambda x: self.get(x, lambda: int(x)), c.args))
      self.additionally_draw(c.name, c_args)

      for points, bs in cdef.basics:
        if points:
          points = self.names2nodes([mapping[p] for p in points])
          points = [p for p in points if p not in plevel_done]
          for p in points:
            p.plevel = plevel
          plevel_done.update(points)
          plevel += 1
        else:
          continue

        for b in bs:
          if b.name != 'rconst':
            args = [mapping[a] for a in b.args]
          else:
            num, den = map(int, b.args[-2:])
            rat, _ = self.get_or_create_const_rat(num, den)
            args = [mapping[a] for a in b.args[:-2]] + [rat.name]

          args = list(map(lambda x: self.get(x, lambda: int(x)), args))

          adds = self.add_piece(name=b.name, args=args, deps=deps)
          basics.append((b.name, args, deps))
          if adds:
            added += adds
            for add in adds:
              self.cache_dep(add.name, add.args, add)

    assert len(plevel_done) == len(new_points)
    for p in new_points:
      p.basics = basics

    return added, plevel

  def all_eqangle_same_lines(self) -> Generator[tuple[Point, ...], None, None]:
    for l1, l2 in utils.perm2(self.type2nodes[Line]):
      for a, b, c, d, e, f, g, h in utils.all_8points(l1, l2, l1, l2):
        if (a, b, c, d) != (e, f, g, h):
          yield a, b, c, d, e, f, g, h

  def all_eqangles_distinct_linepairss(
      self,
  ) -> Generator[tuple[Line, ...], None, None]:
    """No eqangles betcause para-para, or para-corresponding, or same."""

    for measure in self.type2nodes[Measure]:
      angs = measure.neighbors(Angle)
      line_pairss = []
      for ang in angs:
        d1, d2 = ang.directions
        if d1 is None or d2 is None:
          continue
        l1s = d1.neighbors(Line)
        l2s = d2.neighbors(Line)
        # Any pair in this is para-para.
        para_para = list(utils.cross(l1s, l2s))
        line_pairss.append(para_para)

      for pairs1, pairs2 in utils.comb2(line_pairss):
        for pair1, pair2 in utils.cross(pairs1, pairs2):
          (l1, l2), (l3, l4) = pair1, pair2
          yield l1, l2, l3, l4

  def all_eqangles_8points(self) -> Generator[tuple[Point, ...], None, None]:
    """List all sets of 8 points that make two equal angles."""
    # Case 1: (l1-l2) = (l3-l4), including because l1//l3, l2//l4 (para-para)
    angss = []
    for measure in self.type2nodes[Measure]:
      angs = measure.neighbors(Angle)
      angss.append(angs)

    # include the angs that do not have any measure.
    angss.extend([[ang] for ang in self.type2nodes[Angle] if ang.val is None])

    line_pairss = []
    for angs in angss:
      line_pairs = set()
      for ang in angs:
        d1, d2 = ang.directions
        if d1 is None or d2 is None:
          continue
        l1s = d1.neighbors(Line)
        l2s = d2.neighbors(Line)
        line_pairs.update(set(utils.cross(l1s, l2s)))
      line_pairss.append(line_pairs)

    # include (d1, d2) in which d1 does not have any angles.
    noang_ds = [d for d in self.type2nodes[Direction] if not d.neighbors(Angle)]

    for d1 in noang_ds:
      for d2 in self.type2nodes[Direction]:
        if d1 == d2:
          continue
        l1s = d1.neighbors(Line)
        l2s = d2.neighbors(Line)
        if len(l1s) < 2 and len(l2s) < 2:
          continue
        line_pairss.append(set(utils.cross(l1s, l2s)))
        line_pairss.append(set(utils.cross(l2s, l1s)))

    # Case 2: d1 // d2 => (d1-d3) = (d2-d3)
    # include lines that does not have any direction.
    nodir_ls = [l for l in self.type2nodes[Line] if l.val is None]

    for line in nodir_ls:
      for d in self.type2nodes[Direction]:
        l1s = d.neighbors(Line)
        if len(l1s) < 2:
          continue
        l2s = [line]
        line_pairss.append(set(utils.cross(l1s, l2s)))
        line_pairss.append(set(utils.cross(l2s, l1s)))

    record = set()
    for line_pairs in line_pairss:
      for pair1, pair2 in utils.perm2(list(line_pairs)):
        (l1, l2), (l3, l4) = pair1, pair2
        if l1 == l2 or l3 == l4:
          continue
        if (l1, l2) == (l3, l4):
          continue
        if (l1, l2, l3, l4) in record:
          continue
        record.add((l1, l2, l3, l4))
        for a, b, c, d, e, f, g, h in utils.all_8points(l1, l2, l3, l4):
          yield (a, b, c, d, e, f, g, h)

    for a, b, c, d, e, f, g, h in self.all_eqangle_same_lines():
      yield a, b, c, d, e, f, g, h

  def all_eqangles_6points(self) -> Generator[tuple[Point, ...], None, None]:
    """List all sets of 6 points that make two equal angles."""
    record = set()
    for a, b, c, d, e, f, g, h in self.all_eqangles_8points():
      if (
          a not in (c, d)
          and b not in (c, d)
          or e not in (g, h)
          and f not in (g, h)
      ):
        continue

      if b in (c, d):
        a, b = b, a  # now a in c, d
      if f in (g, h):
        e, f = f, e  # now e in g, h
      if a == d:
        c, d = d, c  # now a == c
      if e == h:
        g, h = h, g  # now e == g
      if (a, b, c, d, e, f, g, h) in record:
        continue
      record.add((a, b, c, d, e, f, g, h))
      yield a, b, c, d, e, f, g, h  # where a==c, e==g

  def all_paras(self) -> Generator[tuple[Point, ...], None, None]:
    for d in self.type2nodes[Direction]:
      for l1, l2 in utils.perm2(d.neighbors(Line)):
        for a, b, c, d in utils.all_4points(l1, l2):
          yield a, b, c, d

  def all_perps(self) -> Generator[tuple[Point, ...], None, None]:
    for ang in self.vhalfpi.neighbors(Angle):
      d1, d2 = ang.directions
      if d1 is None or d2 is None:
        continue
      if d1 == d2:
        continue
      for l1, l2 in utils.cross(d1.neighbors(Line), d2.neighbors(Line)):
        for a, b, c, d in utils.all_4points(l1, l2):
          yield a, b, c, d

  def all_congs(self) -> Generator[tuple[Point, ...], None, None]:
    for l in self.type2nodes[Length]:
      for s1, s2 in utils.perm2(l.neighbors(Segment)):
        (a, b), (c, d) = s1.points, s2.points
        for x, y in [(a, b), (b, a)]:
          for m, n in [(c, d), (d, c)]:
            yield x, y, m, n

  def all_eqratios_8points(self) -> Generator[tuple[Point, ...], None, None]:
    """List all sets of 8 points that make two equal ratios."""
    ratss = []
    for value in self.type2nodes[Value]:
      rats = value.neighbors(Ratio)
      ratss.append(rats)

    # include the rats that do not have any val.
    ratss.extend([[rat] for rat in self.type2nodes[Ratio] if rat.val is None])

    seg_pairss = []
    for rats in ratss:
      seg_pairs = set()
      for rat in rats:
        l1, l2 = rat.lengths
        if l1 is None or l2 is None:
          continue
        s1s = l1.neighbors(Segment)
        s2s = l2.neighbors(Segment)
        seg_pairs.update(utils.cross(s1s, s2s))
      seg_pairss.append(seg_pairs)

    # include (l1, l2) in which l1 does not have any ratio.
    norat_ls = [l for l in self.type2nodes[Length] if not l.neighbors(Ratio)]

    for l1 in norat_ls:
      for l2 in self.type2nodes[Length]:
        if l1 == l2:
          continue
        s1s = l1.neighbors(Segment)
        s2s = l2.neighbors(Segment)
        if len(s1s) < 2 and len(s2s) < 2:
          continue
        seg_pairss.append(set(utils.cross(s1s, s2s)))
        seg_pairss.append(set(utils.cross(s2s, s1s)))

    # include Seg that does not have any Length.
    nolen_ss = [s for s in self.type2nodes[Segment] if s.val is None]

    for seg in nolen_ss:
      for l in self.type2nodes[Length]:
        s1s = l.neighbors(Segment)
        if len(s1s) == 1:
          continue
        s2s = [seg]
        seg_pairss.append(set(utils.cross(s1s, s2s)))
        seg_pairss.append(set(utils.cross(s2s, s1s)))

    record = set()
    for seg_pairs in seg_pairss:
      for pair1, pair2 in utils.perm2(list(seg_pairs)):
        (s1, s2), (s3, s4) = pair1, pair2
        if s1 == s2 or s3 == s4:
          continue
        if (s1, s2) == (s3, s4):
          continue
        if (s1, s2, s3, s4) in record:
          continue
        record.add((s1, s2, s3, s4))
        a, b = s1.points
        c, d = s2.points
        e, f = s3.points
        g, h = s4.points

        for x, y in [(a, b), (b, a)]:
          for z, t in [(c, d), (d, c)]:
            for m, n in [(e, f), (f, e)]:
              for p, q in [(g, h), (h, g)]:
                yield (x, y, z, t, m, n, p, q)

    segss = []
    # finally the list of ratios that is equal to 1.0
    for length in self.type2nodes[Length]:
      segs = length.neighbors(Segment)
      segss.append(segs)

    segs_pair = list(utils.perm2(list(segss)))
    segs_pair += list(zip(segss, segss))
    for segs1, segs2 in segs_pair:
      for s1, s2 in utils.perm2(list(segs1)):
        for s3, s4 in utils.perm2(list(segs2)):
          if (s1, s2) == (s3, s4) or (s1, s3) == (s2, s4):
            continue
          if (s1, s2, s3, s4) in record:
            continue
          record.add((s1, s2, s3, s4))
          a, b = s1.points
          c, d = s2.points
          e, f = s3.points
          g, h = s4.points

          for x, y in [(a, b), (b, a)]:
            for z, t in [(c, d), (d, c)]:
              for m, n in [(e, f), (f, e)]:
                for p, q in [(g, h), (h, g)]:
                  yield (x, y, z, t, m, n, p, q)

  def all_eqratios_6points(self) -> Generator[tuple[Point, ...], None, None]:
    """List all sets of 6 points that make two equal angles."""
    record = set()
    for a, b, c, d, e, f, g, h in self.all_eqratios_8points():
      if (
          a not in (c, d)
          and b not in (c, d)
          or e not in (g, h)
          and f not in (g, h)
      ):
        continue
      if b in (c, d):
        a, b = b, a
      if f in (g, h):
        e, f = f, e
      if a == d:
        c, d = d, c
      if e == h:
        g, h = h, g
      if (a, b, c, d, e, f, g, h) in record:
        continue
      record.add((a, b, c, d, e, f, g, h))
      yield a, b, c, d, e, f, g, h  # now a==c, e==g

  def all_cyclics(self) -> Generator[tuple[Point, ...], None, None]:
    for c in self.type2nodes[Circle]:
      for x, y, z, t in utils.perm4(c.neighbors(Point)):
        yield x, y, z, t

  def all_cyclics6(self) -> Generator[tuple[Point, ...], None, None]:
    for c in self.type2nodes[Circle]:
      for x, y, z, t, u, v in utils.perm6(c.neighbors(Point)):
        yield x, y, z, t, u, v
    
  def all_cyclics5(self) -> Generator[tuple[Point, ...], None, None]:
    for c in self.type2nodes[Circle]:
      for x, y, z, t, u in utils.perm5(c.neighbors(Point)):
        yield x, y, z, t, u

  def all_colls(self) -> Generator[tuple[Point, ...], None, None]:
    for l in self.type2nodes[Line]:
      for x, y, z in utils.perm3(l.neighbors(Point)):
        yield x, y, z

  def all_midps(self) -> Generator[tuple[Point, ...], None, None]:
    for l in self.type2nodes[Line]:
      for a, b, c in utils.perm3(l.neighbors(Point)):
        if self.check_cong([a, b, a, c]):
          yield a, b, c

  def all_circles(self) -> Generator[tuple[Point, ...], None, None]:
    for l in self.type2nodes[Length]:
      p2p = defaultdict(list)
      for s in l.neighbors(Segment):
        a, b = s.points
        p2p[a].append(b)
        p2p[b].append(a)
      for p, ps in p2p.items():
        if len(ps) >= 3:
          for a, b, c in utils.perm3(ps):
            yield p, a, b, c

  def two_points_on_direction(self, d: Direction) -> tuple[Point, Point]:
    l = d.neighbors(Line)[0]
    p1, p2 = l.neighbors(Point)[:2]
    return p1, p2

  def two_points_of_length(self, l: Length) -> tuple[Point, Point]:
    s = l.neighbors(Segment)[0]
    p1, p2 = s.points
    return p1, p2
  


def create_consts_str(g: Graph, s: str) -> Union[Ratio, Angle]:
  if 'pi/' in s:
    n, d = s.split('pi/')
    n, d = int(n), int(d)
    p0, _ = g.get_or_create_const_ang(n, d)
  else:
    n, d = s.split('/')
    n, d = int(n), int(d)
    p0, _ = g.get_or_create_const_rat(n, d)
  return p0


def create_consts(g: Graph, p: gm.Node) -> Union[Ratio, Angle]:
  if isinstance(p, Angle):
    n, d = p.name.split('pi/')
    n, d = int(n), int(d)
    p0, _ = g.get_or_create_const_ang(n, d)
  if isinstance(p, Ratio):
    n, d = p.name.split('/')
    n, d = int(n), int(d)
    p0, _ = g.get_or_create_const_rat(n, d)
  return p0  # pylint: disable=undefined-variable
