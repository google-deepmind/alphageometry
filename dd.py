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

"""Implements Deductive Database (DD)."""

# pylint: disable=g-multiple-import,g-importing-member
from collections import defaultdict
import time
from typing import Any, Callable, Generator

import geometry as gm
import graph as gh
import graph_utils as utils
import numericals as nm
import problem as pr
from problem import Dependency, EmptyDependency


def intersect1(set1: set[Any], set2: set[Any]) -> Any:
  for x in set1:
    if x in set2:
      return x
  return None


def diff_point(l: gm.Line, a: gm.Point) -> gm.Point:
  for x in l.neighbors(gm.Point):
    if x != a:
      return x
  return None


# pylint: disable=protected-access
# pylint: disable=unused-argument


def match_eqratio_eqratio_eqratio(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqratio a b c d m n p q, eqratio c d e f p q r u => eqratio a b e f m n r u."""
  for m1 in g.type2nodes[gm.Value]:
    for m2 in g.type2nodes[gm.Value]:
      rats1 = []
      for rat in m1.neighbors(gm.Ratio):
        l1, l2 = rat.lengths
        if l1 is None or l2 is None:
          continue
        rats1.append((l1, l2))

      rats2 = []
      for rat in m2.neighbors(gm.Ratio):
        l1, l2 = rat.lengths
        if l1 is None or l2 is None:
          continue
        rats2.append((l1, l2))

      pairs = []
      for (l1, l2), (l3, l4) in utils.cross(rats1, rats2):
        if l2 == l3:
          pairs.append((l1, l2, l4))

      for (l1, l12, l2), (l3, l34, l4) in utils.comb2(pairs):
        if (l1, l12, l2) == (l3, l34, l4):
          continue
        if l1 == l2 or l3 == l4:
          continue
        if l1 == l12 or l12 == l2 or l3 == l34 or l4 == l34:
          continue
        # d12 - d1 = d34 - d3 = m1
        # d2 - d12 = d4 - d34 = m2
        # => d2 - d1 = d4 - d3 (= m1+m2)
        a, b = g.two_points_of_length(l1)
        c, d = g.two_points_of_length(l12)
        m, n = g.two_points_of_length(l3)
        p, q = g.two_points_of_length(l34)
        # eqangle a b c d m n p q
        e, f = g.two_points_of_length(l2)
        r, u = g.two_points_of_length(l4)
        yield dict(zip('abcdefmnpqru', [a, b, c, d, e, f, m, n, p, q, r, u]))


def match_eqangle_eqangle_eqangle(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqangle a b c d m n p q, eqangle c d e f p q r u => eqangle a b e f m n r u."""
  for m1 in g.type2nodes[gm.Measure]:
    for m2 in g.type2nodes[gm.Measure]:
      angs1 = []
      for ang in m1.neighbors(gm.Angle):
        d1, d2 = ang.directions
        if d1 is None or d2 is None:
          continue
        angs1.append((d1, d2))

      angs2 = []
      for ang in m2.neighbors(gm.Angle):
        d1, d2 = ang.directions
        if d1 is None or d2 is None:
          continue
        angs2.append((d1, d2))

      pairs = []
      for (d1, d2), (d3, d4) in utils.cross(angs1, angs2):
        if d2 == d3:
          pairs.append((d1, d2, d4))

      for (d1, d12, d2), (d3, d34, d4) in utils.comb2(pairs):
        if (d1, d12, d2) == (d3, d34, d4):
          continue
        if d1 == d2 or d3 == d4:
          continue
        if d1 == d12 or d12 == d2 or d3 == d34 or d4 == d34:
          continue
        # d12 - d1 = d34 - d3 = m1
        # d2 - d12 = d4 - d34 = m2
        # => d2 - d1 = d4 - d3
        a, b = g.two_points_on_direction(d1)
        c, d = g.two_points_on_direction(d12)
        m, n = g.two_points_on_direction(d3)
        p, q = g.two_points_on_direction(d34)
        # eqangle a b c d m n p q
        e, f = g.two_points_on_direction(d2)
        r, u = g.two_points_on_direction(d4)
        yield dict(zip('abcdefmnpqru', [a, b, c, d, e, f, m, n, p, q, r, u]))


def match_perp_perp_npara_eqangle(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match perp A B C D, perp E F G H, npara A B E F => eqangle A B E F C D G H."""
  dpairs = []
  for ang in g.vhalfpi.neighbors(gm.Angle):
    d1, d2 = ang.directions
    if d1 is None or d2 is None:
      continue
    dpairs.append((d1, d2))

  for (d1, d2), (d3, d4) in utils.comb2(dpairs):
    a, b = g.two_points_on_direction(d1)
    c, d = g.two_points_on_direction(d2)
    m, n = g.two_points_on_direction(d3)
    p, q = g.two_points_on_direction(d4)
    if g.check_npara([a, b, m, n]):
      if ({a, b}, {c, d}) == ({m, n}, {p, q}):
        continue
      if ({a, b}, {c, d}) == ({p, q}, {m, n}):
        continue

      yield dict(zip('ABCDEFGH', [a, b, c, d, m, n, p, q]))


def match_circle_coll_eqangle_midp(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match circle O A B C, coll M B C, eqangle A B A C O B O M => midp M B C."""
  for p, a, b, c in g.all_circles():
    ab = g._get_line(a, b)
    if ab is None:
      continue
    if ab.val is None:
      continue
    ac = g._get_line(a, c)
    if ac is None:
      continue
    if ac.val is None:
      continue
    pb = g._get_line(p, b)
    if pb is None:
      continue
    if pb.val is None:
      continue

    bc = g._get_line(b, c)
    if bc is None:
      continue
    bc_points = bc.neighbors(gm.Point, return_set=True)

    anga, _ = g._get_angle(ab.val, ac.val)

    for angp in pb.val.neighbors(gm.Angle):
      if not g.is_equal(anga, angp):
        continue

      _, d = angp.directions
      for l in d.neighbors(gm.Line):
        l_points = l.neighbors(gm.Point, return_set=True)
        m = intersect1(bc_points, l_points)
        if m is not None:
          yield dict(zip('ABCMO', [a, b, c, m, p]))


def match_midp_perp_cong(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match midp M A B, perp O M A B => cong O A O B."""
  for m, a, b in g.all_midps():
    ab = g._get_line(a, b)
    for l in m.neighbors(gm.Line):
      if g.check_perpl(l, ab):
        for o in l.neighbors(gm.Point):
          if o != m:
            yield dict(zip('ABMO', [a, b, m, o]))


def match_cyclic_eqangle_cong(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match cyclic A B C P Q R, eqangle C A C B R P R Q => cong A B P Q."""
  for c in g.type2nodes[gm.Circle]:
    ps = c.neighbors(gm.Point)
    for (a, b, c), (x, y, z) in utils.comb2(list(utils.perm3(ps))):
      if {a, b, c} == {x, y, z}:
        continue
      if g.check_eqangle([c, a, c, b, z, x, z, y]):
        yield dict(zip('ABCPQR', [a, b, c, x, y, z]))


def match_circle_eqangle_perp(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match circle O A B C, eqangle A X A B C A C B => perp O A A X."""
  for p, a, b, c in g.all_circles():
    ca = g._get_line(c, a)
    if ca is None:
      continue
    cb = g._get_line(c, b)
    if cb is None:
      continue
    ab = g._get_line(a, b)
    if ab is None:
      continue

    if ca.val is None:
      continue
    if cb.val is None:
      continue
    if ab.val is None:
      continue

    c_ang, _ = g._get_angle(cb.val, ca.val)
    if c_ang is None:
      continue

    for ang in ab.val.neighbors(gm.Angle):
      if g.is_equal(ang, c_ang):
        _, d = ang.directions
        for l in d.neighbors(gm.Line):
          if a not in l.neighbors(gm.Point):
            continue
          x = diff_point(l, a)
          if x is None:
            continue
          yield dict(zip('OABCX', [p, a, b, c, x]))
        break


def match_circle_perp_eqangle(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match circle O A B C, perp O A A X => eqangle A X A B C A C B."""
  for p, a, b, c in g.all_circles():
    pa = g._get_line(p, a)
    if pa is None:
      continue
    if pa.val is None:
      continue
    for l in a.neighbors(gm.Line):
      if g.check_perpl(pa, l):
        x = diff_point(l, a)
        if x is not None:
          yield dict(zip('OABCX', [p, a, b, c, x]))


def match_perp_perp_ncoll_para(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match perp A B C D, perp C D E F, ncoll A B E => para A B E F."""
  d2d = defaultdict(list)
  for ang in g.vhalfpi.neighbors(gm.Angle):
    d1, d2 = ang.directions
    if d1 is None or d2 is None:
      continue
    d2d[d1] += [d2]
    d2d[d2] += [d1]

  for x, ys in d2d.items():
    if len(ys) < 2:
      continue
    c, d = g.two_points_on_direction(x)
    for y1, y2 in utils.comb2(ys):
      a, b = g.two_points_on_direction(y1)
      e, f = g.two_points_on_direction(y2)
      if nm.check_ncoll([a.num, b.num, e.num]):
        yield dict(zip('ABCDEF', [a, b, c, d, e, f]))


def match_eqangle6_ncoll_cong(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqangle6 A O A B B A B O, ncoll O A B => cong O A O B."""
  for a in g.type2nodes[gm.Point]:
    for b, c in utils.comb2(g.type2nodes[gm.Point]):
      if a == b or a == c:
        continue
      if g.check_eqangle([b, a, b, c, c, b, c, a]):
        if g.check_ncoll([a, b, c]):
          yield dict(zip('OAB', [a, b, c]))


def match_eqangle_perp_perp(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqangle A B P Q C D U V, perp P Q U V => perp A B C D."""
  for ang in g.vhalfpi.neighbors(gm.Angle):
    # d1 perp d2
    d1, d2 = ang.directions
    if d1 is None or d2 is None:
      continue
    for d3, d4 in utils.comb2(g.type2nodes[gm.Direction]):
      if d1 == d3 or d2 == d4:
        continue
      # if d1 - d3 = d2 - d4 => d3 perp d4
      a13, a31 = g._get_angle(d1, d3)
      a24, a42 = g._get_angle(d2, d4)
      if a13 is None or a31 is None or a24 is None or a42 is None:
        continue
      if g.is_equal(a13, a24) and g.is_equal(a31, a42):
        a, b = g.two_points_on_direction(d1)
        c, d = g.two_points_on_direction(d2)
        m, n = g.two_points_on_direction(d3)
        p, q = g.two_points_on_direction(d4)
        yield dict(zip('ABCDPQUV', [m, n, p, q, a, b, c, d]))


def match_eqangle_ncoll_cyclic(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqangle6 P A P B Q A Q B, ncoll P Q A B => cyclic A B P Q."""
  for l1, l2, l3, l4 in g.all_eqangles_distinct_linepairss():
    if len(set([l1, l2, l3, l4])) < 4:
      continue  # they all must be distinct.

    p1s = l1.neighbors(gm.Point, return_set=True)
    p2s = l2.neighbors(gm.Point, return_set=True)
    p3s = l3.neighbors(gm.Point, return_set=True)
    p4s = l4.neighbors(gm.Point, return_set=True)

    p = intersect1(p1s, p2s)
    if not p:
      continue
    q = intersect1(p3s, p4s)
    if not q:
      continue
    a = intersect1(p1s, p3s)
    if not a:
      continue
    b = intersect1(p2s, p4s)
    if not b:
      continue
    if len(set([a, b, p, q])) < 4:
      continue

    if not g.check_ncoll([a, b, p, q]):
      continue

    yield dict(zip('ABPQ', [a, b, p, q]))


def match_eqangle_para(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqangle A B P Q C D P Q => para A B C D."""
  for measure in g.type2nodes[gm.Measure]:
    angs = measure.neighbors(gm.Angle)
    d12, d21 = defaultdict(list), defaultdict(list)
    for ang in angs:
      d1, d2 = ang.directions
      if d1 is None or d2 is None:
        continue
      d12[d1].append(d2)
      d21[d2].append(d1)

    for d1, d2s in d12.items():
      a, b = g.two_points_on_direction(d1)
      for d2, d3 in utils.comb2(d2s):
        c, d = g.two_points_on_direction(d2)
        e, f = g.two_points_on_direction(d3)
        yield dict(zip('ABCDPQ', [c, d, e, f, a, b]))


def match_cyclic_eqangle(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match cyclic A B P Q => eqangle P A P B Q A Q B."""
  record = set()
  for a, b, c, d in g_matcher('cyclic'):
    if (a, b, c, d) in record:
      continue
    record.add((a, b, c, d))
    record.add((a, b, d, c))
    record.add((b, a, c, d))
    record.add((b, a, d, c))
    yield dict(zip('ABPQ', [a, b, c, d]))


def rotate_simtri(
    a: gm.Point, b: gm.Point, c: gm.Point, x: gm.Point, y: gm.Point, z: gm.Point
) -> Generator[tuple[gm.Point, ...], None, None]:
  """Rotate points around for similar triangle predicates."""
  yield (z, y, x, c, b, a)
  for p in [
      (b, c, a, y, z, x),
      (c, a, b, z, x, y),
      (x, y, z, a, b, c),
      (y, z, x, b, c, a),
      (z, x, y, c, a, b),
  ]:
    yield p
    yield p[::-1]


def match_cong_cong_cong_cyclic(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match cong O A O B, cong O B O C, cong O C O D => cyclic A B C D."""
  for l in g.type2nodes[gm.Length]:
    p2p = defaultdict(list)
    for s in l.neighbors(gm.Segment):
      a, b = s.points
      p2p[a].append(b)
      p2p[b].append(a)

    for p, ps in p2p.items():
      if len(ps) >= 4:
        for a, b, c, d in utils.comb4(ps):
          yield dict(zip('OABCD', [p, a, b, c, d]))


def match_cong_cong_cong_ncoll_contri(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match cong A B P Q, cong B C Q R, cong C A R P, ncoll A B C => contri* A B C P Q R."""
  record = set()
  for a, b, p, q in g_matcher('cong'):
    for c in g.type2nodes[gm.Point]:
      for r in g.type2nodes[gm.Point]:
        if any([x in record for x in rotate_simtri(a, b, c, p, q, r)]):
          continue
        if not g.check_ncoll([a, b, c]):
          continue
        if g.check_cong([b, c, q, r]) and g.check_cong([c, a, r, p]):
          record.add((a, b, c, p, q, r))
          yield dict(zip('ABCPQR', [a, b, c, p, q, r]))


def match_cong_cong_eqangle6_ncoll_contri(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match cong A B P Q, cong B C Q R, eqangle6 B A B C Q P Q R, ncoll A B C => contri* A B C P Q R."""
  record = set()
  for a, b, p, q in g_matcher('cong'):
    for c in g.type2nodes[gm.Point]:
      if c in (a, b):
        continue
      for r in g.type2nodes[gm.Point]:
        if r in (p, q):
          continue

        in_record = False
        for x in [
            (c, b, a, r, q, p),
            (p, q, r, a, b, c),
            (r, q, p, c, b, a),
        ]:
          if x in record:
            in_record = True
            break

        if in_record:
          continue

        if not g.check_cong([b, c, q, r]):
          continue
        if not g.check_ncoll([a, b, c]):
          continue

        if nm.same_clock(a.num, b.num, c.num, p.num, q.num, r.num):
          if g.check_eqangle([b, a, b, c, q, p, q, r]):
            record.add((a, b, c, p, q, r))
            yield dict(zip('ABCPQR', [a, b, c, p, q, r]))
        else:
          if g.check_eqangle([b, a, b, c, q, r, q, p]):
            record.add((a, b, c, p, q, r))
            yield dict(zip('ABCPQR', [a, b, c, p, q, r]))


def match_eqratio6_eqangle6_ncoll_simtri(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqratio6 B A B C Q P Q R, eqratio6 C A C B R P R Q, ncoll A B C => simtri* A B C P Q R."""
  enums = g_matcher('eqratio6')

  record = set()
  for b, a, b, c, q, p, q, r in enums:  # pylint: disable=redeclared-assigned-name,unused-variable
    if (a, b, c) == (p, q, r):
      continue
    if any([x in record for x in rotate_simtri(a, b, c, p, q, r)]):
      continue
    if not g.check_ncoll([a, b, c]):
      continue

    if nm.same_clock(a.num, b.num, c.num, p.num, q.num, r.num):
      if g.check_eqangle([b, a, b, c, q, p, q, r]):
        record.add((a, b, c, p, q, r))
        yield dict(zip('ABCPQR', [a, b, c, p, q, r]))
    elif g.check_eqangle([b, a, b, c, q, r, q, p]):
      record.add((a, b, c, p, q, r))
      yield dict(zip('ABCPQR', [a, b, c, p, q, r]))


def match_eqangle6_eqangle6_ncoll_simtri(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqangle6 B A B C Q P Q R, eqangle6 C A C B R P R Q, ncoll A B C => simtri A B C P Q R."""
  enums = g_matcher('eqangle6')

  record = set()
  for b, a, b, c, q, p, q, r in enums:  # pylint: disable=redeclared-assigned-name,unused-variable
    if (a, b, c) == (p, q, r):
      continue
    if any([x in record for x in rotate_simtri(a, b, c, p, q, r)]):
      continue
    if not g.check_eqangle([c, a, c, b, r, p, r, q]):
      continue
    if not g.check_ncoll([a, b, c]):
      continue

    mapping = dict(zip('ABCPQR', [a, b, c, p, q, r]))
    record.add((a, b, c, p, q, r))
    yield mapping


def match_eqratio6_eqratio6_ncoll_simtri(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqratio6 B A B C Q P Q R, eqratio6 C A C B R P R Q, ncoll A B C => simtri* A B C P Q R."""
  enums = g_matcher('eqratio6')

  record = set()
  for b, a, b, c, q, p, q, r in enums:  # pylint: disable=redeclared-assigned-name,unused-variable
    if (a, b, c) == (p, q, r):
      continue
    if any([x in record for x in rotate_simtri(a, b, c, p, q, r)]):
      continue
    if not g.check_eqratio([c, a, c, b, r, p, r, q]):
      continue
    if not g.check_ncoll([a, b, c]):
      continue

    mapping = dict(zip('ABCPQR', [a, b, c, p, q, r]))
    record.add((a, b, c, p, q, r))
    yield mapping


def match_eqangle6_eqangle6_ncoll_simtri2(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqangle6 B A B C Q R Q P, eqangle6 C A C B R Q R P, ncoll A B C => simtri2 A B C P Q R."""
  enums = g_matcher('eqangle6')

  record = set()
  for b, a, b, c, q, r, q, p in enums:  # pylint: disable=redeclared-assigned-name,unused-variable
    if (a, b, c) == (p, q, r):
      continue
    if any([x in record for x in rotate_simtri(a, b, c, p, q, r)]):
      continue
    if not g.check_eqangle([c, a, c, b, r, q, r, p]):
      continue
    if not g.check_ncoll([a, b, c]):
      continue

    mapping = dict(zip('ABCPQR', [a, b, c, p, q, r]))
    record.add((a, b, c, p, q, r))
    yield mapping


def rotate_contri(
    a: gm.Point, b: gm.Point, c: gm.Point, x: gm.Point, y: gm.Point, z: gm.Point
) -> Generator[tuple[gm.Point, ...], None, None]:
  for p in [(b, a, c, y, x, z), (x, y, z, a, b, c), (y, x, z, b, a, c)]:
    yield p


def match_eqangle6_eqangle6_ncoll_cong_contri(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqangle6 B A B C Q P Q R, eqangle6 C A C B R P R Q, ncoll A B C, cong A B P Q => contri A B C P Q R."""
  enums = g_matcher('eqangle6')

  record = set()
  for b, a, b, c, q, p, q, r in enums:  # pylint: disable=redeclared-assigned-name,unused-variable
    if not g.check_cong([a, b, p, q]):
      continue
    if (a, b, c) == (p, q, r):
      continue
    if any([x in record for x in rotate_contri(a, b, c, p, q, r)]):
      continue
    if not g.check_eqangle([c, a, c, b, r, p, r, q]):
      continue

    if not g.check_ncoll([a, b, c]):
      continue

    mapping = dict(zip('ABCPQR', [a, b, c, p, q, r]))
    record.add((a, b, c, p, q, r))
    yield mapping


def match_eqratio6_eqratio6_ncoll_cong_contri(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqratio6 B A B C Q P Q R, eqratio6 C A C B R P R Q, ncoll A B C, cong A B P Q => contri* A B C P Q R."""
  enums = g_matcher('eqratio6')

  record = set()
  for b, a, b, c, q, p, q, r in enums:  # pylint: disable=redeclared-assigned-name,unused-variable
    if not g.check_cong([a, b, p, q]):
      continue
    if (a, b, c) == (p, q, r):
      continue
    if any([x in record for x in rotate_contri(a, b, c, p, q, r)]):
      continue
    if not g.check_eqratio([c, a, c, b, r, p, r, q]):
      continue

    if not g.check_ncoll([a, b, c]):
      continue

    mapping = dict(zip('ABCPQR', [a, b, c, p, q, r]))
    record.add((a, b, c, p, q, r))
    yield mapping


def match_eqangle6_eqangle6_ncoll_cong_contri2(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqangle6 B A B C Q R Q P, eqangle6 C A C B R Q R P, ncoll A B C, cong A B P Q => contri2 A B C P Q R."""
  enums = g_matcher('eqangle6')

  record = set()
  for b, a, b, c, q, r, q, p in enums:  # pylint: disable=redeclared-assigned-name,unused-variable
    if not g.check_cong([a, b, p, q]):
      continue
    if (a, b, c) == (p, q, r):
      continue
    if any([x in record for x in rotate_contri(a, b, c, p, q, r)]):
      continue
    if not g.check_eqangle([c, a, c, b, r, q, r, p]):
      continue
    if not g.check_ncoll([a, b, c]):
      continue

    mapping = dict(zip('ABCPQR', [a, b, c, p, q, r]))
    record.add((a, b, c, p, q, r))
    yield mapping


def match_eqratio6_coll_ncoll_eqangle6(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqratio6 d b d c a b a c, coll d b c, ncoll a b c => eqangle6 a b a d a d a c."""
  records = set()
  for b, d, c in g_matcher('coll'):
    for a in g.all_points():
      if g.check_coll([a, b, c]):
        continue
      if (a, b, d, c) in records or (a, c, d, b) in records:
        continue
      records.add((a, b, d, c))

      if g.check_eqratio([d, b, d, c, a, b, a, c]):
        yield dict(zip('abcd', [a, b, c, d]))


def match_eqangle6_coll_ncoll_eqratio6(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqangle6 a b a d a d a c, coll d b c, ncoll a b c => eqratio6 d b d c a b a c."""
  records = set()
  for b, d, c in g_matcher('coll'):
    for a in g.all_points():
      if g.check_coll([a, b, c]):
        continue
      if (a, b, d, c) in records or (a, c, d, b) in records:
        continue
      records.add((a, b, d, c))

      if g.check_eqangle([a, b, a, d, a, d, a, c]):
        yield dict(zip('abcd', [a, b, c, d]))


def match_eqangle6_ncoll_cyclic(
    g: gh.Graph,
    g_matcher: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem,
) -> Generator[dict[str, gm.Point], None, None]:
  """Match eqangle6 P A P B Q A Q B, ncoll P Q A B => cyclic A B P Q."""
  for a, b, a, c, x, y, x, z in g_matcher('eqangle6'):  # pylint: disable=redeclared-assigned-name,unused-variable
    if (b, c) != (y, z) or a == x:
      continue
    if nm.check_ncoll([x.num for x in [a, b, c, x]]):
      yield dict(zip('ABPQ', [b, c, a, x]))


def match_all(
    name: str, g: gh.Graph
) -> Generator[tuple[gm.Point, ...], None, None]:
  """Match all instances of a certain relation."""
  if name in ['ncoll', 'npara', 'nperp']:
    return []
  if name == 'coll':
    return g.all_colls()
  if name == 'para':
    return g.all_paras()
  if name == 'perp':
    return g.all_perps()
  if name == 'cong':
    return g.all_congs()
  if name == 'eqangle':
    return g.all_eqangles_8points()
  if name == 'eqangle6':
    return g.all_eqangles_6points()
  if name == 'eqratio':
    return g.all_eqratios_8points()
  if name == 'eqratio6':
    return g.all_eqratios_6points()
  if name == 'cyclic':
    return g.all_cyclics()
  if name == 'midp':
    return g.all_midps()
  if name == 'circle':
    return g.all_circles()
  raise ValueError(f'Unrecognize {name}')


def cache_match(
    graph: gh.Graph,
) -> Callable[str, list[tuple[gm.Point, ...]]]:
  """Cache throughout one single BFS level."""
  cache = {}

  def match_fn(name: str) -> list[tuple[gm.Point, ...]]:
    if name in cache:
      return cache[name]

    result = list(match_all(name, graph))
    cache[name] = result
    return result

  return match_fn


def try_to_map(
    clause_enum: list[tuple[pr.Clause, list[tuple[gm.Point, ...]]]],
    mapping: dict[str, gm.Point],
) -> Generator[dict[str, gm.Point], None, None]:
  """Recursively try to match the remaining points given current mapping."""
  if not clause_enum:
    yield mapping
    return

  clause, enum = clause_enum[0]
  for points in enum:
    mpcpy = dict(mapping)

    fail = False
    for p, a in zip(points, clause.args):
      if a in mpcpy and mpcpy[a] != p or p in mpcpy and mpcpy[p] != a:
        fail = True
        break
      mpcpy[a] = p
      mpcpy[p] = a

    if fail:
      continue

    for m in try_to_map(clause_enum[1:], mpcpy):
      yield m


def match_generic(
    g: gh.Graph,
    cache: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem
) -> Generator[dict[str, gm.Point], None, None]:
  """Match any generic rule that is not one of the above match_*() rules."""
  clause2enum = {}

  clauses = []
  numerical_checks = []
  for clause in theorem.premise:
    if clause.name in ['ncoll', 'npara', 'nperp', 'sameside']:
      numerical_checks.append(clause)
      continue

    enum = cache(clause.name)
    if len(enum) == 0:  # pylint: disable=g-explicit-length-test
      return 0

    clause2enum[clause] = enum
    clauses.append((len(set(clause.args)), clause))

  clauses = sorted(clauses, key=lambda x: x[0], reverse=True)
  _, clauses = zip(*clauses)

  for mapping in try_to_map([(c, clause2enum[c]) for c in clauses], {}):
    if not mapping:
      continue

    checks_ok = True
    for check in numerical_checks:
      args = [mapping[a] for a in check.args]
      if check.name == 'ncoll':
        checks_ok = g.check_ncoll(args)
      elif check.name == 'npara':
        checks_ok = g.check_npara(args)
      elif check.name == 'nperp':
        checks_ok = g.check_nperp(args)
      elif check.name == 'sameside':
        checks_ok = g.check_sameside(args)
      if not checks_ok:
        break
    if not checks_ok:
      continue

    yield mapping


BUILT_IN_FNS = {
    'cong_cong_cong_cyclic': match_cong_cong_cong_cyclic,
    'cong_cong_cong_ncoll_contri*': match_cong_cong_cong_ncoll_contri,
    'cong_cong_eqangle6_ncoll_contri*': match_cong_cong_eqangle6_ncoll_contri,
    'eqangle6_eqangle6_ncoll_simtri': match_eqangle6_eqangle6_ncoll_simtri,
    'eqangle6_eqangle6_ncoll_cong_contri': (
        match_eqangle6_eqangle6_ncoll_cong_contri
    ),  # pylint: disable=line-too-long
    'eqangle6_eqangle6_ncoll_simtri2': match_eqangle6_eqangle6_ncoll_simtri2,
    'eqangle6_eqangle6_ncoll_cong_contri2': (
        match_eqangle6_eqangle6_ncoll_cong_contri2
    ),  # pylint: disable=line-too-long
    'eqratio6_eqratio6_ncoll_simtri*': match_eqratio6_eqratio6_ncoll_simtri,
    'eqratio6_eqratio6_ncoll_cong_contri*': (
        match_eqratio6_eqratio6_ncoll_cong_contri
    ),  # pylint: disable=line-too-long
    'eqangle_para': match_eqangle_para,
    'eqangle_ncoll_cyclic': match_eqangle_ncoll_cyclic,
    'eqratio6_eqangle6_ncoll_simtri*': match_eqratio6_eqangle6_ncoll_simtri,
    'eqangle_perp_perp': match_eqangle_perp_perp,
    'eqangle6_ncoll_cong': match_eqangle6_ncoll_cong,
    'perp_perp_ncoll_para': match_perp_perp_ncoll_para,
    'circle_perp_eqangle': match_circle_perp_eqangle,
    'circle_eqangle_perp': match_circle_eqangle_perp,
    'cyclic_eqangle_cong': match_cyclic_eqangle_cong,
    'midp_perp_cong': match_midp_perp_cong,
    'perp_perp_npara_eqangle': match_perp_perp_npara_eqangle,
    'cyclic_eqangle': match_cyclic_eqangle,
    'eqangle_eqangle_eqangle': match_eqangle_eqangle_eqangle,
    'eqratio_eqratio_eqratio': match_eqratio_eqratio_eqratio,
    'eqratio6_coll_ncoll_eqangle6': match_eqratio6_coll_ncoll_eqangle6,
    'eqangle6_coll_ncoll_eqratio6': match_eqangle6_coll_ncoll_eqratio6,
    'eqangle6_ncoll_cyclic': match_eqangle6_ncoll_cyclic,
}


SKIP_THEOREMS = set()


def set_skip_theorems(theorems: set[str]) -> None:
  SKIP_THEOREMS.update(theorems)


MAX_BRANCH = 50_000


def match_one_theorem(
    g: gh.Graph,
    cache: Callable[str, list[tuple[gm.Point, ...]]],
    theorem: pr.Theorem
) -> Generator[dict[str, gm.Point], None, None]:
  """Match all instances of a single theorem (rule)."""
  if cache is None:
    cache = cache_match(g)

  if theorem.name in SKIP_THEOREMS:
    return []

  if theorem.name.split('_')[-1] in SKIP_THEOREMS:
    return []

  if theorem.name in BUILT_IN_FNS:
    mps = BUILT_IN_FNS[theorem.name](g, cache, theorem)
  else:
    mps = match_generic(g, cache, theorem)

  mappings = []
  for mp in mps:
    mappings.append(mp)
    if len(mappings) > MAX_BRANCH:  # cap branching at this number.
      break

  return mappings


def match_all_theorems(
    g: gh.Graph, theorems: list[pr.Theorem], goal: pr.Clause
) -> dict[pr.Theorem, dict[pr.Theorem, dict[str, gm.Point]]]:
  """Match all instances of all theorems (rules)."""
  cache = cache_match(g)
  # for BFS, collect all potential matches
  # and then do it at the same time
  theorem2mappings = {}

  # Step 1: list all matches
  for _, theorem in theorems.items():
    name = theorem.name
    if name.split('_')[-1] in [
        'acompute',
        'rcompute',
        'fixl',
        'fixc',
        'fixb',
        'fixt',
        'fixp',
    ]:
      if goal and goal.name != name:
        continue

    mappings = match_one_theorem(g, cache, theorem)
    if len(mappings):  # pylint: disable=g-explicit-length-test
      theorem2mappings[theorem] = list(mappings)
  return theorem2mappings


def bfs_one_level(
    g: gh.Graph,
    theorems: list[pr.Theorem],
    level: int,
    controller: pr.Problem,
    verbose: bool = False,
    nm_check: bool = False,
    timeout: int = 600,
) -> tuple[
    list[pr.Dependency],
    dict[str, list[tuple[gm.Point, ...]]],
    dict[str, list[tuple[gm.Point, ...]]],
    int,
]:
  """Forward deduce one breadth-first level."""

  # Step 1: match all theorems:
  theorem2mappings = match_all_theorems(g, theorems, controller.goal)

  # Step 2: traceback for each deduce:
  theorem2deps = {}
  t0 = time.time()
  for theorem, mappings in theorem2mappings.items():
    if time.time() - t0 > timeout:
      break
    mp_deps = []
    for mp in mappings:
      deps = EmptyDependency(level=level, rule_name=theorem.rule_name)
      fail = False  # finding why deps might fail.

      for p in theorem.premise:
        p_args = [mp[a] for a in p.args]
        # Trivial deps.
        if p.name == 'cong':
          a, b, c, d = p_args
          if {a, b} == {c, d}:
            continue
        if p.name == 'para':
          a, b, c, d = p_args
          if {a, b} == {c, d}:
            continue

        if theorem.name in [
            'cong_cong_eqangle6_ncoll_contri*',
            'eqratio6_eqangle6_ncoll_simtri*',
        ]:
          if p.name in ['eqangle', 'eqangle6']:  # SAS or RAR
            b, a, b, c, y, x, y, z = (  # pylint: disable=redeclared-assigned-name,unused-variable
                p_args
            )
            if not nm.same_clock(a.num, b.num, c.num, x.num, y.num, z.num):
              p_args = b, a, b, c, y, z, y, x

        dep = Dependency(p.name, p_args, rule_name='', level=level)
        try:
          dep = dep.why_me_or_cache(g, level)
        except:  # pylint: disable=bare-except
          fail = True
          break

        if dep.why is None:
          fail = True
          break
        g.cache_dep(p.name, p_args, dep)
        deps.why.append(dep)

      if fail:
        continue

      mp_deps.append((mp, deps))
    theorem2deps[theorem] = mp_deps

  theorem2deps = list(theorem2deps.items())

  # Step 3: add conclusions to graph.
  # Note that we do NOT mix step 2 and 3, strictly going for BFS.
  added = []
  for theorem, mp_deps in theorem2deps:
    for mp, deps in mp_deps:
      if time.time() - t0 > timeout:
        break
      name, args = theorem.conclusion_name_args(mp)
      hash_conclusion = pr.hashed(name, args)
      if hash_conclusion in g.cache:
        continue

      add = g.add_piece(name, args, deps=deps)
      added += add

  branching = len(added)

  # Check if goal is found
  if controller.goal:
    args = []

    for a in controller.goal.args:
      if a in g._name2node:
        a = g._name2node[a]
      elif '/' in a:
        a = create_consts_str(g, a)
      elif a.isdigit():
        a = int(a)
      args.append(a)

    if g.check(controller.goal.name, args):
      return added, {}, {}, branching

  # Run AR, but do NOT apply to the proof state (yet).
  for dep in added:
    g.add_algebra(dep, level)
  derives, eq4s = g.derive_algebra(level, verbose=False)

  branching += sum([len(x) for x in derives.values()])
  branching += sum([len(x) for x in eq4s.values()])

  return added, derives, eq4s, branching


def create_consts_str(g: gh.Graph, s: str) -> gm.Angle | gm.Ratio:
  if 'pi/' in s:
    n, d = s.split('pi/')
    n, d = int(n), int(d)
    p0, _ = g.get_or_create_const_ang(n, d)
  else:
    n, d = s.split('/')
    n, d = int(n), int(d)
    p0, _ = g.get_or_create_const_rat(n, d)
  return p0


def do_algebra(
    g: gh.Graph, added: list[pr.Dependency], verbose: bool = False
) -> None:
  for add in added:
    g.add_algebra(add, None)
  derives, eq4s = g.derive_algebra(level=None, verbose=verbose)
  apply_derivations(g, derives)
  apply_derivations(g, eq4s)


def apply_derivations(
    g: gh.Graph, derives: dict[str, list[tuple[gm.Point, ...]]]
) -> list[pr.Dependency]:
  applied = []
  all_derives = list(derives.items())
  for name, args in all_derives:
    for arg in args:
      applied += g.do_algebra(name, arg)
  return applied
