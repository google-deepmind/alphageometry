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

"""Utilizations for graph representation.

Mainly for listing combinations and permutations of elements.
"""

from geometry import Point


def _cross(elems1, elems2):
  for e1 in elems1:
    for e2 in elems2:
      yield e1, e2


def cross(elems1, elems2):
  return list(_cross(elems1, elems2))


def _comb2(elems):
  if len(elems) < 2:
    return
  for i, e1 in enumerate(elems[:-1]):
    for e2 in elems[i + 1 :]:
      yield e1, e2


def comb2(elems):
  return list(_comb2(elems))


def _comb3(elems):
  if len(elems) < 3:
    return
  for i, e1 in enumerate(elems[:-2]):
    for j, e2 in enumerate(elems[i + 1 : -1]):
      for e3 in elems[i + j + 2 :]:
        yield e1, e2, e3


def comb3(elems):
  return list(_comb3(elems))


def _comb4(elems):
  if len(elems) < 4:
    return
  for i, e1 in enumerate(elems[:-3]):
    for j, e2 in enumerate(elems[i + 1 : -2]):
      for e3, e4 in _comb2(elems[i + j + 2 :]):
        yield e1, e2, e3, e4


def comb4(elems):
  return list(_comb4(elems))


def _perm2(elems):
  for e1, e2 in comb2(elems):
    yield e1, e2
    yield e2, e1


def perm2(elems):
  return list(_perm2(elems))


def _all_4points(l1, l2):
  p1s = l1.neighbors(Point)
  p2s = l2.neighbors(Point)
  for a, b in perm2(p1s):
    for c, d in perm2(p2s):
      yield a, b, c, d


def all_4points(l1, l2):
  return list(_all_4points(l1, l2))


def _all_8points(l1, l2, l3, l4):
  for a, b, c, d in all_4points(l1, l2):
    for e, f, g, h in all_4points(l3, l4):
      yield (a, b, c, d, e, f, g, h)


def all_8points(l1, l2, l3, l4):
  return list(_all_8points(l1, l2, l3, l4))


def _perm3(elems):
  for x in elems:
    for y in elems:
      if y == x:
        continue
      for z in elems:
        if z not in (x, y):
          yield x, y, z


def perm3(elems):
  return list(_perm3(elems))


def _perm4(elems):
  for x in elems:
    for y in elems:
      if y == x:
        continue
      for z in elems:
        if z in (x, y):
          continue
        for t in elems:
          if t not in (x, y, z):
            yield x, y, z, t


def perm4(elems):
  return list(_perm4(elems))
