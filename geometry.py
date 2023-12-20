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

"""Implements geometric objects used in the graph representation."""
from __future__ import annotations
from collections import defaultdict  # pylint: disable=g-importing-member
from typing import Any, Type

# pylint: disable=protected-access


class Node:
  r"""Node in the proof state graph.

  Can be Point, Line, Circle, etc.

  Each node maintains a merge history to
  other nodes if they are (found out to be) equivalent

    a -> b -
            \
         c -> d -> e -> f -> g

  d.merged_to = e
  d.rep = g
  d.merged_from = {a, b, c, d}
  d.equivs = {a, b, c, d, e, f, g}
  """

  def __init__(self, name: str = '', graph: Any = None):
    self.name = name or str(self)
    self.graph = graph

    self.edge_graph = {}
    # Edge graph: what other nodes is connected to this node.
    # edge graph = {
    #   other1: {self1: deps, self2: deps},
    #   other2: {self2: deps, self3: deps}
    # }

    self.merge_graph = {}
    # Merge graph: history of merges with other nodes.
    # merge_graph = {self1: {self2: deps1, self3: deps2}}

    self.rep_by = None  # represented by.
    self.members = {self}

    self._val = None
    self._obj = None

    self.deps = []

    # numerical representation.
    self.num = None
    self.change = set()  # what other nodes' num rely on this node?

  def set_rep(self, node: Node) -> None:
    if node == self:
      return
    self.rep_by = node
    node.merge_edge_graph(self.edge_graph)
    node.members.update(self.members)

  def rep(self) -> Node:
    x = self
    while x.rep_by:
      x = x.rep_by
    return x

  def why_rep(self) -> list[Any]:
    return self.why_equal([self.rep()], None)

  def rep_and_why(self) -> tuple[Node, list[Any]]:
    rep = self.rep()
    return rep, self.why_equal([rep], None)

  def neighbors(
      self, oftype: Type[Node], return_set: bool = False, do_rep: bool = True
  ) -> list[Node]:
    """Neighbors of this node in the proof state graph."""
    if do_rep:
      rep = self.rep()
    else:
      rep = self
    result = set()

    for n in rep.edge_graph:
      if oftype is None or oftype and isinstance(n, oftype):
        if do_rep:
          result.add(n.rep())
        else:
          result.add(n)

    if return_set:
      return result
    return list(result)

  def merge_edge_graph(
      self, new_edge_graph: dict[Node, dict[Node, list[Node]]]
  ) -> None:
    for x, xdict in new_edge_graph.items():
      if x in self.edge_graph:
        self.edge_graph[x].update(dict(xdict))
      else:
        self.edge_graph[x] = dict(xdict)

  def merge(self, nodes: list[Node], deps: list[Any]) -> None:
    for node in nodes:
      self.merge_one(node, deps)

  def merge_one(self, node: Node, deps: list[Any]) -> None:
    node.rep().set_rep(self.rep())

    if node in self.merge_graph:
      return

    self.merge_graph[node] = deps
    node.merge_graph[self] = deps

  def is_val(self, node: Node) -> bool:
    return (
        isinstance(self, Line)
        and isinstance(node, Direction)
        or isinstance(self, Segment)
        and isinstance(node, Length)
        or isinstance(self, Angle)
        and isinstance(node, Measure)
        or isinstance(self, Ratio)
        and isinstance(node, Value)
    )

  def set_val(self, node: Node) -> None:
    self._val = node

  def set_obj(self, node: Node) -> None:
    self._obj = node

  @property
  def val(self) -> Node:
    if self._val is None:
      return None
    return self._val.rep()

  @property
  def obj(self) -> Node:
    if self._obj is None:
      return None
    return self._obj.rep()

  def equivs(self) -> set[Node]:
    return self.rep().members

  def connect_to(self, node: Node, deps: list[Any] = None) -> None:
    rep = self.rep()

    if node in rep.edge_graph:
      rep.edge_graph[node].update({self: deps})
    else:
      rep.edge_graph[node] = {self: deps}

    if self.is_val(node):
      self.set_val(node)
      node.set_obj(self)

  def equivs_upto(self, level: int) -> dict[Node, Node]:
    """What are the equivalent nodes up to a certain level."""
    parent = {self: None}
    visited = set()
    queue = [self]
    i = 0

    while i < len(queue):
      current = queue[i]
      i += 1
      visited.add(current)

      for neighbor in current.merge_graph:
        if (
            level is not None
            and current.merge_graph[neighbor].level is not None
            and current.merge_graph[neighbor].level >= level
        ):
          continue
        if neighbor not in visited:
          queue.append(neighbor)
          parent[neighbor] = current

    return parent

  def why_equal(self, others: list[Node], level: int) -> list[Any]:
    """BFS why this node is equal to other nodes."""
    others = set(others)
    found = 0

    parent = {}
    queue = [self]
    i = 0

    while i < len(queue):
      current = queue[i]
      if current in others:
        found += 1
      if found == len(others):
        break

      i += 1

      for neighbor in current.merge_graph:
        if (
            level is not None
            and current.merge_graph[neighbor].level is not None
            and current.merge_graph[neighbor].level >= level
        ):
          continue
        if neighbor not in parent:
          queue.append(neighbor)
          parent[neighbor] = current

    return bfs_backtrack(self, others, parent)

  def why_equal_groups(
      self, groups: list[list[Node]], level: int
  ) -> tuple[list[Any], list[Node]]:
    """BFS for why self is equal to at least one member of each group."""
    others = [None for _ in groups]
    found = 0

    parent = {}
    queue = [self]
    i = 0

    while i < len(queue):
      current = queue[i]

      for j, grp in enumerate(groups):
        if others[j] is None and current in grp:
          others[j] = current
          found += 1

      if found == len(others):
        break

      i += 1

      for neighbor in current.merge_graph:
        if (
            level is not None
            and current.merge_graph[neighbor].level is not None
            and current.merge_graph[neighbor].level >= level
        ):
          continue
        if neighbor not in parent:
          queue.append(neighbor)
          parent[neighbor] = current

    return bfs_backtrack(self, others, parent), others

  def why_val(self, level: int) -> list[Any]:
    return self._val.why_equal([self.val], level)

  def why_connect(self, node: Node, level: int = None) -> list[Any]:
    rep = self.rep()
    equivs = list(rep.edge_graph[node].keys())
    if not equivs:
      return None
    equiv = equivs[0]
    dep = rep.edge_graph[node][equiv]
    return [dep] + self.why_equal(equiv, level)


def why_connect(*pairs: list[tuple[Node, Node]]) -> list[Any]:
  result = []
  for node1, node2 in pairs:
    result += node1.why_connect(node2)
  return result


def is_equiv(x: Node, y: Node, level: int = None) -> bool:
  level = level or float('inf')
  return x.why_equal([y], level) is not None


def is_equal(x: Node, y: Node, level: int = None) -> bool:
  if x == y:
    return True
  if x._val is None or y._val is None:
    return False
  if x.val != y.val:
    return False
  return is_equiv(x._val, y._val, level)


def bfs_backtrack(
    root: Node, leafs: list[Node], parent: dict[Node, Node]
) -> list[Any]:
  """Return the path given BFS trace of parent nodes."""
  backtracked = {root}  # no need to backtrack further when touching this set.
  deps = []
  for node in leafs:
    if node is None:
      return None
    if node in backtracked:
      continue
    if node not in parent:
      return None
    while node not in backtracked:
      backtracked.add(node)
      deps.append(node.merge_graph[parent[node]])
      node = parent[node]

  return deps


class Point(Node):
  pass


class Line(Node):
  """Node of type Line."""

  def new_val(self) -> Direction:
    return Direction()

  def why_coll(self, points: list[Point], level: int = None) -> list[Any]:
    """Why points are connected to self."""
    level = level or float('inf')

    groups = []
    for p in points:
      group = [
          l
          for l, d in self.edge_graph[p].items()
          if d is None or d.level < level
      ]
      if not group:
        return None
      groups.append(group)

    min_deps = None
    for line in groups[0]:
      deps, others = line.why_equal_groups(groups[1:], level)
      if deps is None:
        continue
      for p, o in zip(points, [line] + others):
        deps.append(self.edge_graph[p][o])
      if min_deps is None or len(deps) < len(min_deps):
        min_deps = deps

    if min_deps is None:
      return None
    return [d for d in min_deps if d is not None]


class Segment(Node):

  def new_val(self) -> Length:
    return Length()


class Circle(Node):
  """Node of type Circle."""

  def why_cyclic(self, points: list[Point], level: int = None) -> list[Any]:
    """Why points are connected to self."""
    level = level or float('inf')

    groups = []
    for p in points:
      group = [
          c
          for c, d in self.edge_graph[p].items()
          if d is None or d.level < level
      ]
      if not group:
        return None
      groups.append(group)

    min_deps = None
    for circle in groups[0]:
      deps, others = circle.why_equal_groups(groups[1:], level)
      if deps is None:
        continue
      for p, o in zip(points, [circle] + others):
        deps.append(self.edge_graph[p][o])

      if min_deps is None or len(deps) < len(min_deps):
        min_deps = deps

    if min_deps is None:
      return None
    return [d for d in min_deps if d is not None]


def why_equal(x: Node, y: Node, level: int = None) -> list[Any]:
  if x == y:
    return []
  if not x._val or not y._val:
    return None
  if x._val == y._val:
    return []
  return x._val.why_equal([y._val], level)


class Direction(Node):
  pass


def get_lines_thru_all(*points: list[Point]) -> list[Line]:
  line2count = defaultdict(lambda: 0)
  points = set(points)
  for p in points:
    for l in p.neighbors(Line):
      line2count[l] += 1
  return [l for l, count in line2count.items() if count == len(points)]


def line_of_and_why(
    points: list[Point], level: int = None
) -> tuple[Line, list[Any]]:
  """Why points are collinear."""
  for l0 in get_lines_thru_all(*points):
    for l in l0.equivs():
      if all([p in l.edge_graph for p in points]):
        x, y = l.points
        colls = list({x, y} | set(points))
        # if len(colls) < 3:
        #   return l, []
        why = l.why_coll(colls, level)
        if why is not None:
          return l, why

  return None, None


def get_circles_thru_all(*points: list[Point]) -> list[Circle]:
  circle2count = defaultdict(lambda: 0)
  points = set(points)
  for p in points:
    for c in p.neighbors(Circle):
      circle2count[c] += 1
  return [c for c, count in circle2count.items() if count == len(points)]


def circle_of_and_why(
    points: list[Point], level: int = None
) -> tuple[Circle, list[Any]]:
  """Why points are concyclic."""
  for c0 in get_circles_thru_all(*points):
    for c in c0.equivs():
      if all([p in c.edge_graph for p in points]):
        cycls = list(set(points))
        why = c.why_cyclic(cycls, level)
        if why is not None:
          return c, why

  return None, None


def name_map(struct: Any) -> Any:
  if isinstance(struct, list):
    return [name_map(x) for x in struct]
  elif isinstance(struct, tuple):
    return tuple([name_map(x) for x in struct])
  elif isinstance(struct, set):
    return set([name_map(x) for x in struct])
  elif isinstance(struct, dict):
    return {name_map(x): name_map(y) for x, y in struct.items()}
  else:
    return getattr(struct, 'name', '')


class Angle(Node):
  """Node of type Angle."""

  def new_val(self) -> Measure:
    return Measure()

  def set_directions(self, d1: Direction, d2: Direction) -> None:
    self._d = d1, d2

  @property
  def directions(self) -> tuple[Direction, Direction]:
    d1, d2 = self._d
    if d1 is None or d2 is None:
      return d1, d2
    return d1.rep(), d2.rep()


class Measure(Node):
  pass


class Length(Node):
  pass


class Ratio(Node):
  """Node of type Ratio."""

  def new_val(self) -> Value:
    return Value()

  def set_lengths(self, l1: Length, l2: Length) -> None:
    self._l = l1, l2

  @property
  def lengths(self) -> tuple[Length, Length]:
    l1, l2 = self._l
    if l1 is None or l2 is None:
      return l1, l2
    return l1.rep(), l2.rep()


class Value(Node):
  pass


def all_angles(
    d1: Direction, d2: Direction, level: int = None
) -> tuple[Angle, list[Direction], list[Direction]]:
  level = level or float('inf')
  d1s = d1.equivs_upto(level)
  d2s = d2.equivs_upto(level)

  for ang in d1.rep().neighbors(Angle):
    d1_, d2_ = ang._d
    if d1_ in d1s and d2_ in d2s:
      yield ang, d1s, d2s


def all_ratios(
    d1, d2, level=None
) -> tuple[Angle, list[Direction], list[Direction]]:
  level = level or float('inf')
  d1s = d1.equivs_upto(level)
  d2s = d2.equivs_upto(level)

  for ang in d1.rep().neighbors(Ratio):
    d1_, d2_ = ang._l
    if d1_ in d1s and d2_ in d2s:
      yield ang, d1s, d2s


RANKING = {
    Point: 0,
    Line: 1,
    Segment: 2,
    Circle: 3,
    Direction: 4,
    Length: 5,
    Angle: 6,
    Ratio: 7,
    Measure: 8,
    Value: 9,
}


def val_type(x: Node) -> Type[Node]:
  if isinstance(x, Line):
    return Direction
  if isinstance(x, Segment):
    return Length
  if isinstance(x, Angle):
    return Measure
  if isinstance(x, Ratio):
    return Value
