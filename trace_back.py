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

"""Implements DAG-level traceback."""

from typing import Any

import geometry as gm
import pretty as pt
import problem


pretty = pt.pretty


def point_levels(
    setup: list[problem.Dependency], existing_points: list[gm.Point]
) -> list[tuple[set[gm.Point], list[problem.Dependency]]]:
  """Reformat setup into levels of point constructions."""
  levels = []
  for con in setup:
    plevel = max([p.plevel for p in con.args if isinstance(p, gm.Point)])

    while len(levels) - 1 < plevel:
      levels.append((set(), []))

    for p in con.args:
      if not isinstance(p, gm.Point):
        continue
      if existing_points and p in existing_points:
        continue

      levels[p.plevel][0].add(p)

    cons = levels[plevel][1]
    cons.append(con)

  return [(p, c) for p, c in levels if p or c]


def point_log(
    setup: list[problem.Dependency],
    ref_id: dict[tuple[str, ...], int],
    existing_points=list[gm.Point],
) -> list[tuple[list[gm.Point], list[problem.Dependency]]]:
  """Reformat setup into groups of point constructions."""
  log = []

  levels = point_levels(setup, existing_points)

  for points, cons in levels:
    for con in cons:
      if con.hashed() not in ref_id:
        ref_id[con.hashed()] = len(ref_id)

    log.append((points, cons))

  return log


def setup_to_levels(
    setup: list[problem.Dependency],
) -> list[list[problem.Dependency]]:
  """Reformat setup into levels of point constructions."""
  levels = []
  for d in setup:
    plevel = max([p.plevel for p in d.args if isinstance(p, gm.Point)])
    while len(levels) - 1 < plevel:
      levels.append([])

    levels[plevel].append(d)

  levels = [lvl for lvl in levels if lvl]
  return levels


def separate_dependency_difference(
    query: problem.Dependency,
    log: list[tuple[list[problem.Dependency], list[problem.Dependency]]],
) -> tuple[
    list[tuple[list[problem.Dependency], list[problem.Dependency]]],
    list[problem.Dependency],
    list[problem.Dependency],
    set[gm.Point],
    set[gm.Point],
]:
  """Identify and separate the dependency difference."""
  setup = []
  log_, log = log, []
  for prems, cons in log_:
    if not prems:
      setup.extend(cons)
      continue
    cons_ = []
    for con in cons:
      if con.rule_name == 'c0':
        setup.append(con)
      else:
        cons_.append(con)
    if not cons_:
      continue

    prems = [p for p in prems if p.name != 'ind']
    log.append((prems, cons_))

  points = set(query.args)
  queue = list(query.args)
  i = 0
  while i < len(queue):
    q = queue[i]
    i += 1
    if not isinstance(q, gm.Point):
      continue
    for p in q.rely_on:
      if p not in points:
        points.add(p)
        queue.append(p)

  setup_, setup, aux_setup, aux_points = setup, [], [], set()
  for con in setup_:
    if con.name == 'ind':
      continue
    elif any([p not in points for p in con.args if isinstance(p, gm.Point)]):
      aux_setup.append(con)
      aux_points.update(
          [p for p in con.args if isinstance(p, gm.Point) and p not in points]
      )
    else:
      setup.append(con)

  return log, setup, aux_setup, points, aux_points


def recursive_traceback(
    query: problem.Dependency,
) -> list[tuple[list[problem.Dependency], list[problem.Dependency]]]:
  """Recursively traceback from the query, i.e. the conclusion."""
  visited = set()
  log = []
  stack = []

  def read(q: problem.Dependency) -> None:
    q = q.remove_loop()
    hashed = q.hashed()
    if hashed in visited:
      return

    if hashed[0] in ['ncoll', 'npara', 'nperp', 'diff', 'sameside']:
      return

    nonlocal stack

    stack.append(hashed)
    prems = []

    if q.rule_name != problem.CONSTRUCTION_RULE:
      all_deps = []
      dep_names = set()
      for d in q.why:
        if d.hashed() in dep_names:
          continue
        dep_names.add(d.hashed())
        all_deps.append(d)

      for d in all_deps:
        h = d.hashed()
        if h not in visited:
          read(d)
        if h in visited:
          prems.append(d)

    visited.add(hashed)
    hashs = sorted([d.hashed() for d in prems])
    found = False
    for ps, qs in log:
      if sorted([d.hashed() for d in ps]) == hashs:
        qs += [q]
        found = True
        break
    if not found:
      log.append((prems, [q]))

    stack.pop(-1)

  read(query)

  # post process log: separate multi-conclusion lines
  log_, log = log, []
  for ps, qs in log_:
    for q in qs:
      log.append((ps, [q]))

  return log


def collx_to_coll_setup(
    setup: list[problem.Dependency],
) -> list[problem.Dependency]:
  """Convert collx to coll in setups."""
  result = []
  for level in setup_to_levels(setup):
    hashs = set()
    for dep in level:
      if dep.name == 'collx':
        dep.name = 'coll'
        dep.args = list(set(dep.args))

      if dep.hashed() in hashs:
        continue
      hashs.add(dep.hashed())
      result.append(dep)

  return result


def collx_to_coll(
    setup: list[problem.Dependency],
    aux_setup: list[problem.Dependency],
    log: list[tuple[list[problem.Dependency], list[problem.Dependency]]],
) -> tuple[
    list[problem.Dependency],
    list[problem.Dependency],
    list[tuple[list[problem.Dependency], list[problem.Dependency]]],
]:
  """Convert collx to coll and dedup."""
  setup = collx_to_coll_setup(setup)
  aux_setup = collx_to_coll_setup(aux_setup)

  con_set = set([p.hashed() for p in setup + aux_setup])
  log_, log = log, []
  for prems, cons in log_:
    prem_set = set()
    prems_, prems = prems, []
    for p in prems_:
      if p.name == 'collx':
        p.name = 'coll'
        p.args = list(set(p.args))
      if p.hashed() in prem_set:
        continue
      prem_set.add(p.hashed())
      prems.append(p)

    cons_, cons = cons, []
    for c in cons_:
      if c.name == 'collx':
        c.name = 'coll'
        c.args = list(set(c.args))
      if c.hashed() in con_set:
        continue
      con_set.add(c.hashed())
      cons.append(c)

    if not cons or not prems:
      continue

    log.append((prems, cons))

  return setup, aux_setup, log


def get_logs(
    query: problem.Dependency, g: Any, merge_trivials: bool = False
) -> tuple[
    list[problem.Dependency],
    list[problem.Dependency],
    list[tuple[list[problem.Dependency], list[problem.Dependency]]],
    set[gm.Point],
]:
  """Given a DAG and conclusion N, return the premise, aux, proof."""
  query = query.why_me_or_cache(g, query.level)
  log = recursive_traceback(query)
  log, setup, aux_setup, setup_points, _ = separate_dependency_difference(
      query, log
  )

  setup, aux_setup, log = collx_to_coll(setup, aux_setup, log)

  setup, aux_setup, log = shorten_and_shave(
      setup, aux_setup, log, merge_trivials
  )

  return setup, aux_setup, log, setup_points


def shorten_and_shave(
    setup: list[problem.Dependency],
    aux_setup: list[problem.Dependency],
    log: list[tuple[list[problem.Dependency], list[problem.Dependency]]],
    merge_trivials: bool = False,
) -> tuple[
    list[problem.Dependency],
    list[problem.Dependency],
    list[tuple[list[problem.Dependency], list[problem.Dependency]]],
]:
  """Shorten the proof by removing unused predicates."""
  log, _ = shorten_proof(log, merge_trivials=merge_trivials)

  all_prems = sum([list(prems) for prems, _ in log], [])
  all_prems = set([p.hashed() for p in all_prems])
  setup = [d for d in setup if d.hashed() in all_prems]
  aux_setup = [d for d in aux_setup if d.hashed() in all_prems]
  return setup, aux_setup, log


def join_prems(
    con: problem.Dependency,
    con2prems: dict[tuple[str, ...], list[problem.Dependency]],
    expanded: set[tuple[str, ...]],
) -> list[problem.Dependency]:
  """Join proof steps with the same premises."""
  h = con.hashed()
  if h in expanded or h not in con2prems:
    return [con]

  result = []
  for p in con2prems[h]:
    result += join_prems(p, con2prems, expanded)
  return result


def shorten_proof(
    log: list[tuple[list[problem.Dependency], list[problem.Dependency]]],
    merge_trivials: bool = False,
) -> tuple[
    list[tuple[list[problem.Dependency], list[problem.Dependency]]],
    dict[tuple[str, ...], list[problem.Dependency]],
]:
  """Join multiple trivials proof steps into one."""
  pops = set()
  con2prem = {}
  for prems, cons in log:
    assert len(cons) == 1
    con = cons[0]
    if con.rule_name == '':  # pylint: disable=g-explicit-bool-comparison
      con2prem[con.hashed()] = prems
    elif not merge_trivials:
      # except for the ones that are premises to non-trivial steps.
      pops.update({p.hashed() for p in prems})

  for p in pops:
    if p in con2prem:
      con2prem.pop(p)

  expanded = set()
  log2 = []
  for i, (prems, cons) in enumerate(log):
    con = cons[0]
    if i < len(log) - 1 and con.hashed() in con2prem:
      continue

    hashs = set()
    new_prems = []

    for p in sum([join_prems(p, con2prem, expanded) for p in prems], []):
      if p.hashed() not in hashs:
        new_prems.append(p)
        hashs.add(p.hashed())

    log2 += [(new_prems, [con])]
    expanded.add(con.hashed())

  return log2, con2prem
