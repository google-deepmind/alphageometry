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

"""Implements the combination DD+AR."""
import time

from absl import logging
import dd
import graph as gh
import problem as pr
from problem import Dependency  # pylint: disable=g-importing-member
import trace_back


def saturate_or_goal(
    g: gh.Graph,
    theorems: list[pr.Theorem],
    level_times: list[float],
    p: pr.Problem,
    max_level: int = 100,
    timeout: int = 600,
) -> tuple[
    list[dict[str, list[tuple[gh.Point, ...]]]],
    list[dict[str, list[tuple[gh.Point, ...]]]],
    list[int],
    list[pr.Dependency],
]:
  """Run DD until saturation or goal found."""
  derives = []
  eq4s = []
  branching = []
  all_added = []

  while len(level_times) < max_level:
    level = len(level_times) + 1

    t = time.time()
    added, derv, eq4, n_branching = dd.bfs_one_level(
        g, theorems, level, p, verbose=False, nm_check=True, timeout=timeout
    )
    all_added += added
    branching.append(n_branching)

    derives.append(derv)
    eq4s.append(eq4)
    level_time = time.time() - t

    logging.info(f'Depth {level}/{max_level} time = {level_time}')  # pylint: disable=logging-fstring-interpolation
    level_times.append(level_time)

    if p.goal is not None:
      goal_args = list(map(lambda x: g.get(x, lambda: int(x)), p.goal.args))
      if g.check(p.goal.name, goal_args):  # found goal
        break

    if not added:  # saturated
      break

    if level_time > timeout:
      break

  return derives, eq4s, branching, all_added


def solve(
    g: gh.Graph,
    theorems: list[pr.Problem],
    controller: pr.Problem,
    max_level: int = 1000,
    timeout: int = 600,
) -> tuple[gh.Graph, list[float], str, list[int], list[pr.Dependency]]:
  """Alternate between DD and AR until goal is found."""
  status = 'saturated'
  level_times = []

  dervs, eq4 = g.derive_algebra(level=0, verbose=False)
  derives = [dervs]
  eq4s = [eq4]
  branches = []
  all_added = []

  while len(level_times) < max_level:
    dervs, eq4, next_branches, added = saturate_or_goal(
        g, theorems, level_times, controller, max_level, timeout=timeout
    )
    all_added += added

    derives += dervs
    eq4s += eq4
    branches += next_branches

    # Now, it is either goal or saturated
    if controller.goal is not None:
      goal_args = g.names2points(controller.goal.args)
      if g.check(controller.goal.name, goal_args):  # found goal
        status = 'solved'
        break

    if not derives:  # officially saturated.
      break

    # Now we resort to algebra derivations.
    added = []
    while derives and not added:
      added += dd.apply_derivations(g, derives.pop(0))

    if added:
      continue

    # Final help from AR.
    while eq4s and not added:
      added += dd.apply_derivations(g, eq4s.pop(0))

    all_added += added

    if not added:  # Nothing left. saturated.
      break

  return g, level_times, status, branches, all_added


def get_proof_steps(
    g: gh.Graph, goal: pr.Clause, merge_trivials: bool = False
) -> tuple[
    list[pr.Dependency],
    list[pr.Dependency],
    list[tuple[list[pr.Dependency], list[pr.Dependency]]],
    dict[tuple[str, ...], int],
]:
  """Extract proof steps from the built DAG."""
  goal_args = g.names2nodes(goal.args)
  query = Dependency(goal.name, goal_args, None, None)

  setup, aux, log, setup_points = trace_back.get_logs(
      query, g, merge_trivials=merge_trivials
  )

  refs = {}
  setup = trace_back.point_log(setup, refs, set())
  aux = trace_back.point_log(aux, refs, setup_points)

  setup = [(prems, [tuple(p)]) for p, prems in setup]
  aux = [(prems, [tuple(p)]) for p, prems in aux]

  return setup, aux, log, refs
