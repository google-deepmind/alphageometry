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

"""Unit testing for the trace_back code."""

import unittest

from absl.testing import absltest
import ddar
import graph as gh
import problem as pr
import trace_back as tb


class TracebackTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)
    cls.rules = pr.Theorem.from_txt_file('rules.txt', to_dict=True)

  def test_orthocenter_dependency_difference(self):
    txt = 'a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b; e = on_line e a c, on_line e b d ? perp a d b c'  # pylint: disable=line-too-long
    p = pr.Problem.from_txt(txt)
    g, _ = gh.Graph.build_problem(p, TracebackTest.defs)

    ddar.solve(g, TracebackTest.rules, p)

    goal_args = g.names2nodes(p.goal.args)
    query = pr.Dependency(p.goal.name, goal_args, None, None)

    setup, aux, _, _ = tb.get_logs(query, g, merge_trivials=False)

    # Convert each predicates to its hash string:
    setup = [p.hashed() for p in setup]
    aux = [p.hashed() for p in aux]

    self.assertCountEqual(
        setup, [('perp', 'a', 'c', 'b', 'd'), ('perp', 'a', 'b', 'c', 'd')]
    )

    self.assertCountEqual(
        aux, [('coll', 'a', 'c', 'e'), ('coll', 'b', 'd', 'e')]
    )


if __name__ == '__main__':
  absltest.main()
