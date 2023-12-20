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

"""Unit tests for graph.py."""
import unittest

from absl.testing import absltest
import graph as gh
import numericals as nm
import problem as pr


MAX_LEVEL = 1000


class GraphTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)
    cls.rules = pr.Theorem.from_txt_file('rules.txt', to_dict=True)

    # load a complex setup:
    txt = 'a b c = triangle a b c; h = orthocenter a b c; h1 = foot a b c; h2 = foot b c a; h3 = foot c a b; g1 g2 g3 g = centroid g1 g2 g3 g a b c; o = circle a b c ? coll h g o'  # pylint: disable=line-too-long
    p = pr.Problem.from_txt(txt, translate=False)
    cls.g, _ = gh.Graph.build_problem(p, GraphTest.defs)

  def test_build_graph_points(self):
    g = GraphTest.g

    all_points = g.all_points()
    all_names = [p.name for p in all_points]
    self.assertCountEqual(
        all_names,
        ['a', 'b', 'c', 'g', 'h', 'o', 'g1', 'g2', 'g3', 'h1', 'h2', 'h3'],
    )

  def test_build_graph_predicates(self):
    gr = GraphTest.g

    a, b, c, g, h, o, g1, g2, g3, h1, h2, h3 = gr.names2points(
        ['a', 'b', 'c', 'g', 'h', 'o', 'g1', 'g2', 'g3', 'h1', 'h2', 'h3']
    )

    # Explicit statements:
    self.assertTrue(gr.check_cong([b, g1, g1, c]))
    self.assertTrue(gr.check_cong([c, g2, g2, a]))
    self.assertTrue(gr.check_cong([a, g3, g3, b]))
    self.assertTrue(gr.check_perp([a, h1, b, c]))
    self.assertTrue(gr.check_perp([b, h2, c, a]))
    self.assertTrue(gr.check_perp([c, h3, a, b]))
    self.assertTrue(gr.check_cong([o, a, o, b]))
    self.assertTrue(gr.check_cong([o, b, o, c]))
    self.assertTrue(gr.check_cong([o, a, o, c]))
    self.assertTrue(gr.check_coll([a, g, g1]))
    self.assertTrue(gr.check_coll([b, g, g2]))
    self.assertTrue(gr.check_coll([g1, b, c]))
    self.assertTrue(gr.check_coll([g2, c, a]))
    self.assertTrue(gr.check_coll([g3, a, b]))
    self.assertTrue(gr.check_perp([a, h, b, c]))
    self.assertTrue(gr.check_perp([b, h, c, a]))

    # These are NOT part of the premises:
    self.assertFalse(gr.check_perp([c, h, a, b]))
    self.assertFalse(gr.check_coll([c, g, g3]))

    # These are automatically inferred by the graph datastructure:
    self.assertTrue(gr.check_eqangle([a, h1, b, c, b, h2, c, a]))
    self.assertTrue(gr.check_eqangle([a, h1, b, h2, b, c, c, a]))
    self.assertTrue(gr.check_eqratio([b, g1, g1, c, c, g2, g2, a]))
    self.assertTrue(gr.check_eqratio([b, g1, g1, c, o, a, o, b]))
    self.assertTrue(gr.check_para([a, h, a, h1]))
    self.assertTrue(gr.check_para([b, h, b, h2]))
    self.assertTrue(gr.check_coll([a, h, h1]))
    self.assertTrue(gr.check_coll([b, h, h2]))

  def test_enumerate_colls(self):
    g = GraphTest.g

    for a, b, c in g.all_colls():
      self.assertTrue(g.check_coll([a, b, c]))
      self.assertTrue(nm.check_coll([a.num, b.num, c.num]))

  def test_enumerate_paras(self):
    g = GraphTest.g

    for a, b, c, d in g.all_paras():
      self.assertTrue(g.check_para([a, b, c, d]))
      self.assertTrue(nm.check_para([a.num, b.num, c.num, d.num]))

  def test_enumerate_perps(self):
    g = GraphTest.g

    for a, b, c, d in g.all_perps():
      self.assertTrue(g.check_perp([a, b, c, d]))
      self.assertTrue(nm.check_perp([a.num, b.num, c.num, d.num]))

  def test_enumerate_congs(self):
    g = GraphTest.g

    for a, b, c, d in g.all_congs():
      self.assertTrue(g.check_cong([a, b, c, d]))
      self.assertTrue(nm.check_cong([a.num, b.num, c.num, d.num]))

  def test_enumerate_eqangles(self):
    g = GraphTest.g

    for a, b, c, d, x, y, z, t in g.all_eqangles_8points():
      self.assertTrue(g.check_eqangle([a, b, c, d, x, y, z, t]))
      self.assertTrue(
          nm.check_eqangle(
              [a.num, b.num, c.num, d.num, x.num, y.num, z.num, t.num]
          )
      )

  def test_enumerate_eqratios(self):
    g = GraphTest.g

    for a, b, c, d, x, y, z, t in g.all_eqratios_8points():
      self.assertTrue(g.check_eqratio([a, b, c, d, x, y, z, t]))
      self.assertTrue(
          nm.check_eqratio(
              [a.num, b.num, c.num, d.num, x.num, y.num, z.num, t.num]
          )
      )

  def test_enumerate_cyclics(self):
    g = GraphTest.g

    for a, b, c, d, x, y, z, t in g.all_cyclics():
      self.assertTrue(g.check_cyclic([a, b, c, d, x, y, z, t]))
      self.assertTrue(nm.check_cyclic([a.num, b.num, c.num, d.num]))

  def test_enumerate_midps(self):
    g = GraphTest.g

    for a, b, c in g.all_midps():
      self.assertTrue(g.check_midp([a, b, c]))
      self.assertTrue(nm.check_midp([a.num, b.num, c.num]))

  def test_enumerate_circles(self):
    g = GraphTest.g

    for a, b, c, d in g.all_circles():
      self.assertTrue(g.check_circle([a, b, c, d]))
      self.assertTrue(nm.check_circle([a.num, b.num, c.num, d.num]))


if __name__ == '__main__':
  absltest.main()
