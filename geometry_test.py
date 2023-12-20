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

"""Unit tests for geometry.py."""
import unittest

from absl.testing import absltest
import geometry as gm


class GeometryTest(unittest.TestCase):

  def _setup_equality_example(self):
    # Create 4 nodes a, b, c, d
    # and their lengths
    a = gm.Segment('a')
    la = gm.Length('l(a)')
    a.connect_to(la)
    la.connect_to(a)

    b = gm.Segment('b')
    lb = gm.Length('l(b)')
    b.connect_to(lb)
    lb.connect_to(b)

    c = gm.Segment('c')
    lc = gm.Length('l(c)')
    c.connect_to(lc)
    lc.connect_to(c)

    d = gm.Segment('d')
    ld = gm.Length('l(d)')
    d.connect_to(ld)
    ld.connect_to(d)

    # Now let a=b, b=c, a=c, c=d
    la.merge([lb], 'fact1')
    lb.merge([lc], 'fact2')
    la.merge([lc], 'fact3')
    lc.merge([ld], 'fact4')
    return a, b, c, d, la, lb, lc, ld

  def test_merged_node_representative(self):
    _, _, _, _, la, lb, lc, ld = self._setup_equality_example()

    # all nodes are now represented by la.
    self.assertEqual(la.rep(), la)
    self.assertEqual(lb.rep(), la)
    self.assertEqual(lc.rep(), la)
    self.assertEqual(ld.rep(), la)

  def test_merged_node_equivalence(self):
    _, _, _, _, la, lb, lc, ld = self._setup_equality_example()
    # all la, lb, lc, ld are equivalent
    self.assertCountEqual(la.equivs(), [la, lb, lc, ld])
    self.assertCountEqual(lb.equivs(), [la, lb, lc, ld])
    self.assertCountEqual(lc.equivs(), [la, lb, lc, ld])
    self.assertCountEqual(ld.equivs(), [la, lb, lc, ld])

  def test_bfs_for_equality_transitivity(self):
    a, _, _, d, _, _, _, _ = self._setup_equality_example()

    # check that a==d because fact3 & fact4, not fact1 & fact2
    self.assertCountEqual(gm.why_equal(a, d), ['fact3', 'fact4'])


if __name__ == '__main__':
  absltest.main()
