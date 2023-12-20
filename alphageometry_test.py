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

"""Unit tests for alphageometry.py."""

import unittest

from absl.testing import absltest
import alphageometry


class AlphaGeometryTest(unittest.TestCase):

  def test_translate_constrained_to_constructive(self):
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'T', list('addb')
        ),
        ('on_dia', ['d', 'b', 'a']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'T', list('adbc')
        ),
        ('on_tline', ['d', 'a', 'b', 'c']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'P', list('bcda')
        ),
        ('on_pline', ['d', 'a', 'b', 'c']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'D', list('bdcd')
        ),
        ('on_bline', ['d', 'c', 'b']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'D', list('bdcb')
        ),
        ('on_circle', ['d', 'b', 'c']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'D', list('bacd')
        ),
        ('eqdistance', ['d', 'c', 'b', 'a']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'C', list('bad')
        ),
        ('on_line', ['d', 'b', 'a']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'C', list('bad')
        ),
        ('on_line', ['d', 'b', 'a']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'O', list('abcd')
        ),
        ('on_circum', ['d', 'a', 'b', 'c']),
    )

  def test_insert_aux_to_premise(self):
    pstring = 'a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b ? perp a d b c'  # pylint: disable=line-too-long
    auxstring = 'e = on_line e a c, on_line e b d'

    target = 'a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b; e = on_line e a c, on_line e b d ? perp a d b c'  # pylint: disable=line-too-long
    self.assertEqual(
        alphageometry.insert_aux_to_premise(pstring, auxstring), target
    )

  def test_beam_queue(self):
    beam_queue = alphageometry.BeamQueue(max_size=2)

    beam_queue.add('a', 1)
    beam_queue.add('b', 2)
    beam_queue.add('c', 3)

    beam_queue = list(beam_queue)
    self.assertEqual(beam_queue, [(3, 'c'), (2, 'b')])


if __name__ == '__main__':
  absltest.main()
