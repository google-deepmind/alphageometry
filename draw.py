import unittest
from absl.testing import absltest
import dd
import graph as gh
import problem as pr

txt = 'f g h i j = pentagon f g h i j; a = intersection_ll a f j g h; b = intersection_ll b f g h i; c = intersection_ll c g h i j; d = intersection_ll d h i f j; e = intersection_ll e f g i j; o1 = circle o1 a f g; o2 = circle o2 b g h; o3 = circle o3 c h i; o4 = circle o4 d j i; o5 = circle o5 e j f; k = intersection_cc k o5 o1 f; l = intersection_cc l o1 o2 g; m = intersection_cc m o2 o3 h; n = intersection_cc n o3 o4 i; o = intersection_cc o o4 o5 j; ox = circle k l m ? cyclic k l m n'
defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)
p = pr.Problem.from_txt(txt,translate=False)
g, _ = gh.Graph.build_problem(p, defs)
goal_args = g.names2nodes(p.goal.args)
gh.nm.draw(
      g.type2nodes[gh.Point],
      g.type2nodes[gh.Line],
      g.type2nodes[gh.Circle],
      g.type2nodes[gh.Segment])