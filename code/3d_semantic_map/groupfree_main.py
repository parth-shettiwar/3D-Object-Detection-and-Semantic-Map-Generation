#!/usr/bin/env python3

import json

from benchbot_api import ActionResult, Agent, BenchBot
from groupfree_helper import groupfree_build, groupfree_detection, groupfree_nms

class MyAgent(Agent):
    def __init__(self):
        self._groupfree = groupfree_build()
        self._raw_results = []
        self.frame = 0

    def is_done(self, action_result):
        return action_result != ActionResult.SUCCESS

    def pick_action(self, observations, action_list):
        self.frame += 1
        print(f"\n######### Frame {self.frame} ######\n")
        results = groupfree_detection(self._groupfree, observations)
        print("Detectesd %d objects in the frame: %s" % 
              (len(results), ",".join(r['class'] for r in results)))
        self._raw_results.append(results)
        return 'move_next', {}

    def save_result(self, filename, empty_results, results_format_fns):
        empty_results['results']['objects'] = groupfree_nms(self._raw_results, self._groupfree, empty_results['results']['class_list'])
        with open(filename, 'w') as f:
            json.dump(empty_results, f)

if __name__ == '__main__':
    print("Welcome to my Semantic SLAM solution!")
    BenchBot(agent=MyAgent()).run()