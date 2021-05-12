#!/usr/bin/env python3

import json

from benchbot_api import ActionResult, Agent, BenchBot
from votenet_helper import votenet_build, votenet_detection, votenet_nms

class MyAgent(Agent):
    def __init__(self):
        self._votenet_0 = votenet_build(flag=0)
        # self._votenet_1 = votenet_build(flag=1)
        self._raw_results = []
        self.frame = 0

    def is_done(self, action_result):
        return action_result != ActionResult.SUCCESS

    def pick_action(self, observations, action_list):
        self.frame += 1
        print(f"\n######### Frame {self.frame} ######\n")
        results_0 = votenet_detection(self._votenet_0, observations, flag=0)
        # results_1 = votenet_detection(self._votenet_1, observations, flag=1)
        ##### TODO: Use NMS here
        results = []
        if not None in results_0:
            results += results_0
        # if not None in results_1:
        #     results += results_1
        # print("%d objects in the frame: \n%s\n" % 
        #       (len(results), ",".join(r['class'] for r in results)))
        self._raw_results.append(results)
        return 'move_next', {}

    def save_result(self, filename, empty_results, results_format_fns):
        empty_results['results']['objects'] = votenet_nms(self._raw_results, empty_results['results']['class_list'])
        with open(filename, 'w') as f:
            json.dump(empty_results, f)

if __name__ == '__main__':
    print("Welcome to my Semantic SLAM solution!")
    BenchBot(agent=MyAgent()).run()