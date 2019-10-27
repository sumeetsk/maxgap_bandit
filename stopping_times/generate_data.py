import pandas as pd
import numpy as np
import copy
import pickle
import multiprocessing
from functools import partial
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pdb
from utils.utils import *
delta = 0.1
sigma = 0.05

class MaxGap(object):
    def __str__(self):
        raise NotImplementedError

    def __init__(self, K):
        self.K = K
        self.total_reward = np.zeros(K)
        self.reward_estimates = np.zeros(K)
        self.pulls = np.zeros(K)
        self.arm_to_play = 0
        self.t = 0.
        self.delta = delta
        self.rcbs = 100*np.ones(K)+1.e-3*np.random.random(K)
        self.lcbs = np.zeros(K)+1.e-3*np.random.random(K)

    def get_arm(self):
        return self.arm_to_play

    def update(self, reward):
        self.total_reward[self.arm_to_play] += reward
        self.pulls[self.arm_to_play] += 1.
        self.t += 1.
        arm = self.arm_to_play
        self.reward_estimates[arm] = self.total_reward[arm] / self.pulls[arm]
        ci = calculate_ci(self.pulls[arm], self.delta/5., self.K, sigma)
        self.rcbs[arm] = self.reward_estimates[arm] + ci
        self.lcbs[arm] = self.reward_estimates[arm] - ci

    def has_stopped(self):
        best_lower_bound = get_best_lower_bound(self.lcbs, self.rcbs)
        left_gap_ub = get_left_gap_ub(self.lcbs, self.rcbs)
        right_gap_ub = get_right_gap_ub(self.lcbs, self.rcbs)
        max_gap_ub = sorted([max(left_gap_ub[i], right_gap_ub[i]) for i in range(len(left_gap_ub))], reverse=True)
        return best_lower_bound - max_gap_ub[2] 


class Top2UCB(MaxGap):
    def __str__(self):
        return 'Top2UCB'

    def __init__(self, K):
        super(Top2UCB, self).__init__(K)
        self.active = [1 for x in range(K)]
        self.left_gap_ub = [100. + np.random.random() for x in range(K)]
        self.right_gap_ub = [100. + np.random.random() for x in range(K)]
        self.best_lower_bound = 0.
        self.ucb_arms = []
        self.flag_current_ucb_finished = 1
        self.counter_current_ucb = 0

    def update(self, reward):
        super().update(reward)
        if self.t < self.K:
            self.arm_to_play = int(self.t)
            return

        if self.flag_current_ucb_finished:
            # left gap ub
            self.left_gap_ub = get_left_gap_ub(self.lcbs, self.rcbs)

            #right gap ub
            self.right_gap_ub = get_right_gap_ub(self.lcbs, self.rcbs)
            max_gap_ub = [(k, max(self.left_gap_ub[k], self.right_gap_ub[k]))
                          for k in range(self.K)]
            ucb_value = max(max_gap_ub, key=lambda s: s[1])[1]
            try:
                second_largest_ucb_value = max([x[1] for x in max_gap_ub if x[
                    1] != ucb_value])
                self.ucb_arms = [x[0] for x in max_gap_ub if x[1] == ucb_value or
                                 x[1] == second_largest_ucb_value]
            except ValueError:
                self.ucb_arms = [x[0] for x in max_gap_ub if x[1] == ucb_value]
            self.arm_to_play = self.ucb_arms[0]
            self.counter_current_ucb += 1
            self.flag_current_ucb_finished = 0
        else:
            self.arm_to_play = self.ucb_arms[self.counter_current_ucb]
            self.counter_current_ucb += 1

        if self.counter_current_ucb == len(self.ucb_arms):
            self.counter_current_ucb = 0
            self.flag_current_ucb_finished = 1
        return


    def get_data(self):
        self.best_lower_bound = get_best_lower_bound(self.lcbs, self.rcbs)
        self.active = get_active_arms(self.left_gap_ub, self.right_gap_ub, self.best_lower_bound)

        sorted_items = sorted(list(enumerate(self.reward_estimates)),
                              key=lambda x:x[1], reverse=True)
        max_gap = max([(i,sorted_items[i][1]-sorted_items[i+1][1]) for i in
                       range(self.K-1)], key=lambda x:x[1])
        top_arms = [sorted_items[i][0] for i in range(max_gap[0]+1)]
        bottom_arms = [sorted_items[i][0] for i in range(max_gap[0]+1,self.K)]
        obj = {'t':self.t,'top_arms':set(top_arms),
               'bottom_arms': set(bottom_arms),'active':self.active[:],
               'lcbs':self.lcbs[:], 'rcbs':self.rcbs[:],
               'emp_means':self.reward_estimates[:],
               'lower_bound': self.best_lower_bound,
               'left_gap_ub':self.left_gap_ub[:],
               'right_gap_ub':self.right_gap_ub[:],
               'pulls':copy.deepcopy(self.pulls[:])}
        return obj

class UCB(MaxGap):
    def __str__(self):
        return 'UCB'

    def __init__(self, K):
        super(UCB, self).__init__(K)
        self.active = [1 for x in range(K)]
        self.left_gap_ub = [100. + np.random.random() for x in range(K)]
        self.right_gap_ub = [100. + np.random.random() for x in range(K)]
        self.best_lower_bound = 0.
        self.ucb_arms = []
        self.flag_current_ucb_finished = 1
        self.counter_current_ucb = 0

    def update(self, reward):
        super().update(reward)
        arm = self.arm_to_play
        self.reward_estimates[arm] = self.total_reward[arm] / self.pulls[arm]
        ci = calculate_ci(self.pulls[arm], self.delta/5., self.K, sigma)
        self.rcbs[arm] = self.reward_estimates[arm] + ci
        self.lcbs[arm] = self.reward_estimates[arm] - ci

        if self.t < self.K:
            self.arm_to_play = int(self.t)
            return

        if self.flag_current_ucb_finished:
            # left gap ub
            self.left_gap_ub = get_left_gap_ub(self.lcbs, self.rcbs)

            #right gap ub
            self.right_gap_ub = get_right_gap_ub(self.lcbs, self.rcbs)
            max_gap_ub = [(k, max(self.left_gap_ub[k], self.right_gap_ub[k]))
                          for k in range(self.K)]
            ucb_value = max(max_gap_ub, key=lambda s : s[1])[1]
            self.ucb_arms = [x[0] for x in max_gap_ub if x[1]==ucb_value]
            self.arm_to_play = self.ucb_arms[0]
            self.counter_current_ucb += 1
            self.flag_current_ucb_finished = 0
        else:
            self.arm_to_play = self.ucb_arms[self.counter_current_ucb]
            self.counter_current_ucb += 1

        if self.counter_current_ucb == len(self.ucb_arms):
            self.counter_current_ucb = 0
            self.flag_current_ucb_finished = 1
        return


    def get_data(self):
        self.best_lower_bound = get_best_lower_bound(self.lcbs, self.rcbs)
        self.active = get_active_arms(self.left_gap_ub, self.right_gap_ub, self.best_lower_bound)

        sorted_items = sorted(list(enumerate(self.reward_estimates)),
                              key=lambda x:x[1], reverse=True)
        max_gap = max([(i,sorted_items[i][1]-sorted_items[i+1][1]) for i in
                       range(self.K-1)], key=lambda x:x[1])
        top_arms = [sorted_items[i][0] for i in range(max_gap[0]+1)]
        bottom_arms = [sorted_items[i][0] for i in range(max_gap[0]+1,self.K)]
        obj = {'t':self.t,'top_arms':set(top_arms),
               'bottom_arms': set(bottom_arms),'active':self.active[:],
               'lcbs':self.lcbs[:], 'rcbs':self.rcbs[:],
               'emp_means':self.reward_estimates[:],
               'lower_bound': self.best_lower_bound,
               'left_gap_ub':self.left_gap_ub[:],
               'right_gap_ub':self.right_gap_ub[:],
               'pulls':copy.deepcopy(self.pulls[:])}
        return obj

    def has_stopped(self):
        sorted_pulls = sorted(self.pulls, reverse=True)
        return sorted_pulls[0]+sorted_pulls[1]-5*(np.sum(sorted_pulls[2:]))


class Elimination(MaxGap):
    def __str__(self):
        return 'Elimination'

    def __init__(self, K):
        super(Elimination, self).__init__(K)
        self.active = [1 for x in range(K)]
        self.left_gap_ub = [1. for x in range(K)]
        self.right_gap_ub = [1. for x in range(K)]
        self.best_lower_bound = 0.

    def update(self, reward):
        super().update(reward)
        arm = self.arm_to_play
        while True:
            self.arm_to_play = (self.arm_to_play+1) % self.K
            if self.arm_to_play == 0:
                self._update()
            if self.active[self.arm_to_play]:
                return

    def _update(self):
        self.reward_estimates = [self.total_reward[k]/self.pulls[k] if self.pulls[k]>0 else 0.\
                                 for k in range(self.K)]
        ci = np.array([calculate_ci(self.pulls[k], self.delta/5., self.K, sigma)
                       for k in range(self.K)])
        self.rcbs = np.array(self.reward_estimates) + ci
        self.lcbs = np.array(self.reward_estimates) - ci

        self.best_lower_bound = get_best_lower_bound(self.lcbs, self.rcbs)
        self.left_gap_ub = get_left_gap_ub(self.lcbs, self.rcbs)
        self.right_gap_ub = get_right_gap_ub(self.lcbs, self.rcbs)

        self.active = get_active_arms(self.left_gap_ub, self.right_gap_ub, self.best_lower_bound)

    def get_data(self):
        sorted_items = sorted(list(enumerate(self.reward_estimates)),
                              key=lambda x:x[1], reverse=True)
        max_gap = max([(i, sorted_items[i][1]-sorted_items[i+1][1]) for i in
                       range(self.K-1)], key=lambda x: x[1])
        top_arms = [sorted_items[i][0] for i in range(max_gap[0]+1)]
        bottom_arms = [sorted_items[i][0] for i in range(max_gap[0]+1,self.K)]
        obj = {'t': self.t, 'top_arms': set(top_arms),
               'bottom_arms': set(bottom_arms), 'active': self.active[:],
               'lcbs': self.lcbs[:], 'rcbs': self.rcbs[:],
               'emp_means':self.reward_estimates[:],
               'lower_bound': self.best_lower_bound,
               'left_gap_ub':self.left_gap_ub[:],
               'right_gap_ub':self.right_gap_ub[:],
               'pulls':copy.deepcopy(self.pulls[:])}
        return obj


class Random(MaxGap):
    def __str__(self):
        return 'Random'

    def __init__(self, K):
        super(Random, self).__init__(K)

    def update(self, reward):
        super().update(reward)
        self.arm_to_play = (self.arm_to_play + 1) % self.K

    def get_data(self):
        best_lower_bound = get_best_lower_bound(self.lcbs, self.rcbs)
        left_gap_ub = get_left_gap_ub(self.lcbs, self.rcbs)
        right_gap_ub = get_right_gap_ub(self.lcbs, self.rcbs)

        active = get_active_arms(left_gap_ub, right_gap_ub, best_lower_bound)

        sorted_items = sorted(list(enumerate(self.reward_estimates)),
                              key=lambda x:x[1], reverse=True)
        max_gap = max([(i, sorted_items[i][1]-sorted_items[i+1][1]) for i in
                       range(self.K-1)], key=lambda x:x[1])
        top_arms = [sorted_items[i][0] for i in range(max_gap[0]+1)]
        bottom_arms = [sorted_items[i][0] for i in range(max_gap[0]+1,self.K)]
        obj = {'t': self.t, 'top_arms': set(top_arms),
               'bottom_arms': set(bottom_arms), 'active':active,
               'lcbs': self.lcbs, 'rcbs': self.rcbs, 'emp_means': self.reward_estimates,
               'lower_bound': best_lower_bound,
               'left_gap_ub': left_gap_ub, 'right_gap_ub': right_gap_ub,
               'pulls':copy.deepcopy(self.pulls[:])}
        return obj

#class Env(object):
#    def __init__(self, thetastar):
#        self.thetastar = thetastar
#
#    def pull_arm(self, k):
        #return np.random.random() < self.thetastar[k]

class Env(object):
    def __init__(self, thetastar):
        self.thetastar = thetastar
        self.stddev = sigma

    def pull_arm(self, k):
        return np.random.normal(self.thetastar[k], self.stddev)


def testrun(thetastar, algorithm, ind):
    alg = copy.deepcopy(algorithm)
    env = Env(thetastar)
    sim_data = {}
    #interval = T // 100
    interval = 100
    #for t1 in range(1, T+1):
    t1 = 1
    while True:
        arm = alg.get_arm()
        reward = env.pull_arm(arm)
        alg.update(reward)
        if t1 % interval == 0:
            stop_gap = alg.has_stopped()
            if stop_gap >= 0:
                break
        t1+=1
    return t1

def single_sim(thetastar, algorithms, ind):
    np.random.seed()
    result = {}
    K = len(thetastar)
    result['thetastar'] = thetastar
    result['simulation_data'] = {}

    for alg in algorithms:
        print(alg)
        alg_object = eval(alg+'(K)')
        result['simulation_data'][alg] = testrun(thetastar, alg_object, ind)
    return result

if __name__ == "__main__":
    thetastar1 = list(np.arange(0,0.11,0.02))
    thetastar2 = list(np.arange(0,0.11,0.02)) + list(np.arange(0.5,0.61,0.02))

    algorithms = ['Top2UCB', 'Elimination', 'Random', 'UCB']
    onecore = False
    nsims = 24

    #single
    all_expt_results = []
    for gap in [0.36, 0.37, 0.38, 0.384, 0.386, 0.388, 0.389, 0.39]:
    #for gap in list(np.logspace(np.log10(0.36),np.log10(0.391),8)):
        thetastar = thetastar1 + [thetastar1[-1]+gap+s for s in thetastar2]
        print(thetastar)

        if onecore:
            results = single_sim(thetastar, T, algorithms, 1)
            with open('test.dat', 'wb') as f:
                pickle.dump([results], f)
        # result is a dictionary. It contains algorithm as key, and a value. The
        # value is also a dictionary, which contains t and err as the keys.
        #    with open('test.dat','wb') as f:
        #        pickle.dump(results, f)
        else:
            pool = multiprocessing.Pool(processes=24)
            iters = [pool.apply_async(single_sim, args=(thetastar,
                                      algorithms, ind)) for ind in range(nsims)]
            results = []

            for result in iters:
                results.append(result.get())
        all_expt_results.append(results)

        with open('test.dat', 'wb') as f:
            pickle.dump(all_expt_results, f)
