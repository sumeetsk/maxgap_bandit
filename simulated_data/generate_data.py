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
delta = 0.1
sigma = 0.05
M_large = 10000 #large number

def calculate_ci(pulls, delta, K):
    beta = 0.5; a = 1.+10./K;
    if pulls > 0:
        ci = (1.+beta) * sigma * np.sqrt(2.*np.log( np.log( 5.*pulls
                                                    /delta)) / pulls)
    else:
        ci = M_large
    return ci


def get_best_lower_bound(lcbs, rcbs):
    best_lower_bound = 0.
    K = len(lcbs)
    leftmost = min([(x, lcbs[x]) for x in range(K)],
                   key=lambda s: s[1])[0]
    for k in range(K):
        # find all arms with rcb to the right of my lcb. They should also
        # have their lcb to the left of my lcb.
        exists_overlap_on_left = [x for x in range(K) if
                          rcbs[x] >= lcbs[k] >= lcbs[x] and x != k]
        if len(exists_overlap_on_left) == 0:
            if leftmost != k:
                # valid lower bound exists
                lower_bound = min([lcbs[k]-rcbs[x] for x in range(K)
                                   if rcbs[x] < lcbs[k]])
                best_lower_bound = max(lower_bound, best_lower_bound)
    return best_lower_bound


def left_gap_ub_fixedpos(lcbs, rcbs, curarm, curpos):
    # return the maximum possible left gap when curarm is at curpos
    maxleftgap = float("inf")
    for i in range(len(lcbs)):
        if rcbs[i] < curpos and i!=curarm:
            maxleftgap = min(maxleftgap, curpos-lcbs[i])

    if maxleftgap == float("inf"):
        lcbs_on_left = [lcbs[i] for i in range(len(lcbs)) if
                        lcbs[i] < curpos and i != curarm]
        if len(lcbs_on_left) > 0:
            maxleftgap = max([curpos-x for x in lcbs_on_left])

    return maxleftgap


def left_gap_ub(lcbs, rcbs, curarm):
    # there are 4 possibilities for any arm's rcb.
    # 1) It lies in curarm's CI
    # 2) It lies to the right of curarm's rcb
    # 3) It lies to the left of curarm's lcb
    # 4) curarm is the leftmost arm
    maxleftgap = 0
    l = lcbs[curarm]; r = rcbs[curarm]
    interesting_positions = [r] + [rcbs[i] for i in range(len(rcbs)) if
                             l <= rcbs[i] <= r and i != curarm]
    for int_pos in interesting_positions:
        maxleftgap = max(maxleftgap,
                     left_gap_ub_fixedpos(lcbs, rcbs, curarm, int_pos))
    if maxleftgap == float("inf"):
        maxleftgap = 0
    return maxleftgap


def right_gap_ub_fixedpos(lcbs, rcbs, curarm, curpos):
    # return the maximum possible right gap when curarm is at curpos
    maxrightgap = float("inf")
    for i in range(len(lcbs)):
        if lcbs[i] > curpos and i != curarm:
            maxrightgap = min(maxrightgap, rcbs[i]-curpos)

    if maxrightgap == float("inf"):
    # there are no arms whose lcb is to my right
        rcbs_on_right = [rcbs[i] for i in range(len(rcbs)) if
                      rcbs[i] > curpos and i!=curarm]
        if len(rcbs_on_right) > 0:
            maxrightgap = max([x-curpos for x in rcbs_on_right])
        # no rcbs on right => it is the rightmost arm and maxrightgap should be
        # infinity
    return maxrightgap


def right_gap_ub(lcbs, rcbs, curarm):
    # there are 4 possibilities for any arm's lcb.
    # 1) inside curarm's CI
    # 2) left of curarm's lcb
    # 3) right of curarm's rcb
    # 4) curarm is the rightmost arm
    maxrightgap = 0.
    l = lcbs[curarm]
    r = rcbs[curarm]
    interesting_positions = [l] + [lcbs[i] for i in range(len(lcbs)) if
                            l <= lcbs[i] <= r and i != curarm]
    for int_pos in interesting_positions:
        maxrightgap = max(maxrightgap,
                      right_gap_ub_fixedpos(lcbs, rcbs, curarm, int_pos))
    if maxrightgap == float("inf"):
        # this is the rightmost arm
        maxrightgap = 0.
    return maxrightgap


def get_left_gap_ub(lcbs, rcbs):
    ub = []
    for k in range(K):
        ub.append(left_gap_ub(lcbs, rcbs, k))
    return ub


def get_right_gap_ub(lcbs, rcbs):
    ub = []
    for k in range(K):
        ub.append(right_gap_ub(lcbs, rcbs, k))
    return ub


class LUCB(object):
    def __str__(self):
        return 'LUCB'

    def __init__(self, K):
        self.K = K
        self.active = [1 for x in range(K)]
        self.t = 0
        self.arm_to_play = 0
        self.rcbs = 100*np.ones(K)+1.e-3*np.random.random(K)
        self.lcbs = np.zeros(K)+1.e-3*np.random.random(K)
        self.pulls = np.zeros(K)
        self.reward_estimates = [0. for x in range(K)]
        self.total_reward = np.zeros(K)
        self.left_gap_ub = [100. + np.random.random() for x in range(K)]
        self.right_gap_ub = [100. + np.random.random() for x in range(K)]
        self.best_lower_bound = 0.
        self.delta = delta
        self.ucb_arms = []
        self.flag_current_ucb_finished = 1
        self.counter_current_ucb = 0

    def get_arm(self):
        return self.arm_to_play

    def update(self, reward):
        arm = self.arm_to_play
        self.total_reward[arm] += reward
        self.pulls[arm] += 1.
        self.t += 1.
        self.reward_estimates[arm] = self.total_reward[arm] / self.pulls[arm]
        ci = calculate_ci(self.pulls[arm], self.delta/5., self.K)
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
        self.active = [1. for x in range(K)]
        for k in range(self.K):
            if max(self.right_gap_ub[k], self.left_gap_ub[k]) \
                    <= self.best_lower_bound:
                self.active[k] = 0.
            else:
                self.active[k] = 1.

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

class UCB(object):
    def __str__(self):
        return 'UCB'

    def __init__(self, K):
        self.K = K
        self.active = [1 for x in range(K)]
        self.t = 0
        self.arm_to_play = 0
        self.rcbs = 100*np.ones(K)+1.e-3*np.random.random(K)
        self.lcbs = np.zeros(K)+1.e-3*np.random.random(K)
        self.pulls = np.zeros(K)
        self.reward_estimates = [0. for x in range(K)]
        self.total_reward = np.zeros(K)
        self.left_gap_ub = [100. + np.random.random() for x in range(K)]
        self.right_gap_ub = [100. + np.random.random() for x in range(K)]
        self.best_lower_bound = 0.
        self.delta = delta
        self.ucb_arms = []
        self.flag_current_ucb_finished = 1
        self.counter_current_ucb = 0

    def get_arm(self):
        return self.arm_to_play

    def update(self, reward):
        arm = self.arm_to_play
        self.total_reward[arm] += reward
        self.pulls[arm] += 1.
        self.t += 1.
        self.reward_estimates[arm] = self.total_reward[arm] / self.pulls[arm]
        ci = calculate_ci(self.pulls[arm], self.delta/5., self.K)
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
        self.active = [1. for x in range(K)]
        for k in range(self.K):
            if max(self.right_gap_ub[k], self.left_gap_ub[k]) \
                    <= self.best_lower_bound:
                self.active[k] = 0.
            else:
                self.active[k] = 1.

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

class Elimination(object):
    def __str__(self):
        return 'Elimination'

    def __init__(self, K):
        self.K = K
        self.active = [1 for x in range(K)]
        self.t = 0
        self.arm_to_play = 0
        self.rcbs = 100*np.ones(K)+1.e-3*np.random.random(K)
        self.lcbs = np.zeros(K)+1.e-3*np.random.random(K)
        self.pulls = np.zeros(K)
        self.reward_estimates = [0. for x in range(K)]
        self.total_reward = np.zeros(K)
        self.left_gap_ub = [1. for x in range(K)]
        self.right_gap_ub = [1. for x in range(K)]
        self.best_lower_bound = 0.
        self.delta = delta

    def get_arm(self):
        return self.arm_to_play

    def update(self, reward):
        arm = self.arm_to_play
        self.total_reward[arm] += reward
        self.pulls[arm] += 1.
        self.t += 1.
        while True:
            self.arm_to_play = (self.arm_to_play+1) % self.K
            if self.arm_to_play == 0:
                self._update()
            if self.active[self.arm_to_play]:
                return

    def _update(self):
        self.reward_estimates = [self.total_reward[k]/self.pulls[k] if self.pulls[k]>0 else 0.\
                                 for k in range(self.K)]
        ci = np.array([calculate_ci(self.pulls[k], self.delta/5., self.K)
                       for k in range(self.K)])
        # ci = np.array([np.sqrt(
        #    2*np.log(self.K * self.pulls[k]**3/self.delta)/self.pulls[k])
        #    for k in range(self.K)])
        self.rcbs = np.array(self.reward_estimates) + ci
        self.lcbs = np.array(self.reward_estimates) - ci

        # lower bound
        self.best_lower_bound = get_best_lower_bound(self.lcbs, self.rcbs)

        # left gap ub
        self.left_gap_ub = get_left_gap_ub(self.lcbs, self.rcbs)

        #right gap ub
        self.right_gap_ub = get_right_gap_ub(self.lcbs, self.rcbs)

        self.active = [1. for x in range(K)]
        for k in range(self.K):
            if max(self.right_gap_ub[k], self.left_gap_ub[k]) \
                <= self.best_lower_bound:
                self.active[k] = 0.
            else:
                self.active[k] = 1.
        return

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


class Random(object):
    def __str__(self):
        return 'Random'

    def __init__(self, K):
        self.K = K
        self.total_reward = [0. for k in range(self.K)]
        self.reward_estimates = [0. for k in range(self.K)]
        self.pulls = [0. for k in range(self.K)]
        self.arm_to_play = 0
        self.t = 0.
        self.delta = delta

    def get_arm(self):
        return self.arm_to_play

    def update(self, reward):
        self.total_reward[self.arm_to_play] += reward
        self.pulls[self.arm_to_play] += 1.
        self.t += 1.
        self.arm_to_play = (self.arm_to_play + 1) % self.K
        return

    def get_data(self):
        self.reward_estimates = [self.total_reward[k]/self.pulls[k] if self.pulls[k]>0 else 0.\
                                 for k in range(self.K)]
        ci = np.array([calculate_ci(self.pulls[k], self.delta/5., self.K)
                       for k in range(self.K)])
        rcbs = np.array(self.reward_estimates) + ci
        lcbs = np.array(self.reward_estimates) - ci
        best_lower_bound = get_best_lower_bound(lcbs, rcbs)
        left_gap_ub = get_left_gap_ub(lcbs, rcbs)
        right_gap_ub = get_right_gap_ub(lcbs, rcbs)

        active = [1. for x in range(self.K)]
        for k in range(self.K):
            if max(right_gap_ub[k], left_gap_ub[k]) \
                    <= best_lower_bound:
                active[k] = 0.
            else:
                active[k] = 1.

        sorted_items = sorted(list(enumerate(self.reward_estimates)),
                              key=lambda x:x[1], reverse=True)
        max_gap = max([(i, sorted_items[i][1]-sorted_items[i+1][1]) for i in
                       range(self.K-1)], key=lambda x:x[1])
        top_arms = [sorted_items[i][0] for i in range(max_gap[0]+1)]
        bottom_arms = [sorted_items[i][0] for i in range(max_gap[0]+1,self.K)]
        obj = {'t': self.t, 'top_arms': set(top_arms),
               'bottom_arms': set(bottom_arms), 'active':active,
               'lcbs': lcbs, 'rcbs': rcbs, 'emp_means': self.reward_estimates,
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


def testrun(thetastar, algorithm, T, ind):
    alg = copy.deepcopy(algorithm)
    env = Env(thetastar)
    sim_data = {}
    interval = T // 100
    for t1 in range(1, T+1):
        arm = alg.get_arm()
        #print(arm)
        reward = env.pull_arm(arm)
        alg.update(reward)
        if t1 % interval == 0:
            obj = alg.get_data()
            sim_data[t1] = obj
    return sim_data

def single_sim(thetastar, T, algorithms, ind):
    np.random.seed()
    result = {}
    K = len(thetastar)
    result['thetastar'] = thetastar
    result['simulation_data'] = {}

    for alg in algorithms:
        alg_object = eval(alg+'(K)')
        result['simulation_data'][alg] = testrun(thetastar, alg_object, T, ind)
    return result

if __name__ == "__main__":
    gaps = [0.1] * 8 + [0.98] + [0.1] * 8 + [1.] + [0.1] * 6
    thetastar = np.cumsum(gaps[::-1])[::-1]
    T = 2*10**3
    K = len(thetastar)
    sorted_index = np.argsort(thetastar)[::-1]
    sorted_scores = np.sort(thetastar)[::-1]
    max_gap = max([(i, sorted_scores[i]-sorted_scores[i+1])
                   for i in range(K-1)], key=lambda x:x[1])
    top_cluster = set(sorted_index[:max_gap[0]+1])
    print(thetastar)
    print(top_cluster)

# plot the means
    fig, ax = plt.subplots()
    plt.scatter(thetastar, np.zeros_like(thetastar), marker='x')
    fig.patch.set_visible(False)
    fig.set_size_inches(5, .3)
    ax.axis('off')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('item_means.pdf', bbox_inches=extent)

    algorithms = ['LUCB', 'UCB', 'Elimination', 'Random']
    onecore = False
    nsims = 120

    #single
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
        iters = [pool.apply_async(single_sim, args=(thetastar, T,
                                  algorithms, ind)) for ind in range(nsims)]
        results = []

        for result in iters:
            results.append(result.get())

        with open('test.dat', 'wb') as f:
            pickle.dump(results, f)

