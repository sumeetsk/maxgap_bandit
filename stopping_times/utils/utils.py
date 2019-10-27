import numpy as np
M_large = 10000 #large number
def calculate_ci(pulls, delta, K, sigma):
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
    K = len(lcbs)
    for k in range(K):
        ub.append(left_gap_ub(lcbs, rcbs, k))
    return ub


def get_right_gap_ub(lcbs, rcbs):
    ub = []
    K = len(lcbs)
    for k in range(K):
        ub.append(right_gap_ub(lcbs, rcbs, k))
    return ub

def get_active_arms(left_gap_ub, right_gap_ub,best_lower_bound):
    K = len(left_gap_ub)
    active = [1. for x in range(K)]
    for k in range(K):
        if max(right_gap_ub[k], left_gap_ub[k]) \
                <= best_lower_bound:
            active[k] = 0.
        else:
            active[k] = 1.
    return active

