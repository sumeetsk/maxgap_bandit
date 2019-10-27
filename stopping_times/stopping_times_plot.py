#!/anaconda3/bin/python
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pdb

def alg_stop_gap(left_gap_ub, right_gap_ub, best_lower_bound):
    max_gap_ub = sorted([max(left_gap_ub[i], right_gap_ub[i]) for i in range(len(left_gap_ub))], reverse=True)
    return max_gap_ub[1] - best_lower_bound

def compute_hardness(thetastar):
    x = sorted(thetastar, reverse=True)
    gaps = [x[i]-x[i+1] for i in range(len(x)-1)]
    maxgap = max(gaps)
    argmaxgap = np.argmax(gaps)
    complexity = []
    #gap of gaps
    for i in range(len(x)):
        if i in [argmaxgap, argmaxgap+1]:
            continue
        else:
            smaller_arms = [y for y in x if y < x[i]]
            larger_arms = [y for y in x if y > x[i]]
            left_gaps = [x[i]-y for y in smaller_arms]
            right_gaps = [y-x[i] for y in larger_arms]
            left_complexities = [min(maxgap-y, y) for y in left_gaps]
            right_complexities = [min(maxgap-y, y) for y in right_gaps]
            left_complexity = max(left_complexities, default=np.infty)
            right_complexity = max(right_complexities, default=np.infty)
            complexity.append(min(left_complexity, right_complexity))
    return np.sum([1/s**2 for s in complexity])


with open('test.dat', 'rb') as f:
    all_expt_results = pickle.load(f)

all_algs = all_expt_results[0][0]['simulation_data'].keys()
all_expt_results_dict = {}
stopping_times = {}
hardness = []
for alg in all_algs:
    stopping_times[alg] = []

for (ind, x) in enumerate(all_expt_results):
    all_expt_results_dict[ind] = {}

    nsims = len(x)
    thetastar = x[0]['thetastar']
    hardness.append(compute_hardness(thetastar[:]))

    all_expt_results_dict[ind]['thetastar'] = thetastar
    all_expt_results_dict[ind]['stop_times'] = {}

    K = len(thetastar)
    sim_data = x[0]['simulation_data']
    algs = list(sim_data.keys())
    #algs = ['Random', 'Elimination', 'UCB', 'LUCB']
    alglabel = {'Random':'Random', 'Elimination':'Elimination', 'UCB':'UCB',
            'Top2UCB':'Top2UCB'}

    for alg in sim_data:
        all_expt_results_dict[ind]['stop_times'][alg] = []

    for sim in range(nsims):
        sim_data = x[sim]['simulation_data']
        for alg in sim_data:
            all_expt_results_dict[ind]['stop_times'][alg].append(sim_data[alg])

    for alg in sim_data:
        stopping_times[alg].append(all_expt_results_dict[ind]['stop_times'][alg])

font={'size': 18}
matplotlib.rc('font', **font)
fig = plt.figure(); ax = fig.add_subplot(111)
for alg in ['Random', 'Elimination', 'UCB', 'Top2UCB']:
    ax.errorbar(hardness,
            np.mean(np.array(stopping_times[alg]), axis=1),
            np.std(np.array(stopping_times[alg]), axis=1)/np.sqrt(len(stopping_times[alg][0])),
            marker='.',
            label=alg)
plt.legend(loc='best')
ax.set_xticks([20e3,25e3,30e3,35e3,40e3])
ax.set_xticklabels(['20K','25K','30K','35K','40K'])
ax.set_yticks([0, 20e3,40e3,60e3,80e3])
ax.set_yticklabels([0, '20K', '40K', '60K', '80K'])
ax.set_xlabel('Hardness parameter')
ax.set_ylabel('No. of samples before stopping')
plt.savefig('stopping_times1.pdf', bbox_inches='tight')

fig = plt.figure(); ax = fig.add_subplot(111)
cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for (i,alg) in enumerate(['Elimination', 'UCB', 'Top2UCB']):
    ax.errorbar(hardness,
            np.mean(np.array(stopping_times[alg]), axis=1),
            np.std(np.array(stopping_times[alg]), axis=1)/np.sqrt(len(stopping_times[alg][0])),
            marker='.', color=cmap[i+1],
            label=alg)
plt.legend(loc='best')
ax.set_xticks([20e3,25e3,30e3,35e3,40e3])
ax.set_xticklabels(['20K','25K','30K','35K','40K'])
ax.set_yticks([0, 5e3,10e3,15e3,20e3])
ax.set_yticklabels([0, '5K', '10K', '15K', '20K'])
ax.set_xlabel('Hardness parameter')
ax.set_ylabel('No. of samples before stopping')
plt.savefig('stopping_times2.pdf', bbox_inches='tight')
