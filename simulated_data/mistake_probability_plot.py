import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open('test.dat', 'rb') as f:
    x = pickle.load(f)
nsims = len(x)
thetastar = x[0]['thetastar']
K = len(thetastar)
sorted_index = np.argsort(thetastar)[::-1]
sorted_scores = np.sort(thetastar)[::-1]
max_gap = max([(i, sorted_scores[i]-sorted_scores[i+1])
               for i in range(K-1)], key=lambda x:x[1])
top_cluster = set(sorted_index[:max_gap[0]+1])
bottom_cluster = set(sorted_index[max_gap[0]+1:])

sim_data = x[0]['simulation_data']
algs = list(sim_data.keys())
alglabel = {'Random':'Random', 'Elimination':'Elimination', 'UCB':'UCB',
        'LUCB':'Top2UCB'}
times = list(sim_data[algs[0]].keys())

misclassification_prob = {}
size_active_set = {}
misclassification_count = {}
for alg in algs:
    misclassification_prob[alg] = np.array([[0. for time in times]
                                   for sim in range(nsims)])
    size_active_set[alg] = np.array([[0. for time in times]
                                    for sim in range(nsims)])
    misclassification_count[alg] = np.array([[0. for time in times]
                                     for sim in range(nsims)])

for sim in range(nsims):
    sim_data = x[sim]['simulation_data']
    for alg in sim_data:
        alg_data = sim_data[alg]
        for tind in range(len(times)):
            t = times[tind]
            misclassification_prob[alg][sim][tind] = \
                float(alg_data[t]['top_arms'] != top_cluster)
            size_active_set[alg][sim][tind] = \
                np.sum(alg_data[t]['active'])
            misclassification_count[alg][sim][tind] = \
                len(alg_data[t]['top_arms'].symmetric_difference(top_cluster))

fig = plt.figure(); ax = fig.add_subplot(111)
for alg in algs:
    ax.errorbar(times, np.mean(misclassification_prob[alg], axis=0),
                np.std(misclassification_prob[alg], axis=0)/np.sqrt(nsims),
                errorevery=1,
                label=alglabel[alg])
ax.set_ylabel('Mistake probability')
ax.set_xlabel('Queries')
plt.legend(loc='best')
plt.savefig('misclassification_probability.pdf')

fig = plt.figure(); ax = fig.add_subplot(111)
for alg in algs:
    ax.errorbar(times, np.mean(size_active_set[alg], axis=0),
                np.std(size_active_set[alg],axis=0)/np.sqrt(nsims),
                errorevery=1,
                label=alg)
ax.set_ylabel('Size of Active Set')
ax.set_xlabel('Queries')
plt.legend(loc='best')
plt.savefig('active_set_size.pdf')

fig = plt.figure(); ax = fig.add_subplot(111)
for alg in algs:
    ax.errorbar(times, np.mean(misclassification_count[alg], axis=0),
                np.std(size_active_set[alg], axis=0)/np.sqrt(nsims),
                errorevery=1,
                label=alglabel[alg])
ax.set_ylabel('Misclassification Count')
ax.set_xlabel('Queries')
plt.legend(loc='best')
plt.savefig('misclassification_count.pdf')
