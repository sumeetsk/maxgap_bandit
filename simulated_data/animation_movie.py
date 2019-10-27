import pickle
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv
import pdb
import time

with open('test.dat', 'rb') as f:
    x = pickle.load(f)
x = x[0]
thetastar = x['thetastar']
K = len(thetastar)
sorted_index = np.argsort(thetastar)[::-1]
sorted_scores = np.sort(thetastar)[::-1]
max_gap = max([(i, sorted_scores[i]-sorted_scores[i+1])
               for i in range(K-1)], key=lambda x:x[1])
top_cluster = set(sorted_index[:max_gap[0]+1])
bottom_cluster = set(sorted_index[max_gap[0]+1:])

sim_data = x['simulation_data']
algs = list(sim_data.keys())[0]

alg = 'UCB'
times = list(sim_data[alg].keys())

plt.ion()
fig = plt.figure(); ax = fig.add_subplot(111)
for t in times:
    pulls = sim_data[alg][t]['pulls']
    plt.cla()
    ax.bar(range(K)[::-1], pulls)
    ax.set_ylim([0,1000])
    #ax.scatter(thetastar*2, np.zeros_like(thetastar), c=np.array(c), marker='.')
    ax.text(0, 275, 't:'+str(t))#+', '+'ucb:'+str(ucb_value))
    fig.canvas.draw()
    time.sleep(0.1)
    plt.pause(0.0001)
#    ucb_arms = sim_data[alg][t]['ucb_arms']
#    c = [1 if i in ucb_arms else 0 for i in range(K)]
#    ax.scatter(thetastar*2, np.zeros_like(thetastar), c=c)
#    fig.canvas.draw()
#    time.sleep(0.1)
#    plt.pause(0.0001)

#f = open(alg+'.txt', 'w')
#for t in times:
#    #ucb_arms = sim_data[alg][t]['ucb_arms']
#    #ucb_value = sim_data[alg][t]['ucb_value']
#    #x = {'t': t, 'ucb_arms': ucb_arms, 'ucb_value': round(ucb_value, 2)}
#    pulls = sim_data[alg][t]['pulls']
#    x = {'t': t, 'pulls': pulls}
#    f.write(str(x)+'\n')
#f.close()



#pdb.set_trace()
#misclassification_prob = {}
#size_active_set = {}
#misclassification_count = {}
#misclassification_prob = np.array([0. for time in times])
#size_active_set = np.array([0. for time in times])
#misclassification_count = np.array([0. for time in times])
#sim_data = x[0]['simulation_data']
#
#alg_data = sim_data[alg]
#for tind in range(len(times)):
#    t = times[tind]
#    misclassification_prob[tind] = \
#        float(alg_data[t]['top_arms'] != top_cluster)
#    size_active_set[tind] = \
#        np.sum(alg_data[t]['active'])
#    misclassification_count[tind] = \
#        len(alg_data[t]['top_arms'].symmetric_difference(top_cluster))
#
#fig = plt.figure(); ax = fig.add_subplot(111)
#for alg in algs:
#    ax.errorbar(times, np.mean(misclassification_prob[alg], axis=0),
#                np.std(misclassification_prob[alg], axis=0)/np.sqrt(nsims),
#                errorevery=1,
#                label=alglabel[alg])
#ax.set_ylabel('Mistake probability')
#ax.set_xlabel('Queries')
#plt.legend(loc='best')
#plt.savefig('misclassification_probability.pdf')


