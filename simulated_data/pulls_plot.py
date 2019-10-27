import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pdb

with open('test.dat', 'rb') as f:
    x = pickle.load(f)
    
Ts = [list(x[i]['simulation_data']['UCB'].keys()) for i in range(len(x))]
trunc_Ts = [Ts[i][:2000] for i in range(len(x))]
npT = np.array(trunc_Ts)

Ds = [[x[j]['simulation_data']['UCB'][trunc_Ts[j][k]]['pulls'] for j in range(npT.shape[0])] for k in range(npT.shape[1])]


avgD = np.average(np.array(Ds), axis=1)
stddevD = np.std(np.array(Ds), axis=1)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3,sharex='col',sharey='row')

x_labels = ['24   ', '19   ', '   18', '10   ', '  9', '1'] 
x_pos = [1,6,7,15,16,24]


t=1
ax1.bar(np.arange(24,0,-1), avgD[t,:], yerr=stddevD[t,:]/np.sqrt(120))
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels)
ax1.text(18,8,'t='+str(trunc_Ts[0][t]))

t=2
ax2.bar(np.arange(24,0,-1), avgD[t,:], yerr=stddevD[t,:]/np.sqrt(120))
ax2.set_xticks(x_pos)
ax2.set_xticklabels(x_labels)
ax2.text(18,8,'t='+str(trunc_Ts[0][t]))

t=3
ax3.bar(np.arange(24,0,-1), avgD[t,:], yerr=stddevD[t,:]/np.sqrt(120))
ax3.set_xticks(x_pos)
ax3.set_xticklabels(x_labels)
ax3.text(18,8,'t='+str(trunc_Ts[0][t]))

t=9
ax4.bar(np.arange(24,0,-1), avgD[t,:], yerr=stddevD[t,:]/np.sqrt(120))
ax4.text(17,190,'t='+str(trunc_Ts[0][t]))
t=19
ax5.bar(np.arange(24,0,-1), avgD[t,:], yerr=stddevD[t,:]/np.sqrt(120))
ax5.text(17,190,'t='+str(trunc_Ts[0][t]))
t=29
ax6.bar(np.arange(24,0,-1), avgD[t,:], yerr=stddevD[t,:]/np.sqrt(120))
ax6.text(17,190,'t='+str(trunc_Ts[0][t]))

fig.suptitle('Profile of samples allocated by MaxGapUCB')
fig.text(0.5,0.01,'Arm index')
fig.text(0.01, 0.5, 'Number of samples', va='center', rotation='vertical')

plt.savefig('pulls_plot.pdf')
