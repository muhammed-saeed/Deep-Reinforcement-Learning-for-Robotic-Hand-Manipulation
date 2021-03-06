import numpy as np
import matplotlib.pyplot as plt

scores = np.loadtxt('/home/muhammed/Documents/Ubuntu%2016%20folders/hindsight-experience-replay-master/HandManipulateBlockRotateZ_v0_nepochs_450_n_cycles_50_n_batches_40_clip_return_100.txt')
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Sucess rate')
plt.xlabel('Epoch')
plt.show()