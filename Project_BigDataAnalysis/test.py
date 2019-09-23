from matplotlib import pyplot as plt
import numpy as np

clusters = [{8: 1, 1: 20, 2: 153, 3: 1},{1: 1, 2: 4, 3: 8, 4: 4, 5: 116, 6: 1, 7: 1, 8: 94, 9: 7},{8: 4, 1: 1, 2: 5, 3: 156, 9: 4},{0: 2, 4: 161, 5: 1, 7: 1},{2: 5, 3: 8, 4: 7, 7: 168, 8: 2, 9: 5},{8: 20, 9: 141, 2: 2, 3: 10, 5: 62},{1: 53, 2: 2, 4: 3, 5: 2, 7: 8, 8: 9, 9: 23},{0: 176, 2: 1, 6: 1},{8: 2, 1: 4, 5: 1, 6: 175},{1: 103, 2: 5, 4: 6, 6: 4, 7: 1, 8: 42}]

idx = 9
cluster = clusters[idx]
x_axis = list(range(1, 11))
num_label = [cluster[i] if i in cluster else 0 for i in range(10)]
plt.bar(x_axis, num_label, width=0.5)
plt.title('Cluster #%d' % (idx+1))
plt.xlabel('Label')
plt.ylabel('Number of Label')
plt.xticks(np.arange(1, 11, 1), ["{}".format(x) for x in np.arange(0, 10)])
plt.savefig("C:\\Users\\hilsu\\Desktop\\빅데분\\hw3\\cluster"+str(idx+1)+".png")