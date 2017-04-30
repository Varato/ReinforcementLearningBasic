import numpy as np
import operator
import matplotlib.pyplot as plt

s_record = np.loadtxt("s_record.csv")
action_record = np.loadtxt("action_record.csv")

arrows = {0:"^",1:">",2:"v",3:"<"}
labelx=[str(xx) for xx in range(10)]
labely=list(reversed([str(xx) for xx in range(10)]))

fig, ax = plt.subplots()
for i in range(11):
    ax.plot([0,10],[i,i],color="k")
    ax.plot([i,i],[0,10],color="k")

ax.plot(s_record[:,1]+0.5, 9.5-s_record[:,0],color="r", linewidth=5)
for i in range(len(action_record)):
	aa = action_record[i]
	ax.plot(s_record[i,1]+0.5, 9.5-s_record[i,0], marker=arrows[aa], color="blue", markersize=10)

plt.axis([-1,11,-1,11])
k=0
for x in np.arange(0.5,10.5,1):
	plt.text(x, 10.5, labelx[k])
	plt.text(-0.5, x, labely[k])
	k+=1
plt.axis("off")
plt.axis("equal")
plt.savefig("optimal_path.png",dpi=200)
plt.show()