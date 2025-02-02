import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

plt.style.use('tableau-colorblind10')


x = [278.37,13.73, 9.05, 40.89, 15.24]
y = [0.55, 0.2, 0.38, 0.24, 0.79]

# size and color:
sizes = [155, 155, 155, 155, 155]
colors = [
       (0, 0.24705882352941178, 0.3607843137254902),#'#003f5c',
       (0.34509803921568627, 0.3137254901960784, 0.5529411764705883), #'#58508d',
       (0.7372549019607844, 0.3137254901960784, 0.5647058823529412), #'#bc5090',
       (1, 0.38823529411764707, 0.3803921568627451), #'"#ff6361',
       (1, 0.6509803921568628, 0) #'#ffa600'
]



labels = ['DFD', 'DD', 'NL', 'UM', 'GPT-4']

# plot
fig, ax = plt.subplots()

scatter = ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

# Create legend
for color, label in zip(colors, labels):
    ax.scatter([], [], color=color, label=label)  # Add invisible points for legend

ax.legend(title="Approaches", loc="upper right")


# Add axis labels
ax.set_xlabel("Execution Time (seconds)", fontsize=12)
ax.set_ylabel("Effectiveness ($F_3$ Score)", fontsize=12)

ax.set(xlim=(0, 290), xticks=np.arange(1, 280, 25),
       ylim=(0, 1), yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


# Save the plot as a PDF
plt.savefig("scatter_plot.png", format="png", bbox_inches="tight")

plt.show()
