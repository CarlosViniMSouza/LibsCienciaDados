import matplotlib.pyplot as mtp
import numpy as np

Number = 5
men_Means = (20, 35, 30, 35, -10, -20)
women_Means = (25, 30, 35, 20, 15)
menStd = (1, 2, 3, 4, 3, 2, 1)
womenStd = (3, 5, 2, 5, 3, 1, 0)
ind = np.arange(Number) # the x locations for the groups
width = 0.35 # the width of the bars: can also be len(x) sequence

fig, ax = mtp.subplots()

check1 = ax.bar(ind, men_Means, width, yerr = menStd, label = 'Mens')
check2 = ax.bar(ind, women_Means, width,
            bottom = men_Means, yerr = womenStd, label = 'Womens')

ax.axhline(0, color = 'black', linewidth = 0.8)
ax.set_ylabel('Results')
ax.set_title('Results - group&gender')
ax.set_xticks(ind)
ax.set_xticklabels(('Group1', 'Group2', 'Group3', 'Group4', 'Group5'))
ax.legend()

# Label with label_type 'center' instead of the default 'edge'
ax.bar_label(check1, label_type = 'center')
ax.bar_label(check2, label_type = 'center')
ax.bar_label(check2)

mtp.show()