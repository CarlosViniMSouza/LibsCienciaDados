import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19587802)

# Example data
people = ('Patrick', 'Dick', 'Harry', 'Robert', 'Jimmy')
y_pos = np.arange(len(people))
performance = 5 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

# Reading Collecting
fig, ax = plt.subplots()

h_bars = ax.barh(y_pos, performance, xerr = error, align = 'center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go moment?!')

# Label with specially formatted floats
ax.bar_label(h_bars, fmt='%.2f')
ax.set_xlim(right=15)  # adjust xlim to fit labels
plt.show()