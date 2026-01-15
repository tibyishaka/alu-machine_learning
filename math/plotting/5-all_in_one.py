#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Create figure and grid
fig = plt.figure(figsize=(8, 8))
fig.suptitle("All in One", fontsize='x-small')

# Plot 1: Cubic
ax1 = plt.subplot(3, 2, 1)
ax1.plot(y0)
ax1.set_title("Cubic", fontsize='x-small')
ax1.set_xlabel("x", fontsize='x-small')
ax1.set_ylabel("y", fontsize='x-small')

# Plot 2: Scatter
ax2 = plt.subplot(3, 2, 2)
ax2.scatter(x1, y1, s=1)
ax2.set_title("Random Scatter", fontsize='x-small')
ax2.set_xlabel("x", fontsize='x-small')
ax2.set_ylabel("y", fontsize='x-small')

# Plot 3: Exponential decay (single)
ax3 = plt.subplot(3, 2, 3)
ax3.plot(x2, y2)
ax3.set_title("C-14 Decay", fontsize='x-small')
ax3.set_xlabel("Time (years)", fontsize='x-small')
ax3.set_ylabel("Fraction Remaining", fontsize='x-small')

# Plot 4: Exponential decay (two lines)
ax4 = plt.subplot(3, 2, 4)
ax4.plot(x3, y31, 'r--', label='C-14')
ax4.plot(x3, y32, 'g-', label='Ra-226')
ax4.set_title("Decay Comparison", fontsize='x-small')
ax4.set_xlabel("Time (years)", fontsize='x-small')
ax4.set_ylabel("Fraction Remaining", fontsize='x-small')
ax4.legend(fontsize='x-small')

# Plot 5: Histogram (spans two columns)
ax5 = plt.subplot(3, 2, (5, 6))
ax5.hist(student_grades, bins=10)
ax5.set_title("Student Grades", fontsize='x-small')
ax5.set_xlabel("Grades", fontsize='x-small')
ax5.set_ylabel("Frequency", fontsize='x-small')

plt.tight_layout()
plt.show()
