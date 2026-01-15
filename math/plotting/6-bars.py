#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# People and positions
people = ['Farrah', 'Fred', 'Felicia']
x = np.arange(len(people))
width = 0.5

# Stack the bars (row order matters)
plt.bar(x, fruit[0], width=width, color='red', label='Apples')
plt.bar(x, fruit[1], width=width, bottom=fruit[0],
        color='yellow', label='Bananas')
plt.bar(x, fruit[2], width=width, bottom=fruit[0] + fruit[1],
        color='#ff8000', label='Oranges')
plt.bar(x, fruit[3], width=width,
        bottom=fruit[0] + fruit[1] + fruit[2],
        color='#ffe5b4', label='Peaches')

# Axis labels and title
plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")

# Ticks and limits
plt.xticks(x, people)
plt.yticks(np.arange(0, 81, 10))
plt.ylim(0, 80)

# Legend
plt.legend()

plt.show()
