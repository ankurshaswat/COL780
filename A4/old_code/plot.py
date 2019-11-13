import matplotlib.pyplot as plt
import sys

x = []
y = []
i = 0
for line in open(sys.argv[1], 'r').readlines():
	l = line.split(' ')
	if i%1 == 0:
		x.append(i)
		y.append(float(l[0]))
	i += 1

plt.title('Validation Loss vs No. of Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.plot(x, y)
plt.show()


