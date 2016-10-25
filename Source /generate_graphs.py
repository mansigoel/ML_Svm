import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand


error_binary = [0.87950, 0.95400, 0.95300, 0.93800, 0.93725, 0.93725]
C = [1e-6, 1e-3, 1, 10, 100,1000]
for i in range(6):
	error_binary[i] = 1-error_binary[i]
plt.xlim(0, 100)
plt.ylim(0, 0.15)
plt.plot(C,error_binary, 'b.-')
plt.ylabel("Cross Validation Error")
plt.xlabel("Parameter C")
plt.title("Binary Class Error vs C")
plt.savefig('CV_1(a).png')
plt.show()


error_multi = [0.33015, 0.33015, 0.0922, 0.0824, 0.09255,0.0939,0.0939]
c_multi = [1e-7, 1e-5, 1e-3, 1, 10, 100, 1000]
plt.xlim(0, 100)
plt.ylim(0.05, 0.1)
plt.plot(c_multi,error_multi, 'b.-')
plt.ylabel("Cross Validation Error")
plt.xlabel("Parameter C")
plt.title("Multi Class Error vs C")
plt.savefig('CV_1(b).png')
plt.show

C1 = [0.96300, 0.19465, 0.10345]
C10 = [0.98515, 0.30800, 0.30525]
C100 = [0.97175, 0.20600, 0.10500]

for i in range(3):
	C1[i] = 1-C1[i]
	C10[i] = 1-C10[i]
	C100[i] = 1-C100[i]
print C1
print C10
print C100

gamma = [0.01, 1, 10]
plt.xlim(-1, 15)
plt.ylim(0, 1)
plt.plot(gamma,C1, 'bo-',linewidth=2.0, label='C = 1')
plt.plot(gamma,C10, 'go-',linewidth=2.0,label='C = 10')
plt.plot(gamma,C100, 'ro-',linewidth=2.0, label='C = 100')
plt.legend(loc=4)
plt.ylabel("Cross Validation Error")
plt.xlabel("Gamma")
plt.title("Multi Class RBF Error vs C")
plt.show()
plt.savefig('CV_1(c).png')