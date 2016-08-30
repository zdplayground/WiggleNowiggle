# made on 07/27/2016 to test np.cov and np.corrcoef
#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

n_rows = 500 
a = np.random.rand(n_rows, n_rows)
cov_a = np.cov(a)
rij_cov_a = np.corrcoef(cov_a)

plt.pcolor(rij_cov_a, cmap='Greys')
plt.colorbar()
plt.show()
