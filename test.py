import numpy as np
EPSILON = 1e-6
adjusted_distances = [0.1,0.2,0.3,0.4,0.5]
#,0.6,0.7,0.8,0.25
adjusted_probabilities = np.asarray([1.0 / max(d, EPSILON) for d in adjusted_distances])
#print(adjusted_probabilities)
adjusted_probabilities /= np.sum(adjusted_probabilities)
#print(adjusted_probabilities)
new_list=np.random.choice(adjusted_distances,p=[0.1,0.05,0.45,0.25,0.15])
print(new_list)
print([None] * 10)