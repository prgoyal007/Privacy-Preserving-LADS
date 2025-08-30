import random
import math

# Uniform Distribution: random int from 0 to max
def get_uni_rank(max):
	return random.randint(0, max)

# Geometric Distribution: number of tails in an unbiased
# coin until we get our first heads
def get_geo_rank():
	geo_rank = 0
	while random.random() < 0.5:
		geo_rank += 1

  return geo_rank

# Bias ranks towards a frequency distribution.
# See https://arxiv.org/abs/2307.07660
def bias_rank(rank, freq, cap, thresholded=False):
  if thresholded:
    freq = max(freq/2, 1/(2 * cap))
  
  return rank + math.floor(math.log2(freq * cap))
