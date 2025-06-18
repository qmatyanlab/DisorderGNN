import json
with open('settings/atomic_embedding_CGCNN.json', 'r') as f:
    atomic_embedding_CGCNN = json.load(f)

from .constants import *

import numpy as np
np.set_printoptions(precision=3, suppress=True)