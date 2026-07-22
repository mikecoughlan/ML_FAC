import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import primesw as psw

propagator = psw.prime()
data = propagator.predict(start="2020-01-01 00:00:00", stop="2020-01-02 00:00:00")

# Using synthetic input
# propagator.predict(input = propagator.build_synth_input(vx=-700))

# Using specified position
# propagator.predict(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00', pos = [13.25, 5, 0])
# All positions are in GSE coordinates with units of Earth Radii. It is not recommended to make predictions outside of the region PRIME was trained on (within 30 Earth radii of the Earth on the dayside).

df = pd.DataFrame(data)
df.to_feather("prime_data.feather")
