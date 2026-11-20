import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from infer.detector import AetherHoundEngine

eng = AetherHoundEngine()
trip = eng.threshold * eng.threat_margin
clean = eng.score_many(np.load("data/baseline_patches.npy"))
attack = eng.score_many(np.load("data/test_patches.npy"))
print("AE trip %.5f" % trip)
print("clean false alarms: %d/%d" % ((clean > trip).sum(), len(clean)))
print("attack caught by AE alone: %d/%d" % ((attack > trip).sum(), len(attack)))
print("attack median %.5f vs trip %.5f" % (np.median(attack), trip))
print("attack max %.5f" % attack.max())
print("trip / attack median: %.1fx" % (trip / np.median(attack)))
print("attack p90 %.5f" % np.quantile(attack, 0.9))
print("trip / attack max: %.2fx" % (trip / attack.max()))