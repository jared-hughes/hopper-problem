
import cProfile
import pstats
from hopperN import hopperN

profiler = cProfile.Profile()
profiler.enable()
for n in range(2, 9):
    print(n, hopperN(n, 1000))
profiler.disable()
with open("profile.log", "w") as stream:
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats()
