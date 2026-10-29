"""Run the simulator from a checkout: ``python runsim.py [CONFIG] [--gui] ...``.

The same as ``python -m crazyflie_sim.runsim`` or, after ``pip install``,
``crazyflie-sim``.
"""

from crazyflie_sim.runsim import main

if __name__ == "__main__":
    main()
