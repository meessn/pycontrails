from openap import prop
from pprint import pprint
from openap import FuelFlow
from openap.kinematic import WRAP

aircraft = prop.aircraft("A20n")
pprint(aircraft)
# wrap = WRAP(ac="A20n")
fuelflow = FuelFlow(ac='A20n')

# FF = fuelflow.at_thrust(acthr=50000, alt=30000)


