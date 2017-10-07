from halo import Halo
import time

spinner = Halo({'text': 'Calculating...', 'color': 'blue', 'spinner': 'dots'})
spinner.start()
counter = 0
while counter < 101:
    counter += 1
    time.sleep(.1)

# spinner.stop()
spinner.succeed()
