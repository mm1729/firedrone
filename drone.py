import firedrone.client as fdc
from firedrone.client.errors import FireDroneClientHttpError

# Create a new client workspace.
# -------------------------------------------------------------------------------------------
key = 'O9P$n7W5p1MhCv7#MmfTh8psPjbASGFuqDHQU#Z?zwr-$Xk@51z8T87aVsyDgWnj'
workspace = fdc.Workspace(key)

# Get a list of available scenes.
# -------------------------------------------------------------------------------------------

try:
    scenes = workspace.get_scenes()
    print(scenes)
except FireDroneClientHttpError as e:
    print(e.status_code)