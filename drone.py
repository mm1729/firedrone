import firedrone.client as fdc
from firedrone.client.errors import FireDroneClientHttpError
import cv2
from firenet import CNN

# initalize CNN
rows = 224
cols = 224
model_filepath = "./firenet.pb"
model = CNN(model_filepath = model_filepath)

# Create a new client workspace.
key = 'O9P$n7W5p1MhCv7#MmfTh8psPjbASGFuqDHQU#Z?zwr-$Xk@51z8T87aVsyDgWnj'
workspace = fdc.Workspace(key)

# Get a list of available scenes.

try:
    scenes = workspace.get_scenes()
    print(scenes)
except FireDroneClientHttpError as e:
    print(e.status_code)

# Start a direct run
try:
    start_result = workspace.directrun_start(21)
    print(start_result)
except FireDroneClientHttpError as e:
    print(e)

run_id = start_result['uniqueId']
print('run id :' + run_id)

frame = workspace.get_drone_fov_image(run_id)
with open('./frame.png', 'wb') as f:
    f.write(frame)

img = cv2.imread('./frame.png')
cv2.imshow('image', img)
cv2.waitKey(0)

# detect fire
small_frame = cv2.resize(img, (rows, cols), cv2.INTER_AREA)
output = model.test(data=small_frame)
print(output[0][0])
cv2.destroyAllWindows()

workspace.directrun_end(run_id)