import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import math


class CNN(object):

    def __init__(self, model_filepath):
        self.model_filepath = model_filepath
        self.load_graph(model_filepath = self.model_filepath)
    
    def load_graph(self, model_filepath):
        print('Loading model...')
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph)

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        
        print('Check out input placeholders:')
        nodes = [n.name + ' => ' + n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)

        self.input = tf.placeholder(np.float32, shape=[None, 224, 224, 3], name='input')

        tf.import_graph_def(graph_def, {'InputData/X': self.input})
        print('Model loading complete!')
    
    def test(self, data):
        #print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
        output_tensor = self.graph.get_tensor_by_name("import/FullyConnected_2/Softmax:0")
        output = self.sess.run(output_tensor, feed_dict={self.input: [data]})
        return output

if __name__ == '__main__':
    model_filepath = "./firenet.pb"
    model = CNN(model_filepath = model_filepath)

################################################################################

    # network input sizes

    rows = 224
    cols = 224

    # display and loop settings

    windowName = "Live Fire Detection - FireNet CNN";
    keepProcessing = True;

################################################################################

    if len(sys.argv) == 2:

        # load video file from first command line argument

        video = cv2.VideoCapture(sys.argv[1])
        print("Loaded video ...")

        # create window

        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);

        # get video properties

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_time = round(1000/fps);

        while (keepProcessing):

            # start a timer (to see how long processing and display takes)

            start_t = cv2.getTickCount();

            # get video frame from file, handle end of file

            ret, frame = video.read()
            if not ret:
                print("... end of video file reached");
                break;

            # re-size image to network input size and perform prediction

            small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)
            output = model.test(data=small_frame)

            # label image based on prediction

            if round(output[0][0]) == 1:
                cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
                cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
            else:
                cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
                cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);

            # stop the timer and convert to ms. (to see how long processing and display takes)

            stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

            # image display and key handling

            cv2.imshow(windowName, frame);

            # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

            key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
            if (key == ord('x')):
                keepProcessing = False;
            elif (key == ord('f')):
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    else:
        print("usage: python firenet.py videofile.ext");

################################################################################
