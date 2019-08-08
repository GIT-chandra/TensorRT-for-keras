import os
import sys
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io

OUTPUT_NODE_PREFIX = 'output_node'
NUMBER_OF_OUTPUTS = 1

outputs = [OUTPUT_NODE_PREFIX + str(i) for i in range(NUMBER_OF_OUTPUTS)]

frozen_graph_path = sys.argv[1]
output_folder = sys.argv[2]

def get_frozen_graph():
  with tf.gfile.FastGFile(frozen_graph_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


frozen_graph_def = get_frozen_graph()
trt_graph_def = trt.create_inference_graph(frozen_graph_def, 
									outputs,
									max_batch_size=16, 
									max_workspace_size_bytes=2<<10<<20, 
									precision_mode='FP32')
tf.reset_default_graph()
g = tf.Graph()
with tf.Session(graph=g) as sess:
	with g.as_default():
		tf.import_graph_def(
  		graph_def=trt_graph_def,
  		name='')
	graph_io.write_graph(g, output_folder 'trt_' + frozen_graph_path.split('/')[-1], as_text=False)
