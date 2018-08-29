import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


#meta_path = 'outputs/checkpoints/c1s_5_c1n_256_c2s_6_c1vl_16_c2d_0.7_c1nf_16_rs_1_c2vl_32_c1s_9_c2n_64_lr_0.0001--TrafficSign--1534256556.0782971.meta'# Your .meta file


input_checkpoint = 'outputs/checkpoints/c1s_5_c1n_256_c2s_6_c1vl_16_c2d_0.7_c1nf_16_rs_1_c2vl_32_c1s_9_c2n_64_lr_0.0001--TrafficSign--1534256556.0782971' 


#Get tensors from graph

output_node_names = ['Conv/biases',
'Conv/biases/Adam',
'Conv/biases/Adam_1',
'Conv/weights',
'Conv/weights/Adam',
'Conv/weights/Adam_1',
'Variable',
'Variable/Adam',
'Variable/Adam_1',
'Variable_1',
'Variable_1/Adam',
'Variable_1/Adam_1',
'Variable_2',
'beta1_power',
'beta2_power',
'conv2d/bias',
'conv2d/bias/Adam',
'conv2d/bias/Adam_1',
'conv2d/kernel',
'conv2d/kernel/Adam',
'conv2d/kernel/Adam_1',
'conv2d_1/bias',
'conv2d_1/bias/Adam',
'conv2d_1/bias/Adam_1',
'conv2d_1/kernel',
'conv2d_1/kernel/Adam',
'conv2d_1/kernel/Adam_1',
'conv2d_2/bias',
'conv2d_2/bias/Adam',
'conv2d_2/bias/Adam_1',
'conv2d_2/kernel',
'conv2d_2/kernel/Adam',
'conv2d_2/kernel/Adam_1',
'conv2d_3/bias',
'conv2d_3/bias/Adam',
'conv2d_3/bias/Adam_1',
'conv2d_3/kernel',
'conv2d_3/kernel/Adam',
'conv2d_3/kernel/Adam_1',
'fully_connected/biases',
'fully_connected/biases/Adam',
'fully_connected/biases/Adam_1',
'fully_connected/weights',
'fully_connected/weights/Adam',
'fully_connected/weights/Adam_1',
'weight',
'weight/Adam',
'weight/Adam_1'] 
print_tensors_in_checkpoint_file(file_name=input_checkpoint, tensor_name='', all_tensors=False)




#{"tf_caps1": "mul:0", "tf_predicted_class": "Reshape_7:0", "tf_decoded": "decoded:0", "tf_correct_prediction": "Equal:0", "tf_accuracy": "Mean_2:0", "tf_reconstruction_loss": "Mean_1:0", "tf_loss_squared_rec": "Square_4:0", "tf_loss": "add_4:0", "tf_margin_loss_sum": "Sum_2:0", "tf_caps2": "Squeeze:0", "tf_tensorboard": "Merge/MergeSummary:0", "tf_images": "images:0", "tf_labels": "labels:0", "tf_optimizer": "Adam", "tf_test": "tf_test:0", "tf_conv_2_dropout": "conv_2_dropout:0", "tf_margin_loss": "Mean:0"}


with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta')

    # Load weights
    saver.restore(sess,input_checkpoint)

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())