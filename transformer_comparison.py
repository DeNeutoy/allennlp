import tensorflow as tf
import numpy
import torch
from torch.autograd import Variable

from Tagger.models.deepatt import encoder
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder

batch_size = 3
sequence_length = 7
embedding_size = 16
num_heads = 4
num_layers = 2
feedforward_size = 5

torch.manual_seed(1234)
numpy.random.seed(1234)
tf.set_random_seed(1234)

def comparison_params():
    params = tf.contrib.training.HParams(
        hidden_size=embedding_size,  # Input size at each layer. Must be divisible by num_heads
        filter_size=feedforward_size, # Hidden dim of the feedforward layer.
        num_heads=num_heads,
        num_hidden_layers=num_layers,
        attention_dropout=0.0,
        residual_dropout=0.0,
        relu_dropout=0.0,
        layer_preprocessor="none",
        layer_postprocessor="layer_norm",
        attention_key_channels=None,
        attention_value_channels=None,
        attention_function="dot_product",
        layer_type="ffn_layer",
    )
    return params

def set_pytorch_variables(parameters, variables):

    parameters["feedforward_0._linear_layers.0.weight"].data = torch.from_numpy(variables["encoder/layer_0/computation/ffn_layer/input_layer/linear/matrix:0"]).t()
    parameters["feedforward_0._linear_layers.0.bias"].data = torch.from_numpy(variables["encoder/layer_0/computation/ffn_layer/input_layer/linear/bias:0"])
    parameters["feedforward_0._linear_layers.1.weight"].data = torch.from_numpy(variables["encoder/layer_0/computation/ffn_layer/output_layer/linear/matrix:0"]).t()
    parameters["feedforward_0._linear_layers.1.bias"].data = torch.from_numpy(variables["encoder/layer_0/computation/ffn_layer/output_layer/linear/bias:0"])
    parameters["feedforward_layer_norm_0.gamma"].data = torch.from_numpy(variables["encoder/layer_0/computation/layer_norm/scale:0"])    
    parameters["feedforward_layer_norm_0.beta"].data = torch.from_numpy(variables["encoder/layer_0/computation/layer_norm/offset:0"])
    parameters["self_attention_0._combined_projection.weight"].data = torch.from_numpy(variables["encoder/layer_0/self_attention/multihead_attention/qkv_transform/matrix:0"]).t()
    parameters["self_attention_0._combined_projection.bias"].data = torch.from_numpy(variables["encoder/layer_0/self_attention/multihead_attention/qkv_transform/bias:0"])
    parameters["self_attention_0._output_projection.weight"].data = torch.from_numpy(variables["encoder/layer_0/self_attention/multihead_attention/output_transform/matrix:0"]).t() 
    parameters["self_attention_0._output_projection.bias"].data = torch.from_numpy(variables["encoder/layer_0/self_attention/multihead_attention/output_transform/bias:0"])
    parameters["layer_norm_0.gamma"].data = torch.from_numpy(variables["encoder/layer_0/self_attention/layer_norm/scale:0"])
    parameters["layer_norm_0.beta"].data = torch.from_numpy(variables["encoder/layer_0/self_attention/layer_norm/offset:0"])

    parameters["feedforward_1._linear_layers.0.weight"].data = torch.from_numpy(variables["encoder/layer_1/computation/ffn_layer/input_layer/linear/matrix:0"]).t()
    parameters["feedforward_1._linear_layers.0.bias"].data = torch.from_numpy(variables["encoder/layer_1/computation/ffn_layer/input_layer/linear/bias:0"]) 
    parameters["feedforward_1._linear_layers.1.weight"].data = torch.from_numpy(variables["encoder/layer_1/computation/ffn_layer/output_layer/linear/matrix:0"]).t()
    parameters["feedforward_1._linear_layers.1.bias"].data = torch.from_numpy(variables["encoder/layer_1/computation/ffn_layer/output_layer/linear/bias:0"])
    parameters["feedforward_layer_norm_1.gamma"].data = torch.from_numpy(variables["encoder/layer_1/computation/layer_norm/scale:0"])
    parameters["feedforward_layer_norm_1.beta"].data = torch.from_numpy(variables["encoder/layer_1/computation/layer_norm/offset:0"])
    parameters["self_attention_1._combined_projection.weight"].data = torch.from_numpy(variables["encoder/layer_1/self_attention/multihead_attention/qkv_transform/matrix:0"]).t()
    parameters["self_attention_1._combined_projection.bias"].data = torch.from_numpy(variables["encoder/layer_1/self_attention/multihead_attention/qkv_transform/bias:0"])  
    parameters["self_attention_1._output_projection.weight"].data = torch.from_numpy(variables["encoder/layer_1/self_attention/multihead_attention/output_transform/matrix:0"]).t()  
    parameters["self_attention_1._output_projection.bias"].data = torch.from_numpy(variables["encoder/layer_1/self_attention/multihead_attention/output_transform/bias:0"])  
    parameters["layer_norm_1.gamma"].data = torch.from_numpy(variables["encoder/layer_1/self_attention/layer_norm/scale:0"])
    parameters["layer_norm_1.beta"].data = torch.from_numpy(variables["encoder/layer_1/self_attention/layer_norm/offset:0"])



def run_tensorflow():

    input_placeholder = tf.placeholder(shape=[batch_size, sequence_length, embedding_size], dtype=tf.float32, name="inputs")
    mask_placeholder = tf.placeholder(shape=[batch_size, sequence_length], dtype=tf.float32, name="mask")
    tensorflow_output = encoder(input_placeholder, mask_placeholder, comparison_params())

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        variables = {x.name: sess.run(x) for x in tf.global_variables()}


        print("Tensorflow number of parameters: ")
        for key, value in variables.items():
            print(key, "   ", value.shape)
        model_input = numpy.random.randn(batch_size, sequence_length, embedding_size)
        variables["model_input"] = model_input

        feed_dict = {input_placeholder: model_input, 
                    mask_placeholder: numpy.ones([batch_size, sequence_length], dtype="float32")}

        intermediate_output_tf = sess.run(feed_dict=feed_dict, fetches=["encoder/layer_0/self_attention/multihead_attention/output_transform/Squeeze:0"])[0]
        
        output = sess.run(feed_dict=feed_dict, fetches=[tensorflow_output])[0]

        print(output.shape)

    return variables, output

def run_pytorch(tensorflow_variables):
    # Pytorch version:
    input_tensor = Variable(torch.from_numpy(tensorflow_variables["model_input"])).float()
    mask_tensor = Variable(torch.ones([batch_size, sequence_length]))
    pytorch_encoder = StackedSelfAttentionEncoder(input_dim=embedding_size,
                                                  hidden_dim=embedding_size,
                                                  projection_dim=embedding_size,
                                                  feedforward_hidden_dim=feedforward_size,
                                                  num_layers=num_layers,
                                                  num_attention_heads=num_heads,
                                                  use_positional_encoding=False, # timing signal is outside the encoder for comparison.
                                                  dropout_prob=0.0)

    set_pytorch_variables({k:v for (k,v) in pytorch_encoder.named_parameters()}, tensorflow_variables)
    for key, value in pytorch_encoder.named_parameters():
        print(key, "   ", value.size())
    pytorch_output = pytorch_encoder(input_tensor, mask_tensor)

    print(pytorch_output)
    print("Final output size: ", pytorch_output.size())

    return pytorch_output.data.numpy()

tensorflow_variables, tensorflow_output = run_tensorflow()
pytorch_output = run_pytorch(tensorflow_variables)

numpy.testing.assert_array_almost_equal(pytorch_output, tensorflow_output)
