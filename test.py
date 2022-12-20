import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
onehot_lenet_path = "cnn/Lenet/lenet_mnist_onehot.h5"

# {'class_name': 'Lambda',
# 'config': {'name': 'lambda', 'trainable': True, 'dtype': 'float32', 'function': ('4wIAAAAAAAAAAAAAAAMAAAAEAAAAQwAAAHMoAAAAdACgAXwAoQF9AnQAoAJ8AHwCoQJ9AHQAoAN8\nAHwBFAChAX0AfABTACkBTikE2gJ0ZtoKcmVkdWNlX21heNoIc3VidHJhY3TaA2V4cCkD2gF4cgQA\nAADaA21heKkAcgcAAAD6LkQ6L3Jlc2VhcmNoZXIvbW9kZWxzL2Nubi9MZW5ldC9sZW5ldF9vbmVo\nb3QucHnaBm9uZWhvdBMAAABzCAAAAAABCgEMAQ4B\n', (100,), None),
#   'function_type': 'lambda',
#   'module': '__main__',
#   'output_shape': [1, 10],
#   'output_shape_type': 'raw',
#   'output_shape_module': None,
#   'arguments': {}},
#   'name': 'lambda',
#   'inbound_nodes': [[['dense_1', 0, 0, {}]]]}

def cnn_onehot_lenet(model_path):
    model = keras.models.load_model(model_path)
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    c = model.get_config()
    X_train = X_train.reshape(-1, 28, 28) / 255
    X_test = X_test.reshape(-1, 28, 28) / 255

    Y_train = np_utils.to_categorical(Y_train, num_classes=10)
    Y_test = np_utils.to_categorical(Y_test, num_classes=10)
    for i in range(100):
        res = model.predict(X_test[i][None])


    model.evaluate(X_train, Y_train, verbose=2)
    print(res)


def get_output_function(model, output_layer_index):
    '''
    model: 要保存的模型
    output_layer_index：要获取的那一个层的索引
    '''
    vector_funcrion = K.function([model.layers[0].input], [model.layers[output_layer_index].output])

    def inner(input_data):
        vector = vector_funcrion([input_data])[0]
        return vector

    return inner


# def print_layer_output(model):
#     inp = model.input  # input
#     outputs = [layer.output for layer in model.layers]  # all layer outputs
#     functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
#
#     # 模型每一层输出
#     layer_outs = [func([inp_batch, 1.]) for func in functors]
#
#     return layer_outs

if __name__ == "__main__":
    cnn_onehot_lenet(onehot_lenet_path)

