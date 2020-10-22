import math
import random


# чтение данных из файла
def read_data(path):
    with open(path, 'r', encoding='windows-1251') as file:
        content = file.read().splitlines()
        data = list()
        data.append(content[0].split('\t'))
        for i in content[1:]:
            data.append(list(map(lambda x: float(x), i.split('\t'))))
        return data


# сигмоида
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def derivative(x):
    return x * (1.0 - x)


# максимум n столбца
def getMax(data, n):
    return max([float(i[n]) for i in data])


# минимум n столбца
def getMin(data, n):
    return min([float(i[n]) for i in data])


# нормализация данных
def norm(data):
    # сразу берем мин и макс для уменьшения сложности алгоритма
    mins = [getMin(data, i) for i in range(length)]
    maxs = [getMax(data, i) for i in range(length)]
    for i in range(data.__len__()):
        for j in range(length):
            data[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j])


class Neuron:
    def __init__(self, prev_layer_count):
        self.weights = [random.random() for _ in range(prev_layer_count)]
        self.offset = random.random()
        self.delta = 0
        self.output = 0

    def activate(self, inputs):
        self.output = sigmoid(self.offset + sum([wi * ii for wi, ii in zip(self.weights, inputs)]))
        return self.output

    def __str__(self):
        return f'Output = {"%.3f" % self.output}  Offset = {"%.3f" % self.offset}  ' \
               f'Delta = {"%.3f" % self.delta}  Weights = {" ".join(["%.3f" % i for i in self.weights])}'


def init_network(layers):
    network = []
    prev_layer = None
    for i, layer in enumerate(layers, 0):
        if i == 0:
            prev_layer = layer
            continue
        network.append([Neuron(prev_layer) for _ in range(layer)])
        prev_layer = layers[i]
    return network


def forward_propagate(network, inputs):
    for layer in network:
        new_inputs = [neuron.activate(inputs) for neuron in layer]
        inputs = new_inputs
    return inputs


def backward_propagate_error(network, expected):
    for i, layer in reversed(list(enumerate(network, 0))):
        errors = []
        if i == len(network) - 1:
            for j, neuron in enumerate(layer, 0):
                errors.append(expected[j] - neuron.output)
        else:
            for j in range(len(layer)):
                errors.append(sum([neuron.weights[j] * neuron.delta for neuron in network[i + 1]]))
        for j, neuron in enumerate(layer, 0):
            neuron.delta = errors[j] * derivative(neuron.output)


def update_weights(network, inputs, sigma):
    for i, layer in enumerate(network, 0):
        if i != 0:
            inputs = [neuron.output for neuron in network[i - 1]]
        for neuron in layer:
            neuron.offset += sigma * neuron.delta
            for j, inp in enumerate(inputs, 0):
                neuron.weights[j] += sigma * neuron.delta * inp


length = 8  # количество измерений
sigma = 0.7  # скорость обучения
layers = [length, 3, 5, 2]

data = read_data('diabetes.txt')[1:]
norm(data)

split = (0.7 * data.__len__()).__int__()
train = data[:split]
test = data[split:]
network = init_network(layers)

steps = 0
error, tmp_error = 0, 0

for _ in range(500):
    error = 0
    steps += 1
    for row in train:
        expected = [0 for i in range(layers[-1])]
        expected[int(row[length])] = 1
        output = forward_propagate(network, row[:length])
        error += sum([(expected[i] - output[i]) ** 2 for i in range(len(expected))])
        backward_propagate_error(network, expected)
        update_weights(network, row[:length], sigma)
    tmp_error = error

accuracy = 0
for i in test:
    pred = forward_propagate(network, i[:length])
    result = pred.index(max(pred))
    # print('val', pred, 'получено', result, 'ожидалось', i[length])
    if result == i[length]:
        accuracy += 1

print('steps:', steps)
print('accuracy:', accuracy / test.__len__())
