# %%
import numpy as np


class Linear:
    def __init__(self, n_input, n_output) -> None:

        self.w = np.random.randn(n_input, n_output)
        assert self.w.shape == (n_input, n_output)

        self.b = np.random.randn(n_output)
        assert self.b.shape == (n_output,)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, grad):
        self.grad_w = np.dot(grad, self.x.T)
        self.grad_b = np.sum(grad, axis=0)
        return np.dot(grad, self.w.T)


class Sigmoid:
    def __init__(self) -> None:
        pass

    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))

    def backward(self, grad):
        return grad * (1 - self.forward(self.x)) * self.forward(self.x)


class ReLU:
    def __init__(self) -> None:
        pass

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.x > 0)


class Softmax:
    def __init__(self) -> None:
        pass

    def forward(self, x):
        self.x = x
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def backward(self, grad):
        return grad * (self.forward(self.x) * (1 - self.forward(self.x)))


class CrossEntropy:
    def __init__(self) -> None:
        pass

    def forward(self, x, y):
        self.x = x
        self.y = y
        return -np.sum(y * np.log(x), keepdims=True)

    def backward(self, grad):
        print('grad.shape', grad.shape)
        return grad * (-self.y / self.x)


class NN:
    def __init__(self, layers) -> None:
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


# %%
if __name__ == "__main__":
    input = np.random.randn(10, 5)
