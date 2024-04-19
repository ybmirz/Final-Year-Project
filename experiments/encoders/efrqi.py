from circuitcomponents import CircuitComponents
import pennylane as qml
from pennylane import numpy as np
import torch
import math

class EFRQI(CircuitComponents):
    
    def __init__(self, filter_length=2):
        if not filter_length == 2:
            raise Exception(f"This Encoding needs filter_length to be 2 not {filter_length}.")
        self.name = "EFRQI"
        self.n = 4 # default 14x14 image input

    def circuit(self, inputs):
        print(inputs.shape)
        inputs = inputs.view(2,2)
        H,W = inputs.shape
        self.n = max(math.ceil(math.log2(inputs)), math.ceil(math.log2(W)))
        self.required_qubits = self.n * 2 + 1

        for wire in range(2*self.n):
            qml.Hadamard(wires=wire)

        for y in range(H):
            for x in range(W):
                pixel_value = inputs[y, x].item()
                if pixel_value > 0:
                    qml.RX(pixel_value/255*np.pi/2, wires=2*self.n)
                else:
                    qml.RX(0, wires=2*self.n)

                for i in range(2*self.n):
                    qml.CNOT(wires=[i, 2*self.n])

        

