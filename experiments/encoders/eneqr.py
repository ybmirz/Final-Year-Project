import pennylane as qml
from pennylane import numpy as np
from pyeda.inter import *
from circuitcomponents import CircuitComponents
import torch

class ENEQR(CircuitComponents):
    """
    Enhanced NEQR (ENEQR) for encoding images into quantum states.
    This class encodes a 2^n x 2^n grayscale image using q+2n qubits, optimizing gate usage and efficiency.
    """
    def __init__(self, q=8, filter_length=2):
        if not 2 ** int(np.log2(filter_length)) == filter_length:
            raise Exception("filter_length must be a power of 2.")
        self.q = q
        self.n = int(np.log2(filter_length))
        self.required_qubits = self.q + 2 * self.n
        self.qubits_ev = list(map(exprvar, [f"q{i}" for i in range(self.required_qubits)]))

    def optimized_Toffoli(self, wires):
        """
        An optimized version of the Toffoli gate that uses fewer auxiliary operations.
        """
        if len(wires) == 2:
            # If only two wires are provided, perform a CNOT gate.
            control, target = wires
            qml.CNOT(wires=[control, target])
        elif len(wires) == 3:
            control0, control1, target = wires
            # Perform an optimized Toffoli gate as previously defined
            qml.PhaseShift(np.pi/4, wires=target)
            qml.CNOT(wires=[control1, target])
            qml.PhaseShift(-np.pi/4, wires=target)
            qml.CNOT(wires=[control0, target])
            qml.PhaseShift(np.pi/4, wires=control1)
            qml.PhaseShift(-np.pi/4, wires=target)
            qml.CNOT(wires=[control1, target])
            qml.PhaseShift(np.pi/4, wires=target)
            qml.CNOT(wires=[control0, target])
            qml.PhaseShift(-np.pi/4, wires=target)
            qml.Hadamard(wires=target)
        else:
            raise ValueError("Incorrect number of wires provided. Expected 2 or 3 wires.")

    def circuit(self, image):
        """
        Main method to encode an image using the ENEQR approach.
        """
        # Initialize the state
        for i in range(self.required_qubits):
            qml.Hadamard(wires=i)
        
        # Prepare image data
        normalized_image = (image.flatten() * (2**self.q - 1)).type(torch.int)
        binaries = [np.binary_repr(pixel, width=self.q) for pixel in normalized_image]

        # Apply the encoding
        for idx, pixel in enumerate(binaries):
            x, y = divmod(idx, 2**self.n)
            x_bits = np.binary_repr(x, width=self.n)
            y_bits = np.binary_repr(y, width=self.n)
            for qubit in range(self.q):
                if pixel[qubit] == '1':
                    control_bits = [int(bit) for bit in (x_bits + y_bits)]
                    control_wires = [i for i, bit in enumerate(control_bits) if bit == 1]
                    # Using optimized Toffoli gates
                    if control_wires:
                        self.optimized_Toffoli(control_wires + [self.n*2 + qubit])
