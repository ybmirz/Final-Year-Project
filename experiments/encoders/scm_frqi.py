from circuitcomponents import CircuitComponents
import pennylane as qml
from pennylane import numpy as np
import torch

class SCMFRQI_for_2x2(CircuitComponents):
    """
    The SCMFRQI (State Connection Modification FRQI) approach for 2x2 images.
    """

    def __init__(self, filter_length=2):
        if not filter_length == 2:
            raise Exception(f"This Encoding needs filter_length to be 2 not {filter_length}.")
        self.name = "SCMFRQI for 2x2 images"
        self.n_qubits = 4
        self.required_qubits = self.n_qubits

    def img_to_theta(self, img):
        """
        Normalize the input image and map the values to the range [0, pi/2].
        """
        img = torch.asin(img)
        return img

    def circuit(self, inputs):
        angles = self.img_to_theta(inputs)
        qubits = list(range(self.n_qubits))

        ### ENCODING ###

        # Apply Hadamard gates to the position qubits
        for qubit in qubits[:-1]:
            qml.Hadamard(wires=qubit)

        for i, theta in enumerate(angles):
            # Flip bits to encode pixel position
            qml.PauliX(qubits[0])
            if i % 2 == 0:  # First in a row
                qml.PauliX(qubits[1])

            # Connect the pixel value and position using the reset gate
            qml.Reset(wires=qubits[3])
            for j in range(2):
                qml.CNOT(control=j, target=qubits[3])
            qml.CNOT(control=qubits[3], target=qubits[3])

            # Apply the controlled rotation
            qml.CRY(2 * theta, wires=[qubits[2], qubits[3]])

            # Reset the pixel value qubits
            for j in range(2):
                if (i >> j) & 1:
                    qml.PauliX(wires=j)

class SCMFRQI_for_4x4(CircuitComponents):
    """
    The SCMFRQI (State Connection Modification FRQI) approach for 4x4 images.
    """

    def __init__(self, filter_length=4):
        if not filter_length == 4:
            raise Exception(f"This Encoding needs filter_length to be 4 not {filter_length}.")
        self.name = "SCMFRQI for 4x4 images"
        self.n_qubits = 9
        self.qubits = list(range(self.n_qubits))
        self.control_qubits = self.qubits[:4]
        self.work_qubits = self.qubits[4:8]
        self.color_qubit = self.qubits[8]
        self.required_qubits = self.n_qubits

    def img_to_theta(self, img):
        """
        Normalize the input image and map the values to the range [0, pi/2].
        """
        img = torch.asin(img)
        return img

    def bitstring_to_numpy(self, bitstring, reverse=False):
        """
        Converts a bitstring to a boolean numpy array.
        """
        if reverse:
            bitstring = reversed(bitstring)
        result = np.array(list(bitstring)).astype(bool)
        return result

    def circuit(self, inputs):
        angles = self.img_to_theta(inputs)

        ### ENCODING ###

        # Apply Hadamard gates to the position qubits
        for qubit in self.control_qubits:
            qml.Hadamard(wires=qubit)

        last_bitstring = self.bitstring_to_numpy('0000')

        for i, theta in enumerate(angles.flatten()):
            # Encode pixel position
            binary_i = self.bitstring_to_numpy(format(15 - i, "b").zfill(4), reverse=True)
            changed_bits = np.logical_xor(binary_i, last_bitstring)
            last_bitstring = binary_i

            for p, flipped in enumerate(changed_bits):
                if flipped:
                    qml.PauliX(wires=self.control_qubits[p])

            # Connect the pixel value and position using the reset gate
            qml.Reset(wires=self.work_qubits[3])
            for j in range(4):
                qml.CNOT(control=self.control_qubits[j], target=self.work_qubits[3])
            qml.CNOT(control=self.work_qubits[3], target=self.work_qubits[3])

            # Apply the controlled rotation
            qml.CRY(2 * theta, wires=[self.work_qubits[3], self.color_qubit])

            # Reset the work qubits
            for j in range(4):
                if (i >> j) & 1:
                    qml.PauliX(wires=self.control_qubits[j])