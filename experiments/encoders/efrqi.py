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
        self.n = 3 # default 14x14 image input
        self.required_qubits = self.n

    @staticmethod
    def img_to_phi(image):
        """
        Convert grayscale image pixel values (normalized between 0 and 1) to phase angles in radians.
        """
        return np.pi * image
    
    @staticmethod
    def img_to_amplitudes(img):
        """
        Convert normalized image pixels directly to amplitudes.
        """
        probabilities = img / np.sum(img)  # Normalize probabilities
        amplitudes = np.sqrt(probabilities)  # Square root to set amplitudes correctly
        return amplitudes

    def circuit(self, inputs):
        # print(inputs.shape)
        # inputs = inputs.view(2,2)
        # H,W = inputs.shape
        # self.n = max(math.ceil(math.log2(H)), math.ceil(math.log2(W)))

        self.required_qubits = self.n
        phi_angles = self.img_to_phi(inputs)
        amplitudes = self.img_to_amplitudes(inputs)

    #    # Apply Hadamard gates to all position qubits to prepare superposition
    #     for i in range(self.required_qubits - 1):
    #         qml.Hadamard(wires=i)

    #     # Apply controlled rotations based on the pixel values
    #     for idx, phi in enumerate(phi_angles):
    #         # Convert pixel index to binary and apply X gates to position qubits accordingly
    #         binary_index = format(idx, f'0{self.required_qubits-1}b')
    #         for qubit, bit in enumerate(binary_index):
    #             if bit == '1':
    #                 qml.PauliX(wires=qubit)

    #         # Apply controlled rotation
    #         control_wires = list(range(self.required_qubits - 1))
    #         target_wire = self.required_qubits - 1
    #         qml.ControlledPhaseShift(phi, wires=[control_wires[-1], target_wire])

    #         # Reset position qubits for next iteration
    #         for qubit, bit in enumerate(binary_index):
    #             if bit == '1':
    #                 qml.PauliX(wires=qubit)

        qml.QubitStateVector(amplitudes, wires=self.required_qubits[:-1])

        # Apply gates to entangle the position with the intensity qubit
        for i in range(4):  # Assuming a 2x2 image
            binary_index = format(i, '02b')
            for idx, bit in enumerate(binary_index):
                if bit == '1':
                    qml.PauliX(wires=idx)

            # Control the intensity qubit based on the position
            qml.CNOT(wires=[self.required_qubits[1], self.required_qubitsits[2]])  # Use the second qubit as control
            qml.CRY(2 * np.pi * amplitudes[i], wires=[self.required_qubits[0], self.required_qubits[2]])

            # Reset the position qubits
            for idx, bit in enumerate(binary_index):
                if bit == '1':
                    qml.PauliX(wires=idx)


        

