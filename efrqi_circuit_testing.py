import pennylane as qml
from pennylane import numpy as np

class EFRQI:
    def __init__(self, num_qubits, device='default.qubit'):
        self.num_qubits = num_qubits
        self.device = qml.device(device, wires=num_qubits)

    @staticmethod
    def img_to_phi(image):
        """
        Convert grayscale image pixel values (normalized between 0 and 1) to phase angles in radians.
        """
        return np.pi * image

    def circuit(self, image):
        """
        Build and return a quantum circuit that encodes the image using the EFRQI method.
        """
        phi_angles = self.img_to_phi(image)
        num_pixels = len(phi_angles)

        @qml.qnode(self.device)
        def efrqi_circuit():
            # Apply Hadamard gates to all position qubits to prepare superposition
            for i in range(self.num_qubits - 1):
                qml.Hadamard(wires=i)

            # Apply controlled rotations based on the pixel values
            for idx, phi in enumerate(phi_angles):
                # Convert pixel index to binary and apply X gates to position qubits accordingly
                binary_index = format(idx, f'0{self.num_qubits-1}b')
                for qubit, bit in enumerate(binary_index):
                    if bit == '1':
                        qml.PauliX(wires=qubit)

                # Apply controlled rotation
                control_wires = list(range(self.num_qubits - 1))
                target_wire = self.num_qubits - 1
                qml.ControlledPhaseShift(phi, wires=[control_wires[-1], target_wire])

                # Reset position qubits for next iteration
                for qubit, bit in enumerate(binary_index):
                    if bit == '1':
                        qml.PauliX(wires=qubit)

            return [qml.state()]

        return efrqi_circuit

# Example Usage
image = np.array([0.0, 0.5, 1.0, 0.25])  # Example image data
efrqi = EFRQI(num_qubits=3)  # Initialize EFRQI with enough qubits to encode the image
circuit = efrqi.circuit(image)  # Get the EFRQI circuit for the image
quantum_state = circuit()  # Execute the circuit to get the quantum state
print(quantum_state)
