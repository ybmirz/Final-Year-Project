import pennylane as qml
from pennylane import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2

# class EFRQI:
#     def __init__(self, num_qubits, device='default.qubit'):
#         self.num_qubits = num_qubits
#         self.device = qml.device(device, wires=num_qubits)

#     @staticmethod
#     def img_to_phi(image):
#         """
#         Convert grayscale image pixel values (normalized between 0 and 1) to phase angles in radians.
#         """
#         return np.pi * image

#     def circuit(self, image):
#         """
#         Build and return a quantum circuit that encodes the image using the EFRQI method.
#         """
#         phi_angles = self.img_to_phi(image)
#         num_pixels = len(phi_angles)

#         @qml.qnode(self.device)
#         def efrqi_circuit():
#             # Apply Hadamard gates to all position qubits to prepare superposition
#             for i in range(self.num_qubits - 1):
#                 qml.Hadamard(wires=i)

#             # Apply controlled rotations based on the pixel values
#             for idx, phi in enumerate(phi_angles):
#                 # Convert pixel index to binary and apply X gates to position qubits accordingly
#                 binary_index = format(idx, f'0{self.num_qubits-1}b')
#                 for qubit, bit in enumerate(binary_index):
#                     if bit == '1':
#                         qml.PauliX(wires=qubit)

#                 # Apply controlled rotation
#                 control_wires = list(range(self.num_qubits - 1))
#                 target_wire = self.num_qubits - 1
#                 qml.ControlledPhaseShift(phi, wires=[control_wires[-1], target_wire])

#                 # Reset position qubits for next iteration
#                 for qubit, bit in enumerate(binary_index):
#                     if bit == '1':
#                         qml.PauliX(wires=qubit)

#             return [qml.state()]

#         return efrqi_circuit

import pennylane as qml
from pennylane import numpy as np
import torch

class EFRQI():
    def __init__(self,num_qubits=3, filter_length=2):
        if filter_length != 2:
            raise ValueError("This Encoding needs filter_length to be 2.")
        self.name = "EFRQI for 2x2 images"
        self.n_qubits =  num_qubits # Two for position, one for color intensity
        self.required_qubits = self.n_qubits
        self.device = qml.device('default.qubit', wires=self.n_qubits)

    def img_to_amplitudes(self, img):
        """
        Convert normalized image pixels directly to amplitudes.
        """
        probabilities = img / np.sum(img)  # Normalize probabilities
        amplitudes = np.sqrt(probabilities)  # Square root to set amplitudes correctly
        return amplitudes

    def circuit(self, inputs):
        amplitudes = self.img_to_amplitudes(inputs)
        qubits = list(range(self.n_qubits))

        @qml.qnode(device=self.device)
        def efrqi_circuit():
            # Initialize the state directly using amplitudes
            print(amplitudes)
            qml.QubitStateVector(amplitudes, wires=qubits[:-1])

            # Apply gates to entangle the position with the intensity qubit
            for i in range(4):  # Assuming a 2x2 image
                binary_index = format(i, '02b')
                for idx, bit in enumerate(binary_index):
                    if bit == '1':
                        qml.PauliX(wires=idx)

                # Control the intensity qubit based on the position
                qml.CNOT(wires=[qubits[1], qubits[2]])  # Use the second qubit as control
                qml.CRY(2 * np.pi * amplitudes[i], wires=[qubits[0], qubits[2]])

                # Reset the position qubits
                for idx, bit in enumerate(binary_index):
                    if bit == '1':
                        qml.PauliX(wires=idx)

            return qml.state()

        return efrqi_circuit


# Example Usage
#image = np.array([0.0, 0.5, 1.0, 0.25])  # Example image data

#Load the MNIST dataset
(_, _), (x_test, _) = mnist.load_data()

# Normalize the images to float values between 0 and 1
x_train_normalized = _.astype('float32') / 255.0
x_test_normalized = x_test.astype('float32') / 255.0


#Choose a random image from the test set
image = x_test[np.random.randint(0, len(x_test))]
image = cv2.resize(image, (14, 14))

# Display the image
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Input Image')

# Retrieve the 2x2 filter part of image
sub_image = qml.numpy.array(image[0:2, 0:2])
# Extract the center 2x2 block
center_x, center_y = 14 // 2, 14 // 2
sub_image = qml.numpy.array(image[center_x-1:center_x+1, center_y-1:center_y+1])
sub_image = sub_image.astype('float32') / 255.0

# Display the sub_image (2x2 convolution filter input)
ax[1].imshow(sub_image, cmap='gray')
ax[1].set_title('2x2 Quanvolutional Filter Input from the Centre')


print(type(sub_image))
sub_image = sub_image.flatten() # Flatten the image to a 1D array
efrqi = EFRQI(num_qubits=3)  # Initialize EFRQI with enough qubits to encode the image
circuit = efrqi.circuit(sub_image)  # Get the EFRQI circuit for the image
quantum_state = circuit()  # Execute the circuit to get the quantum state
print(qml.draw(circuit)())

fig, ax = qml.draw_mpl(circuit)()
# ax[2].set_title('Encoding Circuit')

plt.tight_layout()
plt.show()

print(quantum_state)


