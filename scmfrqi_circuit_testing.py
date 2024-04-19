#from circuitcomponents import CircuitComponents
import pennylane as qml
from pennylane import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset (downloads if necessary)
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Create a DataLoader to easily access images
train_loader = DataLoader(mnist_train, batch_size=1, shuffle=True)

# class SCMFRQI_for_2x2():
#     """
#     The SCMFRQI (State Connection Modification FRQI) approach for 2x2 images.
#     """

#     def __init__(self, filter_length=2):
#         if not filter_length == 2:
#             raise Exception(f"This Encoding needs filter_length to be 2 not {filter_length}.")
#         self.name = "SCMFRQI for 2x2 images"

#         #   n (int): The number of qubits used to represent the position of each pixel.
#         #   q (int): The number of qubits used to represent the value of each pixel
#         # Number of qubits needed for the encoding
#         self.n_qubits = q + 2 * n + 1
#         self.required_qubits = self.n_qubits

#     def img_to_theta(self, img):
#         """
#         Normalize the input image and map the values to the range [0, pi/2].
#         """
#         img = torch.asin(img)
#         return img

#     def reset_gate(wires):
#         """
#         Implement the reset gate using a QubitUnitary operation.
#         """
#         reset_matrix = np.array([[1, 0, 0, 0],
#                                 [0, 1, 0, 0],
#                                 [0, 0, 0, 1],
#                                 [0, 0, 1, 0]])
#         qml.QubitUnitary(reset_matrix, wires=wires)

#     def circuit(self, inputs):
#         angles = self.img_to_theta(inputs)
#         qubits = list(range(self.n_qubits))

#         ### ENCODING ###

#         # Apply Hadamard gates to the position qubits
#         for qubit in qubits[:-1]:
#             qml.Hadamard(wires=qubit)

#         for i, theta in enumerate(angles):
#             # Flip bits to encode pixel position
#             qml.PauliX(qubits[0])
#             if i % 2 == 0:  # First in a row
#                 qml.PauliX(qubits[1])

#             # Connect the pixel value and position using the reset gate
#             self.reset_gate(wires=[qubits[2], qubits[3]])
#             qml.CNOT(wires=[qubits[0], qubits[2]])
#             qml.CNOT(wires=[qubits[1], qubits[3]])
#             #qml.CNOT(wires=[qubits[2], qubits[2]])
#             #qml.CNOT(wires=[qubits[3], qubits[3]])

#             # Apply the controlled rotation
#             qml.CRY(2 * theta, wires=[qubits[2], qubits[3]])

#             # Reset the pixel value qubits
#             for j in range(2):
#                 if (i >> j) & 1:
#                     qml.PauliX(wires=j)

# # Create a QNode for drawing
#  # Adjust device if needed
# @qml.qnode(qml.device('default.qubit', wires=4) )
# def circuit(angles):
#     ### ENCODING ###
#     qubits = list(range(4))

#     ### ENCODING ###

#     # Apply Hadamard gates to the position qubits
#     for qubit in qubits[:-1]:
#         qml.Hadamard(wires=qubit)

#     for i, theta in enumerate(angles):
#         # Flip bits to encode pixel position
#         qml.PauliX(qubits[0])
#         if i % 2 == 0:  # First in a row
#             qml.PauliX(qubits[1])

#         # Connect the pixel value and position using the reset gate
#         SCMFRQI_for_2x2.reset_gate(wires=[qubits[2], qubits[3]])
#         qml.CNOT(wires=[qubits[0], qubits[2]])
#         qml.CNOT(wires=[qubits[1], qubits[3]])
#         qml.CNOT(wires=[qubits[2], qubits[2]])
#         qml.CNOT(wires=[qubits[3], qubits[3]])

#         # Apply the controlled rotation
#         qml.CRY(2 * theta, wires=[qubits[2], qubits[3]])

#         # Reset the pixel value qubits
#         for j in range(2):
#             if (i >> j) & 1:
#                 qml.PauliX(wires=j)

#     return qml.probs(wires=range(4))  # Or any measurement you desire

def scmfrqi_encode(image):
    """
    Encodes a classical image into a quantum state using the SCMFRQI approach.
    
    Args:
        image (np.ndarray): The classical image to be encoded.
        n (int): The number of qubits used to represent the position of each pixel.
        q (int): The number of qubits used to represent the value of each pixel.
    
    Returns:
        qml.QNode: The quantum circuit that encodes the image.
    """
    # Get the shape of the image
    height, width = image.shape

    n = int(np.log2(max(height, width)))
    q = 8
    
    # Define the number of qubits needed
    num_qubits = q + 2 * n + 1
    
    # Define the quantum circuit
    @qml.qnode(qml.device('default.qubit', wires=num_qubits))
    def circuit():
        # Initialize the quantum state to |0>^(q+2n+1)
        for wire in range(num_qubits):
            qml.Hadamard(wires=wire)
        
        # Prepare the pixel values
        for y in range(height):
            for x in range(width):
                # Prepare the pixel value
                pixel_value = image[y, x]
                prepare_pixel_value(pixel_value= pixel_value, wires=[q + i for i in range(q)])
                
                # Prepare the pixel position
                prepare_pixel_position(x, y, wires=[q + q + i for i in range(n)], wires_aux=q + 2 * n)
                
                # Apply the reset gate
                qml.RX(np.pi, wires=q + 2 * n)
        
        return [qml.state()]
    
    return circuit, num_qubits

def prepare_pixel_value(pixel_value, wires):
    """
    Prepares the quantum state representing the pixel value.
    
    Args:
        wires (list): The wires on which to apply the gates.
    """
    # Prepare the pixel value
    for i in range(len(wires)):
        if (pixel_value >> i) & 1:
            qml.X(wires=wires[i])

def prepare_pixel_position(x, y, wires, wires_aux):
    """
    Prepares the quantum state representing the pixel position.
    
    Args:
        x (int): The x-coordinate of the pixel.
        y (int): The y-coordinate of the pixel.
        wires (list): The wires on which to apply the gates.
        wires_aux (int): The auxiliary wire.
    """
    # Prepare the x-coordinate
    for i in range(len(wires)):
        if (x >> i) & 1:
            qml.CNOT(wires=[wires[i], wires_aux])
    
    # Prepare the y-coordinate
    for i in range(len(wires)):
        if (y >> i) & 1:
            qml.CNOT(wires=[wires[i], wires_aux])
    
     # Apply the Toffoli gate
    qml.Toffoli(wires=[wires[0], wires[1], wires_aux])


for image, label in train_loader:
    break  # Just take the first image for this example

#my_encoder = SCMFRQI_for_2x2()
#angles = my_encoder.img_to_theta(image)

# Draw the circuit
#drawer = qml.draw(circuit, show_all_wires=True)(angles)
#print(drawer)

# Load the MNIST dataset
#(_, _), (x_test, _) = mnist.load_data()

# Choose a random image from the test set
#image = x_test[np.random.randint(0, len(x_test))]
image = np.random.randint(0, 256, size=(14, 14))
print(image.shape)

# Encode the image using the SCMFRQI approach
scmfrqi_circuit,num_qubits = scmfrqi_encode(image)
#final_state = scmfrqi_circuit()

# Print the resulting quantum circuit
print(qml.draw(scmfrqi_circuit, show_all_wires=True, wire_order=range(num_qubits))())