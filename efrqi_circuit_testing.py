#from circuitcomponents import CircuitComponents
import pennylane as qml
from pennylane import numpy as np
from tensorflow.keras.datasets import mnist
import cv2
import math


def efrqi_encoding(image, n):
    """
    Encodes a 2^n x 2^n grayscale image using the EFRQI scheme.
    
    Args:
        image (numpy.ndarray): The input grayscale image.
        n (int): The number of qubits used to represent the image.
    
    Returns:
        qml.QNode: The quantum circuit that encodes the image.
    """
    dev = qml.device("default.qubit", wires=2*n+1)

    @qml.qnode(dev)
    def circuit():
        # Initialize the quantum register
        for wire in range(2*n):
            qml.Hadamard(wires=wire)
            
        # Encode the image
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                print("y:", y, "x:", x)  # Monitor the values 
                pixel_value = image[y, x]
                # for i in range(int(pixel_value * 255 / 255)):
                #     qml.RX(np.pi, wires=0)
                # qml.CSWAP(control_wires=[y, x], swap_wires=[0, 2*n])
                if pixel_value > 0:
                    qml.RX(pixel_value/255*np.pi/2, wires=2*n)
                else:
                    qml.RX(0, wires=2*n)

                 # Entangle the position and value qubits
                for i in range(2*n):
                    qml.CNOT(wires=[i, 2*n])

        return [qml.state()]

    return circuit

def eneqr_encoding(image, n, q):
    """
    Encodes a 2^n x 2^n grayscale image using the ENEQR scheme.
    
    Args:
        image (numpy.ndarray): The input grayscale image.
        n (int): The number of qubits used to represent the image position.
        q (int): The number of qubits used to represent the pixel value.
    
    Returns:
        qml.QNode: The quantum circuit that encodes the image.
    """
    dev = qml.device("default.qubit", wires=2*n+q+1)

    @qml.qnode(dev)
    def circuit():
        # Initialize the quantum register
        for wire in range(2*n+q+1):
            qml.Hadamard(wires=wire)

        # Encode the image
        for y in range(2**n):
            for x in range(2**n):
                pixel_value = image[y, x]
                qml.CSWAP(control_wires=[y, x], swap_wires=[2*n, 2*n+1])
                for i in range(q):
                    if (pixel_value >> i) & 1:
                        qml.CNOT(control_wires=2*n+1, target_wires=2*n+2+i)
                qml.CSWAP(control_wires=[y, x], swap_wires=[2*n, 2*n+1])

        return [qml.state()]

    return circuit

#Load the MNIST dataset
(_, _), (x_test, _) = mnist.load_data()

#Choose a random image from the test set
image = x_test[np.random.randint(0, len(x_test))]
image = cv2.resize(image, (14, 14))
# image = np.random.randint(0, 256, size=(14, 14))
# print(image.shape)

#my_encoder = SCMFRQI_for_2x2()
#angles = my_encoder.img_to_theta(image)

# Draw the circuit
#drawer = qml.draw(circuit, show_all_wires=True)(angles)
#print(drawer)



# Encode the image using the SCMFRQI approach
H,W = image.shape
print(image.shape)
n = max(math.ceil(math.log2(H)), math.ceil(math.log2(W)))
q = 8
efrqi_circuit = efrqi_encoding(image, n)

#eneqr_circuit = eneqr_encoding(image, n, q)

# Print the resulting quantum circuit
print(qml.draw(efrqi_circuit, show_all_wires=True, wire_order=range(n))())