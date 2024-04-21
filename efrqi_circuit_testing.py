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

def BRQI(image, n):
    """
    Represents a grayscale image using BRQI, where each bitplane is represented using NEQR.
    
    Args:
        image (2D numpy array): The input grayscale image of size 2^n x 2^k.
        n (int): The total number of qubits (n = n_x + k).
        k (int): The number of qubits representing the y-axis.
        
    Returns:
        qml.QNode: A QNode representing the BRQI state of the image.
    """
    dev = qml.device('default.qubit', wires=n+7)
    
    @qml.qnode(dev)
    def circuit():
        # Prepare the BRQI state
        for wire in range(1,n+7):
            qml.Hadamard(wires=wire)
        for bitplane in range(8):
            # Encode the bitplane using EFRQI
            efrqi_circuit = efrqi_encoding(extract_bitplane(image, bitplane), n)
            # print(np.array(qml.matrix(efrqi_circuit)()).shape)
            efrqi_matrix = np.array(qml.matrix(efrqi_circuit)())
            print(efrqi_matrix)
            print(efrqi_matrix.shape)
            qml.ctrl(qml.QubitUnitary(efrqi_matrix, wires=range(1, 2*n+2)), control=0)
        return qml.state()
    
    return circuit

def extract_bitplane(image, j_bitplane):
    """
    Extracts a bitplane from a grayscale image

    Args:
        image (2D numpy array): input image
        j_bitplane (int): The index of the bitplane to extract (0-)

    Returns:
        2D numpy array: The extracted bitplane
    """

    bitplane_mask = 1 << j_bitplane
    bitplane_image = ((image & bitplane_mask) >> j_bitplane).astype(np.uint8)
    return bitplane_image

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
# q = 8
# efrqi_circuit = efrqi_encoding(image, n)
brqi_circuit = BRQI(image, n)

#eneqr_circuit = eneqr_encoding(image, n, q)

# Print the resulting quantum circuit
print(qml.draw(brqi_circuit, show_all_wires=True, wire_order=range(n))())