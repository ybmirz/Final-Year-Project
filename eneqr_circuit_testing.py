#from circuitcomponents import CircuitComponents
import pennylane as qml
from pennylane import numpy as np
from tensorflow.keras.datasets import mnist
import cv2
import math
import torch

def optimized_Toffoli(wires):
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


def eneqr_encoding(image):
    """
    Encodes a 2^n x 2^n grayscale image using the ENEQR scheme.
    
    Args:
        image (numpy.ndarray): The input grayscale image.
        n (int): The number of qubits used to represent the image position.
        q (int): The number of qubits used to represent the pixel value.
    
    Returns:
        qml.QNode: The quantum circuit that encodes the image.
    """
    q = 8
    n = int(np.log2(2))
    required_qubits = q + 2 * n
    print(required_qubits)
    dev = qml.device("default.qubit", wires=required_qubits)

    @qml.qnode(dev)
    def circuit():
        # Initialize the state
        for i in range(required_qubits):
            qml.Hadamard(wires=i)
        
        # Prepare image data
        normalized_image = (image.flatten() * (2**q - 1)).astype(np.uint8)
        binaries = [np.binary_repr(pixel, width=q) for pixel in normalized_image]

        # Apply the encoding
        for idx, pixel in enumerate(binaries):
            x, y = divmod(idx, 2**n)
            x_bits = np.binary_repr(x, width=n)
            y_bits = np.binary_repr(y, width=n)
            for qubit in range(q):
                if pixel[qubit] == '1':
                    control_bits = [int(bit) for bit in (x_bits + y_bits)]
                    control_wires = [i for i, bit in enumerate(control_bits) if bit == 1]
                    # Using optimized Toffoli gates
                    if control_wires:
                        optimized_Toffoli(control_wires + [n*2 + qubit])


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
            print(image)
            print(image.shape)
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

# Normalize the images to float values between 0 and 1
x_train_normalized = _.astype('float32') / 255.0
x_test_normalized = x_test.astype('float32') / 255.0


#Choose a random image from the test set
image = x_test[np.random.randint(0, len(x_test_normalized))]
image = cv2.resize(image, (14, 14))


# Retrieve the 2x2 filter part of image
sub_image = qml.numpy.array(image[0:2, 0:2])
# Extract the center 2x2 block
center_x, center_y = 14 // 2, 14 // 2
sub_image = qml.numpy.array(image[center_x-1:center_x+1, center_y-1:center_y+1])
sub_image = sub_image.astype('float32') / 255.0

print(type(sub_image))
sub_image = sub_image.flatten()
# Encode the image using the ENEQR approach
# H,W = image.shape
# print(image.shape)
# n = max(math.ceil(math.log2(H)), math.ceil(math.log2(W)))
# q = 8
# efrqi_circuit = efrqi_encoding(image, n)
#brqi_circuit = BRQI(image, n)


eneqr_circuit = eneqr_encoding(sub_image)

# Print the resulting quantum circuit
print(qml.draw(eneqr_circuit, show_all_wires=True, wire_order=range(3))())
print(qml.matrix(eneqr_circuit)())
