import pennylane as qml
from pennylane import numpy as np
from encoders.neqr import NEQR
#import qiskit as qis
from circuitcomponents import CircuitComponents
import torch

class BRQI(CircuitComponents):
    """
    BRQI: Bitplane Representation for Quantum Images
    Varying slightly different that original Li et al paper, by using
    NEQR instead of GNEQR
    """

    def __init__(self, filter_length=2):
        if not 2 ** int(np.log2(filter_length)) == filter_length:
            raise Exception ("filter_length needs to be equal to 2**n for some integer n.")
        self.name = "BRQI"
        self.n = int(np.log2(filter_length))
        self.required_qubits = self.n+7

    def circuit(self, inputs):
        for wire in range(1, self.required_qubits):
            qml.Hadamard(wires=wire)

        for bitplane in range(8):
            inputs = inputs.view(2,2)
            inputs = (inputs * 255).type(torch.uint8)
            bp = self.extract_bitplane(inputs, bitplane)
            # Encode the bitplane using NEQR
            neqr_circuit = NEQR().return_circuit(bp, n=self.n)
            # print(np.array(qml.matrix(neqr_circuit)()).shape)
            # Represent the NEQR circuit as a Unitary operation
            neqr_matrix = np.array(qml.matrix(neqr_circuit)())
            qml.ctrl(qml.QubitUnitary(neqr_matrix, wires=range(1, 2*self.n+2)), control=0)
        

    def extract_bitplane(self, image, j_bitplane):
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
