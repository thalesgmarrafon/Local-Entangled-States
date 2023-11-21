import numpy as np
from math import sqrt
import picos as pc

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------DEFINITION OF STATES AND MATRICES-----------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

ket_0 = np.array([[1],[0]]) #|0>
ket_1 = np.array([[0],[1]]) #|1>
ket_00 = np.array([[1],[0],[0],[0]]) #|00>
ket_01 = np.array([[0],[1],[0],[0]]) #|01>
ket_10 = np.array([[0],[0],[1],[0]]) #|10>
ket_11 = np.array([[0],[0],[0],[1]]) #|11>

ketbra_00 = np.array([[1,0],[0,0]]) #|0><0|
ketbra_01 = np.array([[0,1],[0,0]]) #|0><1|
ketbra_10 = np.array([[0,0],[1,0]]) #|1><0|
ketbra_11 = np.array([[0,0],[0,1]]) #|1><1|

psip = (ket_01 + ket_10)/sqrt(2) #(|01> + |10>)/sqrt(2)
psim = (ket_01 - ket_10)/sqrt(2) #(|01> - |10>)/sqrt(2)
phip = (ket_00 + ket_11)/sqrt(2) #(|00> + |11>)/sqrt(2)
phim = (ket_00 - ket_11)/sqrt(2) #(|00> - |11>)/sqrt(2)

rhopsip = psip@np.transpose(psip) #(|01> + |10>)(<01| + <10|)/2
rhopsim = psim@np.transpose(psim) #(|01> - |10>)(<01| - <10|)/2
rhophip = phip@np.transpose(phip) #(|00> + |11>)(<00| + <11|)/2
rhophim = phim@np.transpose(phim) #(|00> - |11>)(<00| - <11|)/2

rho_00 = ket_00@np.transpose(ket_00) #|00><00|
rho_11 = ket_11@np.transpose(ket_11) #|11><11|
rho_01 = ket_01@np.transpose(ket_01) #|01><01|
rho_10 = ket_10@np.transpose(ket_10) #|10><10|

I = np.array([[1, 0], [0, 1]]) #Identity and Pauli matrices
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])
sigmavec = [sigmax, sigmay, sigmaz]
sigmatensor = np.array([[np.kron(sigmax,sigmax), np.kron(sigmax,sigmay), np.kron(sigmax,sigmaz)],
                        [np.kron(sigmay,sigmax), np.kron(sigmay,sigmay), np.kron(sigmay,sigmaz)],
                        [np.kron(sigmaz,sigmax), np.kron(sigmaz,sigmay), np.kron(sigmaz,sigmaz)]])

rho_I = np.kron(I,I)/4

Ip = pc.Constant("Ip", [1, 0, 0, 1], (2, 2)) #Identity using picos