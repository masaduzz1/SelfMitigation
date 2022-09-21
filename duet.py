# This file is called called duet.py
# It is available at https://github.com/randylewis/SelfMitigation
# It uses IBM's open-source software called qiskit, available from https://qiskit.org/
#
# Self-mitigation helps to mitigate errors on noisy intermediate-scale quantum computer.
# It was introduced in our paper: Self-mitigating Trotter circuits for SU(2) lattice gauge theory on a quantum computer
#                                 A Rahman, Lewis, Mendicelli and Powell
#                                 http://arxiv.org/abs/2205.09247 [hep-lat] (2022)
# If you find this code useful, please cite the paper as well as the code.
# The code below is for time evolution of SU(2) lattice gauge theory on a 2-plaquette lattice having no periodic boundary.
# randy.lewis@yorku.ca

# Define user choices.
myhardware = 2           # Choose the qubit hardware.
myprovider = 2           # Choose 0 for the public account or 2 for a research account.
myqubits = [1,2]         # Choose the specific qubits to be used.
myNtphy = 4              # Choose the number of time steps for the physics circuit.
mydt = 0.08              # Choose the size of each time step.
myruns = 148             # Choose the number of times you want to run this physics circuit. Each run will have a new CNOT randomization.
myshots = 10000          # Choose the number of shots for each run.
myx = 2.0                # Choose the gauge coupling.
myconfidencelevel = 0.95 # Choose 0.95, for example, for error bars that represent a 95% confidence level.
mytime = myNtphy*mydt
print("Input:",myhardware,myprovider,myqubits,myNtphy,mydt,myruns,myshots,myx,myconfidencelevel,mytime)

# Import tools from qiskit and load my IBM Q account.
from qiskit import IBMQ, QuantumRegister, ClassicalRegister, QuantumCircuit, execute, result
from qiskit.tools.monitor import job_monitor
IBMQ.load_account()
if (myprovider==2):
    provider = IBMQ.get_provider(hub='ibm-q-research-2')
else:
    provider = IBMQ.get_provider(hub='ibm-q')

# Import standard python tools.
import datetime
from numpy import pi, sqrt, dot
from math import copysign
from scipy.stats import norm
from scipy.optimize import minimize
from random import random, randrange

# Identify the hardware that will be used.
if myhardware==1:
    chosenhardware = "ibm_perth"
elif myhardware==2:
    chosenhardware = "ibm_lagos"
elif myhardware==3:
    chosenhardware = "ibmq_casablanca"
elif myhardware==4:
    chosenhardware = "ibmq_bogota"
elif myhardware==5:
    chosenhardware = "ibmq_manila"
else:
    chosenhardware = "ibmq_qasm_simulator"
backend = provider.get_backend(chosenhardware)

# This function creates an initial state |01> from the state |00> and inserts the "P,Q" Paulis for the upcoming CNOT gate.
def createright():
    global nextPQ
    nextPQ = [randrange(4),randrange(4)]
    if nextPQ[0]==0:
        circ.x(qreg[0])
    elif nextPQ[0]==2:
        circ.z(qreg[0])
    elif nextPQ[0]==3:
        circ.y(qreg[0])
    if nextPQ[1]==1:
        circ.x(qreg[1])
    elif nextPQ[1]==2:
        circ.y(qreg[1])
    elif nextPQ[1]==3:
        circ.z(qreg[1])

# This function uses the Pauli gates that precede a randomized CNOT to define the Pauli gates that follow the randomized CNOT.
def RSfromPQ(P,Q):
    if P==0 and Q==0:
        R = 0
        S = 0
    elif P==0 and Q==1:
        R = 0
        S = 1
    elif P==0 and Q==2:
        R = 3
        S = 2
    elif P==0 and Q==3:
        R = 3
        S = 3
    elif P==1 and Q==0:
        R = 1
        S = 1
    elif P==1 and Q==1:
        R = 1
        S = 0
    elif P==1 and Q==2:
        R = 2
        S = 3
    elif P==1 and Q==3:
        R = 2
        S = 2
    elif P==2 and Q==0:
        R = 2
        S = 1
    elif P==2 and Q==1:
        R = 2
        S = 0
    elif P==2 and Q==2:
        R = 1
        S = 3
    elif P==2 and Q==3:
        R = 1
        S = 2
    elif P==3 and Q==0:
        R = 3
        S = 0
    elif P==3 and Q==1:
        R = 3
        S = 1
    elif P==3 and Q==2:
        R = 0
        S = 2
    elif P==3 and Q==3:
        R = 0
        S = 3
    return R, S

# This function applies a CNOT gate and records the randomized Pauli gates that follow it.
def applyCNOT(ctrl,targ):
    global thisRS, nextPQ
    thisPQ = nextPQ
    circ.cx(qreg[ctrl],qreg[targ])
    if ctrl==0:
        R,S = RSfromPQ(thisPQ[0],thisPQ[1])
        thisRS[0] = R
        thisRS[1] = S
    elif ctrl==1:
        R,S = RSfromPQ(thisPQ[1],thisPQ[0])
        thisRS[1] = R
        thisRS[0] = S
    nextPQ = [randrange(4),randrange(4)]

# This function applies Pauli then the identity then Pauli.
def applyI(qubit):
    global thisRS, nextPQ
    if thisRS[qubit]==0 and nextPQ[qubit]==1:
        circ.x(qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==2:
        circ.y(qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==3:
        circ.z(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==0:
        circ.x(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==2:
        circ.z(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==3:
        circ.y(qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==0:
        circ.y(qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==1:
        circ.z(qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==3:
        circ.x(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==0:
        circ.z(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==1:
        circ.y(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==2:
        circ.x(qreg[qubit])

# This function applies Pauli then RY(theta) then Pauli.
def applyRY(qubit,theta):
    global thisRS, nextPQ
    if thisRS[qubit]==0 and nextPQ[qubit]==0:
        circ.ry(theta,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==1:
        circ.ry(theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==2:
        circ.ry(pi+theta,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==3:
        circ.ry(pi+theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==0:
        circ.ry(-theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==1:
        circ.ry(-theta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==2:
        circ.ry(pi-theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==3:
        circ.ry(pi-theta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==0:
        circ.ry(pi+theta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==1:
        circ.ry(pi+theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==2:
        circ.ry(theta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==3:
        circ.ry(theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==0:
        circ.ry(pi-theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==1:
        circ.ry(pi-theta,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==2:
        circ.ry(-theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==3:
        circ.ry(-theta,qreg[qubit])

# This function applies Pauli then RZ(theta) then Pauli.
def applyRZ(qubit,theta):
    global thisRS, nextPQ
    if thisRS[qubit]==0 and nextPQ[qubit]==0:
        circ.rz(theta,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==1:
        circ.rz(theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==2:
        circ.rz(pi+theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==3:
        circ.rz(pi+theta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==0:
        circ.rz(-theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==1:
        circ.rz(-theta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==2:
        circ.rz(pi-theta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==3:
        circ.rz(pi-theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==0:
        circ.rz(pi-theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==1:
        circ.rz(pi-theta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==2:
        circ.rz(-theta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==3:
        circ.rz(-theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==0:
        circ.rz(pi+theta,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==1:
        circ.rz(pi+theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==2:
        circ.rz(theta,qreg[qubit])
        circ.x(qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==3:
        circ.rz(theta,qreg[qubit])

# This function applies Pauli then RY(alpha) then RZ(beta) then Pauli.
def applyRYRZ(qubit,alpha,beta):
    global thisRS, nextPQ
    if thisRS[qubit]==0 and nextPQ[qubit]==0:
        circ.ry(alpha,qreg[qubit])
        circ.rz(beta,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==1:
        circ.ry(pi+alpha,qreg[qubit])
        circ.rz(pi-beta,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==2:
        circ.ry(pi+alpha,qreg[qubit])
        circ.rz(-beta,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==3:
        circ.ry(alpha,qreg[qubit])
        circ.rz(pi+beta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==0:
        circ.ry(pi-alpha,qreg[qubit])
        circ.rz(pi+beta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==1:
        circ.ry(-alpha,qreg[qubit])
        circ.rz(-beta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==2:
        circ.ry(-alpha,qreg[qubit])
        circ.rz(pi-beta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==3:
        circ.ry(pi-alpha,qreg[qubit])
        circ.rz(beta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==0:
        circ.ry(pi+alpha,qreg[qubit])
        circ.rz(beta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==1:
        circ.ry(alpha,qreg[qubit])
        circ.rz(pi-beta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==2:
        circ.ry(alpha,qreg[qubit])
        circ.rz(-beta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==3:
        circ.ry(pi+alpha,qreg[qubit])
        circ.rz(pi+beta,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==0:
        circ.ry(-alpha,qreg[qubit])
        circ.rz(pi+beta,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==1:
        circ.ry(pi-alpha,qreg[qubit])
        circ.rz(-beta,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==2:
        circ.ry(pi-alpha,qreg[qubit])
        circ.rz(pi-beta,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==3:
        circ.ry(-alpha,qreg[qubit])
        circ.rz(beta,qreg[qubit])

# This function applies Pauli then RZ(alpha) then RY(beta) then Pauli.
def applyRZRY(qubit,alpha,beta):
    global thisRS, nextPQ
    if thisRS[qubit]==0 and nextPQ[qubit]==0:
        circ.rz(alpha,qreg[qubit])
        circ.ry(beta,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==1:
        circ.rz(pi+alpha,qreg[qubit])
        circ.ry(pi-beta,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==2:
        circ.rz(alpha,qreg[qubit])
        circ.ry(pi+beta,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==3:
        circ.rz(pi+alpha,qreg[qubit])
        circ.ry(-beta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==0:
        circ.rz(pi-alpha,qreg[qubit])
        circ.ry(pi+beta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==1:
        circ.rz(-alpha,qreg[qubit])
        circ.ry(-beta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==2:
        circ.rz(pi-alpha,qreg[qubit])
        circ.ry(beta,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==3:
        circ.rz(-alpha,qreg[qubit])
        circ.ry(pi-beta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==0:
        circ.rz(-alpha,qreg[qubit])
        circ.ry(pi+beta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==1:
        circ.rz(pi-alpha,qreg[qubit])
        circ.ry(-beta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==2:
        circ.rz(-alpha,qreg[qubit])
        circ.ry(beta,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==3:
        circ.rz(pi-alpha,qreg[qubit])
        circ.ry(pi-beta,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==0:
        circ.rz(pi+alpha,qreg[qubit])
        circ.ry(beta,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==1:
        circ.rz(alpha,qreg[qubit])
        circ.ry(pi-beta,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==2:
        circ.rz(pi+alpha,qreg[qubit])
        circ.ry(pi+beta,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==3:
        circ.rz(alpha,qreg[qubit])
        circ.ry(-beta,qreg[qubit])

# This function calculates Pauli then RZ(alpha) then RY(beta) then RZ(alpha) then Pauli.
def applyRZRYRZ(qubit,alpha,beta):
    global thisRS, nextPQ
    if thisRS[qubit]==0 and nextPQ[qubit]==0:
        circ.rz(alpha,qreg[qubit])
        circ.ry(beta,qreg[qubit])
        circ.rz(alpha,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==1:
        circ.rz(alpha,qreg[qubit])
        circ.ry(pi+beta,qreg[qubit])
        circ.rz(pi-alpha,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==2:
        circ.rz(alpha,qreg[qubit])
        circ.ry(pi+beta,qreg[qubit])
        circ.rz(-alpha,qreg[qubit])
    elif thisRS[qubit]==0 and nextPQ[qubit]==3:
        circ.rz(alpha,qreg[qubit])
        circ.ry(beta,qreg[qubit])
        circ.rz(pi+alpha,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==0:
        circ.rz(pi-alpha,qreg[qubit])
        circ.ry(pi+beta,qreg[qubit])
        circ.rz(alpha,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==1:
        circ.rz(-alpha,qreg[qubit])
        circ.ry(-beta,qreg[qubit])
        circ.rz(-alpha,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==2:
        circ.rz(-alpha,qreg[qubit])
        circ.ry(-beta,qreg[qubit])
        circ.rz(pi-alpha,qreg[qubit])
    elif thisRS[qubit]==1 and nextPQ[qubit]==3:
        circ.rz(-alpha,qreg[qubit])
        circ.ry(pi-beta,qreg[qubit])
        circ.rz(alpha,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==0:
        circ.rz(-alpha,qreg[qubit])
        circ.ry(pi+beta,qreg[qubit])
        circ.rz(alpha,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==1:
        circ.rz(pi-alpha,qreg[qubit])
        circ.ry(-beta,qreg[qubit])
        circ.rz(-alpha,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==2:
        circ.rz(-alpha,qreg[qubit])
        circ.ry(beta,qreg[qubit])
        circ.rz(-alpha,qreg[qubit])
    elif thisRS[qubit]==2 and nextPQ[qubit]==3:
        circ.rz(-alpha,qreg[qubit])
        circ.ry(pi+beta,qreg[qubit])
        circ.rz(pi+alpha,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==0:
        circ.rz(pi+alpha,qreg[qubit])
        circ.ry(beta,qreg[qubit])
        circ.rz(alpha,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==1:
        circ.rz(alpha,qreg[qubit])
        circ.ry(pi-beta,qreg[qubit])
        circ.rz(-alpha,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==2:
        circ.rz(pi+alpha,qreg[qubit])
        circ.ry(pi+beta,qreg[qubit])
        circ.rz(-alpha,qreg[qubit])
    elif thisRS[qubit]==3 and nextPQ[qubit]==3:
        circ.rz(alpha,qreg[qubit])
        circ.ry(-beta,qreg[qubit])
        circ.rz(alpha,qreg[qubit])

# This function inserts the "A" portion of the time-evolution circuit.
def Aportion(dt):
    applyCNOT(1,0)
    applyRYRZ(0,-myx*dt/2,-3*dt/8)
    applyI(1)

# This function inserts the "B" portion of the time-evolution circuit.
def Bportion(dt):
    applyCNOT(1,0)
    applyRYRZ(0,-3*myx*dt/2,-9*dt/4)
    applyRZRY(1,-9*dt/8,-3*myx*dt)
    applyCNOT(0,1)
    applyI(0)
    applyRY(1,-myx*dt)
    applyCNOT(0,1)
    applyRY(0,-3*myx*dt/2)
    applyRZ(1,-9*dt/8)
    applyCNOT(1,0)

# This function inserts the "C" portion of the time-evolution circuit.
def Cportion(dt):
    applyRZRYRZ(0,-3*dt/8,-myx*dt)
    applyI(1)

# This function inserts the "D" portion of the time-evolution circuit.
def Dportion(rawdt):
    global thisRS
    dt = rawdt
    applyRZRY(0,-3*dt/8,-myx*dt/2)
    applyI(1)
    applyCNOT(1,0)
    for qubit in range(2):
        if thisRS[qubit]==1:
            circ.x(qreg[qubit])
        elif thisRS[qubit]==2:
            circ.y(qreg[qubit])
        elif thisRS[qubit]==3:
            circ.z(qreg[qubit])

# This function inserts a complete 2nd-order Trotter-Suzuki step if first=last=True.  Otherwise the beginning and end are adjusted.
def TrotterSuzuki(dt,dtforC,first,last):
    if (first):
        Aportion(dt)
    else:
        Cportion(dtforC)
    Bportion(dt)
    if (last):
        Dportion(dt)

# This function accounts for readout errors by using sequential least squares programming (SLSQP).
def ReadoutMitigation(Vector):
    plaq1 = []
    plaq2 = []
    for i in range(myruns):
        def fun(vector):
            return sum((Vector[i] - dot(CalibrationMatrix,vector))**2)
        BestVector = Vector[i]
        constraints = ({'type': 'eq', 'fun': lambda x: myshots-sum(x)})
        bounds = tuple((0, myshots) for x in BestVector)
        leastsquares = minimize(fun, BestVector, method='SLSQP', constraints=constraints, bounds=bounds, tol=1e-6)
        BestVector = leastsquares.x
        plaq1.append(BestVector[1] + BestVector[3])
        plaq2.append(BestVector[2] + BestVector[3])
    return plaq1, plaq2

# This function calculates a central value and error bar from the Wilson score interval.
def WilsonScore(plaq1,plaq2):
    p1 = []
    p2 = []
    sigma1 = []
    sigma2 = []
    z = norm.ppf((1+myconfidencelevel)/2)
    denom = 1 + z**2/myshots
    for i in range(myruns):
        p1hat = plaq1[i]/myshots
        p2hat = plaq2[i]/myshots
        p1.append(( p1hat + z**2/(2*myshots) )/denom)
        p2.append(( p2hat + z**2/(2*myshots) )/denom)
        sigma1.append(z*sqrt( p1hat*(1-p1hat)/myshots + z**2/(4*myshots**2) )/denom)
        sigma2.append(z*sqrt( p2hat*(1-p2hat)/myshots + z**2/(4*myshots**2) )/denom)
    return p1,sigma1,p2,sigma2

# Build the four circuits for mitigation of readout errors.
circlist = []
for i in range(4):
    qreg = QuantumRegister(2)
    creg = ClassicalRegister(2)
    circ = QuantumCircuit(qreg,creg)
    if (i==1 or i==3):
        circ.x(qreg[0])
    if (i==2 or i==3):
        circ.x(qreg[1])
    circ.measure(qreg,creg)
    circlist.append(circ)

# Build the circuits for mitigation of gate errors.
thisRS = [999,999]
for i in range(myruns):
# ----Initialize the circuit.
    qreg = QuantumRegister(2)
    creg = ClassicalRegister(2)
    circ = QuantumCircuit(qreg,creg)
    createright()
# ----Put the forward Trotter steps into the circuit.
    first = True
    last = False
    thisdt = mydt
    thisdtforC = mydt
    for it in range(myNtphy//2-1):
        TrotterSuzuki(thisdt,thisdtforC,first,last)
        first = False
    TrotterSuzuki(thisdt,thisdtforC,first,last)
# ----Put the backward Trotter steps into the circuit.
    first = False
    thisdt = -mydt
    thisdtforC = 0
    circ.barrier(qreg)
    for it in range(myNtphy//2-1):
        TrotterSuzuki(thisdt,thisdtforC,first,last)
        thisdtforC = thisdt
    last = True
    TrotterSuzuki(thisdt,thisdtforC,first,last)
# ----Put the final measurement into the circuit.
    circ.measure(qreg,creg)
    circlist.append(circ)

# Build the physics circuits.
for i in range(myruns):
# ----Initialize the circuit.
    qreg = QuantumRegister(2)
    creg = ClassicalRegister(2)
    circ = QuantumCircuit(qreg,creg)
    createright()
# ----Put the Trotter steps into the circuit.
    first = True
    last = False
    thisdt = mydt
    thisdtforC = mydt
    for it in range(myNtphy-1):
        TrotterSuzuki(thisdt,thisdtforC,first,last)
        first = False
    last = True
    TrotterSuzuki(thisdt,thisdtforC,first,last)
# ----Put the final measurement into the circuit.
    circ.measure(qreg,creg)
    circlist.append(circ)

# Run all of the circuits.
print("Queuing the circuits on the hardware at",datetime.datetime.now())
job = execute(circlist,backend,initial_layout=myqubits,shots=myshots)
job_monitor(job)
print("The hardware has returned results at",datetime.datetime.now())
circoutput = job.result().get_counts()
print("The output from all circuits is")
for i in range(4+2*myruns):
    print(circoutput[i])
readoutdata = circoutput[0:4]
gatedata = circoutput[4:4+myruns]
physicsdata = circoutput[4+myruns:4+2*myruns]

# Collect the counts for the readout error mitigation matrix.
CalibrationMatrix = []
for i in range(4):
    CalibrationMatrix.append([0,0,0,0])
for i in range(4):
    for j,(key,value) in enumerate(readoutdata[i].items()):
        if key=="00":
            CalibrationMatrix[0][i] += value/myshots
        elif key=="01":
            CalibrationMatrix[1][i] += value/myshots
        elif key=="10":
            CalibrationMatrix[2][i] += value/myshots
        elif key=="11":
            CalibrationMatrix[3][i] += value/myshots
print("The calibration matrix is")
for i in CalibrationMatrix:
    print(" ".join(map(str,i)))

# Collect the counts for the gate-error mitigation circuits.
GateVector = []
for i in range(myruns):
    GateVector.append([0,0,0,0])
for i in range(myruns):
    for j,(key,value) in enumerate(gatedata[i].items()):
        if key=="00":
            GateVector[i][0] += value
        elif key=="01":
            GateVector[i][1] += value
        elif key=="10":
            GateVector[i][2] += value
        elif key=="11":
            GateVector[i][3] += value

# Collect the counts for the physics circuits.
PhysicsVector = []
for i in range(myruns):
    PhysicsVector.append([0,0,0,0])
for i in range(myruns):
    for j,(key,value) in enumerate(physicsdata[i].items()):
        if key=="00":
            PhysicsVector[i][0] += value
        elif key=="01":
            PhysicsVector[i][1] += value
        elif key=="10":
            PhysicsVector[i][2] += value
        elif key=="11":
            PhysicsVector[i][3] += value

# Account for readout errors by using sequential least squares programming (SLSQP).
plaq1gate, plaq2gate = ReadoutMitigation(GateVector)
plaq1physics, plaq2physics = ReadoutMitigation(PhysicsVector)

# Get a central value and error bar from the Wilson score interval.
p1gate, sigma1gate, p2gate, sigma2gate = WilsonScore(plaq1gate,plaq2gate)
p1physics, sigma1physics, p2physics, sigma2physics = WilsonScore(plaq1physics,plaq2physics)

# Report the results.
for i in range(myruns):
    print(mytime,p1gate[i],sigma1gate[i],p2gate[i],sigma2gate[i],end=" ")
    print(p1physics[i],sigma1physics[i],p2physics[i],sigma2physics[i])

# How to use the output from this code:
#
# 1. p1gate +/- sigma1gate are the mitigation runs for the left plaquette (red without symbols in Fig. 3 upper panel).
#    p2gate +/- 2sigma2gate are the mitigation runs for the right plaquette (blue without symbols in Fig. 3 upper panel).
#    p1physics +/- sigma1physics are the physics runs for the left plaquette (red symbols in Fig. 3 upper panel).
#    p2physics +/- sigma2physics are the physics runs for the right plaquette (blue symbols in Fig. 3 upper panel).
#    You will want to calculate averages (p1gateave, p2gateave, p1physics ave, p2physicsave) perhaps with bootstrap error bars.
#
# 2. Self-mitigation is implemented by using equation 8 in http://arxiv.org/abs/2205.09247 [hep-lat] (2022).
#    P_true on the left-hand side is what you are trying to calculate.
#    P_computed on the left-hand side is p1physicsave (left plaquette) or p2physicsave (right plaquette).
#    P_true on the right-hand side is 1 (left plaquette) or 0 (right plaquette).
#    P_computed on the right-hand side is p1gateave (left plaquette) or p2gateave (right plaquette).

print("Successful completion of duet.py")
