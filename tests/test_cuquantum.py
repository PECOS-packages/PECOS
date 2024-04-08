from pecos.simulators import CuStateVec
import numpy as np
import cupy as cp
import functools as ft

ident = np.eye(2)

def kron(A, B):
    if len(A) == 0:
        return B
    if len(B) == 0:
        return A
    return np.kron(A,B)

def paulix(q, N):
    return ft.reduce(kron, [
        np.eye(1 << (N-q-1)),
        np.matrix([[0,1],[1,0]], dtype=np.complex64),
        np.eye(1 << q)
    ])
def pauliy(q, N):
    return ft.reduce(kron, [
        np.eye(1 << (N-q-1)),
        np.matrix([[0,-1j],[1j,0]], dtype=np.complex64),
        np.eye(1 << q)
    ])
def pauliz(q, N):
    return ft.reduce(kron, [
        np.eye(1 << (N-q-1)),
        np.matrix([[1,0],[0,-1]], dtype=np.complex64),
        np.eye(1 << q)
    ])
def sqrtzz(qc,qt,N):
    return np.matrix([
        [1, 0,  0,  0],
        [0, 1j, 0,  0],
        [0, 0,  1j, 0],
        [0, 0,  0,  1]
        ], dtype=np.complex64)
def rzz(theta):
    a = np.exp(-1j*theta/2)
    b = np.exp(1j*theta/2)
    return np.matrix([
        [a, 0, 0, 0],
        [0, b, 0,  0],
        [0, 0, b, 0],
        [0, 0, 0, a]
        ], dtype=np.complex64)

def zeropsi(N):
    psi = np.zeros((1 << N), dtype=np.complex64)
    psi[0] = 1
    return psi

# Check X Single qubit
q = 0
N = 3
state = CuStateVec(N)
state.reset()
state.bindings["X"](state, q)
psi = cp.asnumpy(state._psi)
testpsi = np.ravel(paulix(q,N)@zeropsi(N))
# print(psi,testpsi)
# print([state.bindings["measure Z"](state, i) for i in range(N)])
# print({i:np.abs(j)**2 for i,j in enumerate(testpsi) if np.abs(j)**2>1e-4})
assert all(psi==testpsi), "States are not equal"
del state, psi

# Check Y Single qubit
N = 5
state = CuStateVec(N)
state.reset()
state.bindings["Y"](state, 0)
psi = cp.asnumpy(state._psi)
testpsi = np.ravel(pauliy(0,N)@zeropsi(N))
# print(psi,testpsi)
assert all(psi==testpsi), "States are not equal"
del state, psi

# Check Z Single qubit
N = 5
state = CuStateVec(N)
state.reset()
# state.bindings["Z"](state, 0)
psi = cp.asnumpy(state._psi)
testpsi = np.ravel(pauliz(0,N)@zeropsi(N))
assert all(psi==testpsi), "States are not equal"
del state, psi

# Check Two Qubit
N = 3
state = CuStateVec(N)
state.reset()
state.bindings["X"](state, 0)
state.bindings["Y"](state, 2)
psi = cp.asnumpy(state._psi)
testpsi = np.ravel(pauliy(2,N)@paulix(0,N)@zeropsi(N))
assert all(psi==testpsi), "States are not equal"
del state, psi

# Check Two Qubit
N = 4
state = CuStateVec(N)
state.reset()
state.bindings["X"](state, 0)
state.bindings["Y"](state, 1)
state.bindings["Z"](state, 2)
state.bindings["X"](state, 3)
psi = cp.asnumpy(state._psi)
testpsi = np.ravel(paulix(3,N)@pauliz(2,N)@pauliy(1,N)@paulix(0,N)@zeropsi(N))
# print(psi,testpsi)
assert all(psi==testpsi), "States are not equal"
del state, psi

# Check Two Qubit
N = 2
state = CuStateVec(N)
state.reset()
# state.bindings["X"](state, 0)
state.bindings["CNOT"](state, [0,1])
psi = cp.asnumpy(state._psi)
testpsi = np.ravel(paulix(0,N)@zeropsi(N))
print(psi,testpsi)
print([state.bindings["measure Z"](state, i) for i in range(N)])
print({i:np.abs(j)**2 for i,j in enumerate(testpsi) if np.abs(j)**2>1e-4})
assert all(psi==testpsi), "States are not equal"
del state, psi