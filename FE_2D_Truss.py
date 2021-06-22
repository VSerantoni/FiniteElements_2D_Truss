# Problem 2D - Truss

import matplotlib.pylab as plt
import numpy as np

def formStiffness2Dtruss(GDof, numberElements, elementNodes, xx, yy, EA):
    
    stiffness = np.zeros([GDof, GDof])

    # Computation of the stiffness matrix
    for e in range(0, numberElements):
        indice = elementNodes[e, :]
        # calcul x2-x1 et y2-y1
        x2 = xx[indice[1]]-xx[indice[0]]
        y2 = yy[indice[1]] - yy[indice[0]]
        # length of element and cos/sin angle
        Le = np.sqrt(x2**2+y2**2)
        C = x2/Le
        S = y2/Le
        # stiffness matrice for the element
        Ke = EA/Le * np.array([[C**2, C*S, -C**2, -C*S], [C*S, S**2, -C*S, -S**2],
                               [-C**2, -C*S, C**2, C*S], [-C*S, -S**2, C*S, S**2]])

        # injection inside the global matrix
        elementDof = np.array([[indice[0]*2, indice[0]*2+1, indice[1]*2, indice[1]*2+1]])
        stiffness[elementDof, elementDof.T] = stiffness[elementDof, elementDof.T] + Ke

    return stiffness

def solution(GDof, prescribedDof, stiffness, force):
    activeDof = np.setdiff1d(np.arange(0, GDof), prescribedDof)

    U = np.linalg.solve(stiffness[activeDof, activeDof[:, None]], force[activeDof])
    displacements = np.zeros([GDof])
    displacements[activeDof] = U
    return displacements

def stresses2Dtruss(numberElements, elementNodes, xx, yy, displacements, E):
    sigma = np.zeros([numberElements])
    for e in range(0, numberElements):
        indice = elementNodes[e, :]
        elementDof = np.array([[indice[0] * 2, indice[0] * 2 + 1, indice[1] * 2, indice[1] * 2 + 1]])
        # calcul x2-x1 et y2-y1
        x2 = xx[indice[1]]-xx[indice[0]]
        y2 = yy[indice[1]] - yy[indice[0]]
        # length of element and cos/sin angle
        Le = np.sqrt(x2**2+y2**2)
        C = x2 / Le
        S = y2 / Le
        # stress
        sigma[e] = E/Le * np.array([[-C, -S, C, S]])@displacements[elementDof].T

    print(sigma[:, None])


def outputDisplacementsReactions(displacements, stiffness, GDof, prescribedDof):
    # displacements
    print('Displacements')
    jj = np.linspace(0, GDof - 1, GDof).reshape(GDof, 1)
    print(np.append(jj, displacements[:, None], axis=1))

    # reactions
    F = np.matmul(stiffness, displacements)
    reactions = F[prescribedDof]
    print('Reactions')
    print(np.append(prescribedDof[:, None], reactions[:, None], axis=1))

E = 70000
A = 300
EA = E * A
L = 1000
Load1 = -100000
Load2 = -50000

# coordinates and connectivities
numberElements = 11
numberNodes = 6
elementNodes = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [0, 3], [2, 3], [2, 5], [3, 4], [3, 5], [2, 4], [4, 5]])
nodeCoor = np.array([[0, 0], [0, L], [L, 0], [L, L], [2*L, 0], [2*L, L]])
xx = nodeCoor[:, 0]
yy = nodeCoor[:, 1]

# for structure
GDof = 2*numberNodes
displacements = np.zeros([GDof])
force = np.zeros([GDof])
force[3] = Load2
force[7] = Load1
force[11] = Load2

# computation of the system stiffness matrix
stiffness = formStiffness2Dtruss(GDof, numberElements, elementNodes, xx, yy, EA)

# boundary conditions and solution
prescribedDof = np.array([0, 1, 9])
displacements = solution(GDof, prescribedDof, stiffness, force)

# output displacements/reactions
outputDisplacementsReactions(displacements, stiffness, GDof, prescribedDof)

# stresses at elements
stresses2Dtruss(numberElements, elementNodes, xx, yy, displacements, E)

# deformed structure
scaling = 10
new_xx = xx + scaling*displacements[np.arange(0, GDof, 2)]
new_yy = yy + scaling*displacements[np.arange(1, GDof, 2)]

plt.figure()
plt.grid()
for e in range(0,numberElements):
    # initial struture
    indice = elementNodes[e, :]
    x_plot = np.array([xx[indice[0]], xx[indice[1]]])
    y_plot = np.array([yy[indice[0]], yy[indice[1]]])
    plt.plot(x_plot, y_plot, '-ok', lw=1)

    # deformed structure
    x2_plot = np.array([new_xx[indice[0]], new_xx[indice[1]]])
    y2_plot = np.array([new_yy[indice[0]], new_yy[indice[1]]])
    plt.plot(x2_plot, y2_plot, '--ok', lw=1)

plt.title('Deformation of the truss')
plt.legend(['Initial structure', 'Deformed structure (scaled)'])
plt.show()
