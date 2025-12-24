.. _api-standard-gates:

Standard Gates
==============

While some simulators may allow access to other gate sets, the standard gates recognized by PECOS are:

Initializations
---------------

State initializations in Pauli bases:

=============== ========================================
``'init |+>'``  (Re)initiate the state :math:`|+\rangle`
``'init |->'``  (Re)initiate the state :math:`|-\rangle`
``'init |+i>'`` (Re)initiate the state :math:`|+i\rangle`
``'init |-i>'`` (Re)initiate the state :math:`|-i\rangle`
``'init |0>'``  (Re)initiate the state :math:`|0\rangle`
``'init |1>'``  (Re)initiate the state :math:`|1\rangle`
=============== ========================================

Unitaries
---------

Pauli operations:

======= =================================================
``'I'`` :math:`X\rightarrow X`, :math:`Z\rightarrow Z`
``'X'`` :math:`X\rightarrow X`, :math:`Z\rightarrow -Z`
``'Y'`` :math:`X\rightarrow -X`, :math:`Z\rightarrow -Z`
``'Z'`` :math:`X\rightarrow -X`, :math:`Z\rightarrow Z`
======= =================================================

Square-root of Pauli operations:

======== ============================================
``'Q'``  :math:`X \rightarrow X`, :math:`Z \rightarrow -Y`
``'R'``  :math:`X \rightarrow -Z`, :math:`Z \rightarrow X`
``'S'``  :math:`X \rightarrow Y`, :math:`Z \rightarrow Z`
``'Qd'`` :math:`X \rightarrow X`, :math:`Z \rightarrow Y`
``'Rd'`` :math:`X \rightarrow Z`, :math:`Z \rightarrow -X`
``'Sd'`` :math:`X \rightarrow -Y`, :math:`Z \rightarrow Z`
======== ============================================

Hamadard-like:

=============================== ================================
``'H'}, ``'H+z+x'}, or ``'H1'`` Hadamard: :math:`X\leftrightarrow Z`
``'H-z-x'`` or ``'H2'``         :math:`X\leftrightarrow -Z`
``'H+y-z'`` or ``'H3'``         :math:`X\rightarrow Y`, :math:`Z\rightarrow -Z`
``'H-y-z'`` or ``'H4'``         :math:`X\rightarrow -Y`, :math:`Z\rightarrow -Z`
``'H-x+y'`` or ``'H5'``         :math:`X\rightarrow -X`, :math:`\rightarrow Y`
``'H-x-y'`` or ``'H6'``         :math:`X\rightarrow -X`, :math:`Z\rightarrow -Y`
=============================== ================================

Rotations about the face of an octahedron:

========= ===================================================
``'F1'``  :math:`X \rightarrow Y\rightarrow Z \rightarrow X`
``'F2'``  :math:`X \rightarrow -Z`, :math:`Z \rightarrow Y`
``'F3'``  :math:`X \rightarrow Y`, :math:`Z \rightarrow -X`
``'F4'``  :math:`X \rightarrow Z`, :math:`Z \rightarrow -Y`
``'F1d'`` :math:`X\rightarrow Z\rightarrow Y \rightarrow X`
``'F2d'`` :math:`X \rightarrow -Y`, :math:`Z \rightarrow -X`
``'F3d'`` :math:`X \rightarrow -Z`, :math:`Z \rightarrow -Y`
``'F4d'`` :math:`X \rightarrow -Y`, :math:`Z \rightarrow X`
========= ===================================================

Two-qubit gates:

========== =================================================
``'CNOT'`` The controlled-X gate
``'CZ'``   The controlled-Z gate
``'SWAP'`` Swap two qubits
``'G'``    Equivalent to: :math:`CZ_{1,2}\;H_1 \otimes H_2\; CZ_{1,2}`
========== =================================================

Measurements
------------

Measurements in Pauli bases:

=============== =============================================
``'measure X'`` Measure in the :math:`X`-basis
``'measure Y'`` Measure in the :math:`Y`-basis
``'measure Z'`` Measure in the :math:`Z`-basis
=============== =============================================
