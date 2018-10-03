#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import pecos as pc
from pecos.misc.threshold_curve import func


def test_finding_threshold():
    depolar = pc.error_gens.DepolarGen(model_level='code_capacity', perp_errors=True)
    ps = [0.19, 0.17, 0.15, 0.13, 0.11]
    ds = [5, 7, 9]
    plist = np.array(ps * len(ds))

    dlist = []
    for d in ds:
        for _ in ps:
            dlist.append(d)
    dlist = np.array(dlist)

    plog = []
    for d in ds:
        for p in ps:
            surface = pc.qeccs.Surface4444(distance=d)
            mwpm2d = pc.decoders.MWPM2D(surface)
            plog.append(pc.tools.codecapacity_logical_rate(10, surface, d, depolar, error_params={'p': p},
                                                           decoder=mwpm2d, verbose=False)[0])

    plog = np.array(plog)

    print('Finished!')

    try:
        p0 = (0.1, 1.5, 1, 1, 1)
        pc.misc.threshold_fit(plist, dlist, plog, func, p0, maxfev=1000)
    except RuntimeError:
        pass
