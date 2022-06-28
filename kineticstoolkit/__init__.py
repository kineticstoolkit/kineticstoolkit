#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Kinetics Toolkit
================

To get started, please consult Kinetics Toolkit's
[website](https://kineticstoolkit.uqam.ca)

>>> import kineticstoolkit as ktk

"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import os


listing = []
unstable_listing = []


# --- Import released modules and functions
from kineticstoolkit.timeseries import TimeSeries, TimeSeriesEvent  # noqa

listing.append("TimeSeries")
listing.append("TimeSeriesEvent")

from kineticstoolkit.tools import start_lab_mode  # noqa

listing.append("start_lab_mode")

from kineticstoolkit.player import Player  # noqa

listing.append("Player")

from kineticstoolkit.loadsave import load, save  # noqa

listing.append("load")
listing.append("save")

from kineticstoolkit import filters  # noqa

listing.append("filters")

from kineticstoolkit import kinematics  # noqa

listing.append("kinematics")

from kineticstoolkit import pushrimkinetics  # noqa

listing.append("pushrimkinetics")

from kineticstoolkit import cycles  # noqa

listing.append("cycles")

from kineticstoolkit import doc  # noqa

listing.append("doc")

from kineticstoolkit import _repr  # noqa
from kineticstoolkit import gui  # noqa

from kineticstoolkit import geometry  # noqa

listing.append("geometry")


# Load unstable and dev modules (but do not add those to the __dir__ listing)
from kineticstoolkit import dev  # noqa

unstable_listing.append("dev")

try:
    from kineticstoolkit import inversedynamics  # noqa

    unstable_listing.append("inversedynamics")
except:
    pass

try:
    from kineticstoolkit import emg  # noqa

    unstable_listing.append("emg")
except:
    pass

try:
    from kineticstoolkit import anthropometrics  # noqa

    unstable_listing.append("anthropometrics")
except:
    pass

try:
    from kineticstoolkit import ext  # noqa

    unstable_listing.append("ext")
    from kineticstoolkit.ext import _import_extensions as import_extensions

    unstable_listing.append("import_extensions")
except:
    pass


from kineticstoolkit import config  # noqa

listing.append("config")


# Check if a serious warning has been issued on this version.
try:
    from requests_cache import CachedSession  # noqa
    from datetime import timedelta  # noqa
    import json  # noqa
    import warnings  # noqa

    session = CachedSession(
        "kineticstoolkit",
        backend="filesystem",
        use_temp=True,
        expire_after=timedelta(hours=1),
    )
    res = session.get(
        "https://kineticstoolkit.uqam.ca/api/import_check.php",
        params={"version": config.version},
    )
    contents = json.loads(res.content)
    if res.ok and "warning" in contents:
        warnings.warn(contents["warning"])
except Exception:
    pass


def __dir__():
    if config.dev_enabled:
        return listing + unstable_listing
    else:
        return listing


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
