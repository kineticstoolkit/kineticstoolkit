#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2024 Félix Chénier

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
Typing module to typecheck everything in realtime using beartype.
"""
from __future__ import annotations

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2024 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

from typing import NewType, TYPE_CHECKING
from numpy.typing import ArrayLike as npt_ArrayLike
from numbers import Integral, Real, Complex
from beartype import (
    beartype,
    BeartypeConf,
    BeartypeViolationVerbosity,
    BeartypeHintOverrides,
)

typecheck = beartype(
    conf=BeartypeConf(
        hint_overrides=BeartypeHintOverrides(
            {int: Integral, float: Real, complex: Complex}
        ),
        violation_param_type=TypeError,
        violation_return_type=TypeError,
        violation_verbosity=BeartypeViolationVerbosity.MINIMAL,
    )
)

# Define custom types so that beartype, sphinx,
# mypy and the user are all happy
if TYPE_CHECKING:  # mypy is running
    ArrayLike = npt_ArrayLike
else:  # runtime
    ArrayLike = NewType("ArrayLike", npt_ArrayLike)  # mypy cries
