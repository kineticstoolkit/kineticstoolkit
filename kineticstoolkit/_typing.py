#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2023 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typing module for Kinetics Toolkit."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2023 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

# Define custom types to use instead of the builtins, so that beartype, sphinx,
# mypy and the user are all happy (expected ArrayLike is not thrown to the user
# as Union[dsbnjkvlsdbflvjbadsljvbdsljv] but as
# kineticstoolkit._typing.ArrayLike instead.
from numpy.typing import ArrayLike as npt_ArrayLike
from typing import NewType, TYPE_CHECKING

if TYPE_CHECKING:  # mypy is running
    ArrayLike = npt_ArrayLike
else:  # runtime
    ArrayLike = NewType("ArrayLike", npt_ArrayLike)
