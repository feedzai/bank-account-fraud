# coding=utf-8
#
# The copyright of this file belongs to Feedzai. The file cannot be
# reproduced in whole or in part, stored in a retrieval system,
# transmitted in any form, or by any means electronic, mechanical,
# photocopying, or otherwise, without the prior permission of the owner.
#
# (c) 2022 Feedzai, Strictly Confidential

import logging

from copy import deepcopy
from optuna.samplers import RandomSampler
from optuna.trial import FixedTrial, BaseTrial
from optuna.distributions import BaseDistribution
from optuna import distributions
from typing import Iterable, Optional, Any


class RandomValueTrial(FixedTrial):
    """A Trial following optuna's API.
    Does not depend on an optuna.Study and can be used as a standalone object.
    """

    def __init__(
            self,
            number: int = 0,
            seed: Optional[int] = None,
            sampler: Optional[RandomSampler] = None
        ):
        assert not (seed and sampler), \
            f"Must provide at most one of (seed={seed}, sampler={sampler})"
        super().__init__(
            params=None, 
            number=number,
        )
        self.seed = seed
        self.sampler = sampler or RandomSampler(self.seed)

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        """Given a parameter's name and its distribution should return the
        suggested (sampled) value.
        (Template method from the Trial super-class).

        Parameters
        ----------
        name : str
            The parameter's name (so we don't suggest different values when the
            same parameter is sampled more than once).
        distribution : BaseDistribution
            The distribution to draw from.

        Returns
        -------
        The sampled value.
        """

        if name in self._distributions:
            # No need to sample if already suggested.
            distributions.check_distribution_compatibility(
                self._distributions[name], distribution,
            )
            param_value = self._suggested_params[name]

        else:
            if self._is_fixed_param(name, distribution):
                param_value = self.system_attrs["fixed_params"][name]
            elif distribution.single():
                param_value = distributions._get_single_value(distribution)
            else:
                param_value = self.sampler.sample_independent(
                    study=None, trial=self,     # type: ignore
                    param_name=name, param_distribution=distribution,
                )

        self._suggested_params[name] = param_value
        self._distributions[name] = distribution

        return self._suggested_params[name]

    def _is_fixed_param(self, name: str, distribution: BaseDistribution) -> bool:
        """Checks if the given parameter name corresponds to a fixed parameter.
        This implementation does not depend on an optuna.study.

        Parameters
        ----------
        name : str
        distribution : BaseDistribution

        Returns
        -------
        Whether the parameter is a fixed parameter.
        """
        system_attrs = self._system_attrs
        if "fixed_params" not in system_attrs:
            return False

        if name not in system_attrs["fixed_params"]:
            return False

        param_value = system_attrs["fixed_params"][name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)

        contained = distribution._contains(param_value_in_internal_repr)
        if not contained:
            warnings.warn(
                f"Fixed parameter '{name}' with value {param_value} is out of range "
                f"for distribution {distribution}."
            )
        return contained
    
    
def suggest_callable_hyperparams(
        trial: BaseTrial,
        hyperparameter_space: dict,
        param_prefix: str = "learner",
    ) -> dict:
    """Suggests the top-level hyperparameters for a class instantiation, or
    for parameterizing any other callable.

    This includes the classpath/importpath, and the conditional
    hyperparameters to use as kwargs for the class.

    Parameters
    ----------
    trial : BaseTrial
        The trial object to interact with the hyperparameter sampler.
    hyperparameter_space : dict
        A dictionary representing a hyperparameter space.
    param_prefix : str
        The prefix to attach to all parameters' names in order to uniquely
        identify them.

    Returns
    -------
    Dict
        The suggested callable + its suggested key-word arguments.
    """

    # Suggest callable type (top-level callable choice)
    callable_type = trial.suggest_categorical(
        f"{param_prefix}_type", list(hyperparameter_space.keys()),
    )

    hyperparam_subspace = hyperparameter_space[callable_type]
    callable_classpath = hyperparam_subspace['classpath']

    # Suggest callable kwargs (on conditional sub-space)
    callable_kwargs = suggest_hyperparams(
        trial, hyperparam_subspace.get('kwargs', {}),
        param_prefix=f"{param_prefix}_{callable_type}"
    )

    hyperparams = {'classpath': callable_classpath, **callable_kwargs}
    return hyperparams


def suggest_hyperparams(
        trial: BaseTrial,
        hyperparameter_space: dict,
        param_prefix: str = '',
    ) -> dict:
    """Uses the provided hyperparameter space to suggest specific
    configurations using the given Trial object.

    Parameters
    ----------
    trial : BaseTrial
        The trial object to interact with the hyperparameter sampler.
    hyperparameter_space : dict
        A dictionary representing a hyperparameter space.
    param_prefix : str
        The prefix to attach to all parameters' names in order to uniquely
        identify them.

    Returns
    -------
    An instantiation of the given hyperparameter space.
    """
    if not hyperparameter_space:
        logging.warning("Got empty hyperparameter space. Returning '{}'.")
        return dict()
    assert isinstance(hyperparameter_space, dict)

    params = dict()
    for key, value in hyperparameter_space.items():
        param_id = f'{param_prefix}_{key}'

        # Fixed value
        if isinstance(value, (str, int, float)) or not isinstance(value, Iterable):
            params[key] = value

        # Categorical parameter
        elif isinstance(value, (list, tuple)):
            # Categorical parameters must be of one of the following types
            valid_categ_types = (bool, int, float, str)

            # If not, map values to strings for optuna, then remap them back
            if not all(el is None or isinstance(el, valid_categ_types) for el in value):
                categ_map = {str(el): el for el in value}
                suggested_categ_encoded = trial.suggest_categorical(
                    param_id, sorted(categ_map.keys()),
                )
                params[key] = categ_map[str(suggested_categ_encoded)]

            # If categorical values have valid types, use them directly
            else:
                params[key] = trial.suggest_categorical(
                    param_id, value,
                )

        # Numerical parameter
        elif isinstance(value, dict) and 'type' in value:
            params[key] = suggest_numerical_hyperparam(
                trial, value, param_id,
            )

        # Nested parameter
        elif isinstance(value, dict):
            params[key] = suggest_hyperparams(
                trial, value, f"{key}_{param_prefix}",
            )

        else:
            raise ValueError(
                f"Invalid hyperparameter configuration: {key}={value} "
                f"({type(key)} -> {type(value)})."
            )

    return params


def suggest_numerical_hyperparam(
        trial: BaseTrial,
        config: dict,
        param_id: str,
    ) -> float:
    """Helper function to suggest a numerical hyperparameter.

    Parameters
    ----------
    trial : BaseTrial
        The trial object to interact with the hyperparameter sampler.
    config : dict
        The distribution to sample the parameter from.
    param_id : str
        The parameter's name.

    Returns
    -------
    The sampled value.
    """
    # Parameter's type
    config = deepcopy(config)
    param_type = config.pop('type')
    valid_param_types = ('int', 'float', 'uniform', 'discrete_uniform', 'loguniform')
    assert param_type in valid_param_types, \
        f'Invalid parameter type {param_type}, choose one of {valid_param_types}'

    # Parameter's range
    assert 'range' in config and isinstance(config['range'], (list, tuple)), \
        'Must provide a range when searching a numerical parameter'
    low, high = config.pop('range')

    # Remaining suggestion parameters are used as is (e.g., log scale)
    suggest_param_func = getattr(trial, f'suggest_{param_type}')
    return suggest_param_func(
        param_id, low, high, **config,
    )


