"This file is here to avoid relative imports issues. It is not a competitor implementation, just specific utils used for some competitors."

from inspect import signature

def _resolve_callable_param(param, context_kwargs=None):

    """Resolve a parameter that can be either a value or a callable with arguments.
    
    This function handles parameters that can be specified in two ways:
    1. As a direct value (returned as-is)
    2. As a tuple of (callable, dict) where the callable is invoked with the 
       dictionary unpacked as keyword arguments, optionally combined with
       context kwargs
    
    Parameters
    ----------
    param : object or tuple
        The parameter to resolve. Can be:
        - Any object: returned directly without modification
        - A tuple of (callable, dict): the callable is invoked with the dict
          unpacked as keyword arguments (**dict)
    
    context_kwargs : dict, default=None
        Additional keyword arguments to pass to the callable function.
        These are combined with the kwargs from the tuple (if any).
        If both context_kwargs and tuple kwargs contain the same key,
        the tuple kwargs take precedence.
    
    Returns
    -------
    object
        The resolved parameter value. If param was a tuple, returns the result
        of calling the callable with the provided arguments combined with
        context kwargs. Otherwise, returns param unchanged.
    
    Raises
    ------
    ValueError
        If param is a tuple but doesn't have exactly 2 elements, or if the
        first element of the tuple is not callable.
    
    Examples
    --------
    >>> # Direct value - returned as-is
    >>> _resolve_callable_param(10)
    10
    
    >>> # Callable with arguments only
    >>> def multiply(x, factor=2):
    ...     return x * factor
    >>> _resolve_callable_param((multiply, {"x": 5, "factor": 3}))
    15
    
    >>> # Callable with context kwargs
    >>> def add_and_multiply(x, y, factor=2):
    ...     return (x + y) * factor
    >>> context = {"y": 10}
    >>> _resolve_callable_param((add_and_multiply, {"x": 5, "factor": 3}), context)
    45
    """
    if isinstance(param, tuple):
        if len(param) != 2:
            raise ValueError(
                "Parameter must be a tuple of (callable, params) or a direct value."
            )
        if not callable(param[0]):
            raise ValueError("First element of parameter tuple must be callable.")
        
        param_func, args_dict = param
        
        if context_kwargs is None:
            context_kwargs = {}
        
        # Merge kwargs: context_kwargs first, then args_dict to allow override
        final_kwargs = {**context_kwargs, **args_dict}
        # Filter kwargs to only include parameters the function accepts
        sig = signature(param_func)
        param_names = set(sig.parameters.keys())
        
        filtered_kwargs = {k: v for k, v in final_kwargs.items() if k in param_names}
        return param_func(**filtered_kwargs)
    else:
        return param