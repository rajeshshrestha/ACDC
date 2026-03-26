import warnings

__OPERATOR_REGISTERS__ = dict()


'''Function to register operators'''
def register_operator(name: str):
    def decorator(cls):
        '''Check if cls is already registered with the same name'''
        if __OPERATOR_REGISTERS__.get(name, None) and __OPERATOR_REGISTERS__[name] is cls:
            warnings.warn(f"Operator {name} is already registered!", UserWarning)
        else:
            __OPERATOR_REGISTERS__[name]=cls
            cls.name=name
            print(f"Operator {name} is registered successfully.")
            return cls
    return decorator


def get_operator(name: str, **kwargs):
    if __OPERATOR_REGISTERS__.get(name, None):
        return __OPERATOR_REGISTERS__[name](**kwargs)
    else:
        raise NameError(f"Operator {name} is not registered!!!")