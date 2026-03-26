import warnings

__SAMPLER_REGISTERS__ = dict()


'''Function to register samplers'''
def register_sampler(name: str):
    def decorator(cls):
        '''Check if cls is already registered with the same name'''
        if __SAMPLER_REGISTERS__.get(name, None) and __SAMPLER_REGISTERS__[name] is cls:
            warnings.warn(f"Sampler {name} is already registered!", UserWarning)
        else:
            __SAMPLER_REGISTERS__[name]=cls
            cls.name=name
            print(f"Sampler {name} is registered successfully.")
            return cls
    return decorator


def get_sampler(name: str, **kwargs):
    if __SAMPLER_REGISTERS__.get(name, None):
        return __SAMPLER_REGISTERS__[name](**kwargs)
    else:
        raise NameError(f"Sampler {name} is not registered!!!")