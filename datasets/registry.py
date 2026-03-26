import warnings

__DATASET_REGISTERS__ = dict()


'''Function to register datasets'''
def register_dataset(name: str):
    def decorator(cls):
        '''Check if cls is already registered with the same name'''
        if __DATASET_REGISTERS__.get(name, None) and __DATASET_REGISTERS__[name] is cls:
            warnings.warn(f"Dataset {name} is already registered!", UserWarning)
        else:
            __DATASET_REGISTERS__[name]=cls
            cls.name=name
            print(f"Dataset {name} is registered successfully.")
            return cls
    return decorator


def get_dataset(name: str, **kwargs):
    if __DATASET_REGISTERS__.get(name, None):
        return __DATASET_REGISTERS__[name](**kwargs)
    else:
        raise NameError(f"Dataset {name} is not registered!!!")