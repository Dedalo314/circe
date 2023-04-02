import importlib

def import_class(model_class: str):
    return getattr(
        importlib.import_module(
            ".".join(model_class.split(".")[:-1])
        ),
        model_class.split(".")[-1]
    )
