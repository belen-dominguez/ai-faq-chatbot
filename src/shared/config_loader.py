import yaml

_config_cache = None

def load_config():
    global _config_cache

    if _config_cache is None:
        with open("config.yaml", "r") as f:
            _config_cache = yaml.safe_load(f)

    return _config_cache