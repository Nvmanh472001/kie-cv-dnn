import yaml

def get_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as buf:
        config = yaml.safe_load(buf)

    return config