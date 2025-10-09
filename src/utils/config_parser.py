import yaml

def load_yaml(path):
    """خواندن فایل YAML و بازگرداندن دیکشنری پایتون"""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data
