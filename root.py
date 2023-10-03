import os


def root(rel_path: str) -> str:
    self_path = os.path.abspath(__file__)
    root_path = os.path.dirname(self_path)
    return os.path.abspath(os.path.join(root_path, rel_path))
