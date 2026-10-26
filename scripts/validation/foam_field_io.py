"""Shared OpenFOAM field-file parsers for the validation scripts.

Parses the ASCII `internalField` block of a native OpenFOAM field file
into a numpy array. Kept free of repository-relative assumptions: all
callers pass absolute paths.
"""
import numpy as np


def parse_foam_vec(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    idx = content.find('internalField')
    if idx == -1:
        return None
    chunk = content[idx:]
    open_paren = chunk.find('(')
    if open_paren == -1:
        return None
    close_paren = chunk.find('\n)\n;', open_paren)
    if close_paren == -1:
        close_paren = chunk.find('\n);', open_paren)
    if close_paren == -1:
        close_paren = chunk.rfind(')')
    if close_paren == -1:
        return None
    data_str = chunk[open_paren + 1:close_paren].strip()
    vectors = []
    for line in data_str.split('\n'):
        line = line.strip().strip('()').strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 3:
            try:
                vectors.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    return np.array(vectors)


def parse_foam_scalar(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    chunk = content[content.find('internalField'):]
    data_str = chunk[chunk.find('(') + 1:chunk.find(')', chunk.find('('))].strip()
    return np.array([float(x) for x in data_str.split()])
