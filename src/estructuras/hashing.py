"""Funciones hash deterministas (independientes de PYTHONHASHSEED) para las estructuras probabilísticas."""
import hashlib


def hash64(clave) -> int:
    return int.from_bytes(hashlib.blake2b(str(clave).encode(), digest_size=8).digest(), "little")


def hash128(clave):
    d = hashlib.blake2b(str(clave).encode(), digest_size=16).digest()
    return int.from_bytes(d[:8], "little"), int.from_bytes(d[8:], "little") | 1
