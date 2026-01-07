import os
import binascii
import random
from hmac import compare_digest

# Use the secure system generator from the random module
_sysrand = random.SystemRandom()

def randbits(k):
    """Generate an integer with k random bits."""
    return _sysrand.getrandbits(k)

def choice(seq):
    """Choose a random element from a non-empty sequence."""
    return _sysrand.choice(seq)

def randbelow(n):
    """Generate a random integer in the range [0, n)."""
    return _sysrand.randrange(n)

def token_bytes(nbytes=None):
    if nbytes is None:
        nbytes = 32
    return os.urandom(nbytes)

def token_hex(nbytes=None):
    return binascii.hexlify(token_bytes(nbytes)).decode('ascii')

def token_urlsafe(nbytes=None):
    import base64
    return base64.urlsafe_b64encode(token_bytes(nbytes)).rstrip(b'=').decode('ascii')