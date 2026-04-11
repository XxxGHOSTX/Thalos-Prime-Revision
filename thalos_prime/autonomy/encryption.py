"""
End-to-End Encryption Utilities
=================================

Provides symmetric (AES-256-GCM) and asymmetric (RSA-2048) encryption
for inter-node communication and knowledge object storage.

Signing:
  Messages are signed with RSA-PSS (SHA-256) so that recipients can
  verify the authenticity and integrity of every sync packet.
"""

from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class SymmetricEncryption:
    """
    AES-256-GCM symmetric encryption for bulk data transfer.

    A fresh 96-bit nonce is generated for every encryption call to
    guarantee semantic security.
    """

    KEY_LENGTH = 32  # 256 bits

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key or os.urandom(self.KEY_LENGTH)
        self._aesgcm = AESGCM(self._key)

    @property
    def key(self) -> bytes:
        return self._key

    def encrypt(self, plaintext: bytes, aad: bytes = b"") -> bytes:
        """
        Encrypt plaintext.

        Returns ``nonce || ciphertext`` (nonce prepended for convenience).
        """
        nonce = os.urandom(12)  # 96-bit nonce recommended for GCM
        ct = self._aesgcm.encrypt(nonce, plaintext, aad or None)
        return nonce + ct

    def decrypt(self, data: bytes, aad: bytes = b"") -> bytes:
        """
        Decrypt data produced by :meth:`encrypt`.

        Expects ``nonce || ciphertext`` format.
        """
        nonce, ct = data[:12], data[12:]
        return self._aesgcm.decrypt(nonce, ct, aad or None)

    def encrypt_str(self, plaintext: str, aad: bytes = b"") -> str:
        """Encrypt a string and return base-64 encoded ciphertext."""
        raw = self.encrypt(plaintext.encode(), aad)
        return base64.b64encode(raw).decode()

    def decrypt_str(self, b64_data: str, aad: bytes = b"") -> str:
        """Decrypt a base-64 encoded ciphertext and return a plain string."""
        raw = base64.b64decode(b64_data)
        return self.decrypt(raw, aad).decode()


class AsymmetricEncryption:
    """
    RSA-2048 OAEP asymmetric encryption for key exchange and node identity.
    """

    KEY_SIZE = 2048

    def __init__(
        self,
        private_key: rsa.RSAPrivateKey | None = None,
    ) -> None:
        if private_key is not None:
            self._private_key = private_key
        else:
            self._private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=self.KEY_SIZE
            )
        self._public_key = self._private_key.public_key()

    # ------------------------------------------------------------------
    # Key serialisation
    # ------------------------------------------------------------------

    def public_key_pem(self) -> str:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

    def private_key_pem(self, password: bytes | None = None) -> str:
        encryption = (
            serialization.BestAvailableEncryption(password)
            if password
            else serialization.NoEncryption()
        )
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=encryption,
        ).decode()

    @classmethod
    def from_private_pem(cls, pem: str, password: bytes | None = None) -> "AsymmetricEncryption":
        private_key = serialization.load_pem_private_key(pem.encode(), password=password)
        assert isinstance(private_key, rsa.RSAPrivateKey)
        return cls(private_key=private_key)

    # ------------------------------------------------------------------
    # Encrypt / decrypt
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: bytes) -> bytes:
        return self._public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def decrypt(self, ciphertext: bytes) -> bytes:
        return self._private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    @staticmethod
    def encrypt_with_public_pem(plaintext: bytes, public_pem: str) -> bytes:
        """Encrypt using an externally supplied public key (PEM string)."""
        pub_key = serialization.load_pem_public_key(public_pem.encode())
        return pub_key.encrypt(  # type: ignore[union-attr]
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    # ------------------------------------------------------------------
    # Digital signatures (RSA-PSS with SHA-256)
    # ------------------------------------------------------------------

    def sign(self, message: bytes) -> bytes:
        """
        Sign *message* with this node's RSA private key using PSS padding.

        Returns raw signature bytes (not base-64 encoded).
        """
        return self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

    def verify(self, message: bytes, signature: bytes) -> bool:
        """
        Verify *signature* over *message* using this node's public key.

        Returns True on success, False if the signature is invalid.
        """
        try:
            self._public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except InvalidSignature:
            return False

    @staticmethod
    def verify_with_public_pem(message: bytes, signature: bytes, public_pem: str) -> bool:
        """
        Verify *signature* over *message* using a PEM-encoded public key.

        Returns True on success, False on any failure.
        """
        try:
            pub_key = serialization.load_pem_public_key(public_pem.encode())
            pub_key.verify(  # type: ignore[union-attr]
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except (InvalidSignature, Exception):
            return False
