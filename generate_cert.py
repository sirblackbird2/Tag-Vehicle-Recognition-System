from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
import datetime
import ipaddress

# Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

# Generate certificate
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COMMON_NAME, "192.168.1.8"),
])

# Use timezone-aware datetime
now = datetime.datetime.now(datetime.timezone.utc)

cert = x509.CertificateBuilder().subject_name(
    subject
).issuer_name(
    issuer
).public_key(
    private_key.public_key()
).serial_number(
    x509.random_serial_number()
).not_valid_before(
    now
).not_valid_after(
    now + datetime.timedelta(days=365)
).add_extension(
    x509.SubjectAlternativeName([
        x509.IPAddress(ipaddress.IPv4Address("192.168.1.8")),
        x509.DNSName("localhost"),
        x509.DNSName("127.0.0.1"),
    ]),
    critical=False,
).sign(private_key, hashes.SHA256())

# Write certificate
with open("cert.pem", "wb") as f:
    f.write(cert.public_bytes(Encoding.PEM))

# Write private key
with open("key.pem", "wb") as f:
    f.write(private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption()
    ))

print("SSL certificate and key generated successfully!")
print("cert.pem and key.pem saved to your project folder.")