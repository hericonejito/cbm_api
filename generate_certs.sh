#!/bin/bash
# Generate self-signed SSL certificates for development/testing
# For production, use certificates from a trusted CA (e.g., Let's Encrypt)

CERT_DIR="certs"
DAYS_VALID=365

echo "Generating self-signed SSL certificates for development..."

# Create certs directory if it doesn't exist
mkdir -p "$CERT_DIR"

# Generate private key and self-signed certificate
openssl req -x509 -newkey rsa:4096 \
    -keyout "$CERT_DIR/key.pem" \
    -out "$CERT_DIR/cert.pem" \
    -days $DAYS_VALID \
    -nodes \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

if [ $? -eq 0 ]; then
    echo ""
    echo "SSL certificates generated successfully!"
    echo "  Private key: $CERT_DIR/key.pem"
    echo "  Certificate: $CERT_DIR/cert.pem"
    echo ""
    echo "Note: These are self-signed certificates for development only."
    echo "For production, use certificates from a trusted CA."
else
    echo "Failed to generate certificates. Make sure OpenSSL is installed."
    exit 1
fi
