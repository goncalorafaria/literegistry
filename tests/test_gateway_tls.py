"""Tests for the gateway's optional TLS mirror listener."""

import ssl

import pytest

from literegistry.gateway import advertised_gateway_url, ensure_tls_credentials, main


def test_ensure_tls_credentials_generates_loadable_self_signed(tmp_path):
    certfile, keyfile = ensure_tls_credentials(None, None, "mirror.example.internal")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # Raises if the generated pair is malformed or mismatched.
    context.load_cert_chain(certfile, keyfile)


def test_ensure_tls_credentials_passes_through_provided_pair():
    certfile, keyfile = ensure_tls_credentials(None, None, "host-a")
    again = ensure_tls_credentials(certfile, keyfile, "ignored-host")
    assert again == (certfile, keyfile)


def test_ensure_tls_credentials_requires_both_or_neither():
    with pytest.raises(ValueError):
        ensure_tls_credentials("/tmp/only-cert.pem", None, "host")
    with pytest.raises(ValueError):
        ensure_tls_credentials(None, "/tmp/only-key.pem", "host")


def test_ensure_tls_credentials_missing_files_error(tmp_path):
    missing = tmp_path / "nope.pem"
    with pytest.raises(FileNotFoundError):
        ensure_tls_credentials(str(missing), str(missing), "host")


def test_advertised_gateway_url_scheme():
    assert advertised_gateway_url(8080, "node-1").startswith("http://node-1:")
    assert (
        advertised_gateway_url(8443, "node-1", scheme="https")
        == "https://node-1:8443"
    )


def test_main_rejects_tls_port_equal_to_port():
    with pytest.raises(ValueError, match="tls_port must differ"):
        main(port=8080, tls_port=8080)


def test_main_rejects_tls_files_without_tls_port():
    with pytest.raises(ValueError, match="require tls_port"):
        main(port=8080, tls_certfile="/tmp/cert.pem", tls_keyfile="/tmp/key.pem")


def test_main_rejects_tls_in_reload_mode():
    with pytest.raises(ValueError, match="reload"):
        main(port=8080, tls_port=8443, reload=True)
