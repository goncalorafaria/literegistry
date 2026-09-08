"""Repository-wide pytest bootstrap for sibling src-layout packages."""

from importlib import import_module


# Import these before importlib-mode collection creates namespace placeholders
# from their same-named project directories.
for _package_name in (
    "literegistry_podman_client",
    "literegistry_podman_beaker",
    "literegistry_base_deployment",
):
    import_module(_package_name)
