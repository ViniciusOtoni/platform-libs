"""Resolução das dependências que o job precisa instalar no serverless."""

from importlib.metadata import PackageNotFoundError, version

PACKAGE = "mlplatform"
RELEASE_URL = (
    "https://github.com/ViniciusOtoni/platform-libs/releases/download/"
    "{package}-v{version}/{package}-{version}-py3-none-any.whl"
)

# Relativo ao ARQUIVO YAML gerado, que fica em `resources/` — não à raiz do
# bundle. Um `./dist/*.whl` aqui resolveria para `resources/dist/` e o deploy
# falharia com "no files match pattern". Foi exatamente assim que o deploy do
# exemplo-domain quebrou; a constante existe para que a lição fique num lugar só.
DOMAIN_WHEEL_GLOB = "../dist/*.whl"


def framework_version() -> str:
    """A versão realmente instalada.

    Deriva de `importlib.metadata`, e não de uma string escrita à mão em algum
    YAML: a versão instalada é a verdade sobre o que vai rodar, e uma cópia
    manual dessincroniza sem avisar.
    """
    try:
        return version(PACKAGE)
    except PackageNotFoundError as exc:  # pragma: no cover - só fora de instalação
        raise RuntimeError(
            f"'{PACKAGE}' não está instalado — a versão do framework é derivada do "
            "pacote instalado, não configurada à mão"
        ) from exc


def framework_wheel_url(pkg_version: str | None = None) -> str:
    return RELEASE_URL.format(package=PACKAGE, version=pkg_version or framework_version())


def default_dependencies(extra: list[str] | None = None) -> list[str]:
    """Wheel do domínio (buildado pelo `artifacts:` do bundle) + wheel do
    framework (asset do Release), mais o que o componente precisar."""
    return [DOMAIN_WHEEL_GLOB, framework_wheel_url(), *(extra or [])]
