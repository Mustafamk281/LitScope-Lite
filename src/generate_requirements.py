
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

packages = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "matplotlib",
    "seaborn",
    "torch",
    "transformers",
    "sentence-transformers",
    "huggingface-hub",
    "tokenizers",
    "safetensors",
    "streamlit",
    "nbformat",
    "nbclient",
    "ipykernel",
    "pytest",
]

optional_packages = [
    "sentencepiece",
    "jupyter",
    "notebook",
]

requirements = [
    "# LitScope-Lite project dependencies",
    "# Tested environment: Python 3.10.11",
    "",
]

versions = {}

for package in packages:
    installed = version(package)

    if package == "torch":
        installed = installed.split("+")[0]

    versions[package] = installed
    requirements.append(f"{package}=={installed}")

for package in optional_packages:
    try:
        installed = version(package)
        versions[package] = installed
        requirements.append(f"{package}=={installed}")

    except PackageNotFoundError:
        print(f"Optional package not installed: {package}")

(ROOT / "requirements.txt").write_text(
    "\n".join(requirements) + "\n",
    encoding="utf-8",
)

phase1_packages = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "matplotlib",
    "seaborn",
    "nbformat",
    "nbclient",
    "ipykernel",
]

phase1 = [
    "# Phase 1 data-science notebook dependencies",
    "",
]

for package in phase1_packages:
    phase1.append(f"{package}=={versions[package]}")

phase1.append(
    f"jupyter=={versions['jupyter']}"
    if "jupyter" in versions
    else "jupyter>=1.0"
)

(ROOT / "requirements-phase1.txt").write_text(
    "\n".join(phase1) + "\n",
    encoding="utf-8",
)

print("\nUpdated requirements.txt:")
print((ROOT / "requirements.txt").read_text())

print("\nUpdated requirements-phase1.txt:")
print((ROOT / "requirements-phase1.txt").read_text())

print("Dependency files generated successfully.")
print("Installed packages remain unchanged.")
