"""Install the inference package into the selected Python environment."""
from pathlib import Path
from setuptools import setup

ROOT = Path(__file__).parent
requirements = [line.strip() for line in (ROOT/'requirements.txt').read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#')]
setup(name='lynnreal', version='0.1.0', description='LynnReal video inference',
      python_requires='>=3.12,<3.14', install_requires=requirements,
      packages=['lynnreal', 'lynnreal.model'], package_dir={'lynnreal': '.'},
      options={'build': {'build_base': 'build/package'}}, include_package_data=False)
