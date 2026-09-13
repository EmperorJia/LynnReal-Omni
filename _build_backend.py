"""Local pip bootstrap from Conda base; ordinary package builds use setuptools."""
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

BUILD_REQUIRES = ['setuptools>=77', 'wheel']
DIST_INFO = 'lynnreal_bootstrap-0.1.0.dist-info'
METADATA = ('Metadata-Version: 2.1\nName: lynnreal-bootstrap\nVersion: 0.1.0\n'
            'Summary: LynnReal Conda environment installer\nRequires-Python: >=3.9\n')
WHEEL = 'Wheel-Version: 1.0\nGenerator: lynnreal\nRoot-Is-Purelib: true\nTag: py3-none-any\n'


def bootstrap():
    return os.environ.get('CONDA_DEFAULT_ENV') == 'base' and os.environ.get('LYNNREAL_INSTALL_CURRENT') != '1'


def standard_build(hook, *args):
    from setuptools import build_meta
    return getattr(build_meta, hook)(*args)


def get_requires_for_build_wheel(config_settings=None):
    return [] if bootstrap() else BUILD_REQUIRES


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    if not bootstrap():
        return standard_build('prepare_metadata_for_build_wheel', metadata_directory, config_settings)
    folder = Path(metadata_directory)/DIST_INFO
    folder.mkdir(exist_ok=True)
    (folder/'METADATA').write_text(METADATA)
    (folder/'WHEEL').write_text(WHEEL)
    return DIST_INFO


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    if not bootstrap():
        return standard_build('build_wheel', wheel_directory, config_settings, metadata_directory)
    # Never install during metadata resolution, and never leak outer pip hooks.
    env = dict(os.environ, LYNNREAL_INSTALL_CURRENT='1', PYTHONNOUSERSITE='1')
    for key in list(env):
        if key == 'PYTHONPATH' or key.startswith(('_PYPROJECT_HOOKS_', 'PEP517_')):
            env.pop(key)
    subprocess.run([sys.executable, str(Path(__file__).parent/'script/setup_env.py'), '--pypi-only'],
                   env=env, check=True)
    # pip hides successful backend output; a real interactive shell still needs
    # to see which attention backend was installed. Noninteractive logs keep it.
    report = Path(__file__).parent/'output/setup/attention.json'
    if report.is_file():
        backend = json.loads(report.read_text()).get('selected')
        notice = f'\nLynnReal environment ready: conda activate lynnreal (attention={backend}).\n'
        if backend != '_flash_3':
            notice += 'WARNING: FA3 is not active; verify compatibility, speed and sampled videos.\n'
        print(notice, file=sys.stderr, flush=True)
        try:
            with open('/dev/tty', 'w') as terminal:
                terminal.write(notice)
        except OSError:
            pass
    files = {'METADATA': METADATA.encode(), 'WHEEL': WHEEL.encode()}
    if metadata_directory:
        files = {str(p.relative_to(metadata_directory)): p.read_bytes()
                 for p in Path(metadata_directory).rglob('*') if p.is_file()}
    name = 'lynnreal_bootstrap-0.1.0-py3-none-any.whl'
    records = io.StringIO()
    writer = csv.writer(records, lineterminator='\n')
    with zipfile.ZipFile(Path(wheel_directory)/name, 'w', zipfile.ZIP_DEFLATED) as wheel:
        for relative, content in files.items():
            path = DIST_INFO+'/'+relative
            wheel.writestr(path, content)
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode()
            writer.writerow([path, 'sha256='+digest, len(content)])
        writer.writerow([DIST_INFO+'/RECORD', '', ''])
        wheel.writestr(DIST_INFO+'/RECORD', records.getvalue())
    return name


def get_requires_for_build_sdist(config_settings=None):
    return BUILD_REQUIRES


def build_sdist(sdist_directory, config_settings=None):
    return standard_build('build_sdist', sdist_directory, config_settings)


# Editable installs always target the selected environment.
get_requires_for_build_editable = get_requires_for_build_sdist

def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return standard_build('prepare_metadata_for_build_editable', metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return standard_build('build_editable', wheel_directory, config_settings, metadata_directory)
