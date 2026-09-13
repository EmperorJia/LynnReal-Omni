"""Create conda env lynnreal, install this checkout, and probe optional attention."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import platform
import shlex
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FA3 = (
    'git+https://github.com/Dao-AILab/flash-attention.git'
    '@203b9b3dba39d5d08dffb49c09aa622984dff07d'
    '#subdirectory=hopper'
)
FA2 = 'flash-attn==2.7.3'
SUPPORTED_PYTHON = {(3, 12), (3, 13)}


def run(argv, **kwargs):
    print('+ ' + shlex.join(map(str, argv)), flush=True)
    return subprocess.run(list(map(str, argv)), check=True, **kwargs)


def pip_environment(pypi_only=False):
    """Apply an optional index override to child processes, never pip config files."""
    env = dict(os.environ)
    if pypi_only:
        for key in ('PIP_EXTRA_INDEX_URL', 'PIP_NO_INDEX', 'PIP_FIND_LINKS', 'PIP_TRUSTED_HOST'):
            env.pop(key, None)
        env.update(PIP_CONFIG_FILE=os.devnull, PIP_INDEX_URL='https://pypi.org/simple')
    return env


def require_python(version, where):
    if tuple(version[:2]) not in SUPPORTED_PYTHON:
        raise ValueError(f'{where} uses Python {".".join(map(str, version))}; '
                         'LynnReal requires Python 3.12 or 3.13. '
                         'Create a separate environment with '
                         'python script/setup_env.py --env-name lynnreal-py312. '
                         'The existing environment has not been modified.')


def dependencies_ready(python, env):
    """Reuse a complete pinned environment without re-fetching immutable source URLs."""
    code = r"""
import importlib.metadata as md, json, pathlib, subprocess, sys
from packaging.requirements import Requirement
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if not line.strip() or line.lstrip().startswith('#'): continue
    req = Requirement(line)
    if req.marker and not req.marker.evaluate(): continue
    dist = md.distribution(req.name)
    if dist.version not in req.specifier: raise SystemExit(1)
    if req.url and json.loads(dist.read_text('direct_url.json') or '{}').get('url') != req.url:
        raise SystemExit(1)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True, stdout=subprocess.DEVNULL)
"""
    return subprocess.run(python+['-c', code, str(ROOT/'requirements.txt')], env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def prepare_cuda(python, env):
    """Keep package versions fixed; Blackwell needs the CUDA 12.8 wheel variant."""
    probe = "import torch; print(int(torch.cuda.is_available() and torch.cuda.get_device_capability()[0]>=10 and tuple(map(int,(torch.version.cuda or '0.0').split('.')[:2]))<(12,8)))"
    result = subprocess.run(python+['-c', probe], env=env, capture_output=True, text=True, check=True)
    if result.stdout.strip() == '1':
        print('Blackwell detected: installing the pinned CUDA 12.8 PyTorch build.', flush=True)
        index_env = dict(env, PIP_CONFIG_FILE=os.devnull, PIP_INDEX_URL='https://download.pytorch.org/whl/cu128')
        for key in ('PIP_EXTRA_INDEX_URL', 'PIP_NO_INDEX', 'PIP_FIND_LINKS'):
            index_env.pop(key, None)
        run(python+['-m','pip','install','torch==2.7.1+cu128',
                    'torchvision==0.22.1+cu128','torchaudio==2.7.1+cu128'], env=index_env)


def install(python, args):
    env = pip_environment(args.pypi_only)
    env['LYNNREAL_INSTALL_CURRENT'] = '1'
    env.setdefault('TRITON_CACHE_DIR',str(ROOT/'output/.cache/triton'))
    env.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'output/.cache/inductor'))
    wheelhouse = os.environ.get('LYNNREAL_WHEELHOUSE')
    if wheelhouse:
        env['PIP_FIND_LINKS'] = str(Path(wheelhouse).expanduser().resolve(strict=True))
        env['PIP_NO_INDEX'] = '1'
    reuse = ['--no-deps', '--no-build-isolation'] if dependencies_ready(python, env) else []
    run(python+['-m', 'pip', 'install', *reuse, str(ROOT)], env=env)
    prepare_cuda(python, env)
    run(python+['-m', 'pip', 'check'], env=env)
    if not args.skip_attention:
        run(python+[str(Path(__file__).resolve()), '--attention-only', '--report-dir', str(args.report_dir), '--attention', args.attention], env=env)
        try:
            run(python+[str(ROOT/'script/precompile.py'), '--report', str(args.report_dir/'kernels.json')], env=env)
            run(python+[str(ROOT/'script/precompile_profiles.py')], env=env)
        except subprocess.CalledProcessError as error:
            if error.returncode == 2:  # Numerical mismatch must never be accepted as a fallback.
                raise
            print(f'WARNING: optional kernel precompilation failed ({error.returncode}); sampling will probe and compile on demand.', file=sys.stderr)


def bootstrap_conda(prefix):
    system, machine = platform.system(), platform.machine()
    if system not in {'Linux', 'Darwin'} or machine not in {'x86_64', 'aarch64', 'arm64'}:
        raise RuntimeError('Install Conda manually on this platform, then pass --conda')
    prefix = prefix.expanduser().resolve()
    executable = prefix/'bin/conda'
    if executable.is_file():return str(executable)
    if prefix.exists():raise FileExistsError('Existing non-Conda prefix: ' + str(prefix))
    version='25.3.1-0'
    name=f'Miniforge3-{version}-{system}-{machine}.sh'
    url=f'https://github.com/conda-forge/miniforge/releases/download/{version}/{name}'
    with tempfile.TemporaryDirectory() as directory:
        installer=Path(directory)/name
        urllib.request.urlretrieve(url,installer)
        expected=urllib.request.urlopen(url+'.sha256').read().decode().split()[0]
        if hashlib.sha256(installer.read_bytes()).hexdigest()!=expected:
            raise ValueError('Miniforge installer checksum mismatch')
        run(['bash',installer,'-b','-p',prefix])
    return str(executable)


def pinned_fa3_installed():
    import importlib.metadata as metadata
    try:
        dist = metadata.distribution('flash-attn-3')
        direct = json.loads(dist.read_text('direct_url.json') or '{}')
        commit = direct.get('vcs_info', {}).get('commit_id')
        if commit == '203b9b3dba39d5d08dffb49c09aa622984dff07d':
            return True
        audit = json.loads(dist.read_text('lynnreal_build.json') or '{}')
        library = dist.locate_file(audit.get('library', ''))
        return (audit.get('commit') == '203b9b3dba39d5d08dffb49c09aa622984dff07d'
                and library.is_file() and hashlib.sha256(library.read_bytes()).hexdigest() == audit.get('sha256'))
    except (metadata.PackageNotFoundError, OSError, ValueError):
        return False


def install_attention(pip_env=None, requested='auto'):
    import torch
    report = {'torch':torch.__version__, 'cuda':torch.version.cuda, 'attempts':[], 'requested':requested}
    if not torch.cuda.is_available():
        report.update(selected=None, reason='No visible CUDA GPU; re-run --attention-only on a GPU worker')
        return report
    capability = torch.cuda.get_device_capability()
    candidates = []
    cuda = tuple(map(int, (torch.version.cuda or '0.0').split('.')[:2]))
    if capability[0] == 9 and cuda >= (12, 3): candidates.append(('_flash_3', FA3))
    if capability[0] >= 8: candidates.append(('flash', FA2))
    candidates += [('_native_cudnn', None), ('_native_flash', None), ('native', None)]
    if requested != 'auto':
        candidates = [(name, package) for name, package in candidates if name == requested]
    report['capability'] = list(capability)
    for backend, package in candidates:
        # Probe in a fresh process: Diffusers caches package availability at import.
        probe = [sys.executable, '-c',
            'from model.attention import probe_backend; import json; '
            f'print(json.dumps(probe_backend({backend!r})))']
        trial = subprocess.run(probe, cwd=ROOT, capture_output=True, text=True)
        if package and (trial.returncode or (backend == '_flash_3' and not pinned_fa3_installed())):
            try:
                env = dict(pip_env if pip_env is not None else os.environ,
                           MAX_JOBS=os.environ.get('MAX_JOBS', '4'))
                if backend == '_flash_3':
                    env['FLASH_ATTENTION_FORCE_BUILD'] = 'TRUE'
                    # H3 inference uses dense FP16/BF16 forward attention with head dimensions <=128.
                    for feature in ('BACKWARD','SM80','FP8','HDIM192','HDIM256',
                                    'HDIMDIFF64','HDIMDIFF192','SPLIT','PAGEDKV',
                                    'APPENDKV','LOCAL','SOFTCAP','PACKGQA','VARLEN'):
                        env.setdefault('FLASH_ATTENTION_DISABLE_'+feature, 'TRUE')
                run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', '--no-deps', package], env=env)
                trial = subprocess.run(probe, cwd=ROOT, capture_output=True, text=True)
            except subprocess.CalledProcessError as error:
                report['attempts'].append({'backend':backend,'installed':False,'exit_code':error.returncode})
                continue
        report['attempts'].append({'backend':backend,'passed':trial.returncode==0,
                                   'output':trial.stdout, 'error':trial.stderr[-3000:]})
        if trial.returncode == 0:
            report['selected'] = backend
            return report
    # Current H3 uses BF16; legacy FA1 does not satisfy this interface. PyTorch's
    # packaged Flash SDPA is the compatible fallback, then ordinary native SDPA.
    report['selected'] = None
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--no-shell-init', action='store_true', help='do not add Conda initialization to bash/zsh startup files')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--current-env', action='store_true', help='install into the current Python instead of creating conda')
    mode.add_argument('--attention-only', action='store_true', help='probe/install kernels in the current environment')
    p.add_argument('--report-dir', type=Path, default=ROOT/'output/setup', help='attention probe report directory')
    p.add_argument('--attention', choices=('auto', '_flash_3', 'flash', '_native_cudnn', '_native_flash', 'native'),
                   default='auto', help='auto warns on non-FA3; an explicit backend must pass its probe')
    p.add_argument('--env-name', default='lynnreal', help='Conda environment to create or reuse (new environments use Python 3.12)')
    p.add_argument('--pypi-only', action='store_true', help='use only PyPI for pip, ignoring configured extra indexes for this invocation')
    p.add_argument('--skip-attention', action='store_true', help='install core dependencies without probing/building optional GPU kernels')
    p.add_argument('--conda-prefix', type=Path, default=Path.home()/'.local/share/miniforge3', help='install Miniforge here when Conda is absent')
    p.add_argument('--conda', default=os.environ.get('CONDA_EXE') or shutil.which('conda'))
    args = p.parse_args()
    if args.skip_attention and args.attention != 'auto':
        p.error('--skip-attention cannot satisfy an explicit attention requirement')
    if args.attention_only and args.skip_attention:
        p.error('--attention-only and --skip-attention cannot be combined')
    if args.current_env or args.attention_only:
        try:require_python(sys.version_info[:3], sys.executable)
        except ValueError as error:p.error(str(error))
    if args.attention_only:
        destination=args.report_dir; destination.mkdir(parents=True,exist_ok=True)
        # Optional-kernel pip invocations inherit the same explicit index policy.
        report=install_attention(pip_environment(args.pypi_only), args.attention)
        (destination/'attention.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        selected = report['selected']
        if args.attention != 'auto' and selected != args.attention:
            raise SystemExit(f'ERROR: required attention {args.attention} did not pass; see {destination}/attention.json')
        if selected != '_flash_3':
            print(f'WARNING: FA3 IS NOT ACTIVE (selected={selected}). '
                  'Backend changes can affect compatibility, speed and sampled video; '
                  'check the saved probe report and sampling results.', file=sys.stderr, flush=True)
        if selected is None and report.get('capability'):
            raise SystemExit('ERROR: no attention backend passed its numerical probe')
        return
    if args.current_env:
        install([sys.executable],args);return
    if not args.env_name or args.env_name.startswith('-') or any(c in args.env_name for c in '/\\ :'):
        p.error('--env-name must be a Conda environment name, not a path')
    if not args.conda:
        args.conda=bootstrap_conda(args.conda_prefix)
    info=json.loads(subprocess.check_output([args.conda,'env','list','--json'],text=True))
    matches=[x for x in info['envs'] if Path(x).name==args.env_name]
    if len(matches)>1:p.error(f'Multiple environments named {args.env_name}; choose a unique --env-name')
    prefix=[args.conda,'run','--no-capture-output']
    if matches:
        prefix+=['-p',matches[0]]
        version=json.loads(subprocess.check_output(prefix+['python','-c',
            'import json,sys; print(json.dumps(list(sys.version_info[:3])))'],text=True))
        try:require_python(version, matches[0])
        except ValueError as error:p.error(str(error))
    else:
        run([args.conda,'create','-y','-n',args.env_name,'--override-channels','-c','conda-forge','python=3.12','pip'])
        prefix+=['-n',args.env_name]
    install(prefix+['python'],args)
    if not args.no_shell_init:
        run([args.conda,'init','bash','zsh'])
    print('Installation complete. Activate with: conda activate '+shlex.quote(matches[0] if matches else args.env_name))
    print('If Conda is not initialized in this shell: source '+str(Path(args.conda).resolve().parents[1]/'etc/profile.d/conda.sh'))


if __name__=='__main__':main()
