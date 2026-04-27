"""
Figure and data provenance for stellar2d.

Naming convention (see docs/provenance.md):
    {test}_{quantity}_{nr}x{nt}_t{tend}_{commit7}_{date}.{ext}

Example:
    lane_emden_verification_64x32_t0.050_9d6bdbe_20260427.png

Every figure embeds a provenance footer containing the full
reproducing command, git state, and generating script path.
"""
import subprocess, os, datetime


def _run(cmd, fallback='unknown'):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return fallback


def git_commit_short(repo=None):
    cwd = repo or _repo_root()
    return _run(['git', '-C', cwd, 'rev-parse', '--short=7', 'HEAD'])


def git_commit_long(repo=None):
    cwd = repo or _repo_root()
    return _run(['git', '-C', cwd, 'rev-parse', 'HEAD'])


def git_dirty(repo=None):
    cwd = repo or _repo_root()
    status = _run(['git', '-C', cwd, 'status', '--porcelain'])
    return len(status) > 0


def _repo_root():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def datestamp():
    return datetime.datetime.now().strftime('%Y%m%d')


def build_filename(test, quantity, nr, nt, tend, ext='png'):
    """
    Build a traceable filename.

    Parameters
    ----------
    test     : str   – test case name, e.g. 'lane_emden'
    quantity : str   – what the figure shows, e.g. 'verification', 'density_2d'
    nr, nt   : int   – grid resolution
    tend     : float – simulation end time
    ext      : str   – file extension (default 'png')
    """
    commit = git_commit_short()
    date = datestamp()
    tend_str = f'{tend:.3f}'
    return f'{test}_{quantity}_{nr}x{nt}_t{tend_str}_{commit}_{date}.{ext}'


def provenance_string(script_path, run_cmd=None, extra=None):
    """
    Build a multi-line provenance string for figure footers.

    Parameters
    ----------
    script_path : str       – __file__ of the generating script
    run_cmd     : str|None  – the shell command that produced the data
    extra       : dict|None – additional key-value metadata
    """
    repo = _repo_root()
    commit = git_commit_long(repo)
    short = git_commit_short(repo)
    dirty = ' (dirty)' if git_dirty(repo) else ''
    date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    script_rel = os.path.relpath(os.path.abspath(script_path), repo)

    lines = [
        f'commit: {short}{dirty} ({commit[:20]}…)',
        f'script: {script_rel}',
        f'date:   {date}',
    ]
    if run_cmd:
        lines.append(f'cmd:    {run_cmd}')
    if extra:
        for k, v in extra.items():
            lines.append(f'{k}: {v}')
    return '\n'.join(lines)


def add_provenance_footer(fig, script_path, run_cmd=None, extra=None):
    """
    Add a small provenance text block at the bottom of a matplotlib figure.
    """
    text = provenance_string(script_path, run_cmd, extra)
    fig.text(0.01, 0.005, text, fontsize=6, fontfamily='monospace',
             color='#888888', verticalalignment='bottom')
