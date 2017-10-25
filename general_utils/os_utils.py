import subprocess

__author__ = 'marek'

"""Stuff related to operation system, environ etc."""


def check_output(*popenargs, **kwargs):
    """Run command with arguments and return its output as a byte string.
    Backported from Python 2.7 as it's implemented as pure python on stdlib.
    Python 2.6.2. Taken from: https://gist.github.com/edufelipe/1027906
    Example:
    json_str = check_output(["curl", url], stderr=open(os.devnull, 'w'))
    output = delivery_utils.check_output(["curl", '-X', 'POST', url], stderr=open(os.devnull, 'w'))
    """
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        error = subprocess.CalledProcessError(retcode, cmd)
        error.output = output
        raise error
    return output
