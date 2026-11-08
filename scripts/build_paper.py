"""Compile the reviewed LaTeX in a temporary directory and record PDF provenance.

Uses an existing Tectonic, pdfLaTeX or XeLaTeX binary; installs nothing.
Intermediate files and full build logs stay outside paper/.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from check_manuscript import ROOT, PAPER, check_source, sha256


def build(compiler=None):
    digest = check_source()
    executable = compiler or next((p for name in ('tectonic', 'pdflatex', 'xelatex')
                                   if (p := shutil.which(name))), None)
    if executable is None:
        raise RuntimeError('No TeX compiler found. Supply --compiler PATH to Tectonic, pdfLaTeX or XeLaTeX.')
    executable = str(Path(executable).expanduser().absolute()) if Path(executable).expanduser().exists() else shutil.which(executable)
    if not executable:
        raise ValueError('Compiler executable not found')
    name = Path(executable).name
    if name not in ('tectonic', 'pdflatex', 'xelatex'):
        raise ValueError('Supported compiler names: tectonic, pdflatex, xelatex')
    version = subprocess.run([executable, '--version'], capture_output=True, text=True,
                             check=True, timeout=10).stdout.splitlines()[0]
    run_id = 'paper-build-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    directory = ROOT/'logs'/run_id
    directory.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='crish-paper-') as tmp:
        temporary = Path(tmp)
        source = temporary/'manuscript.tex'
        source.write_bytes(PAPER.read_bytes())
        if name == 'tectonic':
            commands = [[executable, '--untrusted', '--keep-logs', '--outdir', str(temporary), str(source)]]
        else:
            command = [executable, '-interaction=nonstopmode', '-halt-on-error', '-no-shell-escape',
                       '-output-directory='+str(temporary), str(source)]
            commands = [command, command]
        record = dict(run_id=run_id, compiler=version, command=commands, status='running',
                      tex_sha256=sha256(PAPER), implementation_sha256=digest)
        try:
            with (directory/'build.log').open('w') as log:
                for command in commands:
                    result = subprocess.run(command, cwd=temporary, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                    result.check_returncode()
            pdf = temporary/'manuscript.pdf'
            if not pdf.is_file() or not pdf.read_bytes().startswith(b'%PDF-'):
                raise RuntimeError('Compiler did not produce a PDF')
            if sha256(PAPER) != record['tex_sha256'] or check_source() != digest:
                raise RuntimeError('Paper or implementation changed during compilation; PDF was not published')
            receipt = dict(tex_sha256=record['tex_sha256'], pdf_sha256=sha256(pdf),
                           implementation_sha256=digest, compiler=version,
                           built_utc=datetime.now(timezone.utc).isoformat())
            staged = ROOT/'paper/manuscript.pdf.tmp'
            shutil.copyfile(pdf, staged)
            os.replace(staged, ROOT/'paper/manuscript.pdf')
            receipt_path = ROOT/'paper/manuscript.build.json'
            staged = receipt_path.with_suffix('.json.tmp')
            staged.write_text(json.dumps(receipt, indent=2)+'\n')
            os.replace(staged, receipt_path)
            record.update(status='passed', **receipt)
        except BaseException as error:
            record.update(status='failed', error=repr(error))
            raise
        finally:
            for log in temporary.glob('*.log'):
                shutil.copyfile(log, directory/log.name)
            (directory/'metadata.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(status='passed', pdf='paper/manuscript.pdf',
                          receipt='paper/manuscript.build.json', log=str((directory/'build.log').relative_to(ROOT)))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', help='Path or name of an existing TeX compiler')
    build(parser.parse_args().compiler)
