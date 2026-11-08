"""Derive dataset-specific DINOv2 settings from pinned source, never run Slurm."""
import ast
import hashlib
import re
import shlex

from gcd_runtime.config import ROOT, TRAIN_SHA, UPSTREAM_SHA


def parser_defaults(path):
    result = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                and node.args[0].value.startswith('--')):
            continue
        key = node.args[0].value[2:]
        for item in node.keywords:
            if item.arg == 'default':
                try:
                    result[key] = ast.literal_eval(item.value)
                except (ValueError, TypeError):
                    pass  # e.g. machine-specific exp_root; not an algorithm setting.
    return result


def decode(value):
    if value in ('True', 'False'):
        return value == 'True'
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def options(tokens):
    result, index = {}, 0
    while index < len(tokens):
        key = tokens[index]
        if not key.startswith('--'):
            raise ValueError(f'Unexpected CLI token: {key}')
        index += 1
        values = []
        while index < len(tokens) and not tokens[index].startswith('--'):
            values.append(decode(tokens[index]))
            index += 1
        result[key[2:]] = values if key == '--eval_funcs' else values[0] if values else True
    return result


def upstream_reference(dataset_name='scars'):
    from gcd_runtime.datasets import dataset_spec
    dataset_spec(dataset_name)
    source = ROOT/'upstream/HypCD/train/train_HypSelEx.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != TRAIN_SHA:
        raise ValueError('Pinned Hyp-SelEx source hash changed')
    script = ROOT/'upstream/HypCD/scripts/train_HypSelEx.cmd'
    text = script.read_text()
    examples = [shlex.split(line.lstrip('# ').strip()) for line in text.splitlines()
                if line.startswith(f'# sbatch scripts/train_HypSelEx.cmd {dataset_name} v2 ')]
    if len(examples) != 1:
        raise ValueError(f'Expected exactly one official {dataset_name}/DINOv2 command')
    positions = examples[0][2:]
    variables = {name: positions[int(position)-1]
                 for name, position in re.findall(r'^(\w+)=\$(\d+)$', text, re.MULTILINE)}
    variables.update(dict(re.findall(r'^(\w+)=([0-9.]+)$', text, re.MULTILINE)))
    command = text.split('srun python -m train.train_HypSelEx', 1)[1].split('>', 1)[0]
    command = command.replace('\\\n', ' ')
    command = re.sub(r'\$\{(\w+)\}|\$(\w+)', lambda m: variables[m[1] or m[2]], command)
    expected = parser_defaults(source)
    expected.update(options(shlex.split(command)))
    return expected, dict(upstream_sha=UPSTREAM_SHA, reference=f'{dataset_name}/DINOv2 released training command',
                         example=' '.join(examples[0]), script_sha256=hashlib.sha256(script.read_bytes()).hexdigest())


def compare_worker(config, *, smoke=False):
    from gcd_runtime.worker import build_flags
    expected, provenance = upstream_reference(config.get('dataset_name', 'scars'))
    effective = parser_defaults(ROOT/'gcd_runtime/trainer.py')
    effective.update(options(build_flags(config)))
    overrides = {'epochs'} if smoke else set()
    differences = {key: dict(expected=value, actual=effective.get(key)) for key, value in expected.items()
                   if key not in overrides and effective.get(key) != value}
    normalization = config.get('use_target_normalization', 'legacy')
    settings = config.get('crish', {})
    extensions = dict(use_target_normalization=normalization,
                      repair_enabled=config.get('method') == 'crish',
                      evidence_sampling=settings.get('evidence_sampling', 'random'),
                      probe_sampling=settings.get('probe_sampling', 'random'),
                      probe_known=settings.get('probe_known', 4),
                      probe_unlabeled=settings.get('probe_unlabeled', 8))
    caveats = ['ID/cluster-order alignment and empty-cluster runtime fixes',
               'isolated evaluation RNG and finite worker deadlines',
               'published table checkpoint mapping not verified']
    if normalization != 'legacy':
        caveats.append('experimental non-self row-normalized USE with positive-view zero-support fallback')
    if config.get('method') == 'crish' and settings.get('evidence_sampling', 'random') != 'random':
        caveats.append('full frozen-feature shortlist; augmented-view neighbors remain local to the cache')
    if config.get('method') == 'crish' and settings.get('probe_sampling', 'random') != 'random':
        caveats.append('experimental labeled-class-balanced probe sampling')
    return dict(**provenance, cli_hyperparameters_match=not differences, differences=differences,
                smoke_overrides={key: dict(expected=expected[key], actual=effective.get(key)) for key in overrides},
                effective_reference_settings={key: effective.get(key) for key in expected},
                checkpoint_selection=config['checkpoint_selection'],
                reference_use_arithmetic_preserved=normalization == 'legacy',
                runtime_extensions=extensions,
                exact_paper_reproduction_verified=False,
                remaining_differences=caveats)
