"""Package a trained RP2K backbone and its inference contract; no GPU required."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gcd_runtime.config import load_config
from gcd_runtime.rp2k import digest


def export(run, output):
    run, output = Path(run), Path(output)
    cfg = load_config(run/'config.json')
    if cfg['dataset_name'] != 'rp2k' or cfg['dino'] != 'v2' or cfg['method'] != 'crish':
        raise ValueError('Expected a CRiSH RP2K DINOv2 run')
    checkpoint = run/'backbone_final.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError('No final backbone; resume.pt is not an inference checkpoint')
    # Avoid emitting deployment weights from a bounded smoke or incomplete run.
    if cfg.get('max_steps_per_epoch') or cfg.get('rp2k_subdir') != 'rp2k':
        raise ValueError('A bounded smoke checkpoint is not a final retail model')
    metadata = json.loads((run/'metadata.json').read_text())
    if metadata.get('status') != 'completed':
        raise ValueError('Run is not completed; inspect/resume it before exporting')
    output.mkdir(parents=True, exist_ok=False)
    target = output/'backbone.pt'
    shutil.copy2(checkpoint, target)
    architecture = ROOT/'upstream/SelEx/models/vision_transformer2.py'
    # The exact pinned definition travels with the weights. Do not assume a
    # torch.hub model has identical state keys, interpolation, or token handling.
    shutil.copy2(architecture, output/'vision_transformer2.py')
    shutil.copy2(ROOT/'upstream/SelEx/LICENSE', output/'SelEx-LICENSE.txt')
    shutil.copy2(ROOT/'NOTICE', output/'NOTICE')
    contract = dict(modelName='crish_rp2k_dinov2_vitb14_reg4', modelVersion=digest(target),
        vectorDimension=768, embeddingType='IMAGE_FEATURE',
        preprocessingVersion='selex-imagenet-224-v1', inputColor='RGB',
        resizeShortSide=256, centerCrop=224, interpolation='bicubic',
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        output='L2-normalized backbone CLS feature; no projector or cluster ID',
        architecture='vision_transformer2.vit_base(patch_size=14,num_register_tokens=4)',
        architecture_sha256=digest(architecture), upstream_sha=cfg['upstream_sha'],
        dataset_manifest_sha256=cfg['rp2k_manifest_sha256'],
        source_git_sha=metadata.get('git_sha'),
        recognition_threshold=None, threshold_status='calibration_required_on_labeled_validation_crops')
    (output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    shutil.copy2(run/'config.json', output/'training_config.json')
    if (run/'evaluation_last.json').exists():
        shutil.copy2(run/'evaluation_last.json', output/'evaluation_last.json')
    return contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.output), indent=2))
