"""The shipped configs: inheritance, validation and the training recipe they encode."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from hae.config import assert_resume_compatible, load_config, validate_config
from hae.schedule import PhaseSchedule

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
SIGLIP2_ARCH = ['model.encoder.arch=' + json.dumps({
    'hidden_size': 1024, 'intermediate_size': 4096, 'num_hidden_layers': 24,
    'num_attention_heads': 16, 'image_size': 256, 'patch_size': 16})]
STAGE_CONFIGS = {
    'stage1_alignment.yaml': [], 'stage2_gan.yaml': [],
    'siglip2/stage1_alignment.yaml': SIGLIP2_ARCH, 'siglip2/stage1_no_hcpe.yaml': SIGLIP2_ARCH,
    'siglip2/stage2_gan.yaml': SIGLIP2_ARCH,
}
EPOCH = 2502


@pytest.mark.parametrize('name', STAGE_CONFIGS)
def test_configs_validate_for_any_gpu_count(name):
    config = load_config(CONFIGS / name, STAGE_CONFIGS[name])
    assert config['training']['global_batch_size'] == 512
    assert config['training']['save_every'] == EPOCH
    for gpus in (1, 2, 4, 8):
        assert validate_config(deepcopy(config), gpus) == 512 // (64 * gpus)
    with pytest.raises(ValueError, match='multiple'):
        validate_config(deepcopy(config), 3)


@pytest.mark.parametrize('name', ['stage1_alignment.yaml', 'siglip2/stage1_alignment.yaml'])
def test_stage1_schedule(name):
    config = load_config(CONFIGS / name, STAGE_CONFIGS[name])
    assert config['stage'] == 1 and config['training']['scope'] == 'cnn_decoder'
    assert config['losses'] == {'l1': 1.0, 'lpips': 1.0, 'alignment': 0.5, 'alignment_type': 'cos'}
    assert config['training']['max_updates'] is None and config['validation']['enabled']
    schedule = PhaseSchedule(config['training']['phases'], EPOCH)
    assert schedule.bounds() == [(0, 3 * EPOCH), (3 * EPOCH, 25 * EPOCH)]
    assert schedule.total_updates() == 25 * EPOCH
    assert schedule.at(0) == (1e-5, (0.9, 0.999))
    assert schedule.at(3 * EPOCH - 1) == (1e-5, (0.9, 0.999))
    assert schedule.at(3 * EPOCH) == (1e-4, (0.9, 0.9))
    assert schedule.at(25 * EPOCH - 1) == (1e-4, (0.9, 0.9))


@pytest.mark.parametrize('name', ['stage2_gan.yaml', 'siglip2/stage2_gan.yaml'])
def test_stage2_recipe(name):
    config = load_config(CONFIGS / name, STAGE_CONFIGS[name])
    training, gan = config['training'], config['gan']
    assert config['stage'] == 2 and training['scope'] == 'decoder' and gan['enabled']
    assert training['learning_rate'] == gan['learning_rate'] == 5.4e-5
    assert training['betas'] == gan['betas'] == [0.5, 0.9]
    assert (training['lr_warmup_updates'], gan['warmup_updates'], training['max_updates']) == (
        EPOCH, 2 * EPOCH, 10 * EPOCH)
    assert (gan['weight'], gan['adaptive'], gan['adaptive_max']) == (0.5, True, 10.0)
    assert config['ema'] == {'decay': 0.9975, 'warmup': True}
    assert config['losses']['alignment'] == 0.0 and 'phases' not in training
    assert config['data']['preprocess'] == 'squash_bilinear' and config['validation']['enabled']
    stage1 = load_config(CONFIGS / name.replace('stage2_gan', 'stage1_alignment'), STAGE_CONFIGS[name])
    assert training['init_from'] == stage1['training']['output_dir'] + '/checkpoint_last.pt'


def test_multiple_bases_merge_in_order():
    dinov3 = load_config(CONFIGS / 'stage1_alignment.yaml')
    siglip2 = load_config(CONFIGS / 'siglip2/stage1_alignment.yaml')
    assert 'encoder' not in dinov3['model'] and siglip2['model']['encoder']['type'] == 'siglip2'
    assert siglip2['model']['encoder']['hcpe_init'] is None
    assert siglip2['training']['hcpe_warmup'] == {'epochs': 2, 'batch_size': 256, 'learning_rate': 1e-3,
                                                  'lr_warmup_updates': 100}
    assert dinov3['training']['hcpe_warmup']['epochs'] == 0
    assert siglip2['logging']['wandb']['group'] == 'siglip2' and dinov3['logging']['wandb']['group'] == 'dinov3'
    assert siglip2['training']['phases'] == dinov3['training']['phases']
    assert 1281167 // 256 * 2 == 10008
    assert siglip2['training']['output_dir'] == 'runs/siglip2/stage1_alignment'
    for key in ('model', 'training', 'logging'):
        siglip2[key], dinov3[key] = None, None
    assert siglip2 == dinov3


def test_no_hcpe_control_freezes_the_encoder():
    control = load_config(CONFIGS / 'siglip2/stage1_no_hcpe.yaml', SIGLIP2_ARCH)
    assert control['model']['encoder']['hcpe_mode'] == 'none'
    assert control['training']['scope'] == 'decoder' and control['losses']['alignment'] == 0.0
    validate_config(deepcopy(control), 4)
    for key, value, message in (('training.scope', 'cnn_decoder', 'scope=decoder'),
                                ('losses.alignment', '0.5', 'alignment must be 0'),
                                ('model.encoder.hcpe_init', 'weights/x.pt', 'hcpe_mode replace')):
        with pytest.raises(ValueError, match=message):
            validate_config(load_config(CONFIGS / 'siglip2/stage1_no_hcpe.yaml', SIGLIP2_ARCH + [f'{key}={value}']), 4)
    hcpe = load_config(CONFIGS / 'siglip2/stage1_alignment.yaml', SIGLIP2_ARCH)
    for current, saved in ((control, hcpe), (hcpe, control)):
        with pytest.raises(ValueError):
            assert_resume_compatible(current, saved)


def test_siglip2_shape_checks():
    name = CONFIGS / 'siglip2/stage1_alignment.yaml'
    for override, message in (('model.decoder.latent_channels=768', 'latent_channels'),
                              ('model.encoder.feature_layer=0', 'feature_layer'),
                              ('model.encoder.hcpe_mode=residual', 'hcpe_mode'),
                              ('model.encoder.type=clip', 'encoder.type')):
        with pytest.raises(ValueError, match=message):
            validate_config(load_config(name, SIGLIP2_ARCH + [override]), 2)


def test_overrides_and_inheritance_errors(tmp_path):
    config = load_config(CONFIGS / 'stage2_gan.yaml', ['training.max_updates=100', 'gan.adaptive_max=5'])
    assert config['training']['max_updates'] == 100 and config['gan']['adaptive_max'] == 5
    with pytest.raises(ValueError, match='Unknown config field'):
        load_config(CONFIGS / 'stage2_gan.yaml', ['training.max_step=100'])
    with pytest.raises(ValueError, match='key=value'):
        load_config(CONFIGS / 'stage2_gan.yaml', ['training.max_updates'])
    (tmp_path / 'a.yaml').write_text('_base_: b.yaml\n')
    (tmp_path / 'b.yaml').write_text('_base_: a.yaml\n')
    with pytest.raises(ValueError, match='Circular'):
        load_config(tmp_path / 'a.yaml')


def test_stage_rules():
    stage1 = load_config(CONFIGS / 'stage1_alignment.yaml')
    stage2 = load_config(CONFIGS / 'stage2_gan.yaml')
    for config, key, value in ((stage1, ('training', 'scope'), 'decoder'), (stage1, ('losses', 'alignment'), 0.0),
                               (stage1, ('gan', 'enabled'), True), (stage1, ('training', 'max_updates'), 100),
                               (stage2, ('gan', 'weight'), 0.0),
                               (stage2, ('gan', 'enabled'), False), (stage2, ('losses', 'alignment'), 0.5),
                               (stage2, ('training', 'max_updates'), None), (stage2, ('ema', 'decay'), 1.0),
                               (stage2, ('data', 'preprocess'), 'center_crop')):
        broken = deepcopy(config)
        broken[key[0]][key[1]] = value
        with pytest.raises(ValueError):
            validate_config(broken, 4)
    discriminator = deepcopy(stage2)
    discriminator['gan']['discriminator']['recipe'] = 'S_16'
    with pytest.raises(ValueError, match='dinodisc'):
        validate_config(discriminator, 4)
