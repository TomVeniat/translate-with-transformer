from pathlib import Path


def get_config():
    return dict(
        batch_size=8,
        num_epochs=20,
        lr=1e-4,
        seq_len=350,
        d_model=512,
        lang_src='en',
        lang_tgt='it',
        model_folder='weights',
        model_basename='tmodel_',
        preload=None,
        tokenizer_file='tokenizer_{0}.json',
        experiment_name='runs/tmodel'
    )


def get_weights_file_path(config, epoch, create=True):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    src_lang = config['lang_src']
    tgt_lang = config['lang_tgt']
    model_filename = f'f{model_basename}{epoch}.pt'
    path = Path('.') / model_folder / f'{src_lang}_{tgt_lang}' / model_filename
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
