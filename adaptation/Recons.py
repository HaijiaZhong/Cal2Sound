import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from music2latent import EncoderDecoder

from model_ablation import build_model
from utils.split_ids import get_target_sm_indices, target_condition_ids


def recons_main(config, datasets=None, encoder_decoder=None, device=None):
    """使用最佳 adaptor 重建当前 (S, M) fold 的7条目标音频。"""
    result_path = Path(config["save_load"]["save_folder"])
    result_path.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        data_path = Path(config["data"]["data_path"])
        neu_dataset = np.load(
            data_path / config["data"]["neu_dataset"], allow_pickle=False
        )
    else:
        neu_dataset = datasets["neu"]

    s_id, m_id = get_target_sm_indices(config)
    test_ids = target_condition_ids(s_id, m_id, neu_dataset.shape[0])
    recons_file_names = [
        f"recon_English_S{s_id}_M{m_id}.wav",
        f"recons_English_S{s_id}_M{m_id}_SM(3.0).wav",
        f"recons_English_S{s_id}_M{m_id}_SM(1.8).wav",
        f"recons_English_S{s_id}_M{m_id}_SM(0.6).wav",
        f"recons_English_S{s_id}_M{m_id}_TM(3.5).wav",
        f"recons_English_S{s_id}_M{m_id}_TM(2.0).wav",
        f"recons_English_S{s_id}_M{m_id}_TM(1.0).wav",
    ]

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    test_neu_data = torch.as_tensor(
        np.asarray(neu_dataset[test_ids]), dtype=torch.float32, device=device
    )

    model_latent_scale = float(config["model"].get("sigma_rescale", 0.06))
    decoder_latent_scale = float(config["loss"]["spectral"]["latent_scale"])
    if model_latent_scale != decoder_latent_scale:
        raise ValueError(
            "model.sigma_rescale and loss.spectral.latent_scale must match for "
            f"reconstruction, got {model_latent_scale} and {decoder_latent_scale}"
        )

    model = build_model(config["model"]).to(device)
    model_path = result_path / config["save_load"]["model_path"]
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        test_output = model(test_neu_data)

    if encoder_decoder is None:
        checkpoint_path = config["loss"]["spectral"].get("checkpoint_path")
        encoder_decoder = EncoderDecoder(
            load_path_inference=checkpoint_path,
            device=device,
        )
    else:
        encoder_decoder.device = device
        encoder_decoder.gen.to(device)

    sample_rate = config["loss"]["spectral"]["sample_rate"]
    noise_seed = int(config["loss"]["spectral"]["noise_seed"])
    output_paths = []
    for index, sample_id in enumerate(test_ids):
        # 与训练频谱路径一致：每个原始样本使用固定的 sample-specific seed。
        torch.manual_seed(noise_seed + int(sample_id))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(noise_seed + int(sample_id))

        latent = test_output[index].reshape(
            -1,
            config["model"]["latent_F_dim"],
            config["model"]["latent_T_dim"],
        )
        waveform = encoder_decoder.decode(
            latent,
            denoising_steps=config["loss"]["spectral"]["denoising_steps"],
        ).squeeze().cpu().numpy()
        output_path = result_path / recons_file_names[index]
        sf.write(output_path, waveform, samplerate=sample_rate)
        output_paths.append(str(output_path))

    del model, test_neu_data, test_output
    return output_paths


if __name__ == "__main__":
    result_path = Path("./results_20260719_S1_M1_ZJM_pixelLevel_fluore/")
    with (result_path / "config.json").open(encoding="utf-8") as f:
        configs = json.load(f)
    recons_main(config=configs)
