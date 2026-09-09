def get_target_sm_indices(config):
    data_loader_config = config.get("data_loader", {})
    s_idx = int(data_loader_config.get("S_idx", 1))
    m_idx = int(data_loader_config.get("M_idx", 1))
    if not 1 <= s_idx <= 10 or not 1 <= m_idx <= 10:
        raise ValueError(
            f"S_idx and M_idx must be in [1, 10], got S={s_idx}, M={m_idx}"
        )
    return s_idx, m_idx


def audio_id_from_sm(s_idx, m_idx):
    return (s_idx - 1) * 10 + (m_idx - 1)


def target_condition_ids(s_idx, m_idx, dataset_size):
    audio_id = audio_id_from_sm(s_idx, m_idx)
    ids = [audio_id]

    if dataset_size == 700:
        ids.extend([audio_id * 3 + 100 + i for i in range(3)])
        ids.extend([audio_id * 3 + 400 + i for i in range(3)])

    return ids


def sametypes_out_ids(s_idx, m_idx, dataset_size):
    base_ids = []
    seen = set()
    for s, m in [(s_idx, m_idx)]:
        audio_id = audio_id_from_sm(s, m)
        base_ids.append(audio_id)
        seen.add(audio_id)

    for i in range(1, 11):
        for s, m in [(s_idx, i), (i, m_idx)]:
            audio_id = audio_id_from_sm(s, m)
            if audio_id not in seen:
                base_ids.append(audio_id)
                seen.add(audio_id)

    ids = list(base_ids)
    if dataset_size == 700:
        for audio_id in base_ids:
            ids.extend([audio_id * 3 + 100 + i for i in range(3)])
            ids.extend([audio_id * 3 + 400 + i for i in range(3)])

    return ids
