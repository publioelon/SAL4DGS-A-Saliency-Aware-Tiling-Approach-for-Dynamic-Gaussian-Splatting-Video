# guess_ntc_config.py
import os, sys, json, itertools
import torch
import tinycudann as tcnn

def try_one(ntc_path, enc, net):
    # NTC model outputs 8 dims (mask + dxyz(3) + drot(4)) pelo seu código
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,
        n_output_dims=8,
        encoding_config=enc,
        network_config=net,
    ).to("cuda")

    sd = torch.load(ntc_path, map_location="cuda")
    # Alguns checkpoints salvam direto state_dict, outros salvam {"state_dict": ...}
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    try:
        model.load_state_dict(sd, strict=True)
        return True
    except Exception:
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python guess_ntc_config.py <NTCs_folder>")
        sys.exit(1)

    ntcs_folder = sys.argv[1]
    ntc0 = os.path.join(ntcs_folder, "NTC_000000.pth")
    if not os.path.isfile(ntc0):
        print("Could not find", ntc0)
        sys.exit(1)

    # ---- Candidatos de encoding (os mais comuns) ----
    enc_candidates = []

    # 1) HashGrid (muito comum em tinycudann)
    for n_levels, n_feat, log2_hash, base_res, per_level_scale in [
        (16, 2, 19, 16, 1.3819),
        (16, 2, 19, 8,  1.4473),
        (16, 2, 18, 16, 1.3819),
        (12, 2, 19, 16, 1.5),
    ]:
        enc_candidates.append({
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_feat,
            "log2_hashmap_size": log2_hash,
            "base_resolution": base_res,
            "per_level_scale": per_level_scale,
        })

    # 2) Frequency (menos comum, mas possível)
    for n_freq in [8, 10, 12]:
        enc_candidates.append({
            "otype": "Frequency",
            "n_frequencies": n_freq,
        })

    # ---- Candidatos de network ----
    net_candidates = []
    for otype in ["FullyFusedMLP", "CutlassMLP"]:
        for n_hidden_layers in [2, 3, 4]:
            for n_neurons in [32, 64, 128]:
                net_candidates.append({
                    "otype": otype,
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": n_neurons,
                    "n_hidden_layers": n_hidden_layers,
                })

    print(f"Trying {len(enc_candidates)} enc configs × {len(net_candidates)} net configs...")

    for i, (enc, net) in enumerate(itertools.product(enc_candidates, net_candidates), start=1):
        ok = try_one(ntc0, enc, net)
        if ok:
            print("\nFOUND MATCH ✅")
            print("Encoding:", enc)
            print("Network:", net)

            out = {"encoding": enc, "network": net}
            out_path = os.path.join(ntcs_folder, "config.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print("\nWrote:", out_path)
            return

        if i % 50 == 0:
            print(f"  tried {i} configs...")

    print("\nNo match found ❌")
    print("Isso geralmente significa que o treino usou um config fora desses candidatos.")
    print("Nesse caso, você precisa exportar o config.json do código de treino.")

if __name__ == "__main__":
    torch.cuda.init()
    main()
