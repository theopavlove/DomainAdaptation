import os
from pathlib import Path
from typing import Union

import pandas as pd


class ScriptGenerator:
    def generate_training_scripts(
        self,
        dir_input: Union[Path, str],
        dir_models: Union[Path, str],
        model_params: Union[Path, str],
        dir_output: Union[Path, str],
        source_genome: str,
        target_genome: str,
        phase: str = 'train',
    ):
        for root_entry in sorted(os.scandir(dir_input), key=lambda x: x.name):
            if str(root_entry.name)[0] == ".":
                continue

            input_prefix = root_entry.name

            print(f"Generating run scripts for {input_prefix}")

            out_script_name = f"{dir_output}/{input_prefix}/{phase}.run"
            if os.path.exists(out_script_name):
                os.remove(out_script_name)

            model_params_df = pd.read_csv(model_params).set_index("model")
            for it in ["bottleneck-dim", "iters-per-epoch", "pretrain-epochs"]:
                model_params_df[it] = model_params_df[it].astype(pd.Int64Dtype())

            with open(out_script_name, "a") as f_out:
                for model in model_params_df.index:
                    src_data_prefix = f"{input_prefix}.{source_genome}"
                    tgt_data_prefix = f"{input_prefix}.{target_genome}"

                    cmd_run = (
                        f"python3 {dir_models}/{model}.py "
                        f"-d {input_prefix}.{source_genome}.{target_genome} "
                        f"-c datasets "
                        f"--source-positive ./data/{src_data_prefix}.fa "
                        f"--source-negative ./data/{src_data_prefix}.random.fa "
                        f"--target-train ./data/{tgt_data_prefix}.random_2x.fa "
                        f"--target-positive ./data/{tgt_data_prefix}.fa "
                        f"--target-negative ./data/{tgt_data_prefix}.random.fa "
                        f"-a hybrid "
                        f"--scratch "
                        f"--seed {model_params_df.loc[model,'seed']} "
                        f"--epochs {model_params_df.loc[model,'epochs']} "
                        f"--log .logs/{model if model != 'erm' else 'src_only'}-{input_prefix}.{source_genome}.{target_genome}-seed-{model_params_df.loc[model,'seed']} "
                        f"--phase {phase}"
                    )

                    for param in [
                        "bottleneck-dim",
                        "trade-off-norm",
                        "trade-off",
                        "iters-per-epoch",
                        "pretrain-epochs",
                    ]:
                        param_value = model_params_df.loc[model, param]
                        if not pd.isna(param_value):
                            cmd_run += f" --{param} {param_value}"

                    f_out.write(" ".join(cmd_run.split()) + "\n")
