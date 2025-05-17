import pandas as pd
import wandb
import numpy as np

from typing import Dict, List, Tuple, Union

api = wandb.Api()


def get_logs(run_name: str) -> Dict:
    # Project is specified by <entity/project-name>
    runs = api.runs(run_name)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )
    runs_df.to_csv(run_name.split("/")[1] + ".csv")
    return runs_df.to_dict("records")


def get_metrics(
    results: List[Dict], keys: List[str] = ["acc@1", "acc@5", "acc@N"]
) -> Dict:
    metrics = {}
    for result in results:
        try:
            for key in keys:
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(result["summary"][key])
        except Exception as e:
            continue
    for key in keys:
        metrics[key] = {
            "mean": np.mean(metrics[key]),
            "std": np.std(metrics[key]),
            "n": len(metrics[key]),
        }
    return metrics


results = get_logs("yuhuiz/VisDiff-BLIP-GPT4-CLIP-GPT4-NewSplit-Purity1.0-Seed0-Rebuttal")
easy_results = [result for result in results if "easy" in result["config"]["config"]]
medium_results = [result for result in results if "medium" in result["config"]["config"]]
hard_results = [result for result in results if "hard" in result["config"]["config"]]
display("easy", get_metrics(easy_results))
display("medium", get_metrics(medium_results))
display("hard", get_metrics(hard_results))