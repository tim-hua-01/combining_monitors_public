# %%
from inspect_ai.log import read_eval_log
from pathlib import Path
import numpy as np
import pandas as pd
from ast import literal_eval

project_path = Path(__file__).parent.parent
output_path = project_path / "data" / "processed_monitor_scores"

# get all the logs in the data/raw_monitor_logs folder
logs = list(project_path.glob("data/raw_monitor_logs/*"))

# %%
def write_processed_monitor_scores(log_path: Path, output_path: Path, subsample_to: int | None = None) -> None:
    log = read_eval_log(str(log_path))
    raw_scores: list[float]  = [s.value for s in log.reductions[0].samples] #mypy: ignore
    raw_scores = np.array(raw_scores)
    raw_scores_fuzzed = raw_scores + np.random.normal(0, 1e-4, len(raw_scores))
    df = pd.DataFrame({"score": raw_scores, "score_fuzzed": raw_scores_fuzzed})
    if subsample_to is not None:
        df = df.sample(subsample_to)
    df.to_csv(output_path, index=False)


def write_processed_monitor_scores_subset(log_path: Path, output_path: Path, first_n: int, subsample_to: int | None = None) -> None:
    log = read_eval_log(str(log_path))
    raw_scores: list[float]  = [np.mean(literal_eval(s.answer)[:first_n]) for s in log.reductions[0].samples] #mypy: ignore
    raw_scores = np.array(raw_scores)
    raw_scores_fuzzed = raw_scores + np.random.normal(0, 1e-4, len(raw_scores))
    df = pd.DataFrame({"score": raw_scores, "score_fuzzed": raw_scores_fuzzed})
    if subsample_to is not None:
        df = df.sample(subsample_to)
    df.to_csv(output_path, index=False)



# %%
"""
All the logs
[PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/o1-mini_120bd_llama_31_8B_mon.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/APPS_llama3.1-70B-instruct-turbo_attack2.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/o1-mini_honest_llama_31_70B_mon.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/APPS_llama3.1-70B-instruct-turbo_attack3.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/o1-mini_honest_gemma-2-9b.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/APPS_llama3.1-8B-instruct-turbo_attack3.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/o1-mini_more-BCB_llama3.1-8b-turbo-instruct_mon.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/APPS_llama3.1-8b-instruct-turbo_attack2.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/APPS_gemma2-9b-it_attack2.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/o1-mini_more-BCB_gemma-2-9b-it_mon.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/APPS_gemma2-9b-it_attack3.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/o1-mini_more-BCB_llama3.1-70b-turbo-instruct_mon.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/o1-mini_120bd_llama_31_70B_mon.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/o1-mini_120bd_gemma-2-9b.eval'),
 PosixPath('/Users/timhua/Library/CloudStorage/OneDrive-Personal/Coding/MARS/new_hierarchical_monitor/data/raw_monitor_logs/o1-mini_honest_llama_31_8B_mon.eval')]


"""
# %%
# hard code the names
if __name__ == "__main__":
    np.random.seed(50)
    # csv should have format {data}_{model}_{number_of_samples}_{attack/honest}.csv
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "o1-mini_120bd_llama_31_8B_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB_llama318B_5_attack.csv")
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "o1-mini_honest_llama_31_8B_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB_llama318B_5_honest.csv")
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "APPS_llama3.1-8b-instruct-turbo_attack2.eval", project_path / "data" / "processed_monitor_scores" / "APPS2_llama318B_5_attack.csv")
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "o1-mini_120bd_llama_31_70B_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB_llama3170B_5_attack.csv")
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "o1-mini_honest_llama_31_70B_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB_llama3170B_5_honest.csv")
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "APPS_llama3.1-70B-instruct-turbo_attack2.eval", project_path / "data" / "processed_monitor_scores" / "APPS2_llama3170B_5_attack.csv")

    # Now with subset and only first 2 samples
    write_processed_monitor_scores_subset(project_path / "data" / "raw_monitor_logs" / "o1-mini_120bd_llama_31_8B_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB_llama318B_2_attack.csv", 2)
    write_processed_monitor_scores_subset(project_path / "data" / "raw_monitor_logs" / "o1-mini_honest_llama_31_8B_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB_llama318B_2_honest.csv", 2)
    write_processed_monitor_scores_subset(project_path / "data" / "raw_monitor_logs" / "o1-mini_120bd_llama_31_70B_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB_llama3170B_2_attack.csv", 2)
    write_processed_monitor_scores_subset(project_path / "data" / "raw_monitor_logs" / "o1-mini_honest_llama_31_70B_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB_llama3170B_2_honest.csv", 2)

    # More backdoors
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "o1-mini_more-BCB_gemma-2-9b-it_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB2_gemma29B_5_attack.csv")
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "o1-mini_more-BCB_llama3.1-8b-turbo-instruct_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB2_llama318B_5_attack.csv")
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "o1-mini_more-BCB_llama3.1-70b-turbo-instruct_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB2_llama3170B_5_attack.csv")

    # More backdoors with only first 2 samples
    write_processed_monitor_scores_subset(project_path / "data" / "raw_monitor_logs" / "o1-mini_more-BCB_gemma-2-9b-it_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB2_gemma29B_2_attack.csv", 2)
    write_processed_monitor_scores_subset(project_path / "data" / "raw_monitor_logs" / "o1-mini_more-BCB_llama3.1-8b-turbo-instruct_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB2_llama318B_2_attack.csv", 2)
    write_processed_monitor_scores_subset(project_path / "data" / "raw_monitor_logs" / "o1-mini_more-BCB_llama3.1-70b-turbo-instruct_mon.eval", project_path / "data" / "processed_monitor_scores" / "BCB2_llama3170B_2_attack.csv", 2)

    # Also the other gemma evals
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "APPS_gemma2-9b-it_attack2.eval", project_path / "data" / "processed_monitor_scores" / "APPS2_gemma29B_5_attack.csv")
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "o1-mini_120bd_gemma-2-9b.eval", project_path / "data" / "processed_monitor_scores" / "BCB_gemma29B_5_attack.csv")
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "o1-mini_honest_gemma-2-9b.eval", project_path / "data" / "processed_monitor_scores" / "BCB_gemma29B_5_honest.csv")

    # And the gemma subsets
    write_processed_monitor_scores_subset(project_path / "data" / "raw_monitor_logs" / "APPS_gemma2-9b-it_attack2.eval", project_path / "data" / "processed_monitor_scores" / "APPS2_gemma29B_2_attack.csv", 2)
    write_processed_monitor_scores_subset(project_path / "data" / "raw_monitor_logs" / "o1-mini_120bd_gemma-2-9b.eval", project_path / "data" / "processed_monitor_scores" / "BCB_gemma29B_2_attack.csv", 2)
    write_processed_monitor_scores_subset(project_path / "data" / "raw_monitor_logs" / "o1-mini_honest_gemma-2-9b.eval", project_path / "data" / "processed_monitor_scores" / "BCB_gemma29B_2_honest.csv", 2)

    # Get attack3 logs, but first subsample the entire dataset to 300 samples. 
    np.random.seed(50)
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "APPS_llama3.1-70B-instruct-turbo_attack3.eval", project_path / "data" / "processed_monitor_scores" / "APPS3_llama3170B_attack.csv", subsample_to=300)
    np.random.seed(50)
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "APPS_llama3.1-8B-instruct-turbo_attack3.eval", project_path / "data" / "processed_monitor_scores" / "APPS3_llama318B_attack.csv", subsample_to=300)
    np.random.seed(50)
    write_processed_monitor_scores(project_path / "data" / "raw_monitor_logs" / "APPS_gemma2-9b-it_attack3.eval", project_path / "data" / "processed_monitor_scores" / "APPS3_gemma29B_attack.csv", subsample_to=300)


