from datasets import load_dataset

# You can pick either "argument_quality_ranking" or "argument_topic"
ds = load_dataset("ibm-research/argument_quality_ranking_30k", "argument_quality_ranking")
print(ds)
        

ds['train'].to_csv("train.csv", index=False)
ds['test'].to_csv("test.csv", index=False)
ds['validation'].to_csv("validation.csv", index=False)