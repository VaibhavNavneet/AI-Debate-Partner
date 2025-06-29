import pandas as pd

# Load the uploaded CSV file
file_path = "train.csv"
df = pd.read_csv(file_path)

def get_system_prompt(stance):
    if stance == 1:
        return "You are debating in favor of the topic."
    elif stance == -1:
        return "You are debating against the topic."
    else:
        return "You are participating in a debate."

# Build the LLaMA2 instruction-tuned text format
def format_llama2(system, topic, argument):
    return (
        "<s>[INST] <<SYS>>\n"
        + system.strip()
        + "\n<</SYS>>\n\n"
        + topic.strip()
        + " [/INST] "
        + argument.strip()
        + " </s>"
    )

# Apply transformation
df["system_prompt"] = df["stance_WA"].apply(get_system_prompt)
df["text"] = df.apply(lambda row: format_llama2(row["system_prompt"], row["topic"], row["argument"]), axis=1)

# Keep only the formatted text
formatted_df = df[["text"]]

# Save to JSONL
output_path = "debate_ai_llama2_formatted.jsonl"
formatted_df.to_json(output_path, orient="records", lines=True)