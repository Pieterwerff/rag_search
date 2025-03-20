# Re-load necessary libraries and data
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the folder where the validation CSV files are stored
folder_path = "validatieresultaten"

# Get all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

# API cost per million tokens (in euros) (input, output)
costs_per_million_tokens = {
    "DeepSeek-V3": (0.27, 1.10),
    "GPT-3.5-Turbo": (0.30, 0.60),
    "GPT-4o-mini": (0.15, 0.60),
    "Llama-3.3-70B-Instruct": (0.59, 0.77),
    "NousResearch Hermes-3-Llama-3.1-405B": (0.80, 0.80),
    "Qwen2.5-72B": (0.35, 0.40)
}

# Token cost per 100 questions
input_tokens_100q = 1577.8 * 100 / 1_000_000  # Convert to million tokens
output_tokens_100q = 203.5 * 100 / 1_000_000  # Convert to million tokens

# Lists to store model names and scores
model_names = []
faithfulness_scores = []
factual_correctness_scores = []
cost_per_100_questions = []

# Loop through each file and extract relevant data
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    # Ensure the required columns exist
    if "faithfulness" in df.columns and "factual_correctness" in df.columns:
        model_name = file.replace("evaluation_results_", "").replace(".csv", "")

        # Normalize model names to match the cost dictionary
        name_mapping = {
            "deepseek-aiDeepSeek-V3": "DeepSeek-V3",
            "gpt-3.5-turbo": "GPT-3.5-Turbo",
            "gpt-4o-mini": "GPT-4o-mini",
            "meta-llamaLlama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
            "NousResearchHermes-3-Llama-3.1-405B": "NousResearch Hermes-3-Llama-3.1-405B",
            "QwenQwen2.5-72B-Instruct": "Qwen2.5-72B"
        }
        model_name = name_mapping.get(model_name, model_name)  # Apply mapping

        # Compute average scores
        avg_faithfulness = df["faithfulness"].mean()
        avg_factual_correctness = df["factual_correctness"].mean()

        # Compute cost per 100 questions
        if model_name in costs_per_million_tokens:
            input_cost, output_cost = costs_per_million_tokens[model_name]
            total_cost = (input_tokens_100q * input_cost) + (output_tokens_100q * output_cost)
        else:
            total_cost = None  # Handle missing cost data

        # Append data
        model_names.append(model_name)
        faithfulness_scores.append(avg_faithfulness)
        factual_correctness_scores.append(avg_factual_correctness)
        cost_per_100_questions.append(total_cost)

# Create DataFrames
df_cost_comparison = pd.DataFrame({
    "Model": model_names,
    "Faithfulness": faithfulness_scores,
    "Factual Correctness": factual_correctness_scores,
    "Cost per 100 Questions (€)": cost_per_100_questions
})

# Create separate DataFrames for faithfulness and factual correctness vs cost
df_faithfulness = df_cost_comparison[["Model", "Faithfulness", "Cost per 100 Questions (€)"]]
df_factual_correctness = df_cost_comparison[["Model", "Factual Correctness", "Cost per 100 Questions (€)"]]

# Function to plot cost vs performance with better clarity
def plot_cost_vs_performance(df, metric_name, color):
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(x=df["Cost per 100 Questions (€)"], y=df[metric_name], s=150, color=color)

    # Annotate model names for clarity
    for i, row in df.iterrows():
        plt.text(row["Cost per 100 Questions (€)"], row[metric_name], row["Model"], fontsize=10, ha="right")

    plt.xlabel("Cost per 100 Questions (€)")
    plt.ylabel(f"{metric_name} Score")
    if metric_name == "Faithfulness":
        plt.ylim(0.6, 1)  # Ensure y-axis always goes from 0 to 1
    else:
        plt.ylim(0.0, 1)  # Ensure y-axis always goes from 0 to 1
    plt.title(f"{metric_name} vs. Cost Across Models")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

# Generate refined visualizations
plot_cost_vs_performance(df_faithfulness, "Faithfulness", "blue")
plot_cost_vs_performance(df_factual_correctness, "Factual Correctness", "orange")
