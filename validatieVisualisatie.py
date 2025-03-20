import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the folder where the validation CSV files are stored
folder_path = "validatieresultaten"  # UPDATE THIS PATH!

# Get all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

# API cost per million tokens (in euros)
costs_per_million_tokens = {
    "DeepSeek-V3": 0.27,  # Input cost in €
    "GPT-3.5-Turbo": 0.30,
    "GPT-4o-mini": 0.150,
    "Llama-3.3-70B-Instruct": 0.59,
    "NousResearch Hermes-3-Llama-3.1-405B": 0.80,
    "Qwen2.5-72B": 0.35
}

# Lists to store model names and scores
model_names = []
faithfulness_scores = []
factual_correctness_scores = []
faithfulness_per_euro = []
factual_correctness_per_euro = []

# Loop through each file and extract relevant data
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    # Ensure the required columns exist
    if "faithfulness" in df.columns and "factual_correctness" in df.columns:
        model_name = file.replace("evaluation_results_", "").replace(".csv", "")
        # Normalize model names to match cost dictionary
        name_mapping = {
            "deepseek-aiDeepSeek-V3": "DeepSeek-V3",
            "gpt-3.5-turbo": "GPT-3.5-Turbo",
            "gpt-4o-mini": "GPT-4o-mini",
            "meta-llamaLlama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
            "NousResearchHermes-3-Llama-3.1-405B": "NousResearch Hermes-3-Llama-3.1-405B",
            "QwenQwen2.5-72B-Instruct": "Qwen2.5-72B"
        }

        # Apply the mapping
        model_name = name_mapping.get(model_name, model_name)  # Fallback to original name if no match

        # Compute average scores
        avg_faithfulness = df["faithfulness"].mean()
        avg_factual_correctness = df["factual_correctness"].mean()

        # Compute performance per euro
        cost_per_million = costs_per_million_tokens.get(model_name, None)
        if cost_per_million:
            faithfulness_value = avg_faithfulness / cost_per_million
            factual_correctness_value = avg_factual_correctness / cost_per_million
        else:
            faithfulness_value = factual_correctness_value = None  # Missing cost data

        # Append data
        model_names.append(model_name)
        faithfulness_scores.append(avg_faithfulness)
        factual_correctness_scores.append(avg_factual_correctness)
        faithfulness_per_euro.append(faithfulness_value)
        factual_correctness_per_euro.append(factual_correctness_value)

# Create DataFrames
df_faithfulness = pd.DataFrame({"Model": model_names, "Faithfulness": faithfulness_scores})
df_factual_correctness = pd.DataFrame({"Model": model_names, "Factual Correctness": factual_correctness_scores})
df_cost_comparison = pd.DataFrame({"Model": model_names, "Faithfulness per €1": faithfulness_per_euro, "Factual Correctness per €1": factual_correctness_per_euro})

# Function to plot bar chart
def plot_metric_chart(df, metric_name, color):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y=metric_name, data=df, color=color)
    
    plt.xlabel("")
    plt.ylabel("Average Score")
    plt.ylim(0, 1)
    plt.title(f"{metric_name} Across Models")
    plt.xticks(rotation=45)
    
    plt.show()

# Function to plot cost-effectiveness
def plot_cost_comparison(df):
    df_long = df.melt(id_vars=["Model"], value_vars=["Faithfulness per €1", "Factual Correctness per €1"], 
                      var_name="Metric", value_name="Value")

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Value", hue="Metric", data=df_long, palette=["blue", "orange"])
    
    plt.xlabel("")
    plt.ylabel("Performance per €1")
    plt.title("Cost-Effective Faithfulness & Factual Correctness")
    plt.xticks(rotation=45)
    plt.legend(loc="upper right")
    # Debugging: Print computed values
    print("Model Names:", model_names)
    print("Faithfulness Scores:", faithfulness_scores)
    print("Factual Correctness Scores:", factual_correctness_scores)
    print("Faithfulness per €1:", faithfulness_per_euro)
    print("Factual Correctness per €1:", factual_correctness_per_euro)
    plt.show()

# Function to plot scatter of faithfulness vs. cost
def plot_scatter_cost_effectiveness():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=[costs_per_million_tokens[m] for m in model_names], y=faithfulness_scores, s=100, color="blue", label="Faithfulness")
    sns.scatterplot(x=[costs_per_million_tokens[m] for m in model_names], y=factual_correctness_scores, s=100, color="orange", label="Factual Correctness")

    plt.xlabel("Cost per 1M Tokens (€)")
    plt.ylabel("Average Score")
    plt.title("Faithfulness & Factual Correctness vs. Cost")
    plt.xscale("log")  # Log scale for better cost visualization
    plt.legend()
    
    plt.show()


# Generate visualizations
plot_metric_chart(df_faithfulness, "Faithfulness", "lightblue")
plot_metric_chart(df_factual_correctness, "Factual Correctness", "sandybrown")
plot_cost_comparison(df_cost_comparison)
plot_scatter_cost_effectiveness()
