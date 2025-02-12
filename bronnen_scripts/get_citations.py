import pandas as pd
import re

import os
print(os.listdir())  # This will list all files in the current folder


def find_references_section(text):
    """
    Checks if a chapter contains the word 'Referenties' and returns True if found.
    
    Args:
        text (str): The full chapter content.

    Returns:
        bool: True if the references section is found, else False.
    """
    return "Referenties" in text

def extract_references(text):
    """
    Extracts references from a chapter's content.

    Args:
        text (str): The full chapter content.

    Returns:
        list: A list of extracted references.
    """
    lines = text.split("\n")
    references = []
    current_reference = ""

    for line in lines:
        line = line.strip()

        # Detect numbered references (e.g., "1.")
        if re.match(r"^\d+\.", line):
            # If a reference is already being built, save it before starting a new one
            if current_reference:
                references.append(current_reference.strip())

            # Start a new reference
            current_reference = line
        else:
            # Append multi-line reference content
            current_reference += " " + line

    # Add the last collected reference
    if current_reference:
        references.append(current_reference.strip())

    return references

# Load the CSV file
csv_filename = "chapters.csv"  # Change this to your actual file
df = pd.read_csv(csv_filename)

# Prepare a list to store extracted references
references_data = []

# Iterate through chapters
for _, row in df.iterrows():
    chapter_number = row["chapter_number"]
    chapter_name = row["chapter_name"]
    content = row["content"]

    # Check if this chapter has a references section
    if find_references_section(content):
        refs = extract_references(content)
        references_data.append({
            "chapter_number": chapter_number,
            "chapter_name": chapter_name,
            "references": refs
        })

# Convert to a DataFrame for better visualization
references_df = pd.DataFrame(references_data)

# Save to CSV
output_filename = "extracted_references.csv"
references_df.to_csv(output_filename, index=False)

print(f"Extracted references saved to {output_filename}")
