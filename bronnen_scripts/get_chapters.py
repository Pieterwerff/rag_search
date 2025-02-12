import re
import pandas as pd

def chunk_manual_by_main_chapter(text):
    # Split the document into lines for line-by-line processing
    lines = text.split("\n")
    
    # Regex pattern for main chapters
    main_chapter_pattern = r"^(?:\*\*\d+\.\*\*|##\s*\d+).*"
    
    chunks = []
    current_main_chapter = None
    
    print("Debugging Output:")
    
    for line in lines:
        # Check for a main chapter heading (e.g., **1.0** Some Title or ## 2)
        if re.match(main_chapter_pattern, line):
            print(f"Detected main chapter: {line.strip()}")
            
            # Save the previous main chapter
            if current_main_chapter:
                chunks.append(current_main_chapter)
            
            # Start a new main chapter
            current_main_chapter = {
                "chapter_name": line.strip(),
                "content": ""
            }
        
        # Append content to the current main chapter
        elif current_main_chapter:
            current_main_chapter["content"] += line + "\n"
    
    # Add the last main chapter to chunks
    if current_main_chapter:
        chunks.append(current_main_chapter)
    
    print("\nDebug Summary:")
    print(f"Total main chapters detected: {len(chunks)}")
    
    return chunks

def visualize_chunks(chunks):
    # Convert chunks into a DataFrame for visualization
    data = [
        {
            "chapter_number": idx,
            "chapter_name": chunk["chapter_name"],
            "content": chunk["content"]  # First 100 chars of content
        }
        for idx, chunk in enumerate(chunks)
    ]
    df = pd.DataFrame(data)
    return df

# Load the document
with open("leidraad.txt", "r", encoding="ISO-8859-1") as file:
    document = file.read()

# Process the document with debugging enabled
chunks = chunk_manual_by_main_chapter(document)

# Visualize chunks
chunk_summary = visualize_chunks(chunks)

# Save the summary to a CSV file
chunk_summary.to_csv("hoofstukken.csv", index=False)
print("\nSummary saved to 'hoofstukken.csv'.")
