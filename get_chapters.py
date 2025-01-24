import re
import pandas as pd

def chunk_manual_with_exact_pattern(text):
    # Split the document into lines for line-by-line processing
    lines = text.split("\n")
    
    # Regex patterns for exact format of main chapters, subchapters, and references
    main_chapter_pattern = r"^(?:\*\*\d+\.\*\*|##\s*\d+).*"
    subchapter_pattern = r"^\*\*\d+\.\d+\*\*.*"
    reference_pattern = r"^\*\*\d+\.\d+\*\* Referenties.*"
    
    chunks = []
    current_main_chapter = None
    current_subchapter = None
    subchapter_list = []  # To group subchapters under the same main chapter

    # Debug counters
    detected_main_chapters = 0
    detected_subchapters = 0
    detected_references = 0
    
    print("Debugging Output:")
    
    for line in lines:
        # Check for a main chapter heading (e.g., **1.0** Some Title or ## 2)
        if re.match(main_chapter_pattern, line):
            print(f"Detected main chapter: {line.strip()}")
            detected_main_chapters += 1
            
            # Save previous main chapter's subchapters as chunks
            if subchapter_list:
                # Assign references to all subchapters of this main chapter
                for subchapter in subchapter_list:
                    subchapter["references"] = current_main_chapter.get("references")
                chunks.extend(subchapter_list)
                subchapter_list = []
            
            # Start a new main chapter
            current_main_chapter = {
                "chapter_name": line.strip(),
                "references": None
            }
        
        # Check for a subchapter heading (e.g., **1.1** Some Subchapter Title)
        elif re.match(subchapter_pattern, line):
            print(f"Detected subchapter: {line.strip()}")
            detected_subchapters += 1

            # Initialize a main chapter if missing
            if current_main_chapter is None:
                print("Warning: No main chapter found. Creating implicit main chapter.")
                current_main_chapter = {
                    "chapter_name": "Implicit Main Chapter",
                    "references": None
                }
            
            # Save the previous subchapter
            if current_subchapter:
                subchapter_list.append(current_subchapter)
            
            # Start a new subchapter
            current_subchapter = {
                "chapter_name": line.strip(),
                "content": "",
                "references": None
            }
        
        # Check for a references section (e.g., **1.6** Referenties)
        elif re.match(reference_pattern, line):
            print(f"Detected references: {line.strip()}")
            detected_references += 1
            if current_main_chapter:
                current_main_chapter["references"] = line.strip()
        
        # Add line to the current subchapter content
        elif current_subchapter:
            current_subchapter["content"] += line + "\n"
    
    # Add the last subchapter and subchapter list to chunks
    if current_subchapter:
        subchapter_list.append(current_subchapter)
    if subchapter_list:
        # Assign references to all remaining subchapters
        for subchapter in subchapter_list:
            subchapter["references"] = current_main_chapter.get("references")
        chunks.extend(subchapter_list)
    
    # Final debug output
    print("\nDebug Summary:")
    print(f"Total main chapters detected: {detected_main_chapters}")
    print(f"Total subchapters detected: {detected_subchapters}")
    print(f"Total references detected: {detected_references}")
    print(f"Total chunks created: {len(chunks)}")
    
    return chunks

def visualize_chunks(chunks):
    # Convert chunks into a DataFrame for visualization
    data = [
        {
            "Subchapter Name": chunk["chapter_name"],
            "References": chunk["references"],
            "Content Snippet": chunk["content"][:100]  # First 100 chars of content
        }
        for chunk in chunks
    ]
    df = pd.DataFrame(data)
    return df

# Load the document
with open("leidraad.txt", "r", encoding="ISO-8859-1") as file:

    document = file.read()

# Process the document with debugging enabled
chunks = chunk_manual_with_exact_pattern(document)

# Visualize chunks
chunk_summary = visualize_chunks(chunks)

# Save the summary to a CSV file
chunk_summary.to_csv("chunked_manual_exact_pattern.csv", index=False)
print("\nSummary saved to 'chunked_manual_exact_pattern.csv'.")
