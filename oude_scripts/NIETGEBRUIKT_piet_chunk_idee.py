import re
import pandas as pd

def chunk_manual_with_full_references_debug(text):
    # Split the document into lines for line-by-line processing
    lines = text.split("\n")
    
    # Regex patterns for main chapters, subchapters, and references
    main_chapter_pattern = r"^\*\*\d+\.\*\*.*"  # **1.0** Some Title
    subchapter_pattern = r"^\*\*\d+\.\d+\*\*.*"  # **1.1** Some Subchapter Title
    reference_pattern = r"^\*\*\d+\.\d+\*\*\s+\*\*Referenties\*\*"  # **2.7** **Referenties**
    chapter_boundary_pattern = r"^##\s+\d+.*"  # ## 3 Validatie van het AIPA or similar
    
    chunks = []
    current_main_chapter = None
    current_subchapter = None
    subchapter_list = []  # To group subchapters under the same main chapter
    capturing_references = False
    references_content = []

    # Debug counters
    detected_main_chapters = 0
    detected_subchapters = 0
    detected_references = 0
    debug_lines_processed = 0

    print("Debugging Output:")
    
    for line in lines:
        debug_lines_processed += 1
        print(f"Processing line {debug_lines_processed}: {line.strip()}")
        
        # Check for a main chapter heading (e.g., **1.0** Some Title)
        if re.match(main_chapter_pattern, line):
            print(f"Detected main chapter: {line.strip()}")
            detected_main_chapters += 1
            
            if capturing_references:
                print("Ending references capture for the previous chapter.")
                current_main_chapter["references"] = "\n".join(references_content).strip()
                references_content = []
                capturing_references = False
            
            if subchapter_list:
                print(f"Saving subchapters for previous main chapter: {current_main_chapter['chapter_name']}")
                for subchapter in subchapter_list:
                    subchapter["references"] = current_main_chapter.get("references")
                chunks.extend(subchapter_list)
                subchapter_list = []
            
            current_main_chapter = {
                "chapter_name": line.strip(),
                "references": None
            }
        
        # Check for the references section (e.g., **2.7** **Referenties**)
        elif re.match(reference_pattern, line):
            print(f"Detected references start: {line.strip()}")
            detected_references += 1
            capturing_references = True
            references_content.append(line.strip())
        
        # If currently capturing references, keep adding lines
        elif capturing_references:
            if re.match(chapter_boundary_pattern, line):
                print("Detected next chapter boundary, ending references capture.")
                current_main_chapter["references"] = "\n".join(references_content).strip()
                references_content = []
                capturing_references = False
            else:
                print(f"Adding line to references: {line.strip()}")
                references_content.append(line.strip())
        
        # Check for a subchapter heading (e.g., **1.1** Some Subchapter Title)
        elif re.match(subchapter_pattern, line):
            print(f"Detected subchapter: {line.strip()}")
            detected_subchapters += 1
            
            if capturing_references:
                print("Ending references capture for subchapter.")
                current_main_chapter["references"] = "\n".join(references_content).strip()
                references_content = []
                capturing_references = False
            
            if current_main_chapter is None:
                print("Warning: No main chapter found. Creating implicit main chapter.")
                current_main_chapter = {
                    "chapter_name": "Implicit Main Chapter",
                    "references": None
                }
            
            if current_subchapter:
                print(f"Saving previous subchapter: {current_subchapter['chapter_name']}")
                subchapter_list.append(current_subchapter)
            
            current_subchapter = {
                "chapter_name": line.strip(),
                "content": "",
                "references": None
            }
        
        # Add content to the current subchapter
        elif current_subchapter:
            current_subchapter["content"] += line + "\n"
    
    # Finalize remaining content
    if capturing_references:
        print("Finalizing remaining references capture.")
        current_main_chapter["references"] = "\n".join(references_content).strip()
    
    if current_subchapter:
        print(f"Saving final subchapter: {current_subchapter['chapter_name']}")
        subchapter_list.append(current_subchapter)
    if subchapter_list:
        print(f"Saving remaining subchapters for main chapter: {current_main_chapter['chapter_name']}")
        for subchapter in subchapter_list:
            subchapter["references"] = current_main_chapter.get("references")
        chunks.extend(subchapter_list)
    
    # Debug summary
    print("\nDebug Summary:")
    print(f"Total lines processed: {debug_lines_processed}")
    print(f"Total main chapters detected: {detected_main_chapters}")
    print(f"Total subchapters detected: {detected_subchapters}")
    print(f"Total references sections detected: {detected_references}")
    print(f"Total chunks created: {len(chunks)}")
    
    return chunks

def visualize_chunks_debug(chunks):
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
with open("leidraad_small.txt", "r", encoding="ISO-8859-1") as file:
    document = file.read()

# Process the document with debugging enabled
chunks_debug = chunk_manual_with_full_references_debug(document)

# Visualize chunks
chunk_summary_debug = visualize_chunks_debug(chunks_debug)

# Save the summary to a CSV file
chunk_summary_debug.to_csv("chunked_manual_with_full_references_debug.csv", index=False)
print("\nSummary saved to 'chunked_manual_with_full_references_debug.csv'.")
