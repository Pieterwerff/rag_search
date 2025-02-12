from openai import OpenAI
import re
import os
from dotenv import load_dotenv

# Laad de API-sleutel
load_dotenv()
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

def chunk_file_with_metadata(document, chunking_strategy="semantic"):
    class AnalyzerAgent:
        def __init__(self):
            self.model = "gpt-4o-mini"

        def analyze(self, text):
            chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
            summaries = []
            for chunk in chunks:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                        {"role": "user", "content": f"Summarize the following text:\n{chunk}"}
                    ],
                    max_tokens=100
                )
                print(response.choices[0].message.content.strip())
                summaries.append(response.choices[0].message.content.strip())
            return summaries

    class ChunkingAgent:
        def chunk(self, text, summaries):
            boundaries = [summary.lower() for summary in summaries]
            chunks = []
            last_idx = 0
            for boundary in boundaries:
                match = re.search(re.escape(boundary), text[last_idx:], re.IGNORECASE)
                if match:
                    start = last_idx + match.start()
                    end = last_idx + match.end()
                    chunks.append((text[last_idx:start].strip(), last_idx, start))
                    last_idx = end
            chunks.append((text[last_idx:].strip(), last_idx, len(text)))
            return chunks

    class ValidationAgent:
        def validate(self, chunks):
            validated_chunks = []
            for chunk, start, end in chunks:
                if len(chunk.split()) > 10:
                    validated_chunks.append({"text": chunk, "start_index": start, "end_index": end})
            return validated_chunks

    def calculate_page_numbers(chunks, document):
        pages = document.split("\f")
        page_starts = []
        current_index = 0
        for page in pages:
            page_starts.append(current_index)
            current_index += len(page)

        for chunk in chunks:
            for i, start in enumerate(page_starts):
                if chunk["start_index"] >= start:
                    chunk["page_number"] = i + 1
                    break

    analyzer = AnalyzerAgent()
    chunker = ChunkingAgent()
    validator = ValidationAgent()

    if chunking_strategy == "semantic":
        summaries = analyzer.analyze(document)
        chunks = chunker.chunk(document, summaries)
        validated_chunks = validator.validate(chunks)
        calculate_page_numbers(validated_chunks, document)
        return validated_chunks
    else:
        raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")

if __name__ == "__main__":
    with open("leidraad_small.txt", "r", encoding="ISO-8859-1") as file:
        document = file.read()
    chunks = chunk_file_with_metadata(document)
    for chunk in chunks:
        print(f"Page {chunk['page_number']}:\n{chunk['text'][:200]}...\n")
