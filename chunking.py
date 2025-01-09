from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

def chunk_recursive (document: str, chunk_size: int) -> list: 

    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=chunk_size,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
    )   

    chunks = text_splitter.create_documents([document])
    return chunks


def chunk_parapgraph (document: str) -> list: 
    # Split the document into lines
    lines = document.splitlines()

    # Group lines into chunks, separated by blank lines
    chunks = []
    paragraph = []

    for line in lines:
        if line.strip():  # Non-blank line
            paragraph.append(line.strip())
        elif paragraph:  # Blank line and there's content in paragraph
            chunks.append(" ".join(paragraph))
            paragraph = []

    # Add the last paragraph if any
    if paragraph:
        chunks.append(" ".join(paragraph))

    return chunks

def combine_sentences(sentences, buffer_size=1):
    # Go through each sentence dict
    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ''

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]['sentence'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['sentence']

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += ' ' + sentences[j]['sentence']

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]['combined_sentence'] = combined_sentence

    return sentences

def chunk_contextual (document:str, breakpoint_percentile_threshold=95) -> list: # gebaseerd op (stap 5 van): https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    # Split the document into sentences using regular expression
    sentences_list = re.split(r'(?<!\w\.[a-zA-Z])(?<=[.!?])\s', document)
    print (f"{len(sentences_list)} sentences were found")

    # maak dictionary van de sentences
    sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(sentences_list)]

    # voeg aan elke dict de zin ervoor en de zin erna toe als een nieuwe waarde
    sentences = combine_sentences(sentences)

    # voeg aan elke dict de embeddings (van de gecombineerde zinnen) toe zodat we die kunnen vergelijken en breakpoints kunnen vinden
    oaiembeds = OpenAIEmbeddings()
    embeddings = oaiembeds.embed_documents([x['combined_sentence'] for x in sentences])
    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]
    

    # bereken alle afstanden naar de volgende zinnen (dat is 1 - de gelijkenis)
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        
        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance

    plt.plot(distances)

    # We need to get the distance threshold that we'll consider an outlier
    # We'll use numpy .percentile() for this
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold) # If you want more chunks, lower the percentile cutoff

    # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list

    # Initialize the start index
    start_index = 0

    # Create a list to hold the grouped sentences
    chunks = []

    # Iterate through the breakpoints to slice the sentences
    for index in indices_above_thresh:
        # The end index is the current breakpoint
        end_index = index

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
        
        # Update the start index for the next group
        start_index = index + 1

    # The last group, if any sentences remain
    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)

    return chunks

    
def chunk_file(document, chunking_strategy, chunk_size=1000):
    """
    Splits a document into chunks of specified size with overlapping text.
    
    Parameters:
        document (str): The full document as a single string.
        chunk_size (int): Size of each chunk in characters.
        chunkingStrategy (String): the strategy to chunk the document.
    
    Returns:
        list: List of document chunks.
    """
    
    if chunking_strategy == 'recursive':
        chunks = chunk_recursive(document, chunk_size)
    
    elif chunking_strategy == 'paragraph':
        chunks = chunk_parapgraph(document)
    
    elif chunking_strategy == 'contextual':
        chunks = chunk_contextual(document)
    
    else:
        raise ValueError(
            f"Invalid chunking_strategy '{chunking_strategy}'. "
            f"Valid options are: recursive, paragraph, contextual"
        )
    return chunks
 
    
