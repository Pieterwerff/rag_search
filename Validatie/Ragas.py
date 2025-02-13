from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset
import sys
import os
# Voeg de bovenliggende map toe aan sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api_calls import query_llm
from vector_storage import  get_chunks
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# --- Settings ---
llm = 'gpt4o-mini' # might become a list of strings to iterate through
leidraad = 'leidraad_ai_in_zorg'
storageStrategy = "Qdrant"
embeddingStrategy = "text-embedding-ada-002"
chunking_strategy = 'paragraph'

sample_queries = [
    "Welke minimale aspecten moeten in het datamanagementplan gedetailleerd worden vastgelegd voor een dataverzameling?",
    "Hoe zit het met de privacy en herleidbaarheid van dataverzameling?",
    "Welke stappen moet een AIPA-ontwikkelaar vastleggen tijdens analyse- en modelontwikkeling?",
    "Hoe kan ik de robuustheid van mijn model onderzoeken volgens de leidraad?",
    "Wat zijn aanbevelingen over versiegeschiedenis van een AI-systeem?",
    "Welke zes vormen van algoritmische bias worden onderscheiden volgens het begrippenkader?",
    "Wat wordt bedoeld met de voorspelkracht van het AIPA-model?",
    "Wat is het uitgangspunt voor het kiezen van de grootte van de dataset voor externe validatie?",
    "Wat wordt bedoeld met een inherent uitlegbaar model?",
    "Waarom is continue monitoring van het AIPA belangrijk?",
    "Wat gebeurt er in fase 5 van de leidraad?",
    "In welk geval moet aanbeveling 5.1c worden gevolgd?",
    "Op welk niveau/categorie van automatisering bevinden zich de meeste AIPA's in softwaretoepassingen?",
    "Hoe kan de meerwaarde van de implementatie van een AIPA op valide wijze worden gekwantificeerd?",
    "Welke bron kan ik raadplegen voor de inrichting van een kwaliteitsmanagementsysteem conform MDR?"
]

expected_responses = [
    "Voor een dataverzameling, moeten tenminste worden vastgelegd: 1. De herkomst van de data, zoals begin- en einddatum van de dataverzameling, locatie(s) van verzameling. 2. Daarnaast moet het originele doel en de context van de beoogte doelgroep, en in die gevallen dat dataverwerking berust op expliete toestemming van patiënt, cliënt of burger, de voorwaarden waaronder de patiënt of burger toestemming heeft verleend. 3. Als laatste moet de procedures van metingen en registratie van data vastgelegd worden. Indien van toepassing de technische eigenschappen van meetinstrumenten (bijv. fabrikant, type nummer)",
    "Ten aanzien van privacy is de geldende regelgeving (de huidige AVG) leidend, ongeacht of de data betrekking heeft op ingezetenen van de Europese Unie. De privacy van personen waar data van is verkregen moet door de ontwikkelaar worden gerespecteerd en gewaarborgd. (1.2.1a). Herleidbaarheid van data naar personen moet worden voorkomen (anonimisering) of beperkt (pseudonimisering). (1.2.1b).",
    "De ontwikkelaar van het model moet alle analyse- en modelontwikkeling stappen vastleggen. Daarbij horen alle voorbereidingsstappen (bijv. initiële data analyse10, feature engineering), gebruikte modelleringstechniek (bijv. neuraal netwerk, random forest, time-toevent, logistische regressie), alle modeleringstappen (bijv. modelselectie, tuning, (her-kalibratie). (2.2a) Het uitgangspunt is dat de achtereenvolgende analyse- en modeleringstappen voldoende gedetailleerd zijn zodat een derde partij op basis van de beschrijving alle analyse- en modelstappen exact zou kunnen reproduceren 7–9, 11.",
    "Om de robuustheid te onderzoeken wordt aanbevolen om diverse sensitiviteitsanalyses uit te voeren. Hierbij kan worden gedacht aan analyses van de achitectuur robuustheid, consistentie van modelvoorspellingen, adversarial robuustheid en domeinshift & outliers.",
    "De versiegeschiedenis (het uiteindelijke model en eventuele updates van het model) moet volledig vastgelegd worden, bijvoorbeeld door het toekennen van een versienummer.",
    "Historical bias, Representation bias, Measurement bias, Aggregation bias, Evaluation bias, Deployment bias",
    "Bij de evaluatie van voorspelkracht moet bij de keuze voor schatters rekening worden gehouden met de schaal waarop de voorspellingen worden gedaan. (3.1.2a) Een juiste schatter of maat van voorspelkracht kan verschillen tussen een binair eindpunt, multi-categorie eindpunt, een survival eindpunt (met mogelijke censoring) of een uitkomst op een continue schaal. Ook moet bij de keuze voor schatters van voorspelkracht rekening worden gehouden met de voorspelde output van het AIPA model. (3.1.2b)",
    "Hoe groter de dataset, hoe preciezer de schattingen die worden gebruikt voor de statistische en medische evaluatie en hoe beter algoritmische bias kan worden onderzocht. Belang van accurate labels (zie sectie 3.4). Voor een berekening van de minimale grootte van een dataset verwijzen we naar de literatuur22,23 De grootte van de dataset voor externe validatie moet worden beargumenteerd. (3.5a) Berekening van de minimale grootte van de dataset, indien mogelijk (zie sectie 2.5), wordt aanbevolen. (3.5b)",
    "Het is een model waarvan direct te interpreteren is hoe de modelvoorspellingen tot stand zijn gekomen. De fabrikant moet informatie over de interpretatie van het model en de modelvoorspellingen beschikbaar maken voor de beoogde eindgebruikers.(4.1.1a)  Dit moet in een de op de eindgebruiker gerichte presentatie van de modelvoorspellingen door de software, dit geldt in het bijzonder als op basis van de modelvoorspellingen medische beslissingen worden genomen. (4.1.1b)",
    "Continue monitoren van het AIPA is een belangrijk onderdeel van kwaliteitsmanagement en een vereiste vanuit de MDR. Voor de inrichting van een kwaliteitsmanagementsysteem conform MDR verwijzen we naar ISO 13485 Medical devices - Quality management systems - Requirements for regulatory purposes.",
    "Het bepalen van de impact of meerwaarde van het gebruik van het AIPA als onderdeel van de software op respectievelijk de beoogde medische praktijk of context , het medisch handelen en de gezondheidsuitkomsten van de beoogde doelgroep (bijv. de patiënt, cliënt of burger)",
    "In het geval van ontwikkeling (van de software met AIPA) binnen een zorganisatie",
    "In de computer assisted en joint decision making categorieën",
    "Er moet een vergelijkende studie worden uitgevoerd waarin de effecten van het gebruik van het AIPA worden afgezet tegen dezelfde context waarin zoveel mogelijk dezelfde zorgprocessen worden toegepast zonder gebruik van het AIPA",
    "ISO 13485 Medical devices - Quality management systems - Requirements for regulatory purposes"
]

collection = "leidraad_ai_in_zorg" + "_" + chunking_strategy

dataset = []

# haal voor alle queries relevante chunks op
for query,reference in zip(sample_queries,expected_responses):

    relevant_docs = [chunk.payload['document'] for chunk in get_chunks(collection, query, storageStrategy, n_chunks=5)]
    response = query_llm(relevant_docs, query, llm)


    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":relevant_docs,
            "response":response,
            "reference":reference
        }
    )
    # print (query + reference + " is toegevoegd aan eval-dataset")

# maak dataset aan, kies een llm om mee te evalueren en start de evaluatie.
evaluation_dataset = EvaluationDataset.from_list(dataset)
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model_name=llm, temperature=0))
result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
print (result)

os.environ["RAGAS_APP_TOKEN"] = "apt.48d0-808b2ffdce1f-53b2-b54c-067a5c88-82c16"
result.upload()