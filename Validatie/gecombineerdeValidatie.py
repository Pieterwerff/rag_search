# Voeg de bovenliggende map toe aan sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vector_storage import get_chunks
from llm_calls.api_calls import query_llm
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset
import sys
import os
from llm_calls.api_calls import query_llm
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, FactualCorrectness, ContextRecall

# --- Settings ---
chunk_size = 1000
chunking_strategy = 'paragraph'

# Lijst met mogelijke LLM's
llm_list = [
    "gpt-4o-mini",
    # "gpt-3.5-turbo",
    # "meta-llama/Llama-3.3-70B-Instruct",
#     # "microsoft/phi-4", - geeft geen json mee
    "deepseek-ai/DeepSeek-V3",
    "NousResearch/Hermes-3-Llama-3.1-405B"
#     # "Qwen/QwQ-32B-Preview", - sprak ineens chinees
#     # "nvidia/Llama-3.1-Nemotron-70B-Instruct", - geeft geen geldige json mee
    # "Qwen/Qwen2.5-72B-Instruct"
#     "01-ai/Yi-34B-Chat",
#     # "databricks/dbrx-instruct" - geeft geen geldige json mee
]

leidraad = 'leidraad_ai_in_zorg'
storageStrategy = "Qdrant"
embeddingStrategy = "text-embedding-ada-002"
collection = "leidraad_ai_in_zorg" + "_" + chunking_strategy

sample_queries = [
    "Welke minimale aspecten moeten in het datamanagementplan gedetailleerd worden vastgelegd voor een dataverzameling?",
    "Hoe zit het met de privacy en herleidbaarheid van dataverzameling?",
    "Welke stappen moet een AIPA-ontwikkelaar vastleggen tijdens analyse- en modelontwikkeling?",
    "Hoe kan ik de robuustheid van mijn model onderzoeken volgens de leidraad?",
    "Wat zijn aanbevelingen over versiegeschiedenis van een AI-systeem?",
    "Welke zes vormen van algoritmische bias worden onderscheiden volgens het begrippenkader?",
    "Hoe kan de voorspelkracht van een AIPA worden gevalideerd?",
    "Wat is het uitgangspunt voor het kiezen van de grootte van de dataset voor externe validatie?",
    "Wat wordt bedoeld met een inherent uitlegbaar model?",
    "Waarom is continue monitoring van het AIPA belangrijk?",
    "Wat gebeurt er in fase 5 van de leidraad?",
    "In welk geval moet aanbeveling 5.1c worden gevolgd?",
    "Op welk niveau/categorie van automatisering bevinden zich de meeste AIPA's in softwaretoepassingen?",
    "Hoe kan de meerwaarde van de implementatie van een AIPA op valide wijze worden gekwantificeerd?",
    "Welke bron kan ik raadplegen voor de inrichting van een kwaliteitsmanagementsysteem conform MDR?",
    "Welke stappen moeten worden doorlopen in de effectbeoordeling van het AIPA?",
    "Wat moet in de risico-inventarisatie worden meegenomen?",
    "Wat houdt een modelmatige Health Technology Assessment (HTA) in?",
    "Waar kan je een routekaart voor de uitvoering van een HTA voor AIPA software vinden?",
    "Welke wet- en regelgeving moet worden gevolgd bij het melden van onverwachte uitkomsten?",
    "Wat moet een implementatieplan voor een AIPA binnen een zorgorganisatie bevatten?",
    "Welke verantwoordelijkheden heeft de fabrikant of ontwikkelende zorgorganisatie na de implementatie van een AIPA?",
    "Wat zijn de verantwoordelijkheden van de zorgorganisatie bij het gebruik van een AIPA?",
    "Welke educatie moet aan de eindgebruiker van een AIPA worden verstrekt?",
    "Wat zijn de rechten en plichten van de zorgverlener bij het gebruik van een AIPA?"
]

expected_responses = [
    "Voor een dataverzameling, moeten tenminste worden vastgelegd: 1. De herkomst van de data, zoals begin- en einddatum van de dataverzameling, locatie(s) van verzameling. 2. Daarnaast moet het originele doel en de context van de beoogte doelgroep, en in die gevallen dat dataverwerking berust op expliete toestemming van patiënt, cliënt of burger, de voorwaarden waaronder de patiënt of burger toestemming heeft verleend. 3. Als laatste moet de procedures van metingen en registratie van data vastgelegd worden. Indien van toepassing de technische eigenschappen van meetinstrumenten (bijv. fabrikant, type nummer)",
    "Ten aanzien van privacy is de geldende regelgeving (de huidige AVG) leidend, ongeacht of de data betrekking heeft op ingezetenen van de Europese Unie. De privacy van personen waar data van is verkregen moet door de ontwikkelaar worden gerespecteerd en gewaarborgd. (1.2.1a). Herleidbaarheid van data naar personen moet worden voorkomen (anonimisering) of beperkt (pseudonimisering). (1.2.1b). Daarnaast moet het principe van dataminimalisatie gevolgd worden, daarmee wordt bedoeld dat niet meer data per subject wordt vastgelegd dan nodig voor de ontwikkeling of het gebruik van het AIPA. Daarnaast moet, indien van toepassing, door de AIPA-ontwikkelaar of -tester expliciet in het datamanagementplan worden vastgelegd hoe om wordt gegaan met eventuele toevalsbevindingen en het recht op vernietiging van data van personen waar data van is verkregen.",
    "De ontwikkelaar van het model moet alle analyse- en modelontwikkeling stappen vastleggen. Daarbij horen alle voorbereidingsstappen (bijv. initiële data analyse10, feature engineering), gebruikte modelleringstechniek (bijv. neuraal netwerk, random forest, time-toevent, logistische regressie), alle modeleringstappen (bijv. modelselectie, tuning, (her-kalibratie). (2.2a) Het uitgangspunt is dat de achtereenvolgende analyse- en modeleringstappen voldoende gedetailleerd zijn zodat een derde partij op basis van de beschrijving alle analyse- en modelstappen exact zou kunnen reproduceren 7–9, 11.",
    "Om de robuustheid te onderzoeken wordt aanbevolen om diverse sensitiviteitsanalyses uit te voeren. Hierbij kan worden gedacht aan analyses van de achitectuur robuustheid, consistentie van modelvoorspellingen, adversarial robuustheid en domeinshift & outliers. Daarnaast kan, om de transparantie van het AIPA te vergroten, ervoor worden gekozen om de invloed van bepaalde inputvariabelen op de voorspelling inzichtelijk te maken met behulp van bijvoorbeeld feature importance methoden (d.w.z. explainable AI[19])",
    "De versiegeschiedenis (het uiteindelijke model en eventuele updates van het model) moet volledig vastgelegd worden, bijvoorbeeld door het toekennen van een versienummer. Deze versiegeschiedenis van het model is een aanvulling op de versiegeschiedenis van de software, zoals bijv. geëist door de MDR.",
    "Historical bias, Representation bias, Measurement bias, Aggregation bias, Evaluation bias, Deployment bias",
    "Bij de evaluatie van voorspelkracht moet bij de keuze voor schatters rekening worden gehouden met de schaal waarop de voorspellingen worden gedaan. (3.1.2a) Een juiste schatter of maat van voorspelkracht kan verschillen tussen een binair eindpunt, multi-categorie eindpunt, een survival eindpunt (met mogelijke censoring) of een uitkomst op een continue schaal. Ook moet bij de keuze voor schatters van voorspelkracht rekening worden gehouden met de voorspelde output van het AIPA model. (3.1.2b)",
    "Het uitgangspunt voor het kiezen van de grootte van de dataset voor externe validatie is: hoe groter, hoe beter.  Hoe groter de dataset, hoe preciezer de schattingen die worden gebruikt voor de statistische en medische evaluatie en hoe beter algoritmische bias kan worden onderzocht. Belang van accurate labels (zie sectie 3.4). Voor een berekening van de minimale grootte van een dataset verwijzen we naar de literatuur22,23 De grootte van de dataset voor externe validatie moet worden beargumenteerd. (3.5a) Berekening van de minimale grootte van de dataset, indien mogelijk (zie sectie 2.5), wordt aanbevolen. (3.5b)",
    "Het is een model waarvan direct te interpreteren is hoe de modelvoorspellingen tot stand zijn gekomen. De fabrikant moet informatie over de interpretatie van het model en de modelvoorspellingen beschikbaar maken voor de beoogde eindgebruikers.(4.1.1a)  Dit moet in een de op de eindgebruiker gerichte presentatie van de modelvoorspellingen door de software, dit geldt in het bijzonder als op basis van de modelvoorspellingen medische beslissingen worden genomen. (4.1.1b)",
    "Continue monitoren van het AIPA is een belangrijk onderdeel van kwaliteitsmanagement en een vereiste vanuit de MDR. Voor de inrichting van een kwaliteitsmanagementsysteem conform MDR verwijzen we naar ISO 13485 Medical devices - Quality management systems - Requirements for regulatory purposes.",
    "Het bepalen van de impact of meerwaarde van het gebruik van het AIPA als onderdeel van de software op respectievelijk de beoogde medische praktijk of context , het medisch handelen en de gezondheidsuitkomsten van de beoogde doelgroep (bijv. de patiënt, cliënt of burger). Ook een een Health technology assessment vindt plaats in deze fase[2].",
    "Aanbeveling 5.1c moet worden gevolgd in het geval van ontwikkeling (van de software met AIPA) binnen een zorganisatie.",
    "De meeste AIPA's in softwaretoepassingen bevinden zich tot nu toe in de 'computer assisted' of 'joint decision making' categorie",
    "Er moet een vergelijkende studie worden uitgevoerd waarin de effecten van het gebruik van het AIPA worden afgezet tegen dezelfde context waarin zoveel mogelijk dezelfde zorgprocessen worden toegepast zonder gebruik van het AIPA",
    "Voor de inrichting van een kwalititeitsmanagementsysteem conform MDR wordt verwezen naar ISO 13485 Medical devices - Quality management systems - Requirements for regulatory purposes",
    "Verwachte effecten: In kaart brengen hoe het AIPA effect zal hebben op het beoogde medische zorgproces en gezondheidsuitkomsten. Risico-inventarisatie: Mogelijke risico’s en onbedoelde effecten voorafgaand aan implementatie inschatten. Mens-machine interactie: Hoe het zorgproces en de zorgverlener interacteren met de software expliciteren. (5.2).",
    "Er moet een risico-inventarisatie worden uitgevoerd om de mogelijke risico’s van het gebruik van het AIPA in de dagelijkse medische praktijk in kaart te brengen, inclusief de verwachte onbedoelde beslissingen en effecten in het gehele zorgproces en redelijkerwijs voorzienbaar verkeerd gebruik.(5.1.2a)",
    "Een modelmatige Health Technology Assessment (HTA) houdt in dat men door middel van een mathematisch model (bijvoorbeeld een Markov-model) een objectieve analyse maakt van de verwachte kosten en baten (meerwaarde) van de introductie van het AIPA in de medische praktijk ten opzichte van de huidige reguliere zorg als benchmark of controle. (5.2a)",
    "In het rapport ‘Waardevolle AI voor gezondheid’ is een routekaart voor de uitvoering van een HTA voor AIPA-software bijgevoegd. Deze routekaart geeft een overzicht van de kosten en de mogelijke financieringsbronnen van HTA-onderzoek voor AI en dus ook voor een AIPA. (5.2a)",
    "MDR Artikel 5.5 (in huis ontwikkelde software), MDR Artikel 10 (kwaliteitssysteem fabrikant), Convenant Veilige Toepassing van Medische Technologie in de Medisch Specialistische Zorg (CMT), MDR Artikel 80 (rapporteren van adverse events tijdens klinisch onderzoek) en MDR Artikel 87 (veiligheidsrapportages voor producten met CE-markering). (5.3.2a)",
    "Het implementatieplan moet bevatten: technische integratie in de IT-infrastructuur en inbedding in werkprocessen, evaluatie van betrouwbaarheid en toepasbaarheid op basis van eerdere studies (6.1b), geleidelijke invoering via een pilot, run-in periode of schaduwdraaien (6.1c), prospectieve risico-inventarisatie (PRI) om risico’s te identificeren en mitigerende maatregelen te nemen (6.1d–6.1f), AVG-gegevensbeschermingseffectbeoordeling (GEB) (6.1g), multidisciplinair implementatieteam, inclusief eindgebruikers, data scientist, IT-specialist en projectleider (6.1h), betrokkenheid van patiënten/cliënten bij het implementatieplan (6.1i) en ondersteuning door bestuurders, inclusief AI-readiness evaluatie (6.1j).",
    "Door de fabrikant moet worden gemonitord op technische fouten in het AIPA en de bijbehorende software, op foutief gebruik, op foutieve voorspellingen, op fairness en op onverwachte neveneffecten van gewoon gebruik van de software in de dagelijkse praktijk (6.2.1a). Aanvullend op de bestaande post-market surveillance (PMS) en vigilantie, monitoring moet zich richten op: foute voorspellingen (bijvoorbeeld miscalibratie, foutpositieve en foutnegatieve classificaties), technische fouten (zoals software-uitval of integratieproblemen in IT-systemen), fairness en bias (ongelijkheden in prestaties van het AIPA tussen verschillende patiëntengroepen) en deployment bias (wordt het AIPA correct en in de juiste doelgroep toegepast?) (6.2.1b – 6.2.1c).",
    "Als een AIPA wordt gebruikt binnen een zorgorganisatie, heeft de zorgorganisatie de verantwoordelijkheid om de juiste werking en het gebruik van het AIPA blijvend te monitoren en moet er een lokaal monitoringsplan worden opgesteld door de zorgorganisatie waar de software wordt geïmplementeerd. Het monitoringsplan moet minimaal de volgende onderdelen bevatten: monitoring of het doel en gewenste effect van het AIPA wordt bereikt, monitoring op foutieve voorspellingen (miscalibratie, foutpositieven, foutnegatieven), monitoring op technische fouten, monitoring op onverwachte effecten voor de zorgverlener, patiënt, organisatie en maatschappij, en monitoring op foutief gebruik, waaronder automation bias (een te zware afhankelijkheid van AIPA-voorspellingen) en deployment bias (verkeerd gebruik of interpretatie van de AIPA-uitkomsten). Daarnaast moet het plan het in kaart brengen van de medische relevantie en kans op fouten, een duidelijke onderbouwing van de verzamelde data en hoe deze in overleg met eindgebruikers wordt vastgelegd, de frequentie van monitoring en de motivatie voor deze keuze, en de meldplicht bij onverwachte uitkomsten waarvan de oorzaak niet direct te herleiden is (6.2.2b) bevatten.",
    "De eindgebruiker (bijvoorbeeld een patiënt of zorgverlener) moet toegang hebben tot informatie over de onderwerpen beschreven in box 6.1, te leveren door de ontwikkelaar of fabrikant (6.3.1a): Volgens Box 6.1 moet de educatie minimaal de volgende onderwerpen bevatten: het bedoelde gebruik van het AIPA, inclusief beperkingen en de gebruikershandleiding zoals verplicht door de MDR, interpretatie van de uitkomsten van het AIPA, mogelijke fouten in gebruik, met aandacht voor transporteerbaarheid en generaliseerbaarheid naar de lokale medische omgeving, beperkingen van het AIPA, zoals het signaleren van afwijkende datapunten of het geven van betrouwbaarheidsintervallen, de mogelijke winst van het toepassen van het AIPA en instructies over data-invoer (inclusief definities en kwaliteit) die van de eindgebruiker wordt verwacht.",
    "Rechten van de zorgverlener omvatten het recht om ondersteund te worden in kennis over specifieke AIPA door de zorgorganisatie en fabrikant, inclusief begrijpelijke eindgebruikerinformatie en training op gebruik en monitoring, transparante communicatie van de fabrikant en/of derden over eerdere validatiestudies en terugkoppeling op gemelde incidenten. Plichten van de zorgverlener omvatten bewust bekwaam zijn in het gebruik van het AIPA, het naleven van het bedoelde gebruik in het belang van de patiënt en fabrikant, implementatie volgens het implementatieplan van de zorgorganisatie, transparantie over het gebruik van AI in prognose of diagnose richting de zorgorganisatie en patiënt (bijvoorbeeld via het dossier), terugmelden van incidenten in het kwaliteitsmanagementsysteem en transparantie over het gebruik van patiëntdata ter verbetering van AI aan de patiënt, inclusief het verkrijgen van geïnformeerde toestemming waar nodig."
]

# --- Stap 1: Laad leidraad uit CSV in een pandas DataFrame ---
document_df = pd.read_csv(r'brondocumenten\hoofstukken.csv')



import os
import json
from dotenv import load_dotenv

# Laad de omgevingsvariabelen en zet RAGAS_APP_TOKEN
load_dotenv()
os.environ["RAGAS_APP_TOKEN"] = os.getenv("RAGAS_APP_TOKEN")

for nchunks in [11, 13, 15, 17, 19]:
    # --- Stap 2: Precompute relevant chunks per sample query ---
    relevant_chunks_dict = {}
    for query in sample_queries:
        relevant_chunks = [chunk.payload['document'] for chunk in get_chunks(collection, query, storageStrategy, n_chunks=nchunks)]
        relevant_chunks_dict[query] = relevant_chunks
        print(f"Precomputed relevant chunks for query: {query}")
    # --- Stap 3: Loop over alle LLM's en voer de evaluatie uit ---
    for llm in llm_list:
        print(f"\nEvaluating with LLM: {llm}")
        dataset = []
        for query, reference in zip(sample_queries, expected_responses):
            # Gebruik de vooraf berekende relevante chunks
            relevant_docs = relevant_chunks_dict[query]
            response = query_llm(relevant_docs, query, llm)
            try:
                object_response = json.loads(response)
                responseText = object_response.get("antwoord")
            except json.JSONDecodeError:
                print(f"Kon de JSON-response niet parsen van {llm} voor query '{query}'")
                print("Ga verder met ongeparsede response")
                responseText = response

            dataset.append({
                "user_input": query,
                "retrieved_contexts": relevant_docs,
                "response": str(responseText),
                "reference": reference
            })
            print(f"Query toegevoegd aan eval-dataset voor {llm}: {query}")
        
        # Maak het evaluatiedataset aan en voer de evaluatie uit
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        # voor een eerlijke vergelijking gebruiken we gpt-4o-mini voor elke evaluatie
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model_name="gpt-4o-mini", temperature=0))
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[Faithfulness(), FactualCorrectness()],
            llm=evaluator_llm
        )
        print(f"Evaluatieresultaat voor {llm} met {nchunks} chunks:")
        print(result)
        
        # Converteer het resultaat naar een pandas DataFrame
        result_df = result.to_pandas()
        
        # Upload het resultaat (zorg dat RAGAS_APP_TOKEN is ingesteld in je .env-bestand)
        # result.upload()

        # Schrijf de resultaten naar een CSV-bestand (zonder index)
        csv_filename = f"./validatieresultaten/aantalchunksvalidatie/evaluation_results_{llm.replace('/', '')}_{nchunks}_chunks.csv"
        result_df.to_csv(csv_filename, index=False)
        print(f"Evaluatieresultaat voor {llm} is opgeslagen in {csv_filename}")
    


print("\nAlle evaluatieresultaten zijn opgeslagen als CSV-bestanden.")

