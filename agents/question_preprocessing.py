# Validation of retrieved information	
# Query pre-processing	

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OpenAI.api_key = os.getenv('OPENAI_API_KEY')

query_mapping = {
    # Algemeen naar Jargon
    "regels en richtlijnen": "Normen, regels, richtlijnen en kaders",
    "wat je moet doen": "Eisen en aanbevelingen",
    "uitleggen waarom iets zo is": "Pas toe of leg uit (comply or explain)",
    "waarom iets belangrijk is": "Criteria en argumentatie",
    "mogelijke problemen": "Risico-inschatting en impactanalyse",
    "nieuwe ontwikkelingen": "Dynamisch updaten en iteratieve evaluatie",
    "hoe goed iets werkt": "Technische robuustheid en voorspelkracht",
    "privacyregels": "Juridische randvoorwaarden en gegevensbescherming",
    "het plan van aanpak": "Datamanagementplan en modelvalidatie",
    "wanneer": "welke fase",

    # Data & Modellering
    "gegevens verzamelen": "Dataverzameling en metadata",
    "duidelijkheid over gegevens": "Transparantie en reproduceerbaarheid",
    "zorgen dat data goed wordt beheerd": "Versiebeheer en FAIR-principes",
    "een model trainen": "Modeleringstappen en trainingsdata-analyse",
    "hoe goed het model werkt op nieuwe data": "Externe validatie en generaliseerbaarheid",
    "hoe betrouwbaar een voorspelling is": "Kalibratie en predictiehorizon",
    "wat een model voorspelt": "Uitkomstvariabele en labeling",
    "wanneer een model niet goed werkt": "Bias, fairness en domeinverschuiving",
    "data gebruiken voor AI": "Artificial Intelligence Prediction Algorithm (AIPA)",

    # Medische & Organisatorische Context
    "gebruikers van het systeem": "Stakeholders en eindgebruikers",
    "AI inzetten in ziekenhuizen": "Implementatie in de zorgpraktijk",
    "de rol van AI in medische beslissingen": "Beslisondersteunende systemen (Clinical Decision Support)",
    "het testen van AI in de praktijk": "Praktijktoets en evaluatie",
    "wat een arts moet weten over AI": "Uitlegbaarheid en educatie",
    "AI en wetgeving": "MDR (Medical Device Regulation) en IVDR (In-Vitro Diagnostic Regulation)",
    "wie verantwoordelijk is voor AI in de zorg": "Verantwoordelijke, zorgorganisaties, fabrikanten en toezichthouders",
    "veiligheid van AI-systemen": "Risico-inventarisatie en mitigerende maatregelen",
    "wanneer AI ethische problemen veroorzaakt": "Medisch-ethische overwegingen en privacybescherming"
}


jargon_text = "\n".join([f'- "{k}" → "{v}"' for k, v in query_mapping.items()])


def preprocessor_agent(user_query):

    """
    Translates the question based on a set of translations from regular language to specific professional language, for example: 

    Bijvoorbeeld: "regels en richtlijnen" -> "Normen en kaders"

    Parameters: 
    user query (Str): The query sent to the prompt optimizer

    Returns: 
    str: optimized prompt 
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
                    {"role": "system", "content": f"""Je bent een AI-assistent gespecialiseerd in medische AI. 
                    Je vertaalt gebruikersvragen naar het juiste jargon uit de Leidraad AI in de zorg.
                    Hier is een lijst met correcte vertalingen:

                    {jargon_text}

                    Als een term niet in de lijst staat, probeer deze dan consistent om te zetten naar jargon op basis van de context. Behoud de acherliggende vraag.

                    Query: "{{query}}"
                    Geef de vertaalde vraag terug"""},

                {"role": "user", "content": f"{user_query}"}
        ]
    )
    print("Vertaald antwoord: " + response.choices[0].message.content)

    return response.choices[0].message.content

