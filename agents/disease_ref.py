"""
agents/disease_ref.py
----------------------
Ground truth reference text for specialist agents.
Contains 3-sentence descriptions for each PHMRC category.
"""

INFECTIOUS_DISEASE = {
    "Pneumonia": "Pneumonia is an acute respiratory infection characterized by cough and fast or difficult breathing as the primary complaint. Key clinical signs include chest indrawing, fever, and crackling breath sounds upon examination. It is not diagnosed if the illness is chronic or if there is a documented underlying wasting condition like AIDS or cancer.",
    "Malaria": "Malaria is a mosquito-borne parasitic infection characterized by cyclical or spiking high fever, especially in endemic regions like sub-Saharan Africa. Common clinical findings include anaemia, splenomegaly, and rapid deterioration in young children. It is not typically the cause of death if there are clear bacterial source signs like a stiff neck or multi-site bleeding.",
    "Meningitis": "Meningitis is a severe infection of the protective membranes covering the brain and spinal cord, with nuchal rigidity being the hallmark sign. Diagnosis is supported by the combination of stiff neck, high fever, and seizures or altered consciousness. It is not diagnosed if stiff neck is explicitly absent from the patient's record.",
    "Encephalitis": "Encephalitis is an acute inflammation of the brain tissue presenting with high fever, seizures, and fluctuating levels of consciousness. The clinical picture is similar to meningitis but crucially lacks the characteristic sign of a stiff neck. It is not chosen over meningitis if there is any documentation of nuchal rigidity.",
    "Sepsis": "Sepsis is a sudden and rapid multi-organ deterioration resulting from a systemic immune response to an overwhelming infection. It presents with extreme temperature changes and a clinical decline that affects multiple systems without one single clear focal site. It is not the preferred diagnosis in malaria-endemic regions when undifferentiated fever is the primary symptom.",
    "Diarrhea/Dysentery": "Diarrhea/Dysentery involves watery or bloody stools as the primary complaint, often accompanied by severe dehydration and cramping. Physical signs of terminal decline include sunken eyes, skin tenting, and rapid weight loss. It is not diagnosed if diarrhea is merely a secondary symptom of another primary respiratory or neurological illness.",
    "Measles": "Measles is a highly contagious viral infection marked by a characteristic maculopapular rash that starts on the face and spreads downwards. Associated symptoms include high fever, cough, conjunctivitis, and Koplik spots inside the mouth. It is not diagnosed unless a visible rash is explicitly documented in the patient dossier.",
    "AIDS": "AIDS is a chronic syndrome caused by HIV infection leading to oral thrush, recurrent infections, and persistent lymphadenopathy. Key indicators include failure to thrive over several months and a history of chronic diarrhea lasting more than four weeks. It is not an acute illness but rather a progressive wasting condition that develops over a long period.",
    "Hemorrhagic fever": "Hemorrhagic fever is a rapid and severe viral illness characterized by high fever and spontaneous bleeding from multiple sites like the nose, gums, and urine. The deterioration is extremely fast and must involve bleeding from more than one anatomical location. It is not diagnosed if bleeding is restricted to only one site or if fever is absent.",
    "Other Infectious Diseases": "Other Infectious Diseases encompasses confirmed pathogens or infectious syndromes like tuberculosis or typhoid that do not fit into specific clinical categories. It represents clear infectious presentations that have their own distinct patterns but are not covered by the other ten groups. It is not used for non-infectious conditions or trauma-related deaths.",
}

EXTERNAL_TRAUMA = {
    "Drowning": "Drowning is a death resulting from submersion or immersion in water or other liquids. The event is typically witnessed or the child is found near a body of water with evidence of asphyxia. It is not a death caused by illness or chronic medical conditions.",
    "Road Traffic": "Road Traffic deaths result from injuries sustained during collisions involving vehicles on public or private roads. Victims present with blunt or penetrating trauma consistent with high-impact forces encountered in such accidents. It is not used for injuries occurring outside the context of a vehicle-related crash.",
    "Falls": "Falls involve fatal injuries sustained after a child falls or collapses from a significant height or ground level into a hazard. The primary cause of death is the physical trauma of the impact, even if neurological signs develop later. It is not classified as a central nervous system disease regardless of the subsequent symptoms.",
    "Fires": "Fires cause death through direct burn injuries to the skin and tissues or through the toxic inhalation of smoke. History usually includes a report of a fire, explosion, or the child being found in a burning structure. It is not the diagnosis if the burns were sustained from hot liquids or non-fire sources without smoke inhalation.",
    "Violent Death": "Violent Death results from physical assault, abuse, or homicide as evidenced by injuries inconsistent with the reported history. Signs may include blunt force trauma, intentional bruising, or other clear indicators of physical violence. It is not used for accidental injuries where there is no suspicion of human-inflicted harm.",
    "Poisonings": "Poisonings occur when a child ingests or is exposed to toxic substances, leading to sudden vomiting or altered consciousness. The diagnosis depends on a history of exposure or a strong suspicion when no other medical cause is apparent. It is not used for allergic reactions or adverse effects of standard medical treatments.",
    "Bite of Venomous Animal": "Bite of Venomous Animal occurs after a witnessed or reported encounter with a snake, scorpion, or other toxic creature. Local signs include swelling and necrosis at the bite site, often followed by rapid systemic toxicity and neurological decline. It is not used for non-venomous animal bites or simple insect stings.",
}

CHRONIC_SYSTEMIC_OTHER = {
    "Other Cardiovascular Diseases": "Other Cardiovascular Diseases includes primary heart pathologies such as congenital malformations, arrhythmias, or cyanotic defects. It is diagnosed when the primary clinical findings are cardiac in nature, such as murmurs or oedema, without evidence of infection. It is not the diagnosis if the heart failure is a secondary complication of trauma or acute infectious disease.",
    "Other Cancers": "Other Cancers refers to chronic progressive illnesses characterized by palpable masses, unexplained weight loss, and a decline over weeks or months. The child is usually unwell for a long period before death without an acute infectious fever pattern. It is not an acute infection and should be distinguished from rapid illnesses like pneumonia.",
    "Other Digestive Diseases": "Other Digestive Diseases covers non-diarrheal abdominal conditions like bowel obstruction, appendicitis, or liver disease. Symptoms include abdominal pain, vomiting, or jaundice without the watery stools characteristic of dysentery. It is not diagnosed if diarrhea is the primary and most prominent symptom.",
    "Other Defined Causes of Child Deaths": "Other Defined Causes of Child Deaths serves as a category for specific, documented etiologies that are not covered elsewhere, such as birth asphyxia or prematurity. It is used when a clear medical cause is identified that does not match any of the established specialist categories. It is not a fallback for cases where the cause of death is simply unknown or poorly documented.",
}

REFS = {
    "Infectious/Disease": INFECTIOUS_DISEASE,
    "External/Trauma": EXTERNAL_TRAUMA,
    "Chronic/Systemic/Other": CHRONIC_SYSTEMIC_OTHER
}

def get_disease_ref(broad_group: str) -> str:
    """
    Returns a formatted string of the relevant group's descriptions.
    """
    group_dict = REFS.get(broad_group, {})
    output = []
    for cat, desc in group_dict.items():
        output.append(f"[{cat}]\n{desc}")
    return "\n\n".join(output)

def get_full_disease_ref() -> str:
    """
    Returns all 21 categories formatted.
    """
    output = []
    for group_dict in REFS.values():
        for cat, desc in group_dict.items():
            output.append(f"[{cat}]\n{desc}")
    return "\n\n".join(output)
