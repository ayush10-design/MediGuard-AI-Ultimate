
"""Reference-based educational medicine information."""

def get_guide(fields):
    def value(key):
        item = (fields or {}).get(key, "")
        if isinstance(item, dict):
            return str(item.get("value") or "")
        return str(item or "")

    composition = " ".join([
        value("composition"),
        value("generic_composition"),
        value("generic_name"),
        value("medicine_name"),
    ]).lower()

    if "ondansetron" in composition:
        return {
            "ingredient": "Ondansetron",
            "uses": (
                "An anti-nausea medicine used to prevent nausea "
                "and vomiting associated with chemotherapy, "
                "radiotherapy and surgery. Doctors may prescribe "
                "it for other causes when appropriate."
            ),
            "working": (
                "Blocks serotonin at 5-HT3 receptors involved "
                "in the nausea and vomiting reflex."
            ),
            "when": (
                "Timing and frequency depend on the reason for "
                "treatment and the prescription. Do not choose "
                "a dose or schedule from this app."
            ),
            "how": (
                "For an orally disintegrating (MD/ODT) tablet, "
                "follow the product instructions: remove with "
                "dry hands, place on the tongue and allow it "
                "to dissolve. Confirm the formulation first."
            ),
            "side_effects": (
                "Possible effects include headache, constipation, "
                "tiredness and dizziness."
            ),
            "precautions": (
                "Discuss heart-rhythm disorders, liver problems, "
                "other medicines, pregnancy and breastfeeding "
                "with a clinician. Some interactions can be serious."
            ),
            "urgent": (
                "Seek urgent medical attention for trouble breathing, "
                "facial swelling, fainting or an abnormal heartbeat."
            ),
            "sources": [
                ("MedlinePlus - Ondansetron",
                 "https://medlineplus.gov/druginfo/meds/a601209.html")
            ],
        }

    has_calcium = "calcium" in composition
    has_d3 = (
        "vitamin d3" in composition
        or "cholecalciferol" in composition
    )

    if has_calcium and has_d3:
        return {
            "ingredient": "Calcium + Vitamin D3",
            "uses": (
                "Used as a supplement when additional calcium "
                "and vitamin D are needed for bone health."
            ),
            "working": (
                "Calcium supports bones and normal muscle function. "
                "Vitamin D helps the intestine absorb calcium."
            ),
            "when": (
                "Take according to the product label or your "
                "clinician's instructions. The appropriate dose "
                "and timing depend on individual needs."
            ),
            "how": (
                "Follow the specific tablet instructions. "
                "Calcium carbonate supplements are commonly "
                "taken with food."
            ),
            "side_effects": (
                "Calcium carbonate can cause constipation, "
                "stomach discomfort or bloating."
            ),
            "precautions": (
                "Ask a clinician if you have kidney disease, "
                "kidney stones, high calcium levels or take "
                "other medicines. Calcium can interfere with "
                "absorption of certain medications."
            ),
            "urgent": (
                "Seek medical advice promptly for severe or "
                "persistent symptoms or suspected overdose."
            ),
            "sources": [
                ("MedlinePlus - Calcium Carbonate",
                 "https://medlineplus.gov/druginfo/meds/a601032.html"),
                ("NIH - Vitamin D",
                 "https://ods.od.nih.gov/factsheets/Vitamind-HealthProfessional/"),
            ],
        }

    return None
