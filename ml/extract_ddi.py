import xml.etree.ElementTree as ET
import pandas as pd
import os

XML_PATH = os.path.join("backend", "data", "full database.xml")
OUT_PATH = os.path.join("ml", "ddi_pairs.csv")

NS = "{http://www.drugbank.ca}"

print("Parsing XML (streaming mode)...")

records = []
current_drug_id = None
inside_drug = False
count = 0

for event, elem in ET.iterparse(XML_PATH, events=("start", "end")):

    if event == "start" and elem.tag == f"{NS}drug":
        inside_drug = True

    if event == "end" and elem.tag == f"{NS}drug":
        primary = elem.find(f"{NS}drugbank-id[@primary='true']")
        if primary is not None:
            current_drug_id = primary.text
            interactions = elem.find(f"{NS}drug-interactions")
            if interactions is not None:
                for inter in interactions.findall(f"{NS}drug-interaction"):
                    d2 = inter.find(f"{NS}drugbank-id")
                    desc = inter.find(f"{NS}description")
                    if d2 is not None:
                        records.append({
                            "drug1_id": current_drug_id,
                            "drug2_id": d2.text,
                            "description": desc.text if desc is not None else ""
                        })
        count += 1
        if count % 500 == 0:
            print(f"  Processed {count} drugs, {len(records)} interactions so far...")
        elem.clear()

df = pd.DataFrame(records)
df = df.drop_duplicates()
df["label"] = 1

print(f"\nDone. Found {len(df)} DDI pairs.")
df.to_csv(OUT_PATH, index=False)
print(f"Saved to {OUT_PATH}")