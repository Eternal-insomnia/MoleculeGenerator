import requests
from bs4 import BeautifulSoup
import re
import time
import csv

page = 0
increment = 50
smiles_list = []

while True:
    start_id = page * increment
    url = f"https://bindingdb.org/rwd/jsp/dbsearch/PrimarySearch_ki.jsp?tag=pol&submit=Search&target=udp-3-o-[3-hydroxymyristoyl]%20n-acetylglucosamine%20deacetylase%20(lpxc)&polymerid=5335,6542&startPg={start_id}&Increment=50"

    print(f"Parsing page {page + 1} with {start_id + 1} - {start_id + increment} SMILES")

    # Send GET request
    response = requests.get(url)
    response.raise_for_status()

    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find table
    table = soup.find('div', class_='index_table')
    if not table:
        print("Table with class 'index_table' not found.")
        exit()

    # Find buttons
    buttons = table.find_all('button', class_='nonb m1fontSize', string=re.compile(r'Copy\s*SMILES', re.IGNORECASE))
    if len(buttons) != 0:
        for button in buttons:
            onclick_attr = button.get('onclick', '')

            # Extract SMILES
            match = re.search(r"setClipboard\('([^']+)'\)", onclick_attr)
            if match:
                smiles = match.group(1)
                clean_smiles = re.sub(r'\|[^|]*\|', '', smiles).strip()

                activity_type = None
                activity_value = None
                activity_unit = None

                # Try to find span.big
                parent = button.find_parent('div')
                if parent:
                    grandparent = parent.find_parent('div')
                    if grandparent:
                        ic50_span = grandparent.find('span', class_='big', string=re.compile(r'(IC50|Ki|Kd):', re.IGNORECASE))

                    if ic50_span:
                        activity_text  = ic50_span.get_text(strip=True)
                        # print(activity_text)
                        for pattern_type in ['IC50', 'Ki', 'Kd']:
                            match = re.search(
                                rf'{pattern_type}:\s*(>?[\d.]+(?:[eE][+-]?\d+)?)\s*(nM|uM|pM|M|mM|µM)', 
                                activity_text, 
                                re.IGNORECASE
                            )
                            if match:
                                activity_type = pattern_type.upper()
                                activity_value = match.group(1)
                                activity_unit = match.group(2)
                                break

                smiles_list.append((clean_smiles, activity_type, activity_value, activity_unit))
            else:
                # Warn if onclick exists but doesn't match
                print(f"No match in: {onclick_attr}")
                pass
    else:
        break

    page += 1
    time.sleep(0.5)

# Print num of SMILES
print(f"\n\nTotal SMILES extracted: {len(smiles_list)}\n")

# Save SMILES to CSV
with open("smiles_list.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["SMILES", "Activity_type", "Activity_value", "Activity_unit"])  # header
    for smiles, act_type, act_val, act_unit in smiles_list:
        writer.writerow([smiles, act_type, act_val, act_unit])

print(f"\nSMILES saved to 'smiles_list.csv'")
