import requests
from bs4 import BeautifulSoup
import json

url = "https://www.tnesevai.tn.gov.in/Pages/ServiceList.aspx"

html = requests.get(url).text

soup = BeautifulSoup(html, "html.parser")

rows = soup.find_all("tr")

data = []

for row in rows:
    cols = row.find_all("td")

    if len(cols) >= 8:
        try:
            department = cols[1].get_text(" ", strip=True)

            service_name = cols[2].get_text(" ", strip=True)

            pdf_link = ""

            link = cols[3].find("a")

            if link:
                pdf_link = link.get("href")

                if not pdf_link.startswith("http"):
                    pdf_link = (
                        "https://www.tnesevai.tn.gov.in/" +
                        pdf_link.lstrip("/")
                    )

            application_fee = cols[5].get_text(" ", strip=True)

            service_charge = cols[6].get_text(" ", strip=True)

            documents = cols[7].get_text(" ", strip=True)

            data.append({
                "department": department,
                "service_name": service_name,
                "pdf_link": pdf_link,
                "application_fee": application_fee,
                "service_charge": service_charge,
                "documents": documents
            })

        except Exception as e:
            print("Error:", e)

with open("tn_esevai_services.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Saved", len(data), "services")