import requests

url = "https://www.tnesevai.tn.gov.in/Pages/ServiceList.aspx"

response = requests.get(url)

with open("service_list.html", "w", encoding="utf-8") as f:
    f.write(response.text)

print("HTML Saved")