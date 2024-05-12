REAL-TIME AUTOMATED SANCTIONS SCREENER
This project is an automated sanctions screening tool using Python. This tool analyzes incoming FIX messages against OFAC (Office of Foreign Assets Control) data to identify and flag sanctioned entities based on security identifiers, names, and other relevant criteria.


Components - 
1. OFAC data integration: 
This is in the form of an XML file downloaded the OFAC website. The data used is the consolidated.xml file retrieved here - sanctionslist.ofac.treas.gov/Home/ConsolidatedList
2. FIX Message Engine : 
FIX_Engine.py simulates live trading conditions by generating FIX messages with varying parameters. It generates both normal and flagged (i.e. containing sanctioned entities) FIX messages. Parameters include client ID (names),  SWIFT code, and ISIN (International Securities Identification Number).
3. Sanctions Screening Model:
model.py uses a Random Forest Classifier to classify incoming FIX messages as either sanctioned or non-sanctioned. It extracts relevant fields from FIX messages using spaCy for Natural Language Processing and then uses the extracted features to train the model. Lastly, it generates a detailed report (FLAGGED_REPORT.txt) listing flagged messages along with relevant details of the sender. 

