#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the spaCy model
nlp = spacy.blank("en")

def extract_sdn_entries(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Define the namespace
    ns = {'ofac': 'https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/XML'}

    # Extract relevant information
    sdn_entries = []
    for sdn_entry in root.findall('ofac:sdnEntry', ns):
        uid = sdn_entry.find('ofac:uid', ns).text if sdn_entry.find('ofac:uid', ns) is not None else None
        first_name = sdn_entry.find('ofac:firstName', ns).text if sdn_entry.find('ofac:firstName', ns) is not None else None
        last_name = sdn_entry.find('ofac:lastName', ns).text if sdn_entry.find('ofac:lastName', ns) is not None else None
        sdn_type = sdn_entry.find('ofac:sdnType', ns).text if sdn_entry.find('ofac:sdnType', ns) is not None else None

        sdn_entries.append({
            'uid': uid,
            'first_name': first_name,
            'last_name': last_name,
            'sdn_type': sdn_type
        })

    # Convert to DataFrame
    df = pd.DataFrame(sdn_entries)
    return df

def parse_fix_message_nlp(fix_message):
    fields = fix_message.split('|')
    parsed_fields = {}
    for field in fields:
        if '=' in field:
            tag, value = field.split('=', 1)
            parsed_fields[tag] = value
    return parsed_fields

def read_fix_messages_from_file(filename):
    with open(filename, 'r') as f:
        fix_messages = [line.strip() for line in f.readlines()]
    return fix_messages

def extract_features(parsed_messages):
    data = []
    for fields in parsed_messages:
        entry = {
            'client_id': fields.get('49', ''),
            'server_id': fields.get('56', ''),
            'order_id': fields.get('11', ''),
            'swift_code': fields.get('50', ''),
            'isin': fields.get('55', '')
        }
        data.append(entry)
    return pd.DataFrame(data)

def parse_xml_file(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bad_sender_compids = []
    bad_swift_codes = []
    bad_isins = []

    ns = {'ofac': 'https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/XML'}

    for sdn_entry in root.findall('ofac:sdnEntry', ns):
        last_name = sdn_entry.find('ofac:lastName', ns)
        if last_name is not None and last_name.text:
            bad_sender_compids.append(last_name.text.strip().lower())
        
        for id_element in sdn_entry.findall(".//ofac:id", ns):
            id_type = id_element.find("ofac:idType", ns)
            id_number = id_element.find("ofac:idNumber", ns)
            if id_type is not None and id_number is not None:
                if id_type.text == "SWIFT/BIC":
                    bad_swift_codes.append(id_number.text.strip().lower())
                if id_type.text == "ISIN":
                    bad_isins.append(id_number.text.strip().lower())

    return list(set(bad_sender_compids)), list(set(bad_swift_codes)), list(set(bad_isins))

def label_message(fields, bad_sender_compids, bad_swift_codes, bad_isins):
    flagged = (
        fields.get('client_id', '').strip().lower() in bad_sender_compids or
        fields.get('swift_code', '').strip().lower() in bad_swift_codes or
        fields.get('isin', '').strip().lower() in bad_isins
    )
    if flagged:
        print(f"Flagged: {fields}")
    return 1 if flagged else 0

def flag_fix_messages(fix_messages, model, X_train):
    parsed_messages = [parse_fix_message_nlp(msg) for msg in fix_messages]
    df = extract_features(parsed_messages)
    df_encoded = pd.get_dummies(df, columns=['client_id', 'server_id', 'swift_code', 'isin'])

    # Ensure the dataframe has the same columns as the training data
    missing_cols = set(X_train.columns) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0
    df_encoded = df_encoded[X_train.columns]  # Reorder columns to match the training data

    predictions = model.predict(df_encoded)
    flagged_messages = [msg for msg, pred in zip(fix_messages, predictions) if pred == 1]
    
    # Print flagged messages for debugging
    for msg, pred in zip(fix_messages, predictions):
        if pred == 1:
            print(f"Flagged message: {msg}")

    return flagged_messages

def generate_report(flagged_messages, df):
    with open('FLAGGED_REPORT.txt', 'w') as report_file:
        for msg in flagged_messages:
            parsed_msg = parse_fix_message_nlp(msg)
            swift_code = parsed_msg.get('50')
            isin = parsed_msg.get('55')
            client_id = parsed_msg.get('49')

            report_file.write(f"FIX Message: {msg}\n")
            report_file.write(f"CLIENT_ID: {client_id}\n")
            report_file.write(f"SWIFT_CODE: {swift_code}\n")
            report_file.write(f"ISIN: {isin}\n")
            report_file.write("\n")

def main():
    xml_file = 'consolidated.xml'
    df = extract_sdn_entries(xml_file)

    if df.empty:
        print("No SDN entries found.")
    else:
        print("Extracted SDN entries:")
        print(df)

    bad_sender_compids, bad_swift_codes, bad_isins = parse_xml_file(xml_file)
    print(f"Bad Sender CompIDs: {bad_sender_compids}")
    print(f"Bad SWIFT Codes: {bad_swift_codes}")
    print(f"Bad ISINs: {bad_isins}")

    # Read FIX messages from file
    fix_messages = read_fix_messages_from_file('fix_log.txt')

    # Parse messages
    parsed_messages = [parse_fix_message_nlp(msg) for msg in fix_messages]

    # Extract features and label messages
    df = extract_features(parsed_messages)
    labels = [label_message(fields, bad_sender_compids, bad_swift_codes, bad_isins) for fields in parsed_messages]
    df['label'] = labels

    # Check if any messages are labeled as 1
    print(f"Labeled Messages: {df[df['label'] == 1]}")

    # Split data into features and target
    X = df.drop(columns=['label'])
    y = df['label']



    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=['client_id', 'server_id', 'swift_code', 'isin'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Flag FIX messages
    new_fix_messages = read_fix_messages_from_file('fix_log.txt')
    flagged_messages = flag_fix_messages(new_fix_messages, model, X_train)

    # Generate report
    generate_report(flagged_messages, df)

if __name__ == "__main__":
    main()
