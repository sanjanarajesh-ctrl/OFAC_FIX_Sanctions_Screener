#!/usr/bin/env python3

import random

# Define lists of SWIFT codes, ISINs, and SenderCompIDs
bad_swift_codes = ["CKLBCNBJ", "VTBRRUMM", "BFEARUMM"]
bad_isins = ["CNE000001857", "CNE000001KB1", "CN000A0B7NS3"]
good_sender_compids = ["ROSE", "SMITHFIELD", "KUNDE", "SMETHLEY", "LEIGHTON"]
bad_sender_compids = ["CENTREX", "RASSVET", "UMANSKI"]

# Define good stocks
good_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "FB"]

def generate_fix_message(is_flagged=False):
    if is_flagged:
        offense_type = random.choice(['swift', 'isin', 'compid'])
        if offense_type == 'swift':
            client_id = random.choice(good_sender_compids)
            swift_code = random.choice(bad_swift_codes)
            isin = random.choice(good_stocks)
        elif offense_type == 'isin':
            client_id = random.choice(good_sender_compids)
            swift_code = "NORMALSWIFT"
            isin = random.choice(bad_isins)
        elif offense_type == 'compid':
            client_id = random.choice(bad_sender_compids)
            swift_code = "NORMALSWIFT"
            isin = random.choice(good_stocks)
    else:
        client_id = random.choice(good_sender_compids)
        swift_code = "NORMALSWIFT"
        isin = random.choice(good_stocks)

    server_id = "SERVER1"
    order_id = str(random.randint(1000000, 9999999))

    fix_message = f"8=FIX.4.4|49={client_id}|56={server_id}|11={order_id}|34=1|50={swift_code}|55={isin}|10=000|"
    return fix_message

def write_fix_messages_to_file(filename, num_messages=1000, bad_ratio=0.05):
    with open(filename, 'w') as f:
        num_bad_messages = int(num_messages * bad_ratio)
        num_good_messages = num_messages - num_bad_messages

        for _ in range(num_good_messages):
            fix_message = generate_fix_message(is_flagged=False)
            f.write(fix_message + '\n')

        for _ in range(num_bad_messages):
            fix_message = generate_fix_message(is_flagged=True)
            f.write(fix_message + '\n')

# Write messages to fix_log.txt
write_fix_messages_to_file('fix_log.txt', num_messages=1000, bad_ratio=1 / 21)
