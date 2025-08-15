import pandas as pd
import requests
import json
import argparse
import sys

def push_to_power_bi(csv_path, power_bi_url):
    df = pd.read_csv(csv_path)
    rows = df.to_dict(orient="records")
    
    payload = {"rows": rows}
    headers = {"Content-Type": "application/json"}

    response = requests.post(power_bi_url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        message = "Data pushed to Power BI successfully."
    else:
        message = f"Error pushing to Power BI: {response.status_code} - {response.text}"
    
    return message

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str)
    parser.add_argument("--power_bi_url", type=str)
    parser.add_argument("--response_message", type=str)
    args = parser.parse_args()

    # Debug print of arguments
    print("=== DEBUG ARGS ===")
    print(f"input_csv: {args.input_csv}")
    print(f"power_bi_url: {args.power_bi_url}")
    print(f"response_message: {args.response_message}")
    print("==================")

    if args.response_message is None:
        print("Error: --response_message argument is missing.")
        sys.exit(1)

    result_message = push_to_power_bi(args.input_csv, args.power_bi_url)

    with open(args.response_message, "w") as f:
        f.write(result_message)

    print(result_message)
