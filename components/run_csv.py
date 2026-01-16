import pandas as pd
from components.csv_loader import load_csv_to_db

CSV_PATH = "data/datas.csv"  # update path if needed

def main():
    df = pd.read_csv(CSV_PATH)
    load_csv_to_db(df)
    print("CSV data loaded into MongoDB")

if __name__ == "__main__":
    main()
