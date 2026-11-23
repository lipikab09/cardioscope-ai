
from pathlib import Path
import time
import wfdb

RECORDS = ["208", "230"]

def main():
    out = Path("data/mitdb")
    out.mkdir(parents=True, exist_ok=True)

    for record in RECORDS:
        success = False

        for attempt in range(1, 6):
            try:
                print(f"Downloading {record}, attempt {attempt}/5")

                wfdb.dl_database(
                    "mitdb",
                    dl_dir=str(out),
                    records=[record],
                    overwrite=False
                )

                success = True
                print(f"SUCCESS: {record}")
                break

            except Exception as e:
                print(f"FAILED: {record}: {e}")
                if attempt < 5:
                    time.sleep(2 ** attempt)

        if not success:
            print(f"Could not download {record}")

    print("Retry process completed.")

if __name__ == "__main__":
    main()
