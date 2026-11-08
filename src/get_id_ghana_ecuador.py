import asyncio
import os
import gfwapiclient as gfw
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("GFW_API_ACCESS_TOKEN")


async def main():
    client = gfw.Client(access_token=token)
    result = await client.references.get_eez_regions()
    df = result.df()

    for name in ["Senegal", "Ghana", "Ecuador"]:
        print(f"\n=== {name} ===")
        matches = df[df["label"].str.contains(name, case=False, na=False)]
        print(matches[["id", "label", "iso3", "isoSov1", "isoSov2", "territory1"]].to_string())
    matches = df[df["label"].str.contains("Seneg", case=False, na=False)]
    print(matches)

if __name__ == "__main__":
    asyncio.run(main())