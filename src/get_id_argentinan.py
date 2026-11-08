import asyncio
import os

import gfwapiclient as gfw
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("GFW_API_ACCESS_TOKEN")


async def main() -> None:
    client = gfw.Client(access_token=token)

    result = await client.references.get_eez_regions()
    df = result.df()

    print("Total EEZ regions:", len(df))
    print("Columns:", df.columns.tolist())

    matches = df[df["label"].str.contains("Argentin", case=False, na=False)]
    print("\nArgentina matches:")
    print(matches.to_string())


if __name__ == "__main__":
    asyncio.run(main())