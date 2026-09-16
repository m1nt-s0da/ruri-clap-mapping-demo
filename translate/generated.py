from dotenv import load_dotenv

load_dotenv()

import os
from tqdm import tqdm
from openai import AsyncOpenAI
import asyncio
import json
from filelock import FileLock
from uuid import uuid7

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.environ["OPENAI_MODEL"]


async def main():
    client = AsyncOpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )
    dst_file = ".var/generated-prompt.json"

    with open("./dataset/gemma-generated.txt", "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    with open(
        os.path.dirname(__file__) + "./generated-prompt.txt", "r", encoding="utf-8"
    ) as f:
        system_prompt = f.read()

    for prompt in tqdm(prompts):
        with FileLock(dst_file + ".lock"):
            if os.path.exists(dst_file):
                with open(dst_file, "r", encoding="utf-8") as f:
                    existing_data = json.loads(f.read())
                    exists = len(
                        [item for item in existing_data if item.get("en_pos") == prompt]
                    )
                    if exists:
                        continue

        completion = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        tqdm.write(f"{completion}")
        content: str = completion.choices[0].message.content
        tqdm.write(f"{content}")
        # if content.startswith("```json"):
        #     content = content[len("```json") :]
        # if content.startswith("```"):
        #     content = content[len("```") :]
        # if content.endswith("```"):
        #     content = content[:-3]
        data = json.loads(content)
        tqdm.write(f"{data}")
        with FileLock(dst_file + ".lock"):
            if os.path.exists(dst_file):
                with open(dst_file, "r", encoding="utf-8") as f:
                    existing_data = json.loads(f.read())
            else:
                existing_data = []
            data["en_pos"] = prompt
            data["id"] = str(uuid7())
            for query in data.get("queries", []):
                if "id" not in query:
                    query["id"] = str(uuid7())
            existing_data.append(data)
            with open(dst_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
