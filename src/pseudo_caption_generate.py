from dotenv import load_dotenv
load_dotenv()

import sys
import json
import textwrap
from pathlib import Path

import openai
from pydantic import BaseModel, Field
from tqdm import tqdm, trange

from helper import get_client, OPENAI_MODELS


class Description(BaseModel):
    category: str = Field(..., title="Category", description="Category of the event. Normal if the event is normal, corresponding action if the event is anomalous.")
    description: str = Field(..., title="Description", description="Description for an anomlous or normal event.")


class SetOfDescriptions(BaseModel):
    normal: Description = Field(..., title="Normal", description="Description for a normal event.")
    anomalous: Description = Field(..., title="Anomalous", description="Description for an anomalous event.")


class ListOfSetsOfDescriptions(BaseModel):
    descriptions: list[SetOfDescriptions] = Field(..., title="Descriptions", description="List of sets of descriptions for normal and anomalous events.")


class Main:
    def __init__(self):
        pass

    def run(
        self,
        host: str = 'localhost', port: int = 50001,
        rank: int = 0, world_size: int = 1,
        model_name: str = 'gpt-4o',
        exp_name: str = '00-rich-context',
        system_prompt: str = """You are solving the video anomaly detection (VAD) problem in a fancy way. As you know, anomalous events are rare but their categories are diverse. You have to generate example scene descriptions both for the anomalous events and normal events. We will use these descriptions to decide if given video clips contain anomalous events by choosing one of the descriptions having the top similarity measured by a multi-modal retrieval model like ImageBind. The descriptions should be short and concise. The entire response should be in the provided json format.""",
        num_seeds: int = 1000,
        num_captions_per_seed: int = 100,
    ):
        print(f"rank: {rank}, world_size: {world_size}")
        print(f"host: {host}, port: {port}")
        print(f"model_name: {model_name}")
        print(f"exp_name: {exp_name}")
        print(f"system_prompt: {system_prompt}")
        print(f"num_seeds: {num_seeds}")
        print(f"num_captions_per_seed: {num_captions_per_seed}")
        print()
        print('#' * 80)
        print(flush=True)

        client = get_client(
            host=host,
            port=port,
            model_name=model_name
        )

        messages = [
            {
                "role": "system",
                "content": textwrap.dedent(system_prompt),
            },
            {
                "role": "user",
                "content": f"Generate {num_captions_per_seed} example scene descriptions.",
            },
        ]

        p_outdir = Path(f"output/psuedo-captions/{model_name.replace('/', '-')}/{exp_name}/captions")
        p_outdir.mkdir(parents=True, exist_ok=True)
        pbar = trange(rank, num_seeds, world_size, mininterval=.001, maxinterval=.5, desc="Seeds", position=1)
        for seed in pbar:
            p_out = p_outdir / f'{seed:08d}.json'
            if p_out.exists():
                pbar.set_postfix_str(f"Skip: {seed}")
                pbar.update()
                continue
            pbar.set_postfix_str(f"Seed: {seed}")

            max_tries = 5
            for current_try in range(max_tries):
                try:
                    if model_name in OPENAI_MODELS:
                        response = client.beta.chat.completions.parse(
                            model=model_name,
                            messages=messages,
                            seed=seed,
                            response_format=ListOfSetsOfDescriptions,
                        )
                    else:
                        response = client.chat.completions.create(
                            model='',
                            messages=messages,
                            seed=seed,
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "foo",
                                    "schema": ListOfSetsOfDescriptions.model_json_schema(),
                                },
                            },
                        )
                except openai.APITimeoutError as e:
                    tqdm.write(f"Timeout: {e}")
                    tqdm.write(f'Try {current_try + 1}/{max_tries}')
                    tqdm.write('Retrying...')
                    continue
                else:
                    break
            else:
                tqdm.write(f"Failed after {max_tries} tries.")
                sys.exit(1)

            response_json = eval(response.choices[0].message.content)
            with p_out.open('w') as f:
                json.dump(response_json, f, indent=2)
            tqdm.write(f"Seed: {seed} saved to {p_out}")
            pbar.update()


if __name__ == '__main__':
    from fire import Fire
    Fire(Main)
