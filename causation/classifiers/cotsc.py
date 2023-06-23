import sys
from pathlib import Path
import pandas as pd
import argparse
import os
from pprint import pprint
from collections import namedtuple

from llm_experiments import CoTSC, SamplingScheme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='chain of thoughts (self consistency) - classification')
    parser.add_argument('--prompt-toml', type=str, help="the path to the toml prompt path.")
    parser.add_argument('--dataset', type=str, help='the path to the dataset.')
    parser.add_argument('--temperature', type=float, help='LLM temperature. 0 >= temp >= 2.0')
    parser.add_argument('--top-p', type=float, help='LLM output probability mass (nucleus sampling). 0 > top-p >= 1.0')
    parser.add_argument('--n-completions', type=int, help='LLM number of completions. Large values may be costly.')
    return parser.parse_args()


if __name__ == '__main__':
    args: argparse.Namespace = parse_args()
    assert Path(args.prompt_toml).exists(), f"{args.prompt_toml} does not exist."
    assert Path(args.dataset).exists(), f"{args.dataset} does not exist."
    assert os.environ['OPENAI_API_KEY'], "OPENAI_API_KEY environment variable is missing."

    assert args.n_completions < 10, f"{args.n_completions=} too large. Hard stopped at 10 to avoid high API costs."

    print(args)

    sampler = SamplingScheme(temperature=args.temperature, top_p=args.top_p, top_k=None)

    cotsc = CoTSC.from_toml(model='text-davinci-003',
                            prompt_toml=args.prompt_toml,
                            sampling_scheme=SamplingScheme(temperature=1.0, top_p=1, top_k=None),
                            n_completions=args.n_completions)

    ROW = namedtuple('ROW', ['query', 'clazz', 'votes', 'steps', 'det', 'se', 'nat', 'hom', 'pos'])
    rows = list()

    df = pd.read_excel(args.dataset)

    # main classifier logic
    counter = 0
    for row in df.itertuples():
        query = row.sentence
        print(f"{query=}")
        votes = cotsc.run(query=query)
        if len(votes) <= 0:
            print(f"No vote results returned.\n{votes=}", file=sys.stderr)
            continue
        best = votes[0]
        clz, stats = best
        num_votes = stats.get('votes')
        steps = stats.get('steps')
        steps_str = '\n'.join(steps)
        r = ROW(query=query, clazz=clz, votes=num_votes, steps=steps_str,
                det=row.det, se=row.se, nat=row.nat, hom=row.hom, pos=row.pos)
        rows.append(r)
        counter += 1
        # if counter > 10:
        #     print("Done.")
        #     break

    results = pd.DataFrame(rows)
    results.to_excel('./cotsc-results.xlsx', index=False)
    print("results wrote to ./cotsc-results.xlsx")
    print(results)
