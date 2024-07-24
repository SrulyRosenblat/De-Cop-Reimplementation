import json
import random
import litellm
from litellm import acompletion
import os
import pandas as pd
from tqdm import tqdm
import asyncio


import asyncio
import time
import random


class AsyncRateLimiter:
    def __init__(self, max_requests_per_minute=60, max_concurrent=10, max_retries=10, base_delay=1):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.request_times = []
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(self, coro):
        async with self.semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    await self._wait_for_rate_limit()
                    
                    start_time = time.time()
                    result = await coro
                    end_time = time.time()

                    self.request_times.append(end_time)
                    return result
                except Exception as e:
                    if attempt == self.max_retries:
                        return Exception(f"Failed after {self.max_retries} retries: {str(e)}")
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Task failed. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)

    async def _wait_for_rate_limit(self):
        current_time = time.time()
        
        self.request_times = [t for t in self.request_times if current_time - t <= 60]
        
        if len(self.request_times) >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def process_batch(self, coros):
        tasks = [self.execute(coro) for coro in coros]
        results = await asyncio.gather(*tasks)
        return results





# model_list = json.load(open("model_prices_and_context_window.json"))


rate_limiter = AsyncRateLimiter(500, 20, 10, 2)


async def run_with_progress(coroutines,name=""):
    total = len(coroutines)
    pbar = tqdm(total=total, desc=f"{name} progress")
    
    async def wrapped_coroutine(coro):
        try:
            result = await coro
        except Exception as e:
            result = e
        pbar.update(1)
        return result

    results = await rate_limiter.process_batch([wrapped_coroutine(c) for c in coroutines])
    print(results)
    pbar.close()
    return results


async def benchmark(input_file_name, output_file_name,statistics_file_name,  models_to_test=[]) -> float:
    """
    Given a text and a list of models, generate alternative texts
    """
    df = pd.read_csv(input_file_name)
    for model in models_to_test:
        responses = []
        rows = [(i, row) for i, row in df.iterrows()]
        for i, row in tqdm(rows):
            prompt = row["Prompt"]
            choices = [row["Example_A"], row["Example_B"], row["Example_C"], row["Example_D"]]
            answer = row["Answer"]
            responses.append(answer_question(prompt, choices, answer, model))

        results = await run_with_progress(responses,f'Benchmarking {model}')
        scores = []
        guess = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                scores.append(None)
                guess.append(None)
            else:
                scores.append(r['grade'])
                guess.append(chr(ord('A') -1 + r['response']))
        model_str = model.split("/")[-1]

        df[f'Model_{model_str}_Score'] = scores
        df[f'Model_{model_str}_Guess'] = guess
        print(f"Model: {model_str} done, waiting 60 seconds")
        time.sleep(60)
    df.to_csv(output_file_name)
    for model in models_to_test:
        model_str = model.split("/")[-1]
        print(f"Model: {model_str}")
        print(df[f'Model_{model_str}_Score'].mean())
    
    

async def answer_question(question,choices,answer, model="gpt-3.5-turbo", retries=5) -> int:
    """
    Given a question, answer it
    """
    result = await acompletion(
        model=model,
        messages=[
            {
            "role": "user",
            "content": question
            }
        ],
        temperature=0,
        max_tokens=2,
        top_p=1,
        frequency_penalty=0,
        stop=["]", ")"],
        
    )

    response = result.choices[0].message.content
    parsed_response = parse(response, len(choices))
    answer = ord(answer)-ord('A')+1
    return {'prompt':question,"response":parsed_response, 'correct_answer':answer,"grade": 1 if answer == parsed_response  else 0}

def parse(text,n_choices) -> int:
    """parses the model response to a float"""
    text = text.strip()
    parsed_answer = int(text[1:])
    if parsed_answer == 0 or parsed_answer > n_choices:
        raise Exception("Invalid answer")
    return parsed_answer
    
def grade_choice(answer:int, options:list) -> bool:
    """Grades the answer based on the options"""
    return options[answer - 1]["real_text"]


# async def benchmark(input_file_name, output_file_name,  model_to_test='openrouter/meta-llama/llama-3-8b-instruct') -> float:
#     """
#     Given a text and a list of models, generate alternative texts
#     """
#     js = json.load(open(input_file_name))
#     means = []
#     for book in js['books']:
#         name = book['name']
#         texts = book['texts']
#         responses = []
#         for original, choices in texts.items():
#             choices.append(	{
# 						"text": original,
# 						"type": "",
# 						"model": "None",
# 						"real": True
# 					},)
#             random.shuffle(choices)

#             prompt = build_multi_choice_prompt(choices,name,'book')
#             responses.append(answer_question(prompt, choices, model=model_to_test))

#         results = await run_with_progress(responses,f'Benchmarking {name}')


#         errors = [r for r in results if isinstance(r, Exception)]
#         valid_results = [r for r in results if not isinstance(r, Exception)]
#         if errors:
#             print(f'Encountered {len(errors)} errors out of {len(results)}')
#             print("grading as if wasn't in benchmark")
#         right_answers = sum(r['grade'] for r in valid_results)
#         if len( valid_results) == 0:
#             print('no valid results')
#             break
        
#         mean = right_answers / len(valid_results)
#         means.append(mean)
#         print(f'score on {name}: {mean}')

#     if len(means) == 0:
#         return
#     print(f"total mean: {sum(means) / len(means)}")