import json
import random
import litellm
from litellm import Router
import os
import pandas as pd
from tqdm import tqdm
import asyncio

litellm.drop_params=True

model_list = [ {
    "model_name": "gpt-3.5-turbo", 
    "litellm_params": { # params for litellm completion/embedding call 
        "model": "gpt-3.5-turbo", 
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "tpm": 100000,
    "rpm": 1000,
    "max_parallel_requests": 10, # 👈 SET PER DEPLOYMENT

},
{
    "model_name": "openrouter/meta-llama/llama-3-8b-instruct",
    "litellm_params": {
        "model": "openrouter/meta-llama/llama-3-8b-instruct",
        "api_key": os.getenv('OPENROUTER_API_KEY'),
    },
    "tpm": 100000,
    "rpm": 600,
    "max_parallel_requests": 10, # 👈 SET PER DEPLOYMENT

},
{
    "model_name": "openrouter/google/gemma-2-9b-it",
    "litellm_params": {
        "model": "openrouter/google/gemma-2-9b-it",
        "api_key": os.getenv('OPENROUTER_API_KEY'),
    },
    "tpm": 100000,
    "rpm": 600,
    "max_parallel_requests": 10, # 👈 SET PER DEPLOYMENT

},
{
    "model_name": "openrouter/mistralai/mistral-7b-instruct-v0.3",
    "litellm_params": {
        "model": "openrouter/mistralai/mistral-7b-instruct-v0.3",
        "api_key": os.getenv('OPENROUTER_API_KEY'),
    },
    "tpm": 100000,
    "rpm": 600,
    "max_parallel_requests": 10, # 👈 SET PER DEPLOYMENT

},
{
    "model_name": "qwen-14b-chat",
    "litellm_params": {
        "model": "openrouter/qwen/qwen-14b-chat",
        "api_key": os.getenv('OPENROUTER_API_KEY'),
    },
    "tpm": 100000,
    "rpm": 600,
    "max_parallel_requests": 10, # 👈 SET PER DEPLOYMENT

}
]

# model_list = json.load(open("model_prices_and_context_window.json"))


router = Router(
    model_list=model_list,
    routing_strategy="usage-based-routing-v2",
    # enable_pre_call_checks=True, 
    allowed_fails=1,
    cooldown_time=5,
    num_retries = 10,

    )
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

    results = await asyncio.gather(*(wrapped_coroutine(c) for c in coroutines), return_exceptions=True)
    pbar.close()
    return results

def plagiarize(text,model="gpt-3.5-turbo",n=1, mode='wordLevel') -> list:
    """
    Given a text and a model, generate alternative texts
    Modes:
        - wordLevel
        - paraphrase
    """
    if n <= 0:
        raise Exception("n must be greater than 0")
    responses = []
    messages =None
    if mode == 'wordLevel':
    
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Paraphrase the following paragraph using word-level perturbations in the style of the original\n```\nThe games follow the adventures of the hapless Guybrush Threepwood as he struggles to become the most notorious pirate in the Caribbean, defeat the plans of the evil undead pirate LeChuck and win the heart of Governor Elaine Marley. The plots often involve the mysterious Monkey Island and its secrets.\n```\n"
                }
            ]
            },
            {
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": "The games trace the misadventures of the unlucky Guybrush Threepwood as he endeavors to become the most infamous pirate in the Caribbean, thwart the schemes of the wicked undead pirate LeChuck, and capture the affection of Governor Elaine Marley. The storylines frequently revolve around the enigmatic Monkey Island and its mysteries."
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Paraphrase the following paragraph using word-level perturbations in the style of the original\n```\nMr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense.\n```\n"
                }
            ]
            },
            {
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": "Mr. and Mrs. Dursley, of number four Privet Drive, were pleased to assert that they were completely ordinary, thank you very much. They were the least likely individuals you'd imagine to be mixed up in anything unusual or enigmatic, because they simply didn’t go in for such absurdities."
                }
            ]
            },
             {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"Paraphrase the following paragraph using word-level perturbations in the style of the original\n```\n{text}\n```"
                }
            ]
            },
        ]
    elif mode == 'paraphrase':
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Paraphrase the following paragraph in the style of the original\n```\nThe games follow the adventures of the hapless Guybrush Threepwood as he struggles to become the most notorious pirate in the Caribbean, defeat the plans of the evil undead pirate LeChuck and win the heart of Governor Elaine Marley. The plots often involve the mysterious Monkey Island and its secrets.\n```\n"
                }
            ]
            },
            {
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": "The games chronicle the misadventures of the inept Guybrush Threepwood on his quest to become the most infamous pirate in the Caribbean, thwart the schemes of the malevolent undead pirate LeChuck and win the affection of Governor Elaine Marley. The narratives frequently revolve around the enigmatic Monkey Island and its hidden truths."
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Paraphrase the following paragraph in the style of the original\n```\nMr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense\n```"
                }
            ]
            },
            {
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": "Mr. and Mrs. Dursley, who resided at number four, Privet Drive, prided themselves on being completely ordinary, thank you very much. They were the very last individuals one would expect to be mixed up in anything curious or uncanny, as they simply had no tolerance for such rubbish."
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"Paraphrase the following paragraph in the style of the original\n```\n{text}\n```"
                }
            ]
            },
        ]
    else:
        raise Exception("mode must be 'wordLevel' or 'paraphrase'")
    
    for i in range(n):
        responses.append( router.acompletion(
        model=model,
        messages=messages,
        temperature=1,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        ))
    return responses

async def generate_alternatives(input_file_name, output_file_name,  models_to_plagiarize=['gpt-3.5-turbo'], generations_per_model=3,mode='wordLevel') -> float:
    """
    Given a text and a list of models, generate alternative texts
    """
    input_df = pd.read_csv(input_file_name)
    df_grouped = input_df.groupby("Name")
    rows = []
    result = {'books':[]}
    if mode not in ['wordLevel','paraphrase']:
        raise Exception("mode must be 'wordLevel' or 'paraphrase'")

    if mode == 'wordLevel':
        print(f'word')
    elif mode == 'paraphrase':
        print(f'paraphrase')
    
    for name, group in df_grouped:
        responses = []
        texts = []
        for i, row in group.iterrows():
            for model in models_to_plagiarize:
                alts = plagiarize(row["Text"], model, generations_per_model,mode=mode)
                responses.extend(alts)
                texts.extend([row["Text"] for _ in range(len(alts))])
        results = await run_with_progress(responses,f'Alternate texts for {name}')
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            print(f'Encountered {len(errors)} errors out of {len(results)}')
        results = [r for r in results if not isinstance(r, Exception)]
        result['books'].append({'name':name,'texts':{},})
        # result['books'][-1]['content'] 
        print(results)

        for i in range(len(results)):
            
            model = results[i].model
            response = results[i].choices[0].message.content
            text = texts[i]
            if text not in result['books'][-1]['texts']:
                result['books'][-1]['texts'][text] = []
            result['books'][-1]['texts'][text].append({'text':response,'type':'text','model':model,'real':False})
            # for choice in choices:
            #     result['books'][-1]['texts'][text].append({'text':choice.message.content,'type':'text','model':model,'real':False})
        json_string = json.dumps(result, indent="\t")

        with open(output_file_name, 'w') as file:
            file.write(json_string)

        
    return result  

async def answer_question(question,choices, model="gpt-3.5-turbo", retries=5) -> int:
    """
    Given a question, answer it
    """
    result = await router.acompletion(
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
    answer = find_answer(choices) + 1   
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
    
def find_answer(list_of_dicts):
    for index, d in enumerate(list_of_dicts):
        if d.get("real") == True:
            return index
    return -1 

def build_multi_choice_prompt(options,name, content_type) -> str:

    """Builds a multiple choice question prompt"""
    prompt =f"""One of the following choices below appears in the {content_type} {name}, it is your job to choose the correct answer.\n"""

    for i, option in enumerate(options):
        prompt += f"\t[{i+1}] {option["text"]}\n"
    prompt += f"please answer with a number between 1 and {len(options)} inclusive surrounded by brackets in the beginning of your answer."
    return prompt.strip()


async def benchmark(input_file_name, output_file_name,  model_to_test='openrouter/meta-llama/llama-3-8b-instruct') -> float:
    """
    Given a text and a list of models, generate alternative texts
    """
    js = json.load(open(input_file_name))
    means = []
    for book in js['books']:
        name = book['name']
        texts = book['texts']
        responses = []
        for original, choices in texts.items():
            choices.append(	{
						"text": original,
						"type": "",
						"model": "None",
						"real": True
					},)
            random.shuffle(choices)

            prompt = build_multi_choice_prompt(choices,name,'book')
            responses.append(answer_question(prompt, choices, model=model_to_test))

        results = await run_with_progress(responses,f'Benchmarking {name}')


        errors = [r for r in results if isinstance(r, Exception)]
        valid_results = [r for r in results if not isinstance(r, Exception)]
        if errors:
            print(f'Encountered {len(errors)} errors out of {len(results)}')
            print("grading as if wasn't in benchmark")
        right_answers = sum(r['grade'] for r in valid_results)
        if len( valid_results) == 0:
            print('no valid results')
            break
        
        mean = right_answers / len(valid_results)
        means.append(mean)
        print(f'score on {name}: {mean}')

    if len(means) == 0:
        return
    print(f"total mean: {sum(means) / len(means)}")