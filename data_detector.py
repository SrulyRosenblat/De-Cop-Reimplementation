import random
import pandas as pd
from tqdm import tqdm
import litellm
from litellm import completion
litellm.drop_params=True
import time


def plagiarize(text,model="gpt-3.5-turbo",n=1) -> list:
    """
    Given a text and a model, generate alternative texts
    """
    if n <= 0:
        raise Exception("n must be greater than 0")
    response = completion(
    model=model,
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Paraphrase the following paragraph\n```\nThe games follow the adventures of the hapless Guybrush Threepwood as he struggles to become the most notorious pirate in the Caribbean, defeat the plans of the evil undead pirate LeChuck and win the heart of Governor Elaine Marley. The plots often involve the mysterious Monkey Island and its secrets.\n```"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "The games chronicle the misadventures of the unlucky Guybrush Threepwood as he aspires to be the most infamous pirate in the Caribbean, thwart the schemes of the wicked undead pirate LeChuck, and capture the affections of Governor Elaine Marley. The storylines frequently revolve around the enigmatic Monkey Island and its hidden secrets."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Paraphrase the following paragraph\n```\nYou had a data interpretation problem, so you tried clustering. Now you have a cluster interpretation problem! There was a suspicion that patterns might exist in the data. Reasonably, the hope was that adding some structure through unsupervised learning would lend some insights. Clusters are the go-to tool for finding structure. Thus, you embarked on your journey. You spend considerable money on computing. You invest a lot of sweat in fiddling with cluster tuning parameters. Just to be sure, you try a few algorithms. But at the end of the day you’re left with rainbow plots of clustered data that might have some meaning — just maybe — if you squint hard enough. You go home with an uneasy suspicion that it was all for naught. Sadly, this is too often the case. Why should this be though?\n```"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "You faced a data interpretation challenge and decided to use clustering as a solution. Unfortunately, this led to a new issue: interpreting the clusters themselves. There was an initial belief that patterns might be present in the data, and it seemed reasonable to think that using unsupervised learning to add some structure would provide insights. Clustering is typically the method of choice for uncovering structure, so you proceeded with it. You spent significant funds on computing resources and put in a lot of effort adjusting the clustering parameters. To cover all bases, you experimented with several algorithms. Yet, at the end of the process, you were left with colorful plots of clustered data that seemed to hold meaning only if you strained to see it. You went home with a nagging feeling that the effort might have been in vain. Unfortunately, this scenario is all too common. But why does this happen so often?"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Paraphrase the following paragraph\n```\n{text}\n```"
            }
        ]
        },
    ],
    temperature=1,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    n=n
    )
    return [a.message.content for a in response.choices]

def answer_question(question,n_choices, model="gpt-3.5-turbo", retries=3) -> int:
    """
    Given a question, answer it
    """
    while True:
        try:
            response = completion(
            model=model,
            messages=[
                {
                "role": "user",
                "content": question
                }
            ],
            temperature=0.1,
            max_tokens=10,
            top_p=1,
            frequency_penalty=0,
            stop=["]", ")"]
            )
            response = response.choices[0].message.content
            parsed_response = parse(response, n_choices)
            return parsed_response
        except Exception as e:

            if retries == 0:
                raise e
            retries -= 1
            time.sleep(1)



def generate_alternatives(text, models, generations_per_model=3) -> list:
    """
    Given a text and a list of models, generate alternative texts
    """
    text_options = []

    for model in models:
        for variation in plagiarize(text, model, generations_per_model):
            text_options.append({"model": model, "text": variation, "real_text": False})
    
    text_options.append({"model": "None", "text": text, "real_text": True})
    random.shuffle(text_options)
    return text_options

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



def build_multi_choice_prompt(options,name, content_type) -> str:

    """Builds a multiple choice question prompt"""
    prompt =f"""One of the following choices below appears in the {content_type} {name}, it is your job to choose the correct answer.\n"""

    for i, option in enumerate(options):
        prompt += f"\t[{i+1}] {option["text"]}\n"
    prompt += f"please answer with a number between 1 and {len(options)} inclusive surrounded by brackets in the beginning of your answer."
    return prompt.strip()


def benchmark(csv_file_name, content_type="text", test_model="gpt-3.5-turbo", models_to_plagiarize=['gpt-3.5-turbo'], generations_per_model=3, guess_chance=0,generations_file_name="generations.csv",wait_between_generations=.1) -> float:
    """
    Given a csv file, benchmark the models
    """

    df = pd.read_csv(csv_file_name)
    generations = []
    
    total = 0

    groups = df.groupby("Name")

    for name, group in groups:
        item_count = len(group)
        item_total = 0
        print(f"Benchmarking {name}")
        for i, row in tqdm(group.iterrows(), total=group.shape[0]):
            alternatives = generate_alternatives(row["Text"], models_to_plagiarize, generations_per_model)
            # print(row["Text"], alternatives)
            prompt = build_multi_choice_prompt(alternatives,name, content_type)
            time.sleep(wait_between_generations)
            answer = answer_question(prompt, len(alternatives), test_model)
            grade = grade_choice(answer, alternatives)
            generations.append({"Name":name,"Options":alternatives, "Question":prompt, "Answer":answer, "Model":test_model, "Grade":grade})
            item_total += grade
            total += grade
            time.sleep(wait_between_generations)
                
        
        if guess_chance > 0:
            print(f"Score: {item_total/item_count} Score - Guess Chance: {item_total/item_count - guess_chance}")
        else:
            print(f"Score: {item_total/item_count}")
    generations = pd.DataFrame(generations)
    generations.to_csv(generations_file_name)
    print(f"Total Score: {total/len(df)}, Score - Guess Chance: {total/len(df) - guess_chance}")
    return total / len(df)
