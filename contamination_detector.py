import requests

def search(query,start_date = '1900-01-01',end_date = '2022-01-01'):
    url = f"https://api.search.brave.com/res/v1/web/search?q={query}&text_decorations=0&result_filter=web&freshness={start_date}to{end_date}"

    payload = {}
    headers = {
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip',
    'X-Subscription-Token': 'BSA1tOCcAYf12yW3A0yHj1BWJ6Ff8Ni',
    'Cookie': 'search_api_csrftoken=.eJwFwdsSQkAAANB_8b4z2l2lR8PkErGFVS9mo9W65B75-s6R3pe8Q6uP7k0BzcnYe-vWU-pVdNQPysKWsdTDJwcVNChLnnYX7EUQu_jhlxWeFWyB-uRkai2aFEaY93DHlDxsEfHkY0yGGyAazs0i-6U9sGzdTqYP3wTMEOuBs5Wa2kbNdxJYhutr4JvBVHKmu8qZrqSGIrZmBIjL-StZhC39AXgqOVg._-I84lPp-MjzA-7XCxthDNNumMk'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    return response.json()

def find_similarity_score(original, suspected, threshold = .90, min_to_search = .40) -> float:
    max_match = SequenceMatcher(None, original, suspected).ratio()
    if max_match < min_to_search:
        return max_match
    for i in range(len(suspected)):
        for j in range(i,len(suspected)):
            match = SequenceMatcher(None, original, suspected[i:j]) 
            ratio = match.ratio()
            max_match = max(max_match, ratio)
            if max_match > .90:
                return max_match
    return max_match

def search_for_contamination(text, threshold = .90, min_to_search = .40) -> float:
    results = search(text)
    original_text = text
    max_similarity = 0
    for result in results['web']['results']:
        title = result['title']
        description = result['description']
        max_similarity = max(max_similarity,find_similarity_score(original_text,title,min_to_search), find_similarity_score(original_text,description,min_to_search))
        if max_similarity > .90:
            break
    return max_similarity * 100
    

    
