from huggingface_hub import InferenceClient
import gradio as gr
import json
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

options = Options()
options.headless = True
driver = webdriver.Chrome(options=options)

driver.get("http://huggingface.co/docs")

def extract_urls(text):
    # Extended regular expression to find URLs within various contexts including angle brackets
    url_pattern = r'https?://[\w\-._~:/?#\[\]@!$&\'()*+,;=%]+'
    # Extract URLs that might be within angle brackets
    bracketed_urls = re.findall(r'<(' + url_pattern + ')>', text)
    # Extract normal URLs not in brackets
    normal_urls = re.findall(url_pattern, text)
    # Combine both lists, avoiding duplicates
    all_urls = list(set(bracketed_urls + normal_urls))
    return all_urls

with open('hf.json', 'r') as f:
    links = json.load(f)

for i in range(len(links)):
    links[i] = links[i][:-1]
    
client = InferenceClient(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    token='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'
    # "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
)

generate_kwargs = dict(
        temperature=0.5,
        max_new_tokens=1000,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True
        # seed=42,
    )

def hf_docs(prompt):
    formatted_prompt = f'''<s>[INST] This is the list of available documentations {str(links)}\n\nTell me the url or urls of the most suitable documentation for the given task.\nTask: {prompt} [/INST]'''

    response = client.text_generation(formatted_prompt, **generate_kwargs, details=True, return_full_text=False)['generated_text']

    urls = extract_urls(response)
    # print(urls)

    docs = ''
    for url in urls:
        text= ''
        driver.get(url)
        try:
            x = driver.find_element(By.CLASS_NAME, 'prose-doc')
        except:
            continue
        driver.implicitly_wait(0.4)
        try:
            if driver.find_elements(By.CSS_SELECTOR, '.absolute.leading-tight.bg-black.text-gray-200.rounded-xl.bottom-12.ring-offset-2'):
                buttons = driver.find_elements(By.CSS_SELECTOR, '.absolute.leading-tight.px-3.bg-black.text-gray-200.rounded-xl.bottom-12.ring-offset-2')
                for button in buttons:
                    button.click()
        except:
            pass
        driver.implicitly_wait(0.4)
        elements = driver.find_element(By.CLASS_NAME, 'prose-doc').find_elements(By.CSS_SELECTOR, "p, h1, h2, h3, h4, h5, h6, ul, pre")
        for element in elements: 
            if element.tag_name == 'ul':
                list_items = element.find_elements(By.TAG_NAME, "li")
                for item in list_items:
                    text += '- ' + item.text + '\n'
            elif element.tag_name =='pre':
                t = element
                text = text + f'```\n{element.text}\n```\n'
            else:
                text += element.text + '\n'
        # print(text)
        docs = docs + text + '\n\n'
    return docs