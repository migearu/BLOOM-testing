from huggingface_hub import inference_api

inference = inference_api.InferenceApi("bigscience/bloom");
import time

def find_nth(string, tofind, n):
    start = string.find(tofind)
    while start >= 0 and n > 1:
        start = string.find(tofind, start+len(tofind))
        n -= 1
    return start

def infer(prompt,
          max_length = 32,
          top_k = 0,
          num_beams = 0,
          no_repeat_ngram_size = 2,
          top_p = 0.9,
          seed=42,
          temperature=0.7,
          greedy_decoding = False,
          return_full_text = False):
    

    top_k = None if top_k == 0 else top_k
    do_sample = False if num_beams > 0 else not greedy_decoding
    num_beams = None if (greedy_decoding or num_beams == 0) else num_beams
    no_repeat_ngram_size = None if num_beams is None else no_repeat_ngram_size
    top_p = None if num_beams else top_p
    early_stopping = None if num_beams is None else num_beams > 0

    params = {
        "max_new_tokens": max_length,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": do_sample,
        "seed": seed,
        "early_stopping":early_stopping,
        "no_repeat_ngram_size":no_repeat_ngram_size,
        "num_beams":num_beams,
        "return_full_text":return_full_text
    }
    
    s = time.time()
    response = inference(prompt, params=params)
    #print(response)
    proc_time = time.time()-s
    print(f"Processing time was {proc_time} seconds")
    return response

# formats a prompt, and questions for a review, and returns the formatted string that can be used by the inference API to generate a response

def review(prompt, questions):
    questionsFormatted = '';
    for q in range(len(questions)): # format the questions
        questionsFormatted += (f"{q+1}: {questions[q]}\n")
    prompt = f'**Review:**\n{prompt}\n\n**Questions:**\n{questionsFormatted}\n\n**Answers:**\n1.' # format the prompt
    return prompt # return the prompt

# formats a question into a form that can be used by the inference API to answer the question

def answer(question): # Set greedy_decoding to True to get more accurate answers when you call infer()
    prompt = f'**Question:**\n{question}\n\n**Answer:**\n' # format the prompt
    return prompt # return the prompt

# the primer basically helps to get the conversation started, like the overall feeling of the bot.
# primer should be formatted like this:
# [
#     {
#         "You": "How are you doing?",
#         "Bot": "I'm doing fine, thanks for asking."
#     },
#     {
#         "You": "I feel like going outside today.",
#         "Bot": "That's great!"
#     }
# ]

def chat(primer, chat_turns):
    # prepare the primer
    primer_string = ''
    for i in range(len(primer)):
        primer_string += (f"You: {primer[i]['P']}\nBot: {primer[i]['A']}\n")
    print(primer_string)
    # start the chat
    for i in range(chat_turns):
        prompt = input("You: ") # get the user's input
        response = infer(primer_string + f"You: {prompt}\nBot: ", max_length=100) # get the bot's response
        resp = response[0]['generated_text']
        resp_formatted = resp[resp.find(f"You: {prompt}")+len(f"You: {prompt}")+1:find_nth(resp, 'You:', i + len(primer) + 2)] # format the response
        print(resp_formatted) # print the response
        primer_string += f"You: {prompt}\n" + resp_formatted # add the response to the primer string (to remember the conversation, to leverage later responses)

def main():
    # to use: uncomment one of the examples below, and run the script.
    # requirements: huggingface_hub, to install: pip install huggingface_hub

    # CHATBOT EXAMPLE
    # primer = [
    #     {
    #         "P": "How are you doing?",
    #         "A": "I'm doing fine, thanks for asking."
    #     },
    #     {
    #         "P": "I feel like going outside today.",
    #         "A": "That's great!"
    #     }
    # ]
    # chat(primer, 5)

    # QUESTION ANSWER EXAMPLE
    # prompt = """What is the Periodic Table?"""
    # input = answer(prompt)
    # response = infer(input, max_length=50, greedy_decoding=True) # notice that greedy_decoding is set to True, to get more accurate answers

    # REVIEW EXAMPLE
    # prompt = """Google Translate est un service de traduction automatique neuronale multilingue développé par Google pour traduire du texte, des documents et des sites Web d'une langue à une autre. Il offre une interface de site Web, une application mobile pour Android et iOS et une API qui aide les développeurs à créer des extensions de navigateur et des applications logicielles."""
    # questions = ["What is the topic about?", "Who was the creator of Google Translate?", "What is the purpose of Google Translate?", "Does it have a website?"]
    # input = review(prompt, questions)
    # response = infer(input, max_length=100)
    # print(response[0]['generated_text'])

    # personal note: this is really impressive, it reads the french text, and formulates it's answers in english...

if __name__ == "__main__": # this program will error if you don't uncomment one of the examples above
    main()