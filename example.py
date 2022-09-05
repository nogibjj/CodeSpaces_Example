import openai

#write a function that use openai api to answer questions
def answer(question):
    openai.api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    response = openai.Completion.create(
    engine="davinci",
    prompt="This is a test",
    temperature=0.9,
    max_tokens=5,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["\n", " Human:", " AI:"]
    )
    return response.choices[0].text