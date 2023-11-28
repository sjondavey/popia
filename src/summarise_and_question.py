# Update the relevant summary row
from openai import OpenAI
client = OpenAI()

system_content_summerise = "You are summarising sections of the 'Protection of Personal Information Act' (POPIA) for a 'responsible party'. When summerising, do not add filler words like 'the act says ...' or 'section x says ...', just provide the summary, using the minimum amount of legalese, of the section without any context or explanation. Try to minimise the use of lists or bullet points and use paragraphs instead.\n\
Note: In this act the term 'data subject' means the person (natural or, where applicable, juristic) to whom personal information relates. Where possible, try to avoid using the term 'data subject' in the summary. Rather use words like 'client' or 'customer'\n\
Note: 'responsible party' is defined as a public or private body or any other person which, alone or in conjunction with others, determines the purpose of and means for processing personal information; Try to avoid using the words 'responsible party' in the summary. Instead use words like 'you'"

# system_content_summerise = "You are summarising sections of the 'Protection of Personal Information Act' (POPIA) for a 'responsible party'. When summerising, do not add filler words like 'the act says ...' or 'section x says ...', just provide the summary, using the minimum amount of legalese, of the section without any context or explanation. Try to minimise the use of lists or bullet points and use paragraphs instead.\n\
# Note: In this act the term 'data subject' means the person (natural or, where applicable, juristic) to whom personal information relates. Where possible, try to avoid using the term 'data subject' in the summary. Rather use words like 'client' or 'customer'"


system_content_question = "You are assisting a 'responsible party' to comply with the 'Protection of Personal Information Act' (POPIA) by preparing a set of Frequently Asked Questions (FAQs). You will be provided with the Answer, your role to create one or two Questions for the Answer. The questions should be high level and not focus on any detail in the answer. Do not use the phrases like 'According to POPIA ...' or 'Under the act ...'. List your questions as a pipe delimited string."


def get_summary_and_questions_for(text, model):

    user_context = text
    response = client.chat.completions.create(
                        model=model,
                        temperature = 1.0,
                        max_tokens = 500,
                        messages=[
                            {"role": "system", "content": system_content_summerise},
                            {"role": "user", "content": user_context},
                        ]
                    )
    summary = response.choices[0].message.content


    user_context_question = summary
    response = client.chat.completions.create(
                        model=model,
                        temperature = 1.0,
                        max_tokens = 500,
                        messages=[
                            {"role": "system", "content": system_content_question},
                            {"role": "user", "content": user_context_question},
                        ]
                    )
    questions = response.choices[0].message.content

    return summary, questions

