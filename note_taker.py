import openai
import os
import math

openai.api_key = os.getenv("OPENAI_API_KEY")

file_name = input("Enter name of transcription file: ")
if file_name == "":
    file_name = "RussianRev_transcription.txt"
transcript_file = open(file_name, "r")
transcription_text = ""

for i in transcript_file:
    if (i == "\n"):
        continue
    transcription_text += i

transcript_file.close()

length_of_transcription = len(transcription_text)

max_char_length_api = 5500
max_word_length_api = math.ceil(max_char_length_api / 3.1)


transcription_fragments = []

current_text_index = 0

for char in range(1, math.ceil(length_of_transcription / max_char_length_api) + 1):

    if (int(max_char_length_api) * char >= length_of_transcription):
        text_fragment = transcription_text[max_char_length_api *
                                           (char - 1): length_of_transcription]
    else:
        text_fragment = transcription_text[max_char_length_api *
                                           (char - 1): max_char_length_api * char]

    transcription_fragments.append(text_fragment)

if ".txt" in file_name:
    file_name = file_name.replace(".txt", "")

summaries_file = open(file_name+"_summaries.txt", "w+")


def summarize(transcript, word_limit, tolerance=15):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="""

        In paragraph form, write some of the main ideas, specific examples of the main ideas, and definitions of key terms expressed in this transcription. Please write at least {min_words} but not more than {max_words}.

        """.format(max_words=word_limit+tolerance, min_words=word_limit-tolerance) + transcript,
        temperature=0.5,
        max_tokens=1000,
        frequency_penalty=0.0,
        presence_penalty=1
    )

    return response["choices"][0]["text"]


summary_text = ""

for fragment in transcription_fragments:
    summary_number = transcription_fragments.index(fragment)
    summary_text += summarize(fragment, math.ceil(max_word_length_api /
                              len(transcription_fragments))) + "\n"

summaries_file.write(summary_text)
summaries_file.close()


def main_topics(text):
    relevance_prompt = "After each of the identified topics, include a number from 1 to 10 that represents the topic's relevance."

    gpt_main_topics_ask = openai.Completion.create(
        model="text-davinci-003",
        prompt="""

            What are 7 main topics of this text that, on a scale of 1-10 (with 10 being the highest and 1 being the lowest), have a high relevance that is based on the frequency in which they are mentioned/alluded to in the text (organize them in a comma-seperated list)? Do NOT include the relevance rating in the output
            
            """+text,
        temperature=0.0,
        max_tokens=1000,
        frequency_penalty=0.0,
        presence_penalty=1
    )

    text_topics = gpt_main_topics_ask["choices"][0]["text"]

    text_topics = text_topics.replace("\n", "")

    text_topics = text_topics.replace(
        "The seven main topics of this text, in order of relevance, are: ", "")

    text_topics = text_topics.replace(
        "Seven main topics: ", "")

    text_topics = text_topics.replace(
        "Seven Main Topics: ", "")

    text_topics = text_topics.replace(
        "7 Main Topics: ", "")

    numbers_array = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    for i in text_topics:
        if i in numbers_array:
            text_topics.replace(i, "")

    text_topics = text_topics.split(",")

    for i in range(0, len(text_topics)):
        item = text_topics[i]
        if (item[0] == " "):
            text_topics[i] = item[1:]
        if (item[len(item)-1] == " "):
            text_topics[i] = item[1:]

    text_topics = text_topics[0:7]
    print(text_topics)
    return text_topics


def bullet_points_specific(topic, bullet_limit, verbosity):

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="""
        
        Based on the text below, jot down {bullets}-bullet points for the topic of {topic}. Each bullet MUST NOT CONTAIN more than {words} words but should contain at least 6 words. Include the topic as a header and the bullet points underneath the header

        """.format(topic=topic, bullets=bullet_limit, words=verbosity)+summary_text,
        temperature=0.5,
        max_tokens=1000,
        frequency_penalty=0.0,
        presence_penalty=1
    )

    return response["choices"][0]["text"]


def bullet_points_general(main_topics, bullet_limit, verbosity):

    main_topics_string = ""

    for item in main_topics:
        main_topics_string += item
        if main_topics.index(item) != len(main_topics) - 1:
            main_topics_string += ","

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="""

        What are {bullets} details in this text that do NOT conform to these main topics --- {topics}? Include these details as bullet points, each with a maximum of {words} words, underneath the heading General Bullet Points:

        """.format(topics=main_topics_string, bullets=bullet_limit, words=verbosity)+summary_text,
        temperature=0.5,
        max_tokens=1000,
        frequency_penalty=0.0,
        presence_penalty=1
    )

    return response["choices"][0]["text"]


def key_definitions(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="""

        What are the definitions of the key terms covered in this summary? Base these definitions solely off of the text below. Include the key definitions below the heading, Key Definitions.

        """+text,
        temperature=0.0,
        max_tokens=1000,
        frequency_penalty=0.0,
        presence_penalty=1
    )

    return response["choices"][0]["text"]


notes_text = ""


main_topics = main_topics(summary_text)

notes_text += bullet_points_general(main_topics, 8, 10) + "\n\n"

for i in range(0, len(main_topics)):
    topic = main_topics[i]
    notes_text += bullet_points_specific(topic, 6, 10) + "\n\n"

notes_text += "\n" + key_definitions(summary_text) + "\n\n"

if ".txt" in file_name:
    file_name = file_name.replace(".txt", "")

notes_file = open(file_name+"_notes.txt", "w")
notes_file.write(notes_text)
