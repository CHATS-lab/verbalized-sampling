import os
import json
import csv
from ast import literal_eval
import random

# def main():
#     with open("data/state_name.json", "r") as f:
#         data = json.load(f)

#     print(len(data))

#     count = 0
#     key_list = []
#     for idx, (key, value) in enumerate(data.items()):
#         # print(key)
#         if len(value['answers']) >= 30:
#             count += 1
#             key_list.append(key)
#     print(count)

    
#     # key = 'Name a country in Europe that is located in the Central European Time zone. Only provide the answer without explanation or punctuation.'
#     # print(data[key]['answers'])
#     # print(len(data[key]['answers']))
    
#     # Save the key list to state_name.txt
#     with open("data/state_name.txt", "w") as f:
#         for key in key_list:
#             f.write(key + "\n")
    


# simple_qa.csv
def main():
    data = []
    with open("data/simple_qa.csv", "r", encoding="utf-8") as f:
        csv_reader = csv.DictReader(f)
        data = list(csv_reader)
    
    print(len(data))
    
    topic_list = []
    topic_question_list = {}
    for row in data:
        metadata = row['metadata']
        if isinstance(metadata, str):
            metadata = literal_eval(metadata)
        topic = metadata['topic']
        topic_list.append(topic)
        question = row['problem']
        if topic not in topic_question_list:
            topic_question_list[topic] = []
        topic_question_list[topic].append(question)

    topic_list = list(set(topic_list))
    # for topic in topic_list:
    #     print(f"Topic: {topic}")
    #     print(f"Number of questions: {len(topic_question_list[topic])}")
    #     print("-" * 50)


    # Set random seed for reproducibility
    random.seed(42)

    # Select 50 random problems from each topic
    selected_problems = []
    for topic in topic_list:
        questions = topic_question_list[topic]
        if len(questions) >= 30:
            # Randomly select 50 questions from this topic
            selected = random.sample(questions, 30)
        else:
            # If less than 50 questions available, take all of them
            selected = questions
            print(f"Warning: Topic '{topic}' only has {len(questions)} questions")
        
        # Add selected questions to the list
        for question in selected:
            selected_problems.append(question)

    print(f"Total selected problems: {len(selected_problems)}")

    # # Save selected problems to file
    # with open("data/simple_qa.txt", "w", encoding="utf-8") as f:
    #     for problem in selected_problems:
    #         f.write(problem + "\n")

    # print(f"Saved {len(selected_problems)} problems to data/simple_qa.txt")


if __name__ == "__main__":
    main()