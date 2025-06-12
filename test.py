import pandas as pd


def get_answer_by_question(df, question):
    print(question)
    question = question.strip()
    print(question)
    for index, row in df.iterrows():
        if row["problem"] == question:
            return row["answer"]
    return None


def main():
    df = pd.read_csv("data/simple_qa.csv")
    print(df.head())
    examples = [row['problem'] for _, row in df.iterrows()]
    print(len(examples))
    with open("data/simple_qa.txt", "w") as f:
        for example in examples:
            f.write(example + "\n")



    # for index, row in df.iterrows():
    #     print(row["metadata"])
    #     print(row["problem"])
    #     print(row["answer"])
    #     print("-" * 100)

    # question = """
    # What month and year was Miranda Lambert's album "Revolution" certified platinum by the RIAA?
    # """
    # answer = get_answer_by_question(df, question)
    # print(answer)

if __name__ == "__main__":
    main()