import os
import json

def main():
    with open("data/state_name.json", "r") as f:
        data = json.load(f)
    
    # Iterate through the dictionary and print its structure
    print("Data structure:")
    print("=" * 50)
    
    key = 'Name a country in Europe that is located in the Central European Time zone. Only provide the answer without explanation or punctuation.'
    print(data[key]['answers'])
    print(len(data[key]['answers']))
    
    # Save the key list to state_name.txt
    # with open("data/state_name.txt", "w") as f:
    #     for key in key_list:
    #         f.write(key + "\n")
    

if __name__ == "__main__":
    main()