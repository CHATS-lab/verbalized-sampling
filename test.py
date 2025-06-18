import os
import json

def main():
    with open("data/CoverageQA.json", "r") as f:
        data = json.load(f)
    
    # Iterate through the dictionary and print its structure
    print("Data structure:")
    print("=" * 50)
    
    key_list = []
    for key, value in data.items():
        print(f"Key: {key}")
        key_list.append(key)
        # print(f"Type: {type(value)}")
        # print(f"Value: {value}")
        # print("-" * 30)
    
    # Save the key list to state_name.txt
    with open("data/state_name.txt", "w") as f:
        for key in key_list:
            f.write(key + "\n")
    

if __name__ == "__main__":
    main()