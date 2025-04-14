import json

def check_unique_query_ids(file_path):
    query_ids = set()
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            query_id = data.get("query_id")
            if query_id in query_ids:
                print(f"Duplicate query_id found: {query_id}")
                return False
            query_ids.add(query_id)
    print("All query_ids are unique.")
    return True

if __name__ == "__main__":
    file_path = "data/mad_data_for_cone/data/mad_data/test.jsonl"
    check_unique_query_ids(file_path)