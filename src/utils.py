import pandas as pd
import json

def prepare_dataset(path, include_oos=False):
    with open(path, "r") as f:
        data = json.loads(f.read())
    
    data_formatted = {"datatype": [], "query": [], "intent": []}
    for datatype, queries in data.items():
        for query_intent in queries:
            query, intent = query_intent
            if (intent != "oos") or include_oos:
                data_formatted["datatype"] += [datatype]
                data_formatted["query"] += [query]
                data_formatted["intent"] += [intent]
                
    df = pd.DataFrame(data_formatted)
    
    intents = df.intent.unique()
    intents_dict = dict((intent, i) for i, intent in enumerate(intents))
    df["intent_idx"] = df.intent.apply(lambda x: intents_dict[x])

    return df, intents_dict

if __name__ == "__main__":
    path = "../oos-eval/data/data_full.json"
    data, intents_dict = prepare_dataset(path)
    print(intents_dict)

    print(data)