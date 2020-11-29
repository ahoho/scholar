####################################################################
#
# python add_label_element.py < in.jsonl > out.jsonl
#
#  Takes in jsonlines and adds key/value pair {"dummy_label":"0"}
#  to every line so that there is always a label for every document
#
####################################################################
import json
import sys

for line in sys.stdin:
    obj                = json.loads(line)
    obj['dummy_label'] = "0"
    new                = json.dumps(obj)
    print(new)

    
