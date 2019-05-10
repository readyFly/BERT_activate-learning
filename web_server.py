# coding:utf-8
from flask import Flask
from flask import request,jsonify
from flask_cors import CORS
import json
from read_txt import predict

test_path = "./glue_data/EE/the_last_data/single_data/test.txt"
txt_output_path = "./1.ann"

app = Flask("my_web")

@app.route("/",methods=['GET', 'POST'])

def hello():

    json_resp = {}
    # for example:
    # json_resp = {"1": {"offsets": [[43, 49]], "type": "TITLE", "texts": ["Oracle"]},
    #              "2": {"offsets": [[88, 96]], "type": "Start-Org", "texts": ["starting"]}}

    data_text = request.data.decode('utf8')

    with open(test_path,"w",encoding="utf-8")as f:
        f.write(data_text)

    predict(test_path, txt_output_path) # model predict 

    with open(txt_output_path, "r", encoding="utf-8") as fr:
        content = fr.read()
        lines = content.split("\n")
        cid = 1
        for line in lines:
            if len(line) > 0:
                type = line.split("\t")[1].split(" ")[0]
                offsets = []
                offsets.append(int(line.split("\t")[1].split(" ")[1]))
                offsets.append(int(line.split("\t")[1].split(" ")[2]))
                text = line.split("\t")[2]
                if cid not in json_resp:
                    json_resp[cid] = {"offsets": [offsets], "type": type, "texts": [text]}
                cid = cid + 1

    for cid, ann in ((i, a) for i, a in json_resp.items()):

        offsets = ann['offsets']
        _type = ann['type']
        texts = ann['texts']

        print(offsets, _type, texts)

    return jsonify(json_resp)

CORS(app, supports_credentials=True)
app.run(host='0.0.0.0', port=8000)

