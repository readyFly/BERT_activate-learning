# read a txt file and change it to test.txt perform
import os
from stanfordcorenlp import StanfordCoreNLP

def get_pos(token, nlp_res):
    token_list = []
    res_pos_list = []
    contect_split = nlp_res.split("Tokens:\n")

    if len(contect_split) == 1:
        # print("can't find token:  " + token)
        # print (nlp_res)
        res_pos = "None"
    else:
        for item in contect_split[1].split("\n"):
            if item != "":
                token = item.split("Text=")[1].split(" ")[0]
                res_pos = item.split("PartOfSpeech=")[1].replace("]", "")
                token_list.append(token)
                res_pos_list.append(res_pos)
    return token_list, res_pos_list

def select_file(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            if filename[-4:] == ".txt":
                apath = os.path.join(maindir, filename)
                result.append(apath)

    return list(set(result))


# input_dir = "/home/feng/brat-v1.3_Crunchy_Frog/data/ace_input/"
# output_dir = "/home/feng/brat-v1.3_Crunchy_Frog/data/ace_output/"
# test_data_path = "/home/feng/bert_code_1/glue_data/EE/the_last_data/single_data/test000.txt"
#
# res = select_file(input_dir)
# for path in res:
#     with open(path,"r",encoding="utf-8") as fr:
#         all_content = fr.read()
#         content = all_content.replace("\n", "").split("\n")
#         print("content:",content)
#     # os.system("mv " + path + " " + output_dir)
#
#     nlp = StanfordCoreNLP(r'/home/feng/stanford-corenlp-full-2018-10-05/')
#     props_sentence = {'annotators': 'pos', 'pipelineLanguage': 'zh', 'outputFormat': 'text'}
#
#     with open(test_data_path, 'w',encoding="utf-8")as f1:
#         f1.write('sentence_id' + '\t' + '/' + '\t' + 'token' + '\t' + 'token_mark' + '\t' + 'pos' + '\t' + 'event_type' + '\t' + 'token_start' + '\n')
#         f1.write('-' * 10 + '\n')
#         index = 0
#         for item in content:
#             if item != "":
#                 words = item.strip().split("，")
#                 for word in words:
#                     nlp_res_sentence = nlp.annotate(word, properties=props_sentence)
#                     tokens, pos = get_pos(word, nlp_res_sentence)
#                     for i in range(len(tokens)):
#                         index = index + all_content[index:].find(tokens[i])
#                         if index != -1:
#                             f1.write(tokens[i] + "\t" + "token" + "\t" + pos[i] + "\t" + "None" + "\t" + str(index))
#                             f1.write("\n")
#                     f1.write("----------")
#                     f1.write("\n")
#
#     # os.system("python /home/feng/bert_code_1/EventExtract_trigger_cross_validation.py --iterations=0")

test_data_path = "./glue_data/EE/the_last_data/single_data/test.txt"
def read_txt_To_test_data(txt_path):

    with open(txt_path,"r",encoding="utf-8") as fr:
        all_content = fr.read()
        content = all_content.replace("\n", "").split("\n")
        print("content:",content)
    # os.system("mv " + path + " " + output_dir)

    nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05/')
    props_sentence = {'annotators': 'pos', 'pipelineLanguage': 'zh', 'outputFormat': 'text'}

    with open(test_data_path, 'w',encoding="utf-8")as f1:
        f1.write('sentence_id' + '\t' + '/' + '\t' + 'token' + '\t' + 'token_mark' + '\t' + 'pos' + '\t' + 'event_type' + '\t' + 'token_start' + '\n')
        f1.write('-' * 10 + '\n')
        index = 0
        for item in content:
            if item != "":
                words = item.strip().split("，")
                for word in words:
                    nlp_res_sentence = nlp.annotate(word, properties=props_sentence)
                    tokens, pos = get_pos(word, nlp_res_sentence)
                    for i in range(len(tokens)):
                        index = index + all_content[index:].find(tokens[i])
                        if index != -1:
                            f1.write(tokens[i] + "\t" + "token" + "\t" + pos[i] + "\t" + "None" + "\t" + str(index))
                            f1.write("\n")
                    f1.write("----------")
                    f1.write("\n")


def predict(txt_input_path, txt_output_path):
    read_txt_To_test_data(txt_input_path)
    os.system("python EventExtract_trigger_cross_validation.py --out_put_ann=" + str(txt_output_path))
    print("model predict success !")

#
# txt_input_path = "/home/feng/brat-master/data/ace_input/1.txt"
# txt_output_path = "/home/feng/brat-master/data/ace_input/1.ann"
# predict(txt_input_path, txt_output_path)