from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
import json

app = Flask(__name__)
app.config["DEBUG"] = True

class Model:
    def __init__(self):
        self.model = SentenceTransformer('./model')

model = Model().model

def calculate_relevance_scores(sentences, keyword):
    # Encode keyword once for efficiency
    keyword_embedding = model.encode(keyword, convert_to_tensor=True)

    # Encode all sentences
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Calculate cosine similarity scores for all sentences
    cosine_scores = util.pytorch_cos_sim(sentence_embeddings, keyword_embedding)

    # Convert PyTorch tensor to a list of scores
    relevance_scores = cosine_scores.cpu().numpy().tolist()[0][0]
    return relevance_scores

def get_result(input_dict: dict) -> dict:
    score_ids = []
    keyword = input_dict['keyword']
    for key, value in input_dict['experiences'].items():
        score = 0
        sentences = value.split('\n')
        for sent in sentences:
            score += calculate_relevance_scores(sent, keyword)

        # score is the average of all sentences in a given experience
        score_ids.append((score/len(sentences), key))
    score_ids.sort(reverse=True)
    
    result = {
        "keyword": f"{keyword}",
        "most_relevant": f"{score_ids[0][1]}",
        "scores": score_ids,
    }

    return result

@app.route('/rel_score', methods=['POST'])
def get_score():
    try:
        input_dict = request.get_json()
        result = get_result(input_dict)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=2222)