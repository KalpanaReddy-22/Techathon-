from flask import Flask, request, jsonify
from gnn_model import GNNModel
from rl_agent import RLAgent
import torch

app = Flask(__name__)
gnn_model = GNNModel()
gnn_model.load_state_dict(torch.load('gnn_model.pth'))
gnn_model.eval()

rl_agent = RLAgent()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = torch.tensor(data['features'], dtype=torch.float)
        edges = torch.tensor(data['edges'], dtype=torch.long)
        gnn_data = {'x': features, 'edge_index': edges}
        pred = gnn_model(gnn_data).detach().numpy()
        action = rl_agent.predict(pred)
        return jsonify({'prediction': action.tolist()})
     except Exception as e:
        return jsonify({'error': str(e)}), 400
if __name__ == '__main__':
    app.run(debug=True)
