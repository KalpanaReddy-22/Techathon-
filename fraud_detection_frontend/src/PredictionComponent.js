import React, { useState } from 'react';
import axios from 'axios';

function PredictionComponent() {
    const [features, setFeatures] = useState([]);
    const [edges, setEdges] = useState([]);
    const [prediction, setPrediction] = useState(null);

    const handlePredict = async () => {
        try {
            const response = await axios.post('http://localhost:5000/predict', {
                features: features,
                edges: edges
            });
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error('Error making prediction:', error);
        }
    };

    return (
        <div>
            <h1>Fraud Detection</h1>
            <button onClick={handlePredict}>Predict</button>
            {prediction && <div>Prediction: {prediction}</div>}
        </div>
    );
}

export default PredictionComponent;