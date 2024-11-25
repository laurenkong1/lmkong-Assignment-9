from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from neural_networks import visualize


app = Flask(__name__)

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle experiment parameters and trigger the experiment
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    try:
        activation = request.json.get('activation')
        lr = float(request.json.get('lr'))
        step_num = int(request.json.get('step_num'))

        # Validate input parameters
        if step_num <= 0:
            return jsonify({"error": "step_num must be greater than 0."}), 400
        if lr <= 0:
            return jsonify({"error": "Learning rate must be greater than 0."}), 400

        print(f"Running experiment with activation={activation}, lr={lr}, step_num={step_num}")

        # Run the visualization
        visualize(activation, lr, step_num)

        result_gif = "results/visualize.gif"
        if not os.path.exists(result_gif):
            return jsonify({"error": "Result GIF could not be generated."}), 500

        return jsonify({"result_gif": result_gif})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred.", "details": str(e)}), 500

# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)