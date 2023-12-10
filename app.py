from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/recommend_product', methods=['GET'])
def recommend_product():
    try:
        # Your actual processing logic should go here
        # ...

        # For demonstration purposes, let's assume some processing and a successful response
        response = {
            'message': 'Recommendation generated successfully.',
            'status': 'success'
        }

        return jsonify(response), 200

    except Exception as e:
        # Handle exceptions or errors
        response = {
            'message': 'Error processing the request: {}'.format(str(e)),
            'status': 'error'
        }

        return jsonify(response), 500

if __name__ == '__main__':
    app.run(debug=True)
