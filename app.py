from flask import Flask, jsonify, request
from flask_cors import CORS
from forecasting import model_predict, create_enhanced_model
import time
from data_offline import filtered_data 


print("starting model...")
enhanced_model = create_enhanced_model()
print("model trained!")
app = Flask(__name__)
CORS(app)
docs_data = filtered_data

@app.route("/")
def home_route():
    start_time = time.time()
    print("/ executed")
    response_data = {
        "status": 200,
        "message": "Forecasting API Basic ready to launch!"
    }
    
    end_time = time.time()
    response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
    
    return jsonify(response_data), 200

@app.route("/api/data/forecast", methods=["GET"])
def get_forecast_data():
    start_time = time.time()
    
    try:
        periods = request.args.get('periods', default=30, type=int)
        
        # Validate periods parameter
        if periods <= 0 or periods > 3000:
            response_data = {
                "status": 400,
                "message": "Periods must be between 1 and 365"
            }
            end_time = time.time()
            response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
            return jsonify(response_data), 400
        
        forecast = model_predict(periods, enhanced_model)
        
        if forecast is not None:
            # Convert forecast to a serializable format
            forecast_dict = forecast.to_dict('records')
            response_data = {
                "status": 200,
                "message": "Forecast data successfully returned",
                "periods": periods,
                "data": forecast_dict
            }
        else:
            response_data = {
                "status": 500,
                "message": "Forecast data unreachable or empty"
            }
    
    except Exception as e:
        response_data = {
            "status": 500,
            "message": f"Error generating forecast: {str(e)}"
        }
    
    end_time = time.time()
    response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
    
    return jsonify(response_data), response_data["status"]

@app.route("/api/data/real", methods=["GET"])
def get_real_data():
    start_time = time.time()
    
    try:
        page = request.args.get('page', default=1, type=int)
        total_data = request.args.get('total_data', default=10, type=int)
        
        # Validate parameters
        if page < 1:
            response_data = {
                "status": 400,
                "message": "Page number must be greater than 0"
            }
            end_time = time.time()
            response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
            return jsonify(response_data), 400
            
        if total_data < 1 or total_data > 50:
            response_data = {
                "status": 400,
                "message": "total_data must be between 1 and 50"
            }
            end_time = time.time()
            response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
            return jsonify(response_data), 400
        
        # Get fresh data
        data_list = docs_data
        total_items = len(data_list)
        
        # Calculate pagination
        start_index = (page - 1) * total_data
        end_index = start_index + total_data
        
        # Handle case when no data exists
        if total_items == 0:
            response_data = {
                "status": 200,
                "message": "No data available",
                "total": 0,
                "page": page,
                "total_data": total_data,
                "total_pages": 0,
                "data": []
            }
            end_time = time.time()
            response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
            return jsonify(response_data), 200
        
        # Handle invalid page numbers
        if start_index >= total_items:
            response_data = {
                "status": 400,
                "message": "Page number exceeds available data"
            }
            end_time = time.time()
            response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
            return jsonify(response_data), 400
        
        # Get paginated data
        paginated_data = data_list[start_index:end_index]
        
        response_data = {
            "status": 200,
            "message": "Data successfully returned",
            "total": total_items,
            "page": page,
            "total_data": total_data,
            "total_pages": (total_items + total_data - 1) // total_data,
            "data": paginated_data
        }
        
    except Exception as e:
        response_data = {
            "status": 500,
            "message": f"Error retrieving data: {str(e)}"
        }
    
    end_time = time.time()
    response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
    
    return jsonify(response_data), response_data.get("status", 500)
    start_time = time.time()
    
    try:
        # Check if request has JSON data
        if not request.is_json:
            response_data = {
                "status": 400,
                "message": "Request must contain JSON data"
            }
            end_time = time.time()
            response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
            return jsonify(response_data), 400
        
        json_data = request.get_json()
        
        # Validate JSON data
        if not json_data:
            response_data = {
                "status": 400,
                "message": "No JSON data provided"
            }
            end_time = time.time()
            response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
            return jsonify(response_data), 400
        
        # Ensure json_data is a list
        if not isinstance(json_data, list):
            response_data = {
                "status": 400,
                "message": "JSON data must be a list of items"
            }
            end_time = time.time()
            response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
            return jsonify(response_data), 400
        
        # Process each item
        processed_items = []
        for item in json_data:
            if not isinstance(item, dict):
                response_data = {
                    "status": 400,
                    "message": "Each item must be a dictionary"
                }
                end_time = time.time()
                response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
                return jsonify(response_data), 400
            
            # Check if 'time' field exists
            if 'time' not in item:
                response_data = {
                    "status": 400,
                    "message": "Each item must have a 'time' field"
                }
                end_time = time.time()
                response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
                return jsonify(response_data), 400
            
            # Add to Firestore
            doc_ref = firebase_db.collection("forecasting_data").document(str(item['time']))
            doc_ref.set(item)
            processed_items.append(item['time'])
        
        response_data = {
            "status": 200,
            "message": "Data added successfully",
            "processed_count": len(processed_items),
            "processed_items": processed_items
        }
        
    except Exception as e:
        response_data = {
            "status": 500,
            "message": f"An error occurred while processing data: {str(e)}"
        }
    
    end_time = time.time()
    response_data["duration_ms"] = round((end_time - start_time) * 1000, 2)
    
    return jsonify(response_data), response_data.get("status", 500)

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": 404,
        "message": "Endpoint not found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "status": 405,
        "message": "Method not allowed"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    print(error)
    return jsonify({
        "status": 500,
        "message": "Internal server error"
    }), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080)