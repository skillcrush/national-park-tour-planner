from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import logging
import requests
from datetime import datetime
import json
import os
import io
from langchain_openai import ChatOpenAI
from langchain.agents import create_json_chat_agent, AgentExecutor, tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import StructuredTool
from langchain import hub
from fuzzywuzzy import fuzz, process
from langchain_openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# app will run at: http://127.0.0.1:5000/

# Initialize logging
logging.basicConfig(filename="app.log", level=logging.INFO)
log = logging.getLogger("app")

# Initialize the Flask application
app = Flask(__name__)

# Initialize the OpenAI language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=4000)

# Configure the Flask application with the SQLite database
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", 'default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nature_nook.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create the database object
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Define the Park model
class Park(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    code = db.Column(db.String(10), unique=True, nullable=False)

# Define the User model for the database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Define the Trip model for the database
class Trip(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    trip_name = db.Column(db.String(150), nullable=False)
    location = db.Column(db.String(150), nullable=False)
    trip_start = db.Column(db.Date, nullable=False)
    trip_end = db.Column(db.Date, nullable=False)
    traveling_with = db.Column(db.String(150), nullable=False)
    lodging = db.Column(db.String(150), nullable=False)
    adventure = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    typical_weather = db.Column(db.String(150), nullable=True)
    itinerary = db.Column(db.Text, nullable=True)
    important_things_to_know = db.Column(db.Text, nullable=True)

    user = db.relationship('User', backref=db.backref('trips', lazy=True))

# Define the user loader function
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Define the route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

# Define the route for the logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Define the route for the signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created!', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

# Define the route for the home page
@app.route("/", methods=["GET"])
@login_required
def index():
    """Renders the main page."""
    return render_template("index.html", user=current_user)

# Define the route for the plan trip page
@app.route("/plan_trip", methods=["GET", "POST"])
@login_required
def plan_trip():
    """Renders the trip planning page or handles trip editing."""
    trip_id = request.args.get('trip_id')
    trip = None
    if trip_id:
        trip = Trip.query.get_or_404(trip_id)

    parks = Park.query.all()

    return render_template("plan-trip.html", parks=parks, user=current_user, trip=trip)

# Define function to get list of national parks
def get_parks():
    """Fetches the entire list of national parks and historic parks from the NPS API."""
    url = "https://developer.nps.gov/api/v1/parks"
    params = {
        "api_key": os.environ.get("NPS_API_KEY"),
        "limit": 75,  # Adjust this number based on the API's limit
        "start": 0,
        "designation": "National Park,National Historic Park",  # Filter by designations
    }
    parks = []
    while True:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            for park in data["data"]:
                designation = park.get("designation", "")
                if "National Park" in designation or "National Historic Park" in designation:
                    parks.append({"name": park["fullName"], "code": park["parkCode"]})
            if len(data["data"]) < params["limit"]:
                break
            params["start"] += params["limit"]
        else:
            break
    return parks


# Define the route for viewing the generated trip itinerary
@app.route("/view_trip", methods=["POST"])
@login_required
def view_trip():
    """Handles the form submission to view the generated trip itinerary."""    
    # Extract form data
    trip_id = request.form.get('trip_id')  # Retrieve trip_id from the form
    location = request.form["location-search"]
    trip_start_str = request.form["trip-start"]
    trip_end_str = request.form["trip-end"]
    trip_start = datetime.strptime(trip_start_str, '%Y-%m-%d').date()
    trip_end = datetime.strptime(trip_end_str, '%Y-%m-%d').date()
    traveling_with = ", ".join(request.form.getlist("traveling-with"))
    lodging = ", ".join(request.form.getlist("lodging"))
    adventure = ", ".join(request.form.getlist("adventure"))
    trip_name = request.form["trip-name"]

    # Create the input string with the user's unique trip information
    input_data = generate_trip_input(trip_name, location, trip_start_str, trip_end_str, traveling_with, lodging, adventure)
    
    # Create a tool for the agent to use that utilizes Wikipedia's run function
    wikipedia_tool = create_wikipedia_tool()

    # Define and register a custom tool for retrieving data from the National Park Service API
    nps_tool = create_nps_tool()

    # Pull a tool prompt template from the hub
    prompt = hub.pull("hwchase17/react-chat-json")

    # Create our agent that will utilize tools and return JSON
    agent = create_json_chat_agent(llm=llm, tools=[wikipedia_tool, nps_tool], prompt=prompt)

    # Create a runnable instance of the agent
    agent_executor = AgentExecutor(agent=agent, tools=[wikipedia_tool, nps_tool], verbose=True, handle_parsing_errors="Check your output and make sure it conforms to the expected JSON structure.")

    # Invoke the agent with the input data
    response = agent_executor.invoke({"input": input_data})

    output = response["output"]

    # Query the database for the existing trip or create a new one
    existing_trip = Trip.query.get(trip_id) if trip_id else None
    
    if existing_trip:
        # Update the existing trip
        existing_trip.trip_name = trip_name
        existing_trip.location = location
        existing_trip.trip_start = trip_start
        existing_trip.trip_end = trip_end
        existing_trip.traveling_with = traveling_with
        existing_trip.lodging = lodging
        existing_trip.adventure = adventure
        existing_trip.typical_weather = output["typical_weather"]
        existing_trip.itinerary = json.dumps(output["itinerary"])
        existing_trip.important_things_to_know = output["important_things_to_know"]
        db.session.commit()
        trip = existing_trip
    else:
        # Create a new trip
        new_trip = Trip(user_id=current_user.id, trip_name=trip_name, location=location, trip_start=trip_start, 
                        trip_end=trip_end, traveling_with=traveling_with, lodging=lodging, adventure=adventure,
                        typical_weather=output["typical_weather"], itinerary=json.dumps(output["itinerary"]),
                        important_things_to_know=output["important_things_to_know"])
        db.session.add(new_trip)
        db.session.commit()
        trip = new_trip

    log.info(response["output"])

    # Render the response on the view-trip.html page
    return render_template("view-trip.html", output=output, user=current_user, trip_id=trip.id)

# Define the route for viewing the saved trips
@app.route("/my_trips", methods=["GET"])
@login_required
def my_trips():
    """Renders the saved trips page."""
    trips = Trip.query.filter_by(user_id=current_user.id).all()
    return render_template("my-trips.html", trips=trips, user=current_user)

# Define the route for deleting a trip
@app.route("/delete_trip/<int:trip_id>", methods=["POST"])
@login_required
def delete_trip(trip_id):
    """Handles the deletion of a trip."""
    trip = Trip.query.get_or_404(trip_id)
    if trip.user_id != current_user.id:
        flash("You do not have permission to delete this trip.", "danger")
        return redirect(url_for('my_trips'))
    
    db.session.delete(trip)
    db.session.commit()
    flash("Trip deleted successfully.", "success")
    log.info("Trip deleted: %s", flash)
    return redirect(url_for('my_trips'))

# Define the route for viewing a saved trip
@app.route("/view_trip/<int:trip_id>", methods=["GET"])
@login_required
def view_saved_trip(trip_id):
    """Renders the detailed view for a saved trip."""
    trip = Trip.query.get_or_404(trip_id)
    output = {
        "trip_name": trip.trip_name,
        "location": trip.location,
        "trip_start": trip.trip_start,
        "trip_end": trip.trip_end,
        "typical_weather": trip.typical_weather,
        "traveling_with": trip.traveling_with,
        "lodging": trip.lodging,
        "adventure": trip.adventure,
        "itinerary": json.loads(trip.itinerary) if trip.itinerary else [],
        "important_things_to_know": trip.important_things_to_know
    }
    return render_template("view-trip.html", output=output, user=current_user, trip_id=trip.id)

# Define the route for the PDF download
@app.route("/download_pdf", methods=["POST"])
@login_required
def download_pdf():
    """Handles the PDF download of the generated trip itinerary."""
    output = request.json

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph(f"<b>Trip Name:</b> {output['trip_name']}", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Location:</b> {output['location']}", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Dates:</b> {output['trip_start']} - {output['trip_end']}", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Typical Weather:</b> {output['typical_weather']}", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Traveling With:</b> {output['traveling_with']}", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Lodging:</b> {output['lodging']}", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Activities:</b> {output['adventure']}", styles['Normal']))
    elements.append(Spacer(1, 24))

    elements.append(Paragraph("<b>Itinerary:</b>", styles['Normal']))
    elements.append(Spacer(1, 12))
    for day in output['itinerary']:
        elements.append(Paragraph(f"<b>Day {day['day']}:</b> {day['date']}", styles['Normal']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"<b>Morning:</b> {day['morning']}", styles['Normal']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"<b>Afternoon:</b> {day['afternoon']}", styles['Normal']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"<b>Evening:</b> {day['evening']}", styles['Normal']))
        elements.append(Spacer(1, 24))

    elements.append(Paragraph(f"<b>Important Things to Know:</b> {output['important_things_to_know']}", styles['Normal']))

    doc.build(elements)

    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="itinerary.pdf", mimetype='application/pdf')

# Define a function to generate the input string
def generate_trip_input(trip_name, location, trip_start, trip_end, traveling_with, lodging, adventure):
    """
    Generates a structured input string for the trip planning agent.
    """
    return f"""
    Create an itinerary for a trip by the name of "{trip_name}" to {location}.
    The trip starts on: {trip_start}
    The trip ends on: {trip_end}
    I will be traveling with {traveling_with}
    I would like to stay in {lodging}
    I would like to do the following activities: {adventure}

    Please generate a complete and detailed trip itinerary with the following JSON data structure:

    {{
      "trip_name": "String - Name of the trip",
      "location": "String - Location of the trip",
      "trip_start": "String - Start date of the trip",
      "trip_end": "String - End date of the trip",
      "typical_weather": "String - Description of typical weather for the trip",
      "traveling_with": "String - List of travel companions, verbatim from user",
      "lodging": "String - List of lodging arrangements, verbatim from user",
      "adventure": "String - List of planned activities, verbatim from user",
      "itinerary": [
        {{
          "day": "Integer - Day number",
          "date": "String - Date of this day",
          "morning": "String - Description of morning activities",
          "afternoon": "String - Description of afternoon activities",
          "evening": "String - Description of evening activities"
        }}
      ],
      "important_things_to_know": "String - Any important things to know about the park being visited."
    }}

    The trip should be appropriate for those listed as traveling, themed around the interests specified, and that last for the entire specified duration of the trip.
    Include realistic and varied activities for each day, considering the location, hours of operation, and typical weather.
    Make sure all itinerary fields are filled with appropriate and engaging content.
    Include descriptive information about each day's activities and destination.
    Respond only with a valid parseable JSON object representing the itinerary.
    """
    
# Define a function to create the Wikipedia tool
def create_wikipedia_tool():
    """
    Creates a built-in langchain tool for querying Wikipedia.
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return StructuredTool.from_function(
        func=wikipedia.run,
        name="Wikipedia",
        description="Useful for Wikipedia searches about national parks."
    )

# Define a function to create the National Park Service tool
def create_nps_tool():
    """
    Creates a custom tool for retrieving data from the National Park Service (NPS) API.  
    """
    base_url = "https://developer.nps.gov/api/v1"
    # Load your API key from an environment variable
    api_key = os.environ.get("NPS_API_KEY")

    # Define a function to fetch data from the NPS API
    def fetch_data(endpoint, params):
        """
        Fetches data from the NPS API given an endpoint and parameters.
        """
        url = f"{base_url}/{endpoint}"
        params['api_key'] = api_key
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Failed to fetch data from {endpoint}, status code: {response.status_code}"}

    # Define a function to search for parks by name using the NPS API
    def search_parks_by_name(park_name):
        """
        Searches for parks by name.
        """
        return fetch_data("parks", {"q": park_name}).get("data", [])

    # Define a function to find the best matching park
    def find_best_matching_park(park_name, parks):
        """
        Finds the best matching park using fuzzy search.
        """
        park_names = [park['fullName'] for park in parks]
        best_match_name, _ = process.extractOne(park_name, park_names, scorer=fuzz.partial_ratio)
        for park in parks:
            if park['fullName'] == best_match_name:
                return park
        return None

    # Define a function to find related data for a park
    def find_related_data_for_park(park):
        """
        Finds related data for a park from various NPS API endpoints.
        """
        park_code = park["parkCode"]
        endpoints = [
            "activities/parks" # Add more endpoints as needed. Be mindful of model input token limits these endpoints provide significant amounts of information that could exceed the context window. See https://www.nps.gov/subjects/developer/api-documentation.htm. 
        ]
        related_data = {endpoint: fetch_data(endpoint, {"parkCode": park_code}) for endpoint in endpoints}
        return related_data
    
    # Define a tool for searching for parks and related data
    @tool
    def search_park_and_related_data(input: str) -> str:
        """
        Searches for a park and finds related data.
        """
        park_name = input.strip()
        parks = search_parks_by_name(park_name)
        if parks:
            best_matching_park = find_best_matching_park(park_name, parks)
            if best_matching_park:
                combined_data = {
                    "park": best_matching_park,
                    "related_data": find_related_data_for_park(best_matching_park)
                }
            else:
                combined_data = {"error": f"Exact park named '{park_name}' not found in search results."}
        else:
            combined_data = {"error": f"Park named '{park_name}' not found."}
        return json.dumps(combined_data, indent=4)

    return search_park_and_related_data

# Create a Flask CLI command for initializing the database
# Run "flask init-db" from the command line  to initialize the database
@app.cli.command("init-db")
def init_db():
    db.create_all()
    parks = get_parks()
    for park in parks:
        existing_park = Park.query.filter_by(code=park["code"]).first()
        if not existing_park:
            new_park = Park(name=park["name"], code=park["code"])
            db.session.add(new_park)
    db.session.commit()
    print("Database initialized!")

# Run the Flask server
if __name__ == "__main__":
    app.run()
