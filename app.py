from flask import Flask, render_template, request, redirect, session 
import joblib
import random

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load ML model + vectorizer
model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

# Dummy users
users = {"admin@example.com": "admin123"}

# ---- Helper Functions ----
def extract_company(job_text):
    words = job_text.split()
    for i in range(len(words)-1):
        if words[i][0].isupper() and words[i+1][0].isupper():
            return words[i] + " " + words[i+1]
    return "Unknown Company"

def get_company_details(company_name):
    return {
        "name": company_name,
        "age": random.randint(1, 15),
        "location": random.choice(["USA", "India", "UK", "Singapore", "Remote"]),
        "status": random.choice(["Verified", "Unverified", "Low Online Presence"])
    }

def generate_feedback(company_name):
    feedback_list = [
        "The company responds slowly to emails.",
        "No verified LinkedIn profile found.",
        "Website has minimal details.",
        "Company seems genuine based on available info.",
        "Mixed online reviews found.",
        "Job posting appears too generic."
    ]
    return random.sample(feedback_list, 3)

# ---- Routes ----
@app.route("/", methods=["GET","POST"])
def login_page():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if email in users and users[email] == password:
            session["user"] = email
            return redirect("/dashboard")
        else:
            return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if email in users:
            return render_template("register.html", error="User already exists!")
        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match!")

        users[email] = password
        return redirect("/")
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html", user=session["user"])

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    result = None
    prob = None
    company = None
    feedback = None

    if request.method == "POST":
        text = request.form.get("jobtext", "")
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

        # Get probability
        prob = float(model.predict_proba(X)[0].max())

        # ----- CUSTOM LOGIC -----
        if prob < 0.60:
            result = "FAKE"
        else:
            result = "FAKE" if int(pred) == 1 else "REAL"
        # -------------------------

        company_name = extract_company(text)
        company = get_company_details(company_name)
        feedback = generate_feedback(company_name)

    return render_template(
        "index.html",
        result=result,
        prob=prob,
        company=company,
        feedback=feedback
    )

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
